"""Hugging Face backend for the real model tiers.

Imports of torch and transformers are deferred to construction so that importing
silentwall on a machine with no GPU stack still works. That matters because the test
suite and CI never touch this file's dependencies.

Note on typing: transformers ships incomplete annotations for generate() and the
tokenizer call, so strict mode flags calls that are correct at runtime. The ignores in
this file are all of that kind and each names the specific code rather than blanket
silencing the module. This is the only file in the package that needs them, which is a
direct consequence of keeping the model interaction behind one boundary.

The throughput lever here is num_return_sequences: k samples for one probe share a
single prompt encoding and one forward pass over the prompt, which is most of the
cost at short generation lengths.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from ..errors import BackendOOMError, ConfigError
from ..types import Generation, TokenTrace
from .base import BaseBackend, GenerationRequest

__all__ = ["HfBackend", "TIER_SETTINGS"]

#: Quantization per tier. The 8B tier uses 4-bit NF4 because that is what makes a 7B to
#: 8B model fit alongside activations in 16GB of free-tier accelerator.
#:
#: dtype is resolved at load time rather than fixed here, see pick_dtype. Hardcoding
#: bfloat16 was a bug: a free Colab or Kaggle T4 is Turing, which has no native bf16, so
#: it either falls back to slow paths or makes bitsandbytes complain. The correct dtype
#: depends on the card you actually got, which is not knowable when this table is
#: written.
TIER_SETTINGS: dict[str, dict[str, Any]] = {
    "cpu-0p5b": {"quantize": None, "dtype": "float32", "device": "cpu"},
    "gpu-1p5b": {"quantize": None, "dtype": "auto", "device": "cuda"},
    "gpu-8b-nf4": {"quantize": "nf4", "dtype": "auto", "device": "cuda"},
}


def pick_dtype(torch_mod: Any, requested: str) -> tuple[Any, str]:
    """Choose a compute dtype the current device can actually run.

    bfloat16 needs compute capability 8.0 or newer, meaning Ampere and later. That
    covers A100, L4, and the Ada and Hopper cards. It does not cover the T4 that free
    Colab and Kaggle hand out, which is Turing at 7.5, so float16 is the right answer
    there.

    Returns the torch dtype and a human readable name for logging.
    """
    if requested != "auto":
        return getattr(torch_mod, requested), requested

    if not torch_mod.cuda.is_available():
        return torch_mod.float32, "float32"

    major, minor = torch_mod.cuda.get_device_capability()
    if major >= 8:
        return torch_mod.bfloat16, "bfloat16"
    return torch_mod.float16, "float16"


class HfBackend(BaseBackend):
    def __init__(
        self,
        tier: str,
        model_id: str | None = None,
        batch_size: int = 8,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        if tier not in TIER_SETTINGS:
            raise ConfigError(
                f"tier {tier!r} has no HF settings, expected one of {sorted(TIER_SETTINGS)}"
            )

        from ..config import TIER_MODELS

        self.tier = tier
        self.model_id = model_id or TIER_MODELS[tier]
        self.settings = TIER_SETTINGS[tier]
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self.dtype_name = "unknown"
        self._logit_processors: list[Any] = []

        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        want_cuda = self.settings["device"] == "cuda"
        if want_cuda and not torch.cuda.is_available():
            raise ConfigError(
                f"tier {self.tier} needs a CUDA device but none is visible. "
                f"Use tier cpu-0p5b locally, or stub for tests."
            )

        print(f"loading tokenizer for {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        kw: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        dtype, dtype_name = pick_dtype(torch, self.settings["dtype"])
        self.dtype_name = dtype_name

        if want_cuda:
            name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability()
            print(f"gpu: {name}, compute capability {major}.{minor}, using {dtype_name}")
            if major < 8 and self.settings["quantize"] == "nf4":
                print(
                    "note: this card predates Ampere, so 4-bit matmuls run without native "
                    "bf16 support and throughput will be lower than the projection assumes"
                )

        if self.settings["quantize"] == "nf4":
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            kw["device_map"] = "auto"
            print("loading model in 4-bit, first run downloads weights so give it a few minutes")
        else:
            kw["torch_dtype"] = dtype
            if want_cuda:
                kw["device_map"] = "auto"
            print(f"loading model in {dtype_name}")

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kw)
        self.model.eval()

        if torch.cuda.is_available() and want_cuda:
            used = torch.cuda.memory_allocated() / 1e9
            print(f"model ready, vram used {used:.2f} GB")
        else:
            print("model ready on cpu")

        self.set_fingerprint_extra("quantize", self.settings["quantize"])
        # The resolved dtype, not the requested one. fp16 and bf16 give different
        # generations, so a cache built on a T4 must not be reused on an A100.
        self.set_fingerprint_extra("dtype", dtype_name)
        self.set_fingerprint_extra("tokenizer", getattr(self.tokenizer, "name_or_path", ""))

    def supports_logprobs(self) -> bool:
        return True

    def add_logit_processor(self, processor: Any) -> None:
        """Attach a logit processor, used by the reference defense."""
        self._logit_processors.append(processor)
        self.set_fingerprint_extra("n_logit_processors", len(self._logit_processors))

    def clear_logit_processors(self) -> None:
        self._logit_processors.clear()
        self.set_fingerprint_extra("n_logit_processors", 0)

    def _chat(self, req: GenerationRequest) -> str:
        """Render through the model's chat template when it has one."""
        messages = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        user = req.prompt
        if req.context_docs:
            user = "Context:\n" + "\n---\n".join(req.context_docs) + "\n\nQuestion:\n" + user
        if req.tools_exposed:
            user = f"Tools available: {', '.join(sorted(req.tools_exposed))}\n\n" + user
        messages.append({"role": "user", "content": user})

        if getattr(self.tokenizer, "chat_template", None):
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # tokenize=False returns a single string, but the annotation is a union
            return rendered if isinstance(rendered, str) else str(rendered)
        return req.full_prompt()

    def generate(self, requests: Sequence[GenerationRequest]) -> list[Generation]:
        out: list[Generation] = []
        bs = self.batch_size
        i = 0
        while i < len(requests):
            chunk = list(requests[i : i + bs])
            try:
                out.extend(self._generate_batch(chunk))
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or bs == 1:
                    raise BackendOOMError(str(exc)) from exc
                self._torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"hit oom, dropping batch size to {bs} and retrying")
                continue
            i += len(chunk)
        return out

    def _generate_batch(self, reqs: list[GenerationRequest]) -> list[Generation]:
        torch = self._torch
        from transformers import LogitsProcessorList

        prompts = [self._chat(r) for r in reqs]
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        sampling = reqs[0].sampling
        # One seed per batch. Per-sample determinism still holds because sample_index
        # feeds the request seed and each sample_index lands in its own request.
        torch.manual_seed(reqs[0].seed)

        processors = LogitsProcessorList(self._logit_processors) if self._logit_processors else None

        started = time.perf_counter()
        # return_dict_in_generate=True guarantees a ModelOutput with .sequences and
        # .scores, but the declared return type is a union that includes a bare tensor,
        # so the attribute access below cannot be proven safe by the checker.
        # getattr rather than attribute access: the transformers class hierarchy
        # declares generate on a mixin whose self type does not match the loaded
        # model class, so the bound method access itself is what fails to check.
        generate: Any = getattr(self.model, "generate")  # noqa: B009
        with torch.no_grad():
            result: Any = generate(
                **enc,
                do_sample=sampling.temperature > 0,
                temperature=max(1e-5, sampling.temperature),
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                max_new_tokens=sampling.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=processors,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0 / max(1, len(reqs))

        prompt_len = enc["input_ids"].shape[1]
        sequences = result.sequences[:, prompt_len:]
        scores = result.scores  # tuple over steps, each (batch, vocab)

        gens: list[Generation] = []
        for row, req in enumerate(reqs):
            token_ids = [int(t) for t in sequences[row] if int(t) != self.tokenizer.pad_token_id]
            decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            text = (decoded if isinstance(decoded, str) else str(decoded)).strip()
            trace = self._extract_trace(
                scores, sequences, row, len(token_ids), sampling.topm_logprobs
            )

            gens.append(
                Generation(
                    cache_key="",
                    probe_id=req.probe_id,
                    entity_id=req.entity_id,
                    sample_index=req.sample_index,
                    seed=req.seed,
                    text=text,
                    n_tokens=len(token_ids),
                    backend_fp=self.fingerprint(),
                    containment_fp="",
                    trace=trace,
                    tool_calls=tuple(_parse_tool_calls(text)),
                    latency_ms=elapsed_ms,
                    latency_trustworthy=True,
                )
            )
        return gens

    def _extract_trace(
        self, scores: Any, sequences: Any, row: int, n_tokens: int, topm: int
    ) -> TokenTrace | None:
        if not scores or n_tokens == 0:
            return None
        torch = self._torch
        chosen: list[float] = []
        topm_rows: list[tuple[float, ...]] = []

        for step in range(min(n_tokens, len(scores))):
            logits = scores[step][row]
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            tok = int(sequences[row, step])
            chosen.append(float(logprobs[tok]))
            top = torch.topk(logprobs, k=min(topm, logprobs.shape[-1]))
            topm_rows.append(tuple(float(x) for x in top.values))

        ids = tuple(int(sequences[row, s]) for s in range(min(n_tokens, len(scores))))
        return TokenTrace(
            token_ids=ids, chosen_logprobs=tuple(chosen), topm_logprobs=tuple(topm_rows)
        )


def _parse_tool_calls(text: str) -> list[str]:
    """Detect tool invocations in free text.

    Deliberately simple. The tool families use an explicit call syntax in the prompt,
    and over-detecting here would inflate the tool features.
    """
    found: list[str] = []
    lowered = text.lower()
    for name in ("search_docs", "calc"):
        idx = lowered.find(name + "(")
        while idx >= 0:
            found.append(name)
            idx = lowered.find(name + "(", idx + 1)
    return found
