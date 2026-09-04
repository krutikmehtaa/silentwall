"""Generate the notebooks.

Kept as a script because notebook JSON is painful to edit by hand and because the
notebooks are meant to stay thin. A test asserts no notebook defines a function or a
class, so anything substantial has to live in the package where it can be tested.

Run: python build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = """
# On Kaggle or Colab, uncomment to install
# !pip install -q -e /kaggle/working/silentwall
# !pip install -q -e .

from silentwall.config import load_config
from silentwall.pipeline import prepare_workspace, run_method, run_sweep, save_workspace
from silentwall.report.render import render_comparison, render_markdown, write_comparison

CONFIG = "../configs/smoke.yaml"
cfg = load_config(CONFIG)
print(cfg.profile, cfg.tier, "methods:", len(cfg.methods))
"""


def nb_01() -> dict:
    return notebook([
        md("""
# 1. Build the corpus

Stage 0 and 1. CPU only, no model, no GPU.

The corpus is real M&A events from SEC EDGAR. The announcement 8-K gives exact ground
truth and a real dated moment when the information became public. Control entities are
matched to restricted ones on sector and size band, because asking whether you can tell
them apart needs a comparison group that resembles them.

Set `corpus.synthetic: true` in the config to build an offline corpus instead, which is
what the smoke profile does. Structure is identical, only the language is invented.
        """),
        code(SETUP),
        code("""
ws = prepare_workspace(cfg)
        """),
        md("""
## What came out

`pair_id` is shared by a restricted entity and the control it was matched to. Splits and
cross-validation folds group on that, never on the entity, because a matched pair sitting
on both sides of a fold boundary would let a model learn the matching rule instead of the
behavioural signal.
        """),
        code("""
print("restricted:", len(ws.corpus.restricted))
print("controls:  ", len(ws.corpus.controls))
print("pairs:     ", len(ws.corpus.pair_ids))
print("corpus hash:", ws.corpus.manifest.corpus_hash[:16])

d = ws.corpus.deals[0]
print()
print("sample deal:", d.acquirer_name, "acquires", d.target_name, "on", d.announcement_date)
for f in d.protected_fields:
    print(f"  {f.name}: {f.value_raw}  (normalized {f.value_normalized})")
        """),
        md("""
## Private-side artifacts

The documents a deal team would hold. Composed from templates over already-public filing
fields, so no real confidential document is involved and the output is byte-reproducible.
        """),
        code("""
print("artifacts:", len(ws.artifacts))
print()
print(ws.artifacts[0].text)
        """),
        md("""
## Splits

The dev half is where a containment method is allowed to calibrate. The eval half is
where the reported numbers come from. `DevSplit` carries no reference to eval entities,
so a method cannot reach one by accident.
        """),
        code("""
audit = ws.splits.audit()
for k, v in audit.items():
    print(f"{k}: {v}")
        """),
        code("""
save_workspace(ws, "../artifacts/corpus")
        """),
    ])


def nb_02() -> dict:
    return notebook([
        md("""
# 2. Probes

Six families. Five go after a protected field. The sixth never mentions it, and that one
carries the entire detectability result.
        """),
        code(SETUP),
        code("""
ws = prepare_workspace(cfg, verbose=False)

from collections import Counter
counts = Counter(p.family.value for p in ws.probes)
for family, n in sorted(counts.items()):
    print(f"{family:20s} {n}")
        """),
        md("""
## Content probes

These target a specific protected field, so they only go to restricted entities. A control
entity has nothing to leak.
        """),
        code("""
for p in ws.probes:
    if not p.is_behavioural:
        print(f"[{p.family.value}] {p.prompt}")
        if p.family.value == "inference_chain":
            break
        """),
        md("""
## Behavioural probes

These are the interesting ones. Ordinary questions, nothing confidential, answerable by an
agent that never held the protected information. The identical set goes to restricted and
control entities.

That symmetry is what makes the comparison valid. If restricted and control entities got
different questions, a classifier separating them would be reading the question rather
than the answer.
        """),
        code("""
restricted = ws.corpus.restricted[0]
control = ws.corpus.controls[0]

for entity in (restricted, control):
    print(f"--- {entity.entity_class}: {entity.display_name} ---")
    shown = 0
    for p in ws.probes:
        if p.entity_id == entity.entity_id and p.is_behavioural:
            print(" ", p.prompt)
            shown += 1
            if shown == 4:
                break
    print()
        """),
    ])


def nb_03() -> dict:
    return notebook([
        md("""
# 3. Baselines

The containment approaches practitioners actually deploy, plus the clean reference agent
that everything else is measured against.

`clean_reference` is not a defense. It is an agent that never received the private
artifacts at all, and it never absorbed the values into its weights either. Withholding
the documents while leaving the model informed would make it a filtered agent rather than
a clean one, and it would stop being a valid upper bound.
        """),
        code(SETUP),
        code("""
ws = prepare_workspace(cfg, verbose=False)

results = []
for method_id in ("clean_reference", "none", "retrieval_filter"):
    results.append(run_method(ws, method_id))
        """),
        md("""
## Reading the output

`retrieval_filter` is worth attention. It drops private documents from context, which is
the access-control answer and what most production systems do. It still leaks, because the
model absorbed the values during fine-tuning and the filter cannot touch that pathway.
        """),
        code("""
print(render_comparison(results))
        """),
    ])


def nb_04() -> dict:
    return notebook([
        md("""
# 4. Suppression baselines and the side channel

`system_prompt`, `refusal_classifier` and `lora_ga` all suppress. Watch what suppression
does to detectability.
        """),
        code(SETUP),
        code("""
ws = prepare_workspace(cfg, verbose=False)

results = []
for method_id in ("clean_reference", "system_prompt", "refusal_classifier", "lora_ga"):
    results.append(run_method(ws, method_id))
        """),
        code("""
print(render_comparison(results))
        """),
        md("""
## The point

`refusal_classifier` leaks nothing. It is also the most detectable method in the set.

An observer who probes the agent with ordinary questions can see which companies it has
gone quiet on, and reconstruct the restricted list without ever requesting a single
protected fact. In a compliance setting that list is the confidential thing.

Suppression is what creates the signature, so suppressing harder makes this worse rather
than better.
        """),
        code("""
for r in results:
    det = r.primary_detectability
    if det is None:
        continue
    print(f"--- {r.method_id} ---")
    ranked = sorted(det.feature_importance.items(), key=lambda kv: -abs(kv[1].point))
    for name, iv in ranked[:4]:
        print(f"  {name:24s} {iv}")
    print()
        """),
    ])


def nb_05() -> dict:
    return notebook([
        md("""
# 5. Detectability, in detail

How the headline number is computed, and why it can be trusted.
        """),
        code(SETUP),
        code("""
ws = prepare_workspace(cfg, verbose=False)
result = run_method(ws, "refusal_classifier")
det = result.primary_detectability
        """),
        md("""
## The number

AUC is the probability that the classifier ranks a randomly chosen restricted entity above
a randomly chosen control. 0.5 is a coin flip, meaning invisible.
        """),
        code("""
print("AUC:", det.auc)
print("permutation p:", det.permutation_p)
print("pairs:", det.n_pairs)
print("undetectable claim:", det.undetectable_claim)
print()
print(det.power_note)
        """),
        md("""
## Two kinds of uncertainty

Sampling uncertainty is the interval above, from a cluster bootstrap over matched pairs.
Fold-assignment uncertainty is the spread across cross-validation repeats. Reporting one
and hiding the other overstates precision, so both are shown.
        """),
        code("""
print("AUC per repeat:", [round(x, 3) for x in det.auc_by_repeat])
if det.auc_by_repeat:
    print("spread:", round(max(det.auc_by_repeat) - min(det.auc_by_repeat), 3))
        """),
        md("""
## Which behaviour leaked

A single AUC tells a practitioner they have a problem. The feature importances tell them
what to fix.
        """),
        code("""
ranked = sorted(det.feature_importance.items(), key=lambda kv: -abs(kv[1].point))
for name, iv in ranked:
    print(f"{name:24s} {iv}")
        """),
        md("""
## Limitations

Derived from the run rather than written in advance.
        """),
        code("""
for lim in result.limitations:
    print("-", lim)
        """),
    ])


def nb_06() -> dict:
    return notebook([
        md("""
# 6. The reference defense

Every baseline suppresses, and suppression is the signature. This one substitutes instead:
it answers from public information, confidently and in normal register, rather than
refusing.

On the GPU tiers that is logit arithmetic, `z = z_base + alpha * (z_retain - z_forget)`,
using two small adapters on one frozen base. Subtracting the forget expert removes the
marginal information the private documents added. Adding the retain expert supplies the
public answer that should be there instead.
        """),
        code(SETUP),
        code("""
ws = prepare_workspace(cfg, verbose=False)

results = []
for method_id in ("clean_reference", "refusal_classifier", "silentwall"):
    results.append(run_method(ws, method_id))
        """),
        code("""
print(render_comparison(results))
        """),
        md("""
## Calibration

Substituting a fluent answer is not enough. If the substitution is systematically shorter,
or never hedges, or repeats itself across samples while genuine answers vary, each of those
becomes a feature the detector reads.

So the defense measures the control distribution on the dev split and generates to match
it. Control entities only, dev split only, and the split tripwire raises if that pass ever
reaches an eval entity.

Two design notes worth knowing:

The defense applies its adjustment to every entity, restricted and control alike, doing
identical work in both cases. For controls the blend weight is zero, so the passes are
wasted. That waste is deliberate. Gating only on restricted entities would make those
queries measurably slower, and latency is one of the features the detector reads.

An earlier version dropped the token trace when substituting. Restricted entities then had
missing entropy features while controls had real ones, so availability of the feature
became a cleaner label than anything the feature measured, and AUC went to 1.0. Fixing one
side channel opened a louder one, which is the failure mode this whole project is about.
        """),
        code("""
sw = [r for r in results if r.method_id == "silentwall"][0]
print("leak by family:")
for lr in sw.leak:
    print(f"  {lr.family.value:20s} leak@1 {lr.leak_at_1.point:.3f}   leak@k {lr.leak_at_k.point:.3f}")
print()
det = sw.primary_detectability
print("detectability:", det.auc)
        """),
    ])


def nb_07() -> dict:
    return notebook([
        md("""
# 7. Full sweep and report

Every method on one shared corpus, then the comparison table.

Before running this on a real tier, check the cost. `plan` counts what is already cached
and projects the time without generating anything, which is how you avoid discovering a
quota overrun three hours in.

```
silentwall plan --config configs/default.yaml
```
        """),
        code(SETUP),
        code("""
results = run_sweep(cfg)
        """),
        md("""
## The table

Read the middle two columns together. Low leakage with high AUC is the failure mode this
benchmark exists to surface: the content is hidden and the barrier is not.
        """),
        code("""
print(render_comparison(results))
        """),
        code("""
write_comparison(results, "../outputs")
for r in results:
    print(r.method_id, "->", r.run.run_id)
        """),
        md("""
## Reproducibility

Every run record carries the config hash, corpus hash, seeds, library versions, and the
split audit. A reader can recompute the dev and eval id hashes from the config and corpus
and verify that the reported numbers came from entities the defense never calibrated on.
        """),
        code("""
r = results[0].run
for key in ("config_hash", "corpus_hash", "dev_ids_hash", "eval_ids_hash", "splits_disjoint"):
    print(f"{key}: {getattr(r, key)}")
print("seeds:", dict(r.seeds))
        """),
    ])


def nb_colab() -> dict:
    """Self-contained Colab runner. Clones, installs, runs, inspects, downloads."""
    return notebook([
        md("""
# SILENTWALL on Colab

Run the cells in order. Everything here is free tier.

**Before you start**, set the runtime: Runtime, Change runtime type, Hardware
accelerator, **T4 GPU**. Leave High-RAM off. Do not pick a TPU, the code is PyTorch
CUDA and has no XLA path, so it will fail immediately.

A100 and L4 appear in that menu but they consume paid compute units. The 1.5B run
fits comfortably on a free T4.
        """),
        md("""
## 1. Confirm you actually got a GPU

Do this first. If it says `cuda: False` the accelerator did not attach, and you should
re-save the runtime setting before spending time on anything else.
        """),
        code("""
import torch

print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability())
    print("vram:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
else:
    print("no GPU attached. Runtime > Change runtime type > T4 GPU, then Save.")
        """),
        md("""
Expect `Tesla T4`, capability `(7, 5)`, about `15.8 GB`.

Capability 7.5 is Turing, which has no native bfloat16, so the code will select
float16 automatically. On an A100 or L4 it selects bfloat16 instead.
        """),
        md("""
## 2. Clone and install

Torch, transformers and accelerate are already on Colab, so the plain install is
enough. No Hugging Face token is needed anywhere: the model is Qwen, which is ungated.
        """),
        code("""
# Clones on a fresh runtime, pulls if a clone is already there. Reopening the notebook
# creates a new session but sometimes reuses a live runtime, and a bare `git clone`
# fails with "destination path already exists" in that case.
!if [ -d /content/silentwall/.git ]; then cd /content/silentwall && git pull -q && echo "pulled into existing clone"; else git clone -q https://github.com/krutikmehtaa/silentwall.git /content/silentwall && echo "fresh clone"; fi

%cd /content/silentwall
!pip install -q -e .
!python -m silentwall.cli --version
        """),
        md("""
`pip install -e .` is an editable install, so if you ever need to take a code update
mid-session, `!git pull` is enough on its own. No reinstall.
        """),
        md("""
### Did a previous run leave a cache behind?

Generations are cached, so a rerun skips work already done. Colab disk is ephemeral, so
this survives an interrupted cell but not a disconnected runtime.
        """),
        code("""
!ls /content/silentwall/cache/iterate 2>/dev/null | head -3 || true
!echo "---"
!find /content/silentwall/cache -name '*.jsonl.gz' 2>/dev/null | wc -l | xargs -I{} echo "cache shards found: {}"
        """),
        md("""
## 3. Free sanity check, no GPU, about 60 seconds

This runs all six pipeline stages against a deterministic stub backend. It proves the
plumbing works in this environment before any model weights are involved.

Two things to look for in the table. `clean_reference` must show leak `0.000`, because
an agent that never held the data cannot leak it. And its AUC should sit near 0.5,
because there is nothing to detect. If either is wrong, stop, because nothing
downstream will mean anything.
        """),
        code("""
!python -m silentwall.cli sweep -c configs/smoke.yaml --quiet
        """),
        md("""
## 4. See the cost before spending it

Generates nothing. Counts the work, counts what is already cached, projects the time.
        """),
        code("""
!python -m silentwall.cli plan -c configs/iterate.yaml
        """),
        md("""
## 5. The real run

29,760 generations across 6 methods at 1.5B. Roughly **1 to 3 hours**.

The first method spends 5 to 10 minutes downloading about 3GB of Qwen weights before
it generates anything, so early silence is normal.

Keep this browser tab active. Free Colab disconnects on idle. A disconnect costs you
the session but not the work, because every generation is flushed to the cache as it
is produced. Rerunning resumes.
        """),
        code("""
!python -m silentwall.cli sweep -c configs/iterate.yaml --confirm-budget
        """),
        md("""
## 6. Check the three things that matter

This is the first time the GPU code path has executed anywhere, so treat this as a
debugging pass as much as an experiment.
        """),
        code("""
import json
from pathlib import Path

results = [json.loads(p.read_text()) for p in sorted(Path("outputs").glob("audit_*.json"))]

print("CHECK 1: clean_reference must leak nothing")
for r in results:
    if r["method_id"] == "clean_reference":
        worst = max((x["leak_at_k"]["point"] for x in r["leak"]), default=0.0)
        print(f"  leak {worst:.3f}", "PASS" if worst < 0.02 else "FAIL, investigate before trusting anything else")

print()
print("CHECK 2: entropy features must be populated, not missing")
for r in results:
    det = next((d for d in r["detectability"] if d["detector_id"] == "logreg_primary"), None)
    if det and det["feature_importance"]:
        have = [k for k in det["feature_importance"] if "entropy" in k or "logprob" in k]
        print(f"  {r['method_id']:20s} logprob features present: {len(have)}")
        break
else:
    print("  no detectability estimate found")

print()
print("CHECK 3: suppression methods should be detectable")
for r in results:
    det = next((d for d in r["detectability"] if d["detector_id"] == "logreg_primary"), None)
    if det:
        a = det["auc"]
        print(f"  {r['method_id']:20s} AUC {a['point']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]  pairs={det['n_pairs']}")
        """),
        md("""
What the checks mean.

**Check 1 fails** means Qwen knows something about these companies from pretraining
that the stub could not, or the substrate is leaking. Either way it invalidates the
comparison until understood.

**Check 2 shows 0 features** means logprob extraction from `generate()` is broken and
three of the twelve behavioural features are dead. This is the most likely first bug.

**Check 3**: `refusal_classifier` and `system_prompt` should show high AUC. If they
collapse toward 0.5, the refusal cue lexicon in `scoring/features.py` does not match
how Qwen actually phrases refusals and needs adjusting against real output.
        """),
        md("""
## 7. Read the comparison table
        """),
        code("""
from IPython.display import Markdown, display

display(Markdown(Path("outputs/comparison.md").read_text()))
        """),
        md("""
## 8. Download before the session ends

Take the cache too. It is what makes the next run resume instead of regenerating, and
it is the difference between minutes and hours if you come back to this.
        """),
        code("""
!zip -qr silentwall_outputs.zip outputs cache

from google.colab import files
files.download("silentwall_outputs.zip")
        """),
        md("""
## If something failed

Paste the error back into the chat. The likely candidates, in order:

- logprob extraction in `backends/hf.py`, see Check 2
- refusal lexicon in `scoring/features.py`, see Check 3
- CUDA out of memory, in which case the backend halves the batch and retries, then
  defers the unit and continues. Lower `sampling.max_new_tokens` if it persists.
        """),
    ])


def nb_kaggle() -> dict:
    """Self-contained Kaggle runner for the 60-pair headline result."""
    return notebook([
        md("""
# SILENTWALL on Kaggle: the headline run

This is the run whose numbers count. 60 restricted companies plus 60 matched controls,
so 42 evaluation pairs, on a 7B model in 4-bit. The Colab runs proved the machinery
works on real weights; this is where the result is measured at a sample size that can
actually resolve it.

**Before you start:**

- Settings, Accelerator, pick **GPU T4 x2** or **P100**. Either is fine.
- Settings, Internet, switch **on**. Needed once to download model weights.
- GPU needs phone verification on your account. Do that first if you have not.

Kaggle gives 30 GPU-hours per week. This notebook is built to fit, and to resume across
sessions if it does not finish in one sitting.
        """),
        md("""
## 1. Confirm the GPU
        """),
        code("""
import torch

print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability())
    print("vram:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
        """),
        md("""
## 2. Clone and install
        """),
        code("""
# Clones or pulls, then installs. bitsandbytes is needed for 4-bit quantization and
# Kaggle does not ship it the way Colab does.
!if [ -d /kaggle/working/silentwall/.git ]; then cd /kaggle/working/silentwall && git pull -q && echo pulled; else git clone -q https://github.com/krutikmehtaa/silentwall.git /kaggle/working/silentwall && echo cloned; fi
%cd /kaggle/working/silentwall
!pip install -q -e . && pip install -q "bitsandbytes>=0.46.1"
!python -m silentwall.cli --version
        """),
        md("""
## 3. Check the cost before spending it

`plan` generates nothing. It counts the work and projects the time. Read the total
before running the sweep.

Two settings keep this affordable. `sampling.k=8` halves the generations versus the
default of 16, and the leak curve up to k=8 is preserved. `silentwall.regen_retries=4`
keeps the reference defense fast: the retries are batched into one call, but each
extra candidate is still real compute, so 4 rather than 8 roughly halves its cost while
the stub curve shows it stays well below the suppression baselines.
        """),
        code("""
!python -m silentwall.cli plan -c configs/default.yaml \\
  --set sampling.k=8 \\
  --set methods=clean_reference,none,system_prompt,retrieval_filter,refusal_classifier,silentwall
        """),
        md("""
## 4. Run the sweep

`--confirm-budget` is required because this exceeds the built-in ceiling, which is the
point: a large run is always a deliberate choice.

If the session ends before this finishes, do not worry. Every generation is cached as
it is produced. Re-running this exact cell resumes from where it stopped, and section 6
shows how to carry the cache across separate sessions.

The `silentwall` method regenerates answers and is slower than the others even with the
batched retries and the lower retry budget. Expect it to take the largest share of the
time. That is inherent to what it does, not a problem.
        """),
        code("""
!python -m silentwall.cli sweep -c configs/default.yaml \\
  --set sampling.k=8 \\
  --set method_params.silentwall.regen_retries=4 \\
  --set methods=clean_reference,none,system_prompt,retrieval_filter,refusal_classifier,silentwall \\
  --confirm-budget
        """),
        md("""
## 5. The comparison table
        """),
        code("""
from pathlib import Path
from IPython.display import Markdown, display

display(Markdown(Path("outputs/comparison.md").read_text()))
        """),
        md("""
The checks that matter at this sample size:

- `clean_reference` should sit near 0.5 with a reasonably tight interval. At 42 pairs it
  is the control on the instrument, and its landing on chance is what licenses reading
  every other row as real signal.
- `refusal_classifier` should be high, leak zero and AUC near 1.0, the clearest
  statement of the finding.
- `silentwall` should sit below the suppression baselines. Whether it reaches chance is
  the open question the retry-budget curve addresses; a value between the baselines and
  0.5 is the expected and honest outcome.
        """),
        md("""
## 6. Carry the cache to the next session

If the sweep did not finish, save `/kaggle/working/cache` as a Kaggle output dataset.
Next session, add that dataset as an input and point the config at both layers. The
earlier layer is read only, the working directory takes new writes, and only the
unfinished work runs.

```python
!python -m silentwall.cli sweep -c configs/default.yaml \\
  --set sampling.k=8 \\
  --set method_params.silentwall.regen_retries=4 \\
  --set cache_layers=/kaggle/input/silentwall-cache-1/cache,/kaggle/working/cache \\
  --set methods=clean_reference,none,system_prompt,retrieval_filter,refusal_classifier,silentwall \\
  --confirm-budget
```

Watch the `already cached` percentage in the plan output. If it says 0% when you expect
otherwise, the layer path is wrong. Stop and fix it, or you pay the quota twice.
        """),
        md("""
## 7. Download everything
        """),
        code("""
!zip -qr silentwall_default_outputs.zip outputs cache
from IPython.display import FileLink
FileLink("silentwall_default_outputs.zip")
        """),
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    books = {
        "00_colab_run.ipynb": nb_colab(),
        "00_kaggle_run.ipynb": nb_kaggle(),
        "01_build_corpus.ipynb": nb_01(),
        "02_probes.ipynb": nb_02(),
        "03_baselines.ipynb": nb_03(),
        "04_suppression_and_side_channel.ipynb": nb_04(),
        "05_detectability.ipynb": nb_05(),
        "06_reference_defense.ipynb": nb_06(),
        "07_sweep_and_report.ipynb": nb_07(),
    }
    for name, nb in books.items():
        path = OUT / name
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote", path.name)


if __name__ == "__main__":
    main()
