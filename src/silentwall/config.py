"""Configuration loading.

Tunable numbers live in YAML under configs/, never inline in code. That is what
makes a run record meaningful: the config hash plus the corpus hash plus the seeds
pin down every number in the report.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from .errors import ConfigError
from .hashing import hash_obj

__all__ = [
    "ModelTier",
    "Profile",
    "SamplingConfig",
    "CorpusConfig",
    "SplitConfig",
    "StatsConfig",
    "SilentwallConfig",
    "load_config",
    "config_hash",
]

ModelTier = Literal["stub", "cpu-0p5b", "gpu-1p5b", "gpu-8b-nf4"]
Profile = Literal["smoke", "iterate", "default"]

#: Model id per tier. Qwen is deliberate: ungated on the Hub, so a run is never
#: blocked waiting on an access approval queue.
TIER_MODELS: dict[str, str] = {
    "stub": "stub://deterministic",
    "cpu-0p5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "gpu-1p5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "gpu-8b-nf4": "Qwen/Qwen2.5-7B-Instruct",
}


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    k: int = 16
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 128
    base_seed: int = 20260101
    topm_logprobs: int = 5

    def validate(self) -> None:
        if self.k < 1:
            raise ConfigError("sampling.k must be at least 1")
        if not 0.0 < self.temperature <= 4.0:
            raise ConfigError("sampling.temperature must be in (0, 4]")
        if not 0.0 < self.top_p <= 1.0:
            raise ConfigError("sampling.top_p must be in (0, 1]")
        if self.max_new_tokens < 1:
            raise ConfigError("sampling.max_new_tokens must be at least 1")


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    quarters: tuple[str, ...] = ("2019Q1",)
    form_types: tuple[str, ...] = ("8-K",)
    target_restricted: int = 60
    control_ratio: float = 1.0
    size_metric: str = "Revenues"
    sic_match_digits: int = 2
    user_agent: str = ""
    synthetic: bool = False
    synthetic_seed: int = 7
    max_fetch_failure_rate: float = 0.25

    def validate(self) -> None:
        if self.target_restricted < 1:
            raise ConfigError("corpus.target_restricted must be at least 1")
        if self.control_ratio <= 0:
            raise ConfigError("corpus.control_ratio must be positive")
        if not self.synthetic and not self.user_agent.strip():
            raise ConfigError(
                "corpus.user_agent is required for live EDGAR access. "
                "SEC asks for a descriptive string with a contact address, "
                "for example 'SilentWall research you@example.com'. "
                "Set corpus.synthetic: true to build an offline corpus instead."
            )


@dataclass(frozen=True, slots=True)
class SplitConfig:
    split_seed: int = 991
    dev_fraction: float = 0.5

    def validate(self) -> None:
        if not 0.0 < self.dev_fraction < 1.0:
            raise ConfigError("split.dev_fraction must be strictly between 0 and 1")


@dataclass(frozen=True, slots=True)
class StatsConfig:
    bootstrap_resamples: int = 10_000
    cv_folds: int = 5
    cv_repeats: int = 10
    permutation_draws: int = 2_000
    fdr_q: float = 0.10
    ci_level: float = 0.95
    undetectable_threshold: float = 0.60
    leak_curve_k: tuple[int, ...] = (1, 2, 4, 8, 16)

    def validate(self) -> None:
        if self.cv_folds < 2:
            raise ConfigError("stats.cv_folds must be at least 2")
        if self.bootstrap_resamples < 100:
            raise ConfigError("stats.bootstrap_resamples must be at least 100")
        if not 0.0 < self.fdr_q < 1.0:
            raise ConfigError("stats.fdr_q must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class SilentwallConfig:
    profile: str = "smoke"
    tier: str = "stub"
    corpus: CorpusConfig = CorpusConfig()
    sampling: SamplingConfig = SamplingConfig()
    split: SplitConfig = SplitConfig()
    stats: StatsConfig = StatsConfig()
    methods: tuple[str, ...] = ("none",)
    #: Per-method constructor overrides, keyed by method id. Lets a config or a CLI
    #: override tune one method without a bespoke flag, for example
    #: method_params.silentwall.regen_retries. Kept a plain mapping rather than typed
    #: dataclasses because each method takes different parameters.
    method_params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    cache_layers: tuple[Path, ...] = (Path("cache"),)
    artifacts_dir: Path = Path("artifacts")
    data_dir: Path = Path("data")
    logs_dir: Path = Path("logs")
    max_generations: int = 5_000
    behavioural_templates: int = 8
    content_probes_per_family: int = 3
    version: str = "0.1.0"
    template_pack_version: str = "tp1"

    def validate(self) -> None:
        if self.tier not in TIER_MODELS:
            raise ConfigError(f"unknown tier {self.tier!r}, expected one of {sorted(TIER_MODELS)}")
        if not self.methods:
            raise ConfigError("methods must not be empty")
        if not self.cache_layers:
            raise ConfigError("cache_layers must not be empty")
        if self.max_generations < 1:
            raise ConfigError("max_generations must be at least 1")
        self.corpus.validate()
        self.sampling.validate()
        self.split.validate()
        self.stats.validate()

    @property
    def model_id(self) -> str:
        return TIER_MODELS[self.tier]

    @property
    def writable_cache(self) -> Path:
        """Last layer is the writable overlay. Earlier layers are read-only."""
        return self.cache_layers[-1]

    def with_overrides(self, **kw: Any) -> SilentwallConfig:
        return replace(self, **kw)


_SECTION_TYPES: dict[str, type] = {
    "corpus": CorpusConfig,
    "sampling": SamplingConfig,
    "split": SplitConfig,
    "stats": StatsConfig,
}

_TUPLE_FIELDS = {"quarters", "form_types", "methods", "cache_layers", "leak_curve_k"}
_PATH_FIELDS = {"artifacts_dir", "data_dir", "logs_dir"}


def _build_section(name: str, raw: Mapping[str, Any]) -> Any:
    cls = _SECTION_TYPES[name]
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown keys in section {name!r}: {sorted(unknown)}")
    kw: dict[str, Any] = {}
    for key, value in raw.items():
        kw[key] = tuple(value) if key in _TUPLE_FIELDS and isinstance(value, list) else value
    return cls(**kw)


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> SilentwallConfig:
    """Load a config file, apply dotted overrides, validate, return it frozen.

    Overrides use dotted keys so the CLI can pass things like
    ``--set sampling.k=4`` without knowing the dataclass layout.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")

    raw = dict(raw)
    for dotted, value in (overrides or {}).items():
        _apply_dotted(raw, dotted, value)

    top_known = {f.name for f in fields(SilentwallConfig)}
    unknown = set(raw) - top_known
    if unknown:
        raise ConfigError(f"unknown top-level config keys: {sorted(unknown)}")

    kw: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _SECTION_TYPES:
            if not isinstance(value, dict):
                raise ConfigError(f"section {key!r} must be a mapping")
            kw[key] = _build_section(key, value)
        elif key == "cache_layers":
            kw[key] = tuple(Path(p) for p in _as_list(key, value))
        elif key in _PATH_FIELDS:
            kw[key] = Path(value)
        elif key in _TUPLE_FIELDS:
            kw[key] = tuple(_as_list(key, value))
        else:
            kw[key] = value

    cfg = SilentwallConfig(**kw)
    cfg.validate()
    return cfg


def _as_list(key: str, value: Any) -> list[Any]:
    """Coerce a config value into a list for a field that must be a sequence.

    A bare string is accepted and split on commas, because
    ``--set methods=none,silentwall`` from a shell is far less error prone than
    getting a quoted JSON array past PowerShell. Without this branch a string would
    fall through and be iterated character by character, which passes validation and
    then fails much later with a baffling error about a method named "n".
    """
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            raise ConfigError(f"{key} was given an empty value")
        return parts
    if isinstance(value, list | tuple):
        return list(value)
    raise ConfigError(
        f"{key} must be a list or a comma separated string, got {type(value).__name__}"
    )


def _apply_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = target
    for part in parts[:-1]:
        nxt = node.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise ConfigError(f"cannot descend into {part!r} while applying override {dotted!r}")
        node = nxt
    node[parts[-1]] = _coerce(value)


def _coerce(value: Any) -> Any:
    """Turn a CLI string into the obvious Python value, leave anything else alone."""
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def config_hash(cfg: SilentwallConfig) -> str:
    """Content hash over everything that affects results.

    Directory locations are excluded on purpose: moving the cache to a different
    path does not change what the numbers mean, and including it would make every
    Kaggle session look like a different configuration.
    """
    return hash_obj(
        cfg.profile,
        cfg.tier,
        cfg.corpus,
        cfg.sampling,
        cfg.split,
        cfg.stats,
        sorted(cfg.methods),
        cfg.method_params,
        cfg.max_generations,
        cfg.behavioural_templates,
        cfg.content_probes_per_family,
        cfg.version,
        cfg.template_pack_version,
    )
