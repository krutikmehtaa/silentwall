# SILENTWALL

**Your information barrier works. Can anyone tell it is there?**

When an LLM agent handles confidential information, some facts must reach one internal
audience and not another. A deal team holds merger details; the trading desk must not.
Teams implement this with retrieval filters, system-prompt rules, and refusals.

Nobody checks whether the barrier itself is visible. If the agent noticeably changes
behaviour on restricted companies, an observer can locate the barrier without ever
seeing protected content, and thereby recover the restricted list. In a financial
compliance setting that list is itself the confidential thing.

SILENTWALL measures that. Point it at an agent, a restricted entity list, and a matched
control list, and it answers one question:

> Can a classifier separate restricted from control entities using only the agent's
> observable behaviour on neutral questions?

The answer is an AUC. 0.5 means the barrier is invisible. 1.0 means an observer can read
your restricted list straight off the agent's behaviour.

## Results

Headline numbers from the full run: **Qwen2.5-7B-Instruct in 4-bit NF4** on an A100,
60 restricted companies, 60 matched controls, 42 evaluation pairs, k=8 samples per probe.

| method | worst-family leak@k | detectability AUC | verdict |
|---|---|---|---|
| `clean_reference` | 0.000 | 0.767 [0.667, 0.863] | model-prior floor |
| `none` | 0.872 | 0.783 [0.683, 0.876] | leaks everything |
| `system_prompt` | 0.872 | 0.967 [0.926, 0.991] | leaks and detectable |
| `retrieval_filter` | 0.000 | 0.723 [0.609, 0.829] | contained, barrier visible |
| `refusal_classifier` | 0.000 | **1.000** [1.000, 1.000] | contained, perfectly visible |
| `silentwall` | 0.000 | 0.811 [0.710, 0.896] | contained, barrier visible |

Read the middle two columns together. Low leakage with high AUC is the failure mode
this benchmark exists to surface: the content is hidden and the barrier is not.

**`refusal_classifier`** achieves perfect containment (zero leak across all five probe
families) and is perfectly detectable (AUC 1.000 with interval [1.000, 1.000]). An
observer needs only ordinary, non-confidential questions to reconstruct the restricted
list. The top features driving detection are `refusal_rate` and `length_mean`, both
direct artifacts of suppression behaviour.

**`clean_reference`** at 0.767 shows that Qwen 7B carries enough incidental knowledge
about entity types to create a baseline separation even without private information.
This is itself a finding: larger models raise the detectability floor through parametric
priors alone. The 1.5B model showed 0.460 and the stub showed 0.521, both near chance.

**`silentwall`** at 0.811 sits between the clean reference floor (0.767) and the
suppression baselines (0.967–1.000). The defense eliminated the refusal signature
(`refusal_rate` = 0.000) and shifted the residual signal to `length_cv` and
`token_entropy`, which are artifacts of the censored-distribution effect from
regeneration.

Full audit reports with feature importances, confidence intervals, and per-family
breakdowns are in [`results/`](results/).

### Validation run (1.5B, Colab free tier)

A separate run at Qwen2.5-1.5B-Instruct on a free T4 confirmed the pattern at a smaller
scale. Results are in [`results/colab_1p5b/`](results/colab_1p5b/).

## Quick start

```bash
git clone https://github.com/krutikmehtaa/silentwall.git
cd silentwall
pip install -e ".[dev]"
```

Core install needs only numpy, scipy, scikit-learn, PyYAML and requests. For GPU tiers
add `pip install -e ".[gpu]"` and `pip install bitsandbytes>=0.46.1`.

### Smoke test (no GPU, seconds)

```bash
silentwall sweep --config configs/smoke.yaml
```

Runs all six pipeline stages against a deterministic stub backend. Produces a full
comparison table. This is how you verify the plumbing before spending compute.

### Plan a real run

```bash
silentwall plan --config configs/default.yaml
```

Shows the generation count, cache hits, and projected time without generating anything.

### Full sweep

```bash
silentwall sweep --config configs/default.yaml --confirm-budget
```

### Other commands

```bash
silentwall corpus --config configs/default.yaml   # stages 0–1 only, CPU
silentwall audit --method silentwall -c configs/iterate.yaml
silentwall methods                                # list containment methods
```

## Reproduce the headline result

The easiest path is the Colab Pro notebook, which runs the full sweep in about 3 hours
on an A100:

1. Open [`notebooks/00_colab_pro_run.ipynb`](https://colab.research.google.com/github/krutikmehtaa/silentwall/blob/main/notebooks/00_colab_pro_run.ipynb)
2. Set runtime to **A100 GPU**
3. Run all cells top to bottom

The notebook clones this repo, installs, runs the stub sanity check, then the full
6-method sweep at 60 companies on Qwen 7B 4-bit. Results download as a zip at the end.

For free-tier Colab (T4), use [`notebooks/00_colab_run.ipynb`](https://colab.research.google.com/github/krutikmehtaa/silentwall/blob/main/notebooks/00_colab_run.ipynb)
which runs the 1.5B tier. For Kaggle, use [`notebooks/00_kaggle_run.ipynb`](notebooks/00_kaggle_run.ipynb).

## Compute tiers

| tier | model | hardware | cost |
|---|---|---|---|
| `stub` | deterministic, no weights | any CPU | free, seconds |
| `gpu-1p5b` | Qwen2.5-1.5B-Instruct | free Colab T4 | free |
| `gpu-8b-nf4` | Qwen2.5-7B-Instruct, 4-bit NF4 | A100 / T4 x2 | Colab Pro or Kaggle |

Qwen rather than Llama on purpose: ungated on the Hub, so a run is never blocked on an
access approval queue.

### Surviving free-tier sessions

Sessions die without warning. Every generation is flushed to the cache before use, and
the cache is layered: point `cache_layers` at previous sessions' outputs and a
multi-session sweep becomes additive.

```yaml
cache_layers:
  - /kaggle/input/silentwall-cache-run1   # read only
  - /kaggle/working/cache                 # writable overlay
```

## How it works

Six stages, each reading and writing disk so any one can run alone.

```
corpus → artifacts → splits → agent → generation → scoring → report
 CPU       CPU        CPU      CPU       GPU         CPU      CPU
```

**Corpus.** Real M&A events from SEC EDGAR (public domain, no API key). The announcement
8-K gives exact ground truth and a real dated moment when information became public.
Controls are matched on sector and size band. An offline generator produces a
structurally identical synthetic corpus for tests.

**Private artifacts.** Documents a deal team would hold, composed from templates over
already-public filing fields. Byte-reproducible, no real confidential document used.

**Probes.** Six families. Five target a protected field: direct, paraphrase, inference
chain, tool-mediated, memory retrieval. The sixth is behavioural: ordinary questions,
never mentions protected information, identical for restricted and control entities.
That sixth family carries the entire detectability result.

**Scoring.** `leak@k` uses the unbiased combinatorial estimator. Twelve behavioural
features feed the detector. Every interval is a cluster bootstrap over matched pairs.

## Why the numbers can be trusted

**The detector that trains is not the detector that reports.** The reference defense
calibrates against a dev-split adversary. The reported AUC comes from the held-out eval
split. If those shared entities, the reported AUC would be training loss in costume. The
`DevSplit` type cannot carry evaluation entity IDs; requesting one raises
`SplitLeakageError`. Every run record carries hashes of both ID sets plus a disjointness
assertion you can recompute.

**Splits and folds group on the matched pair, not the entity.** Splitting a pair across
folds would let the model learn the matching rule instead of the behavioural signal.

**Scaling, imputation and selection happen inside training folds.** Fitting any of them
on the full dataset before cross-validation is the standard way to manufacture a high AUC.

**Power is stated, not hidden.** At 42 pairs the standard error on AUC is about 0.063.
An undetectability claim means the upper confidence bound sits at or below 0.60.

## The reference defense

Every baseline suppresses. Suppression creates the signature the detector reads, so a
method that suppresses harder cannot win. The `silentwall` defense substitutes instead:
it regenerates the answer from public context using the same model, producing real
logprobs and natural hedging behaviour rather than a canned refusal.

The regeneration uses batched resampling: all retry candidates go in one GPU call, and
the first clean draw is kept. This is statistically identical to sequential resampling
but roughly 4x faster.

The residual detectability comes from the censored-distribution effect: restricted
entities need the "did not leak" filter and controls never do, so surviving outputs are
drawn from a truncated version of the model's natural distribution. The truncation is
itself the fingerprint. More retries dilute it but cannot remove it, which is evidence
that reaching chance requires removing the fact from the parameters rather than
filtering at inference time.

## Tests

```bash
pytest -m "not network and not gpu and not slow"
```

72 tests, about two minutes, no GPU. Property-based tests via Hypothesis cover the
functions where a silent bug produces a plausible wrong answer: the `leak@k` estimator,
AUC algebra, cache keys, fold hygiene, corpus determinism, and end-to-end
reproducibility. One test scans the source to assert that correctness errors
(`SplitLeakageError`, `BudgetExceededError`) appear in no `except` clause.

## Repository structure

```
src/silentwall/
  backends/      stub and Hugging Face model tiers
  cache/         content-addressed keys, layered persistent store
  containment/   method interface + seven implementations
  corpus/        EDGAR retrieval, parsing, control matching, synthetic generation
  probes/        six probe families, offline tool environment, memory store
  runner/        planning, budget guard, checkpointing, execution
  scoring/       leak@k, twelve features, statistics, detector
  report/        audit assembly and markdown rendering
  pipeline.py    stage orchestration
  cli.py         command-line interface

configs/           smoke, iterate, default profiles
experiments/       retry-budget curve script
notebooks/         Colab, Colab Pro, and Kaggle runners; walkthrough notebooks 01–07
results/
  a100_7b_nf4/    headline run: Qwen 7B 4-bit on A100, 42 eval pairs
  colab_1p5b/     validation run: Qwen 1.5B on free T4, 10 eval pairs
references/        papers, project brief, and related work
tests/
  properties/      end-to-end and statistical property tests
  unit/            component-level tests
```

## Scope and ethics

This is a defensive compliance control. The probes measure whether a barrier holds,
which is how any security control gets evaluated. Nothing here helps extract protected
information from a system you do not control.

All source data is public-domain SEC filings. Private-side artifacts are synthesized
from already-public deal terms. No real confidential document is used at any point.

## Background

Related work this builds on and departs from: agentic unlearning across parameters and
memory ([2602.17692](https://arxiv.org/abs/2602.17692)), workflow-level skill revocation
([OBLIVION, 2608.08264](https://arxiv.org/abs/2608.08264)), tool-mediated recovery of
unlearned knowledge ([2608.21544](https://arxiv.org/abs/2608.21544)), relearning
robustness ([2502.05374](https://arxiv.org/html/2502.05374)), and the finding that
unlearning is often reversible in seconds
([2507.07754](https://arxiv.org/abs/2507.07754)). All of that work targets global
erasure: forget it for everyone, permanently. SILENTWALL targets an audience-scoped
barrier where the fact stays fully available to the entitled side, and where the barrier
being observable is itself a failure.

## Author

**Krutik Mehta**

## License

MIT
