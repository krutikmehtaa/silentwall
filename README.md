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

## The finding

Every containment method we tested suppresses information and advertises the barrier
while doing it. Suppression is what creates the signature, so a method that suppresses
harder cannot escape it.

Reference numbers from the offline smoke profile, 8 deals and 4 evaluation pairs:

| method | worst-family leak | detectability AUC | reading |
|---|---|---|---|
| `clean_reference` | 0.000 | 0.438 | never held the data, so nothing to detect |
| `none` | 0.708 | 1.000 | undefended, tells you outright |
| `retrieval_filter` | 0.792 | 0.688 | blocks documents, misses weight-resident knowledge |
| `system_prompt` | 0.333 | 1.000 | naming the company in the rule is the leak |
| `lora_ga` | 0.333 | 1.000 | suppresses the value, changes the behaviour |
| `refusal_classifier` | 0.000 | 1.000 | perfect containment, perfectly visible barrier |
| `silentwall` | 0.000 | 0.625 | substitutes instead of refusing |

Read the middle two columns together. `refusal_classifier` leaks nothing and is trivially
locatable. That combination is the failure mode this tool exists to surface.

These are small-sample illustrations on synthetic data, not headline results. Confidence
intervals at 4 pairs are very wide and the report says so.

## Install

```bash
git clone https://github.com/<you>/silentwall.git
cd silentwall
pip install -e ".[dev]"
```

Core install needs only numpy, scipy, scikit-learn, PyYAML and requests. For the real
model tiers add `pip install -e ".[gpu]"`.

## Run it

Start with the smoke profile. No GPU, no weights, no network, a few seconds:

```bash
silentwall sweep --config configs/smoke.yaml
```

That runs all six pipeline stages against a deterministic stub backend and produces a
full report. It is how you check the plumbing before spending anything.

Then see what a real run would cost, without generating anything:

```bash
silentwall plan --config configs/default.yaml
```

Then the real thing:

```bash
silentwall sweep --config configs/default.yaml --confirm-budget
```

Other commands:

```bash
silentwall corpus --config configs/default.yaml   # stage 0 and 1 only, CPU
silentwall audit --method silentwall -c configs/iterate.yaml
silentwall methods                                # list containment methods
```

## Compute

Built for a zero-dollar budget. Three tiers, switched by config.

| tier | model | where | cost |
|---|---|---|---|
| `stub` | none, deterministic | laptop CPU | free, seconds |
| `gpu-1p5b` | Qwen2.5-1.5B-Instruct | free Colab | free |
| `gpu-8b-nf4` | Qwen2.5-7B-Instruct in 4-bit NF4 | Kaggle 16GB | free tier quota |

The default profile is roughly 28,800 generations per method, about 1.5 to 2.5 GPU-hours
on a T4 or P100. A seven-method sweep fits inside one weekly Kaggle allowance only
because nothing is ever regenerated.

Qwen rather than Llama on purpose: it is ungated on the Hub, so a run is never blocked
waiting for an access approval.

### Surviving free-tier sessions

Sessions die without warning. Two mechanisms handle it. Every generation is written and
flushed to the cache before it is used, and completed work units are appended to a
checkpoint one line at a time, so a kill costs at most one unit. And the cache is
layered: point `cache_layers` at previous sessions' output datasets, with the working
directory last, and a multi-session sweep becomes additive instead of repetitive.

```yaml
cache_layers:
  - /kaggle/input/silentwall-cache-run1   # read only
  - /kaggle/input/silentwall-cache-run2   # read only
  - /kaggle/working/cache                 # writable overlay
```

## How it works

Six stages, each reading and writing disk so any one can run alone.

```
corpus  ->  artifacts  ->  splits  ->  agent  ->  generation  ->  scoring  ->  report
 CPU          CPU           CPU        CPU          GPU            CPU         CPU
```

**Corpus.** Real M&A events from SEC EDGAR, which needs no API key and is public domain,
so the benchmark is redistributable. The announcement 8-K gives exact ground truth and a
real dated moment when the information became public. Control entities are matched to
restricted ones on sector and size band, because "can you tell these apart" needs a
comparison group that resembles them. An offline generator produces a structurally
identical corpus for tests.

**Private artifacts.** The documents a deal team would hold, composed from templates over
already-public filing fields. No model inference, byte-reproducible, and no real
confidential document is used anywhere.

**Probes.** Six families. Five go after a protected field: direct, paraphrase, inference
chain, tool-mediated, memory retrieval. The sixth is behavioural, never mentions the
protected information, and is answerable by an agent that never held it. That sixth
family carries the whole detectability result.

**Scoring.** `leak@k` uses the unbiased combinatorial estimator, so `leak@1` and the full
curve over k come out of one sampling budget. Twelve behavioural features feed the
detector. Every interval is a cluster bootstrap over matched pairs, never over probes.

## Why the numbers can be trusted

Three pieces of discipline that change the answer if dropped.

**The detector that trains is not the detector that reports.** The reference defense is
calibrated against a wall detector, and the same metric is the headline. If those shared
entities, the reported AUC would be training loss in costume. So `ContainmentMethod.fit`
receives a `DevSplit` type that cannot carry evaluation entity ids, requesting one raises
`SplitLeakageError`, and every run record carries hashes of both id sets plus a
disjointness assertion you can recompute.

**Splits and folds group on the matched pair, not the entity.** A restricted company and
its control were chosen to resemble each other. Splitting them lets the model learn the
matching rule instead of the behavioural signal.

**Scaling, imputation and selection happen inside training folds.** Fitting any of them on
the full dataset before cross validation is the standard way to manufacture a high AUC.

Also: the primary feature set is fixed in advance at eleven features, one primary
hypothesis per method, everything else FDR-adjusted as exploratory. And the report states
the power limit rather than burying it. At 60 pairs the standard error on AUC is about
0.06, so 0.5 can be distinguished from 0.7 but not from 0.58. An undetectability claim
means the upper confidence bound sits at or below 0.60, not that no signal exists.

## Two bugs worth knowing about

Both were caught by the test suite and both illustrate the thesis.

The first version of the reference defense dropped the token trace when it substituted an
answer. Restricted entities then had missing entropy features while controls had real
ones, so *availability of the feature* became a cleaner label than anything the feature
measured, and AUC went to 1.0. Fixing one side channel opened a louder one.

And the defense now applies its adjustment to every entity, restricted and control alike,
doing identical work in both cases. For controls the blend weight is zero, so the passes
are wasted. That waste is deliberate: gating only on restricted entities would make those
queries measurably slower, and latency is one of the features the detector reads.

## Tests

```bash
pytest -m "not network and not gpu and not slow"
```

55 tests, about three minutes, no GPU and no network. Property-based tests using
Hypothesis cover the functions where a silent bug produces a plausible wrong answer:
the `leak@k` estimator against brute-force enumeration, AUC algebra, cache key
correctness, fold hygiene, corpus determinism, and end-to-end reproducibility. One test
scans the source to assert that `SplitLeakageError` and `BudgetExceededError` appear in no
`except` clause anywhere in the package.

Marked tests excluded from CI: `network` for live EDGAR access, `gpu` for the 4-bit
backend, `slow` for a bootstrap coverage simulation.

## Scope and ethics

This is a defensive compliance control. The probes measure whether a barrier holds, which
is how any security control gets evaluated. Nothing here helps extract protected
information from a system you do not control.

All source data is public-domain SEC filings. Private-side artifacts are synthesized from
already-public deal terms. No real confidential document is used at any point.

## Layout

```
src/silentwall/
  corpus/        EDGAR retrieval, parsing, control matching, artifacts, splits
  probes/        six probe families, offline tool environment, memory store
  backends/      stub, Hugging Face tiers
  cache/         content-addressed keys, layered persistent store
  containment/   the method interface plus seven implementations
  runner/        planning, budget guard, checkpointing, execution
  scoring/       leak@k, twelve features, statistics, detector
  report/        audit assembly and rendering
  pipeline.py    stage orchestration
configs/         smoke, iterate, default
notebooks/       thin, import from the package
tests/           property, unit, end to end
```

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

The regulatory grounding is real. US broker-dealers must maintain policies preventing
misuse of material non-public information under Exchange Act Section 15(g), and the
controls built for that are what the industry calls information barriers. See
[SIFMA on MNPI](https://www.sifma.org/wp-content/uploads/2020/03/TA6-NEW-Protecting-Firm-and-Client-Information-MNPI-and-Client-Confidentiality.pdf)
and [Skadden on AI models accessing nonpublic information](https://www.skadden.com/insights/publications/2026/07/when-ai-models-access-nonpublic-information).
Content was paraphrased from the linked sources for compliance with licensing
restrictions.

## License

MIT.
