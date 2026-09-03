# Runbook

Every command uses `python -m silentwall.cli` rather than the bare `silentwall`
script. The script gets installed into a Scripts directory that is often not on PATH,
and the module form works everywhere without setup.

---

## Stage 0. Local setup and validation

Free, no GPU, about five minutes including the install.

```bash
cd SilentWall
pip install -e ".[dev]" -c constraints.txt
```

Confirm it imported:

```bash
python -m silentwall.cli --version
python -m silentwall.cli methods
```

You should see seven methods listed.

### Run the whole pipeline

```bash
python -m silentwall.cli sweep -c configs/smoke.yaml
```

This runs all six stages for all seven containment methods against a deterministic
stub backend. No model weights are downloaded and no network is touched. Takes about
60 seconds cold, 27 seconds once the cache is warm.

You are looking for a table like this at the end:

```
| method                | leak  | AUC   |
| clean_reference       | 0.000 | 0.438 |
| none                  | 0.708 | 1.000 |
| refusal_classifier    | 0.000 | 1.000 |
| silentwall            | 0.000 | 0.625 |
```

Two sanity checks on that output. `clean_reference` must show leak 0.000, because an
agent that never held the data cannot leak it. And its AUC should sit near 0.5, because
there is nothing to detect. If either is wrong, something is broken and nothing
downstream will mean anything.

### Confirm the cache works

Run the same command again. The second run should report `already cached 100%` and
finish in roughly half the time, with identical numbers. This matters more than it
looks: the cache is what makes a multi-session GPU run affordable.

### Run the tests

```bash
pytest -m "not network and not gpu and not slow"
```

56 tests, about two minutes, no GPU and no network.

---

## Stage 1. Build the real corpus

Two options. Start with the offline one, it needs nothing.

### Offline corpus, no network

Already the default in every config (`corpus.synthetic: true`). Structure, sectors,
size bands and field values all vary the way real ones do; only the language is
generated. Good enough for every methodological result.

```bash
python -m silentwall.cli corpus -c configs/default.yaml
```

Two seconds, CPU only. Writes to `artifacts/default/corpus/`.

### Live SEC EDGAR corpus

Needs internet but no API key and no signup. SEC asks automated clients to identify
themselves, so a descriptive User-Agent with a contact address is required and the
client refuses to start without one.

Edit `configs/default.yaml`:

```yaml
corpus:
  synthetic: false
  user_agent: "SilentWall research your.email@example.com"
  quarters: ["2019Q1", "2019Q2", "2019Q3", "2019Q4"]
```

Then:

```bash
python -m silentwall.cli corpus -c configs/default.yaml
```

This takes a while on the first run because it fetches filings one at a time under a
rate limit. Every response is cached to disk, so a rerun is offline and instant.

If it stops with `MatchingInfeasibleError` saying it only extracted N deals, widen
`corpus.quarters` and run again. The HTTP cache is preserved, so nothing refetches.

---

## Stage 2. Check the cost before spending anything

Always do this before a GPU run.

```bash
python -m silentwall.cli plan -c configs/default.yaml
```

This generates nothing. It counts the work, counts what is already cached, and prints a
projection per method plus a sweep total.

At the shipped defaults that total is **208,320 generations**, which the tool projects
at 14.5 hours. Treat that projection as optimistic. It assumes 4 generations per second
for a 7B model in 4-bit, and 4-bit on a T4 has no native int4 acceleration, so 1.5 to
3 per second is more realistic. The honest range is **15 to 40 hours**.

Kaggle gives 30 GPU-hours per week, so the shipped default may not fit in one week.

### Cut it down

The single best lever is halving `k`:

```bash
python -m silentwall.cli plan -c configs/default.yaml --set sampling.k=8
```

That gives 104,160 generations, so 7 to 20 hours. You keep `leak@1` and the leak curve
up to k=8, and detectability barely moves because the AUC comes from 8 behavioural
templates per entity regardless of k.

Dropping the one method that needs training as well:

```bash
python -m silentwall.cli plan -c configs/default.yaml \
  --set sampling.k=8 \
  --set methods=clean_reference,none,system_prompt,retrieval_filter,refusal_classifier,silentwall
```

89,280 generations. This is the configuration I would actually run.

Note the comma separated form for `methods`. A JSON array works too but fighting
PowerShell quoting is not worth it.

---

## Stage 3. Colab run at 1.5B

Do this before the 8B run. One to three hours, free tier, and if the AUC pattern holds
here you already have a defensible result.

New Colab notebook, set Runtime to T4 GPU, then:

```python
!git clone https://github.com/<you>/silentwall.git
%cd silentwall
!pip install -q -e .
```

Torch and transformers are already present on Colab, so no `[gpu]` extra is needed.

```python
!python -m silentwall.cli plan -c configs/iterate.yaml
```

Confirm the projection looks sane, then:

```python
!python -m silentwall.cli sweep -c configs/iterate.yaml --confirm-budget
```

29,760 generations across 6 methods at 1.5B. The first run downloads about 3GB of
weights for Qwen2.5-1.5B-Instruct. No Hugging Face token is needed, which is why Qwen
was chosen over Llama.

Save the results before the session ends:

```python
from google.colab import files
!zip -r outputs.zip outputs cache
files.download("outputs.zip")
```

---

## Stage 4. Kaggle run at 7B

### Set up the notebook

1. New Kaggle notebook.
2. Settings, Accelerator, pick **GPU T4 x2** or **P100**.
3. Settings, Internet, switch **on**. Needed to download model weights.

GPU access requires phone verification on your Kaggle account. Do that first if you
have not, it is the step most likely to block you unexpectedly.

### First session

```python
!git clone https://github.com/<you>/silentwall.git /kaggle/working/silentwall
%cd /kaggle/working/silentwall
!pip install -q -e .
```

Check the plan, then start:

```python
!python -m silentwall.cli plan -c configs/default.yaml --set sampling.k=8
```

```python
!python -m silentwall.cli sweep -c configs/default.yaml \
  --set sampling.k=8 \
  --set methods=clean_reference,none,system_prompt,retrieval_filter,refusal_classifier,silentwall \
  --confirm-budget
```

`--confirm-budget` is required because the run exceeds the configured ceiling. That
gate exists so a large run is always a deliberate choice.

If the session dies partway, nothing is lost. Every generation is flushed to the cache
before use.

### Carry the cache to the next session

This is the part that makes a multi-session sweep affordable rather than repetitive.

At the end of a session, save `/kaggle/working/cache` as a Kaggle output dataset. Then
in the next session, add that dataset as an input and point the config at it:

```python
!python -m silentwall.cli sweep -c configs/default.yaml \
  --set sampling.k=8 \
  --set cache_layers=/kaggle/input/silentwall-cache-1/cache,/kaggle/working/cache \
  --confirm-budget
```

Earlier layers are read only, the last one takes the writes. The plan output will show
a high `already cached` percentage, and only the remaining work runs. Add another layer
per session as you go.

Verify it worked: if `already cached` says 0% when you expect otherwise, the layer path
is wrong. Fix it before letting the run continue, or you will spend the quota twice.

---

## Stage 5. Read the results

```
outputs/
  comparison.md          the table across all methods
  comparison.json        machine readable, plus FDR adjusted secondary p-values
  audit_<method>.md      per method detail
  audit_<method>.json
```

Read `comparison.md` first, and read the leak column and the AUC column **together**.

- Low leak with high AUC is the finding: content hidden, barrier visible.
- `clean_reference` is the reference point. Leak must be 0.000 and AUC near 0.5.
- Any method whose AUC is far from `clean_reference` is advertising its own barrier.

Then in a per-method report, the feature importance table tells you *which* behaviour
leaked, which is the actionable part. A large coefficient on `refusal_rate` means the
agent is visibly clamming up. A large one on `length_mean` means answers get shorter.

### On interpreting a null result

If a method shows AUC near 0.5, check whether `undetectable` says True. That flag means
the upper confidence bound sits at or below 0.60. It does not mean no signal exists.
At 60 pairs the standard error on AUC is about 0.06, so 0.5 can be told apart from 0.7
but not from 0.58. The report prints this next to the claim rather than in a footnote.

---

## Stage 6. Publish

```bash
git init
git add -A
git commit -m "SILENTWALL: detectability auditing for LLM agent information barriers"
git branch -M main
git remote add origin https://github.com/<you>/silentwall.git
git push -u origin main
```

`.gitignore` already excludes `cache/`, `logs/`, `artifacts/`, `data/` and `outputs/`.
If you want the results in the repo, commit `outputs/` deliberately:

```bash
git add -f outputs/comparison.md outputs/comparison.json
```

CI runs lint, type checking and the test suite on push, all on CPU, no GPU or network
needed.

---

## Notebooks

The seven notebooks under `notebooks/` walk the same pipeline with explanation, and are
the better artifact for a reader arriving from the README. They import from the package
and define nothing themselves, which a test enforces.

Run them in order, 01 through 07. They default to the smoke config, so change `CONFIG`
in the first cell to point at `iterate.yaml` or `default.yaml` on a GPU machine.

---

## Troubleshooting

**`already cached 0%` when you expect hits.** The cache key covers the model, the
containment method and its parameters, and every sampling parameter. Changing any of
them correctly invalidates the cache. If nothing changed, check that `cache_layers`
points where you think.

**`BudgetExceededError`.** Working as intended. Either lower the work with
`--set sampling.k=8`, raise `max_generations` in the config, or pass
`--confirm-budget`.

**`MatchingInfeasibleError`.** Not enough control entities to fill the target. Widen
`corpus.quarters` for a live corpus, or lower `corpus.target_restricted`. This is fatal
rather than a warning because a smaller corpus silently widens every interval.

**`SplitLeakageError`.** A containment method tried to look at evaluation data while
calibrating. This is never recoverable and never caught, because the numbers it would
produce still look plausible. It indicates a bug in a method, not a configuration
problem.

**CUDA out of memory.** The backend halves the batch size and retries up to three
times, then defers the unit and continues. Deferred units are reported at the end and
retried on the next run. To avoid it, lower `sampling.max_new_tokens` or use the
`gpu-1p5b` tier.

**Slower than projected.** Expected. The throughput table in `runner/plan.py` is an
estimate and has never been validated against a real GPU. Once you have a real
measurement, put it there and the projections get honest.
