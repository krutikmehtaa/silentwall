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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    books = {
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
