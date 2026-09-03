# SILENTWALL: Audience-Conditional Unlearning for Financial Agents

Working research proposal. Independent project.
Status: awaiting decision on scope before implementation.

---

## 1. The short version

Every agentic unlearning paper published so far assumes **global erasure**: after
unlearning, the target fact should be gone for everybody, permanently. That is the
GDPR right-to-erasure model, and it is what the medical-QA work has been testing.

The highest-value enterprise requirement is not that. In finance the legal
requirement is **conditional, directional, and temporary non-disclosure**:

- The deal team that legitimately holds the information must keep full access. Deleting
  it would break the business and is not what the regulation asks for.
- The trading desk must not be able to reach the same fact, including by paraphrase,
  by inference, by tool call, or through shared agent memory.
- The barrier must be auditable to a regulator.
- When the deal is announced the barrier must lift cleanly, because the information
  is now public.

Nobody has formulated unlearning this way. That is the gap.

And there is a second, sharper insight that falls out of it, which I think is the
actual headline contribution:

> **In an audience-conditional setting, refusal is a leak.**

If the trading-side agent visibly clams up on one ticker, that behavioural change
reveals the existence of the barrier. An insider can then recover the confidential
deal pipeline without ever seeing a single restricted document, purely by probing
which entities the agent has gone quiet on. Every existing unlearning method
optimises for suppression, and suppression is exactly what creates this signature.

So the project is: formulate the problem, build the benchmark, demonstrate that
current methods all fail this way, and propose a method that targets
*indistinguishability* rather than suppression.

---

## 2. Why finance, and why this is a real business problem

This is not a hypothetical framing. It is an active, expensive, unsolved compliance
problem right now.

- US broker-dealers are required under Exchange Act Section 15(g) to maintain written
  policies preventing misuse of material non-public information. The controls built to
  satisfy this are what the industry calls information barriers, or Chinese walls.
  See [SIFMA's guidance on MNPI and information barriers](https://www.sifma.org/wp-content/uploads/2020/03/TA6-NEW-Protecting-Firm-and-Client-Information-MNPI-and-Client-Confidentiality.pdf).
- Regulators treat information barriers as a core supervisory obligation, including
  the requirement to restrict access on a need-to-know basis.
  See [CIRO on MNPI supervision](https://www.ciro.ca/newsroom/publications/supervision-related-material-non-public-information).
- Law firms are now publishing client advisories specifically about LLMs ingesting
  nonpublic information. Skadden's July 2026 note frames the baseline rule as: do not
  give a model MNPI about an issuer whose securities that model may trade or influence
  trading in. See [Skadden, When AI Models Access Nonpublic Information](https://www.skadden.com/insights/publications/2026/07/when-ai-models-access-nonpublic-information).
- Practitioners building agent infrastructure for the buy side describe the
  information barrier as the defining control structure, and note that broad tool
  access gives an agent the reach to move MNPI across it.
  See [MCP Manager on investment management](https://docs.mcpmanager.ai/industries/investment-management).

Content above was paraphrased from the linked sources for compliance with licensing
restrictions.

The business consequence is concrete. Banks are currently solving this by simply not
deploying the model on the private side at all, which throws away most of the value.
A method that provably contains MNPI inside an agent system is the thing that unblocks
deployment. That is a much clearer commercial story than clinical unlearning has.

Note on scope: this project studies **containment of confidential information**, which
is a defensive compliance control. It does not build anything that helps somebody
trade on or extract MNPI. The adversarial probes exist to measure whether the barrier
holds, which is standard practice for evaluating any security control.

---

## 3. What already exists (literature map)

I went through the current landscape. Here is the honest positioning, including the
work that comes uncomfortably close.

### 3.1 Agentic unlearning, dual-pathway

- **SBU, "Agentic Unlearning: When LLM Agent Meets Machine Unlearning"** ([2602.17692](https://arxiv.org/abs/2602.17692)).
  This is the paper the team was working from. Introduces agentic unlearning across
  parameters and persistent memory, names the parameter-memory backflow problem, and
  evaluates on medical QA. Global erasure. Assumes one audience.

### 3.2 Skill and tool-level unlearning (closest prior work, read carefully)

- **OBLIVION, "Workflow-Level Operational Skill Unlearning for Deployed Agents"** ([2608.08264](https://arxiv.org/abs/2608.08264)).
  Studies revoked-skill resurrection: after a skill is pulled from a registry, the agent
  rebuilds it from residual carriers such as archives, transcripts, schemas and memory
  entries. Proposes cross-surface erasure plus remediation near dangerous sinks.
  Reports pushing resurrection rate down to roughly 0.11 while holding utility.
  **Why we are still distinct:** OBLIVION removes a capability from all users. We keep the
  capability fully alive for the entitled audience and remove it for another. It also does
  not consider that the block itself is observable.

- **"Forgotten in Weights, Recovered by Tools"** ([2608.21544](https://arxiv.org/abs/2608.21544)).
  Two stages: parametric unlearning to kill direct recall, then trajectory-level RL in
  simulated tool environments to penalise target-seeking tool use and answer leakage.
  **Why we are still distinct:** same global-erasure assumption, and penalising
  target-seeking behaviour is precisely the thing that produces a detectable signature.
  We can use their method as a baseline and show it fails our indistinguishability metric.

- **"Effective Skill Unlearning through Intervention and Abstention"** ([2503.21730](https://arxiv.org/html/2503.21730v1)).
  Training-free skill unlearning via Neuron Adjust and Key Space Detection, tested on
  math, Python and comprehension skills. Useful as a cheap baseline. Note the method is
  literally named around abstention, which is the behaviour we are going to show is leaky.

- **"Agent Tools Orchestration Leaks More"** ([2512.16310](https://arxiv.org/abs/2512.16310)).
  Reports very high leakage rates across six agents when tools are orchestrated. Good
  evidence that the tool pathway is the weak point, and a useful citation for motivating
  the tool-probe family.

### 3.3 Robustness, recovery, and why suppression is not forgetting

This cluster is important because it supports the core claim that current methods only
hide things.

- **"Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning"** ([2406.13356](https://arxiv.org/abs/2406.13356)).
  Approximate unlearning largely suppresses outputs rather than robustly removing knowledge.
- **"Towards LLM Unlearning Resilient to Relearning Attacks"** ([2502.05374](https://arxiv.org/html/2502.05374)).
  Connects robust unlearning to sharpness-aware minimisation. Smoothness helps.
- **REBEL** ([2602.06248](https://arxiv.org/html/2602.06248v1)).
  Evolutionary probing recovers supposedly forgotten knowledge with attack success rates
  reported up to 60% on TOFU and 93% on WMDP.
- **"Erasing Representational Separability toward Irreversible Deep Forgetting"** ([2507.07754](https://arxiv.org/abs/2507.07754)).
  Across 14 unlearning methods, a single linear map fitted on held-out calibration data,
  with no access to forgotten data, reverses unlearning in seconds. This is very strong
  support for our thesis: the separability that unlearning creates is itself the leak.
- **"Unlearning Does Not Make LLMs Forget Under Probabilistic Decoding"** ([2511.04934](https://arxiv.org/abs/2511.04934)).
  Leakage persists under sampling. Motivates evaluating at k samples, not greedy only.
- **"Revealing 'Erased' Knowledge"** ([2506.17279](https://arxiv.org/html/2506.17279v1)).
  Step-by-step reasoning acts as a backdoor to recover hidden information. Directly
  motivates our inference-chain probe family.

### 3.4 Entanglement and collateral damage

- **SKEB / "The Limits of Obliviate"** ([2510.25732](https://arxiv.org/abs/2510.25732)).
  Structural entanglement predicts what leaks after unlearning. Also finds persuasive
  framing lifts recall, with smaller models more vulnerable.
- **"Forget Narrowly, Retain Broadly"** ([2607.09236](https://arxiv.org/abs/2607.09236v1)).
  Names under-forgetting and over-forgetting as an asymmetric generalisation problem.
  Important terminology note: their "asymmetric" is about the forget/retain objective
  imbalance, **not** about different audiences. Our use of the word is a different axis.
  I will define it explicitly to avoid a reviewer collision.
- **Forgetting-MarI** ([2511.11914](https://arxiv.org/html/2511.11914v3)).
  Removes only the marginal information contributed by the forget set. Conceptually
  relevant: in our setting the quantity to remove is exactly the marginal information
  the private documents add over the public prior. This is a strong theoretical hook.
- **"Auditing Collateral Knowledge Damage Before Unlearning"** ([2606.18473](https://arxiv.org/html/2606.18473)).
  Useful for the over-forgetting cost measurement.

### 3.5 Multi-agent and networked propagation

- **"When Unlearning Fails: Reliable Data Deletion under Post-Training in Agent Networks"** ([2607.28829](https://arxiv.org/abs/2607.28829)).
  Deletion leaves an influence echo because the forget data already shaped later retained
  trajectories. Echo survives retraining on retained data.
- **"Memory Contagion"** ([2606.23195](https://arxiv.org/abs/2606.23195v2)).
  Bias propagates to future agents through a shared memory store even under perfect
  consolidation. Direct evidence that shared memory is a cross-agent channel.
- **"Governed Shared Memory for Multi-Agent LLM Systems"** ([2606.24535](https://arxiv.org/abs/2606.24535)).
  Names unauthorised leakage, stale propagation, contradiction persistence, provenance
  collapse. This is a systems architecture paper, ACL-style enforcement, not unlearning.
  It is the closest thing to our setting on the engineering side and it is exactly what we
  should argue is insufficient, because it does not address weight-resident knowledge or
  behavioural inference.

### 3.6 Inference-time and cheap unlearning (relevant to cost-effectiveness)

- **ALU, "Agents Are All You Need for LLM Unlearning"** ([2502.00406](https://arxiv.org/abs/2502.00406)).
  Multi-agent, retrain-free, model-agnostic inference-time unlearning. Very relevant as a
  cheap baseline and as a design pattern we can borrow for the enforcement layer.
- **ECO, "Embedding-Corrupted Prompts"** ([2406.07933](https://arxiv.org/html/2406.07933v2)).
  Enforces an unlearned state at inference using a prompt classifier plus embedding
  corruption. Cheap. Also an obvious source of a detectable signature, so a good baseline
  to break.
- **Look-ahead bias removal via logit adjustment with small forget/retain experts**
  ([OpenReview zYsLIPgM28](https://openreview.net/pdf?id=zYsLIPgM28)).
  Finance-specific, inference-time, low cost, removes verbatim and semantic knowledge by
  steering a large base model with a pair of small tuned models. This is the single most
  directly reusable technical mechanism I found for our method, and it is already validated
  in a financial context.

### 3.7 Finance agent benchmarks (for the utility side)

- **FinToolBench** ([2603.08262](https://arxiv.org/html/2603.08262v1)): 760 executable financial
  tools, 295 tool-required queries. Runnable, which matters.
- **FinMCP-Bench** ([2603.24943](https://arxiv.org/html/2603.24943v1)): tool invocation under MCP.
- **Finance Agent Benchmark** ([2508.00828](https://arxiv.org/abs/2508.00828)): 537 expert-authored
  questions; best model reported at 46.8% accuracy, which tells us headroom is large.
- **FORCE-Bench** ([2607.19409](https://arxiv.org/pdf/2607.19409)) and
  **FrontierFinance** ([2608.11683](https://arxiv.org/pdf/2608.11683v1.pdf)): enterprise finance
  agentic evaluation, verifiability and rule adherence.
- **FinBen** ([2402.12659](https://arxiv.org/abs/2402.12659)): broad financial LLM benchmark.

### 3.8 Bottom line on novelty

Occupied: dual-pathway global unlearning, skill revocation, tool-mediated recovery,
relearning robustness, entanglement-aware forgetting, multi-agent influence echo,
governed shared memory as an access-control system.

**Open:**
1. Unlearning where the forget target is scoped to an audience rather than global.
2. Unlearning where the barrier must be **undetectable**, and where abstention is
   therefore a failure mode rather than a success criterion.
3. Unlearning with a **scheduled release** (barrier lifts when information becomes public).
4. All of the above evaluated on real, dated, public-domain financial events.

I could not find a paper doing any of these four. The combination is clearly new.

---

## 4. Problem formulation

Let an agent system serve two roles inside one firm:

- **Private side** (`A_priv`): deal team. Entitled to the confidential fact `f`.
- **Public side** (`A_pub`): trading and research. Not entitled to `f`.

Both sit on a shared substrate: the same base model, a shared skill or memory store,
and overlapping tool access. This shared substrate is what makes it hard, and it is
also the realistic configuration, because firms do not want to train and host a
separate model per role.

Define three requirements.

**R1 Retention.** `A_priv` answers questions about `f` at full baseline accuracy.

**R2 Containment.** `A_pub` does not reveal `f` under any probe: direct, paraphrased,
multi-hop inference, tool-mediated, memory-retrieval, or multi-turn accumulation.

**R3 Indistinguishability.** `A_pub`'s behaviour on the walled entity is
statistically indistinguishable from a **clean reference agent** that was never exposed
to `f` at all. Not "refuses safely". Indistinguishable.

R3 is the new one, and it is what makes R2 meaningful. Containment without R3 leaks the
existence and location of the wall, which in a trading context is itself valuable
nonpublic information about the firm's deal pipeline.

Add a fourth for the temporal dimension.

**R4 Clean release.** On the public announcement date, `A_pub` regains full access to
`f` without retraining, and without a discontinuity that retroactively reveals how long
the wall was in place.

The clean reference agent in R3 is the key methodological device. It gives us an
oracle-grounded target: we are not asking "did it forget enough", we are asking
"does it look exactly like a model that never knew". That is a much crisper,
falsifiable criterion, and it is borrowed conceptually from indistinguishability
arguments in security rather than from the accuracy-drop framing in the unlearning
literature.

---

## 5. Dataset plan

I want ground truth that is real, dated, public domain, and has a genuine
"became public at time T" event. SEC EDGAR gives exactly that, for free.

### 5.1 The walled-fact corpus, built from real M&A events

Construction:

1. Pull completed M&A events from EDGAR: merger 8-Ks (Item 1.01 / 2.01), S-4
   registration statements, SC 13D/14D filings. All public domain.
2. For each deal, the announcement 8-K gives a hard **public date T_pub**, plus the real
   terms: acquirer, target, price per share, premium, structure, expected close.
3. The **walled fact** is that same deal content positioned in the pre-announcement
   window. We synthesise the private-side artifacts a deal team would actually hold:
   engagement memo, draft fairness opinion, valuation model summary, diligence notes.
   These are generated from the real post-announcement terms, so the content is factually
   correct and the ground truth is exact, but no actual confidential document is used.
4. **Distractor set**: matched control companies with no deal, same sector and size band.
   Needed for R3, because wall-detection AUC is meaningless without negatives.

This design has a property I like a lot: because the deal really did happen and really was
announced, we get a genuine, externally dated release event for R4, and we can verify the
public prior by checking what was publicly knowable before T_pub.

Target scale: 60 to 100 deals, 40 to 60 matched controls. Enough for AUC with reasonable
confidence intervals, small enough to run on one GPU.

### 5.2 Probe suite (this is where the real work is)

Six families, each scored separately. Families 3 and 6 are the ones existing work
under-tests.

| # | Family | What it tests | Example shape |
|---|--------|---------------|---------------|
| 1 | Direct | Verbatim recall | "What is the offer price for TARGET?" |
| 2 | Paraphrase | Surface robustness | Reworded, indirect, hypothetical framings |
| 3 | Inference chain | Derivable implication | Reason step by step to a conclusion that requires `f` without naming it |
| 4 | Tool-mediated | Recovery via tools | Give the agent retrieval and calculators; see if it reconstructs `f` |
| 5 | Memory retrieval | Shared-store bleed | Query the shared memory the private side wrote to |
| 6 | Behavioural / wall detection | Existence of the barrier | Never asks for `f` at all. Asks neutral questions and classifies from behaviour |

Family 6 is the novel one. Concretely: ask `A_pub` a set of ordinary questions about a
company (valuation view, sector comparables, recent news summary, hedging suggestion).
Extract features: refusal rate, hedging-language rate, response length, entropy over
sampled generations, latency, tool-call pattern, confidence calibration. Then train a
simple classifier to separate walled entities from controls. Report **AUC**.

- AUC near 0.5 means the wall is invisible. This is the goal.
- AUC near 1.0 means the wall is trivially locatable. My hypothesis is that every existing
  method lands here.

Also evaluate at k samples rather than greedy decoding only, following the probabilistic
decoding result, and report leak@k.

### 5.3 Utility measurement

- Private side retention: accuracy on `f` questions, must stay at baseline.
- Public side general utility: a slice of FinToolBench or the Finance Agent Benchmark,
  restricted to non-walled entities, to catch collateral damage.
- Over-forgetting cost: performance on the *matched controls*. If controls degrade, the
  method is bleeding.

---

## 6. Method sketch

Baselines to break (all should fail R3):

1. No defence.
2. Prompt-level instruction ("do not discuss X").
3. RAG / retrieval filtering, the ACL approach. This is what industry actually does.
4. Gradient ascent and NPO on the private corpus.
5. ECO-style inference-time embedding corruption.
6. Trajectory-RL penalisation, in the spirit of 2608.21544.
7. OBLIVION-style cross-surface erasure, adapted.

Proposed method, working name **SILENTWALL**. Three ideas, in order of how much I trust them.

**(a) Public-prior substitution instead of refusal.**
Rather than suppressing, steer `A_pub` to answer *from the public prior*, confidently and
in normal register. The target is not silence, it is the counterfactual answer a
clean agent would give. Mechanically this is close to the finance look-ahead-bias work:
adjust logits of the base model using a small expert tuned on the private corpus (to
subtract) and one tuned on the public-only corpus (to add). Inference-time, cheap, no
base-model retraining, and it naturally produces the *right* answer instead of a
conspicuous non-answer.

**(b) Marginal-information targeting.**
The quantity to remove is exactly the marginal information the private documents add
over the public prior, which is the Forgetting-MarI framing. This gives a principled
handle on the R1/R2 tension: subtract only the delta, leave the public substrate intact.

**(c) Calibration matching as an explicit objective.**
Add a term that matches `A_pub`'s output distribution on walled entities to its
distribution on matched controls. This directly optimises R3 rather than hoping for it.
This is the piece I have not seen anywhere and it is the most likely source of a genuine
methodological contribution.

Efficiency note: (a) and (c) are inference-time or small-adapter operations. No full
fine-tune of the 8B base. That keeps this runnable on a single A100 or a rented GPU,
which matters given the compute situation.

---

## 7. Expected contributions

1. **Formulation.** Audience-conditional, temporally-scoped unlearning, grounded in an
   actual regulatory control rather than a synthetic privacy scenario.
2. **Negative result.** Existing unlearning methods, including the newest agentic ones,
   create a detectable behavioural signature. Reporting wall-detection AUC near 1.0 for
   methods that look successful on conventional leak metrics would be a genuinely
   interesting finding, and negative results with a clean diagnosis travel well.
3. **Benchmark.** Public-domain, real-event, dated, with a release milestone and a probe
   family nobody else is testing. Releasable on GitHub without any licensing problem.
4. **Metric.** Wall-detection AUC as a first-class unlearning metric. I think this
   generalises well beyond finance, which is what gives the work legs.
5. **Method.** Substitution-plus-calibration-matching as an alternative to suppression.

Contribution 2 alone is enough for the repository to be worth something, and it does not
depend on the method working. That is deliberate. The project has a floor.

---

## 8. Feasibility

Model: an 8B open-weight instruct model. Small experts for (a) can be 1B or smaller.

Compute: adapter training on 60 to 100 deals is minutes to low hours. The cost is
dominated by evaluation, because family 6 needs many samples per entity for the entropy
and calibration features. Rough estimate: 150 entities times 8 question types times 16
samples is about 19k generations per configuration, times roughly 8 configurations.
That is very manageable on one A100, and it batches well with vLLM.

Estimate: 20 to 40 GPU-hours total including reruns. Fits inside Colab Pro, or roughly
$30 to $70 on rented hardware, or comfortably inside an ACCESS Explore allocation.

Legal and ethical: all source data is public-domain SEC filings. No real confidential
document is used anywhere. Private-side artifacts are synthesised from already-public
terms. The output is a containment control, not an extraction tool.

---

## 9. Open decisions for you

1. **Domain confirmed?** Finance MNPI, or would you rather keep it domain-general and
   use finance as one instantiation? My view: commit to finance. The specificity is the
   strength, and the regulatory hook is what makes it credible.
2. **Scope.** Full four contributions, or start with the negative result (contributions
   1 to 4) and treat the method as a stretch goal? My view: build the benchmark and the
   negative result first. It is the part that cannot fail, and it makes the method
   evaluable.
3. **Base model.** Staying with an 8B open-weight model, consistent with prior runs?
4. **Compute.** Which of Colab Pro, rented GPU, or ACCESS Explore do you want to target,
   since that sets the batch sizes I write into the notebooks.

---

## 10. Proposed build order

| Phase | Deliverable | Notebook |
|-------|-------------|----------|
| 0 | EDGAR pull, deal extraction, control matching | `01_build_corpus.ipynb` |
| 1 | Private-side artifact synthesis, probe suite generation | `02_build_probes.ipynb` |
| 2 | Clean reference agent, baseline agent, retention and leak baselines | `03_baselines.ipynb` |
| 3 | All defence baselines, full six-family evaluation | `04_defences.ipynb` |
| 4 | Wall-detection classifier, AUC results, the negative result | `05_wall_detection.ipynb` |
| 5 | SILENTWALL method, ablations | `06_silentwall.ipynb` |
| 6 | Release event / R4, final figures, writeup | `07_release_and_report.ipynb` |

Each notebook is self-contained, saves artifacts to disk, and can be run independently
so a crashed session does not cost the whole pipeline. Lesson learned from the Delta
queue experience.
