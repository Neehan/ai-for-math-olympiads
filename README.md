# Paper 1: When More Inference Cannot Replace a Missing Strategy

**Question.** Can additional inference substitute for missing strategic information on hard, fresh olympiad proofs?

We independently intervene on strategic information and inference compute, then compare the resulting failure boundary across Sonnet 5 → Opus 4.8 → Fable 5. The predicted result is that a ≤25-word strategy unlocks failures that resist both parallel sampling and sequential revision, while stronger models disproportionately solve those failures unaided. Hints are diagnostic interventions, not a solving method.

**Pilot.** On 13 development-only combinatorics problems, Opus 4.8 + hint succeeds on 9/13. Sonnet 5 improves from 1/13 at baseline to 5/13 with the same frozen hints, rescuing four baseline failures. The ~50-word outline is only a development ceiling. All pilot problems stay outside the confirmatory headline analysis.

## Confirmatory design

Profiles are frozen from the two 1× cells before any scaling run.

| 1× profile | no hint | ≤25-word hint | parallel 8× prediction | sequential 8× prediction |
|---|---|---|---|---|
| ceiling | solved | — | — | — |
| **hint-responsive** | failed | solved | **low response** | **low response** |
| **hint-unresponsive** | failed | failed | **higher response** | **higher response** |

A hint-responsive problem failing both separate 8× protocols is **robustly compute-flat under the two tested protocols**. We do not claim that it can never be solved.

Pre-registered predictions:

1. Hint response predicts lower success under parallel 8×.
2. Hint response separately predicts lower success under sequential 8×.
3. Nearly all unaided compute rescues observed through 8× occur by 4×.
4. Fable disproportionately solves Opus hint-responsive / robustly compute-flat failures unaided.
5. On surviving cases, unaided IdeaSearch-8 also fails to recover many oracle-supplied strategies.

Supplying a strategy is a controlled intervention. Hint-response labels are frozen before held-out compute runs, so their relationship with later scaling is predictive, not a causal decomposition of the original failure. The cross-model boundary shift is supporting observational evidence.

## Problems and models

**Problems.** 35 post-cutoff 2026 contest problems in algebra, combinatorics, and number theory. Selection criteria and human hardness labels are frozen before model runs. Geometry is excluded because reliable proof auditing is difficult for long coordinate derivations. No retrieval corpus is available.

Every solver's published cutoff must predate the earliest contest (February 2026).

- **Full design:** Opus 4.8 + GPT-5.5. These provide the primary within-model tests and cross-lab replication. GPT-5.6 is excluded as contaminated.
- **Capability ladder:** Sonnet 5 → Opus 4.8 → Fable 5, using matched 1× baseline and frozen-hint cells. The load-bearing boundary test is Fable on Opus failures.
- **Fable:** baseline 1× on all 35 and hint 1× on baseline failures; no full compute sweep.
- **Open-weight replication:** baseline + hint, followed by targeted 8× scaling on hint-responsive failures, using a model with a verified pre-February-2026 cutoff.

## Interventions and compute

- **Hint:** one frozen, audited ≤25-word problem-specific strategy. It may supply the key idea but not the final answer or a derivation.
- **Placebo:** length-matched non-strategic text.
- **Outline:** audited ~50-word development ceiling; never used on the held-out 35.
- Hints are written from the statement and official solution, independently audited for leakage, committed, hashed, and published verbatim.

Compute is total output tokens: thinking, visible text, and tool-call text. Reasoning effort stays fixed at high. Provider quota rejection is transport recovery: the same local conversation UUID resumes through the next available credential with one fixed continuation message, all streamed output remains charged to the original budget, and every reconnect is logged.

- **1×:** 200k output tokens.
- **2× / 4× / 8×:** 400k / 800k / 1.6M.
- **Parallel:** eight independent 1× attempts, reported as pass@1/2/4/8.
- **Sequential:** one work → self-review → revise trajectory per seed, observed at 2×/4×/8× cuts; no external grade or ground truth is shown. It stops early after two consecutive exact `NO GENUINE GAP FOUND` critiques and carries that proof forward. An audited failure that self-converged requires forced full-budget follow-up before it can be called compute-flat.
- **IdeaSearch-8 robustness:** an adaptation of [IdeaSearch](https://proceedings.iclr.cc/paper_files/paper/2025/hash/071a637d41ea290ac4360818a8323f33-Abstract-Conference.html) (Wang et al., ICLR 2025) to proof generation. Each of eight independent branches uses a fresh same-model planner (≤20k tokens), then a fresh executor given only the problem and that branch's plan (≤180k). The candidate plan is not an oracle; the executor may repair or abandon it. Branches share no context and the bank totals 8×. Run only on primary-model hint rescues that fail both standard 8× protocols.

Parallel and sequential are separate experiments. Each receives 8×; running both spends 16× and is never reported as one 8× cell. Verifier-guided tree search is outside scope because no validated proof-level process verifier exists.

## Arms

| Arm | Condition | Gating | Runs |
|---|---|---|---|
| `baseline` | no hint, 1× | all 35 | 3 |
| `baseline-parallel` | parallel scaling | baseline failures | +5; 8 samples total |
| `baseline-sequential` | sequential 2×/4×/8× cuts | baseline failures | 3 trajectories |
| `baseline-ideasearch` | IdeaSearch-8 | primary-model robustly compute-flat hint rescues | 8 branches |
| `placebo-hint` | placebo, 1× | non-ceiling | 3 |
| `hint` | strategy hint, 1× | non-ceiling | 3 |
| `hint-sequential` | strategy hint, sequential 8× | hint failures | 3 |

The standard design has at most 20 completed trajectories per problem × main model. `baseline-ideasearch` adds one eight-branch bank only on surviving headline cases.

## Analysis and grading

- A run succeeds at audit score ≥5. In three-run cells, **solved = ≥2/3**, **failed = 0/3**; 1/3 cases are reported separately and excluded from discrete profiles. Parallel success is pass@k over its eight-sample bank; IdeaSearch-8 succeeds if any of its eight audited proofs passes.
- **Primary tests:** compare hint-responsive with hint-unresponsive problems separately at parallel 8× and sequential 8× using one-sided problem-level permutation tests with Holm correction.
- **Boundary shift:** among Opus baseline failures, compare Fable baseline success on Opus hint-responsive / robustly compute-flat problems versus remaining failures, conditioning on pre-run human difficulty and baseline pass rate. Also report Sonnet → Opus → Fable transition matrices.
- **Saturation:** freeze a practical-equivalence margin on development data and test whether the aggregate 4×→8× gain remains below it.
- **Continuous analysis:** model hint response and parallel/sequential compute response separately, conditioning on baseline success; use model fixed effects, problem random effects, and problem bootstrap uncertainty.
- **Operational sensitivity:** report all session reconnects; if a headline cell depends on a recovered attempt, repeat it uninterrupted or show that excluding recovered attempts leaves the conclusion unchanged.
- Any `baseline-ideasearch` rescue narrows the conclusion to ordinary parallel sampling and sequential revision.

**Proof grading.** Scores are 7 (complete), 6 (one obvious one-line gap), 5 (two or three such gaps), or 0. Auditors see only the proposed proof, not its arm or hint. Fable grades sub-frontier outputs; GPT-5.6 Sol grades Fable outputs.

**Human validation.** Two medalists independently grade 45 solutions stratified across models, arms, and automated outcomes; a third adjudicates. Report precision/recall at ≥5, one-sided precision lower bound, Cohen's κ, and human agreement. If the precision criterion fails, humans re-grade pivotal cells.

Include one fully read case study per observed profile, especially failures where the model receives the correct object but executes the wrong invariant.

## Budget and discipline

The full standard design costs at most 62 token-units per problem × main model, or ≈868M output tokens across 35 problems × Opus and GPT, before gating. IdeaSearch-8 adds 1.6M per surviving Opus case. Each baseline + hint ladder/open-weight model adds at most ≈42M before targeted follow-up.

Nothing runs unless it feeds a primary test, the boundary-shift result, a figure, or a frozen prediction.
