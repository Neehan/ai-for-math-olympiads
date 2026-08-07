# Paper 1: Minimal Strategy Probes Predict Test-Time Scaling Boundaries

**Question.** When a model fails a hard, fresh olympiad proof, can additional inference recover it, or is missing strategic information the bottleneck?

**Hypothesis.** Strategic information and inference effort are not generally fungible. A frozen, audited, ≤25-word correct-strategy probe measures whether a model can execute the key idea. Probe-negative failures should be less likely to be rescued by unaided inference scaling, while some probe-positive failures should still resist unaided scaling because the model does not discover the strategy itself. Strategy-conditioned sequential inference tests whether the remaining failures require both resources.

The headline claim is predictive, not an unrestricted claim about capability:

> A minimal strategy probe predicts which proof failures remain unsolved through the tested test-time scaling protocols and budgets.

Supplying the strategy and extending inference are controlled interventions. Comparisons across models are supporting evidence that the boundary moves with model capability; model identity is not itself a controlled intervention.

## Experimental design

The core design crosses strategic information with sequential inference. Every cell uses three seeds.

| Arm | Strategic information | Inference | Gating |
|---|---|---|---|
| `baseline` | none | 1× | all problems |
| `hint` | ≤25-word strategy | 1× | baseline non-ceiling problems |
| `baseline-sequential` | none | exact-prefix continuation to 8× | baseline failures |
| `hint-sequential` | same frozen strategy | exact-prefix continuation to 8× | hint failures |

For each baseline failure, the four outcomes define operational response profiles:

A problem is **probe-positive** when the hint arm passes and **probe-negative** when it fails under the frozen three-seed rule.

| Profile | Hint 1× | Unaided sequential 8× | Hint sequential 8× | Interpretation |
|---|---:|---:|---:|---|
| strategy-only | pass | fail | pass carried forward | strategy substitutes for tested inference |
| inference-only | fail | pass | any | tested inference succeeds without the supplied strategy |
| complementary | fail | fail | pass | strategy and inference are jointly required |
| robust failure | fail | fail | fail | neither intervention succeeds within budget |
| dual-responsive | pass | pass | pass carried forward | either intervention can rescue the failure |

These are response profiles under specified interventions, not claims about an unknowable intrinsic cause. In particular, “robust failure” means failure through the tested protocols and 8× budget, never failure under unlimited computation.

### Predictions

1. Among baseline failures, probe-negative problems have a lower unaided scaling rescue rate than probe-positive problems and the probe adds predictive value beyond pre-run human difficulty.
2. A nontrivial subset is strategy-only: the ≤25-word probe succeeds although unaided continuation and independent sampling do not.
3. Strategy-conditioned continuation rescues some problems that neither strategy nor continuation solves alone, demonstrating complementarity.
4. The probe boundary shifts across Sonnet 5 → Opus 4.8 → Fable 5: stronger models solve more weaker-model probe-resistant failures unaided.

Predictions and analysis choices are frozen after development and before runs on the held-out 35.

## Problems and models

**Problems.** The confirmatory set contains 35 post-cutoff 2026 contest problems in algebra, combinatorics, and number theory. Selection criteria, hints, and human-hardness labels are frozen before confirmatory model runs. Geometry is excluded because long coordinate proofs are difficult to audit reliably. Every solver's published cutoff must predate the earliest contest.

- **Primary within-model tests:** Opus 4.8 and a GPT-family model with a valid cutoff.
- **Capability ladder:** Sonnet 5 → Opus 4.8 → Fable 5, using matched baseline and frozen-hint cells.
- **Open-weight replication:** baseline and hint, followed by targeted scaling under the same frozen protocol.

The controlled results are always reported within model. Cross-model boundary shifts are observational supporting analyses because models also differ in training, architecture, and post-training.

## Interventions and inference protocols

- **Strategy probe:** one frozen, independently audited, ≤25-word problem-specific hint. It may state the key strategic idea, but not the final answer, a substantial intermediate derivation, or a proof sketch.
- **Placebo control:** length-matched non-strategic text under the same instruction to use the supplied information. This tests whether effects come from mathematical content rather than prompting or authority.
- **Development outline:** an audited ~50-word ceiling used only to develop and validate shorter probes; never part of the confirmatory headline analysis.

Hints are written from the statement and official solution, checked for correctness and leakage, then committed, hashed, and published verbatim before confirmatory runs.

Compute is accumulated output tokens, including hidden reasoning, visible text, and tool-call text. Reasoning effort remains fixed.

- **1×:** 200k output tokens.
- **Sequential depth:** resume the exact failed 1× session and continue the same transcript to cumulative 2×/4×/6×/8× cuts. The model receives self-review instructions but no external grade, reference answer, or ground truth. A success is carried forward to later cuts.
- **Parallel breadth:** eight independent baseline-prompt attempts capped at 1× each, reported as pass@1/2/4/8 together with realized token usage. Because attempts may stop early, this is a fixed-sample experiment, not guaranteed to consume exactly 8× tokens.
- **IdeaSearch-8 robustness:** on selected cases failing both standard protocols, eight independent same-model planners each produce a plan for a fresh executor. This tests whether explicit strategy diversification changes the conclusion.

Sequential depth and parallel breadth are separate interventions and are never pooled into one “8×” result. Provider reconnections preserve the same conversation UUID and cumulative output-token accounting and are fully logged.

## Analysis and grading

- A proof succeeds at audit score ≥5: 7 is complete, 6 has one obvious one-line gap, 5 has two or three such gaps, and 0 is a substantive failure.
- In three-seed cells, **solved = ≥2/3**, **failed = 0/3**, and 1/3 is reported as unstable rather than forced into a discrete profile.
- The primary predictive analyses compare unaided rescue rates for probe-positive and probe-negative baseline failures, separately for sequential depth and parallel breadth. Report problem-level uncertainty and whether the probe adds predictive value beyond pre-run human difficulty.
- The interaction analysis tests whether hint-sequential success exceeds what is explained by hint-only and sequential-only success. All four-arm transition tables are reported, not only aggregate accuracy.
- Capability-boundary shifts are reported as cross-model transition matrices and supporting regressions, not causal effects of model scale.
- Auditors see only the proposed proof, not its model, arm, hint, or response profile. Pivotal cells receive independent human grading.

The paper succeeds only if the frozen held-out experiment shows a reproducible predictive separation or a clear strategy–inference interaction. An aggregate gain from hints or reviewers alone is not sufficient.
