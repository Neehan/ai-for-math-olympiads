# Paper 1: Inference Scaling Saturates Before Conditional Capability Is Exhausted

**Question.** When unaided inference scaling stops producing new proofs, has the model exhausted its reasoning capability, or has it failed to discover the strategy needed to use that capability?

**Hypothesis.** Unaided inference rescues are front-loaded: nearly every proof recovered through an 8× sequential budget is already recovered by 4×, after which marginal returns are negligible. Yet a frozen, audited, ≤25-word strategy solves a reproducible fraction of failures surviving the full unaided budget. Strategy-conditioned sequential inference tests whether further survivors become solvable once the missing direction is supplied.

The headline claim is bounded by the tested protocols and budgets:

> Unaided inference scaling reaches an observed plateau before the model's conditional proof capability is exhausted.

Supplying the strategy and extending inference are controlled interventions. A hint rescue after unaided 8× demonstrates conditional capability given that strategy; it does not identify an intrinsic cause of the original failure. Comparisons across models are supporting evidence that the plateau moves with model capability, not a controlled intervention on capability.

## Experimental design

The core design crosses strategic information with sequential inference. Every cell uses three seeds.

| Arm | Strategic information | Inference | Gating |
|---|---|---|---|
| `baseline` | none | 1× | all problems |
| `hint` | ≤25-word strategy | 1× | baseline non-ceiling problems |
| `baseline-sequential` | none | exact-prefix continuation to 8× | baseline failures |
| `hint-sequential` | same frozen strategy | exact-prefix continuation to 8× | hint failures |

For each baseline failure, the four outcomes define operational response profiles:

| Profile | Hint 1× | Unaided sequential 8× | Hint sequential 8× | Interpretation |
|---|---:|---:|---:|---|
| beyond-plateau strategy rescue | pass | fail | pass carried forward | one strategy sentence succeeds beyond the unaided plateau |
| unaided inference rescue | fail | pass | any | tested inference succeeds without the supplied strategy |
| strategy-conditioned rescue | fail | fail | pass | structured inference succeeds once strategy is supplied |
| surviving failure | fail | fail | fail | neither intervention succeeds within budget |
| dual-responsive | pass | pass | pass carried forward | either intervention can rescue the failure |

These are response profiles under specified interventions, not intrinsic problem types. “Surviving failure” means failure through the tested protocols and 8× budget, never failure under unlimited computation.

### Predictions

1. Nearly all unaided sequential rescues observed through 8× occur by 4×, and the aggregate 4×→8× gain is below a frozen practical-equivalence margin.
2. A nontrivial fraction of failures surviving unaided 8× is solved by the ≤25-word strategy at 1×.
3. Strategy-conditioned continuation rescues some failures that survive both the strategy at 1× and unaided continuation at 8×.
4. The observed plateau is not specific to one inference protocol: late returns are also small under independent parallel sampling, and selected survivors remain difficult under IdeaSearch-8.
5. The plateau shifts across Sonnet 5 → Opus 4.8 → Fable 5: stronger models solve more weaker-model survivors unaided.

Predictions and analysis choices are frozen after development and before runs on the held-out 35.

## Problems and models

**Problems.** The confirmatory set contains 35 post-cutoff 2026 contest problems in algebra, combinatorics, and number theory. Selection criteria, hints, and human-hardness labels are frozen before confirmatory model runs. Geometry is excluded because long coordinate proofs are difficult to audit reliably. Every solver's published cutoff must predate the earliest contest.

- **Primary within-model tests:** Opus 4.8 and a GPT-family model with a valid cutoff.
- **Capability ladder:** Sonnet 5 → Opus 4.8 → Fable 5, using matched baseline and frozen-hint cells.
- **Open-weight replication:** baseline and hint, followed by targeted scaling under the same frozen protocol.

The controlled results are always reported within model. Cross-model boundary shifts are observational supporting analyses because models also differ in training, architecture, and post-training.

## Interventions and inference protocols

- **Strategy intervention:** one frozen, independently audited, ≤25-word problem-specific hint. It may state the key strategic idea, but not the final answer, a substantial intermediate derivation, or a proof sketch. The cap defines one low-bandwidth treatment; it is not claimed to be a minimal or optimal hint length.
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
- **Primary saturation analysis:** plot cumulative audited success and first-passing budget at 1×/2×/4×/6×/8×. Test the frozen practical-equivalence claim for the 4×→8× gain and report realized tokens, problem-clustered uncertainty, and every late rescue.
- **Conditional-capability analysis:** among failures surviving unaided 8×, estimate the fraction rescued by the strategy at 1×. Report the full four-arm transition table rather than only aggregate accuracy.
- **Strategy-conditioned analysis:** report whether hint-sequential rescues failures that survive both component interventions, and at which audited token cut each rescue first appears.
- Capability-boundary shifts are reported as cross-model transition matrices and supporting regressions, not causal effects of model scale.
- Auditors see only the proposed proof, not its model, arm, hint, or response profile. Pivotal cells receive independent human grading.

The paper succeeds only if the frozen held-out experiment shows both an unaided late-budget plateau and reproducible strategy rescues beyond it. Aggregate gains from hints, reviewers, or stronger models alone are not sufficient.
