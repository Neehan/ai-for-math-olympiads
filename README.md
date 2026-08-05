# Paper 1: One-Line Hints Beat 8× Compute (ICLR 2027, ~7 weeks)

**Claim.** On hard, novel olympiad problems, baseline failures split operationally into two kinds: problems a one-line hint rescues at 1× compute stay unsolved at 8× compute without the hint, while problems no hint rescues are mostly closed by compute alone. Labels are assigned from the hint axis at 1× only; the prediction is tested on the independent compute axis at 8× — a double dissociation, not a re-labeling. Cognitive terms (capability/rigor, recognition/execution) appear only as interpretation in the discussion. Hints are oracle probes, not a proposed method.

## Design: 2×2 factorial, {no-hint, hint} × {1×, 8×}

Every non-ceiling problem runs in all four cells, k = 3 independent runs per cell per model. Labels come from the two 1× cells; the two 8× cells are the pre-registered predictions (bold = the test).

| Label | no-hint 1× | hint 1× | no-hint 8× (predicted) | hint 8× (predicted) |
|---|---|---|---|---|
| ceiling | solved | — | — | — |
| **capability gap** | failed | solved | **failed** | solved |
| **rigor gap** (provisional) | failed | failed | **solved** | solved |
| both gaps | failed | failed | failed | solved |
| beyond | failed | failed | failed | failed |

The headline figure is the saturation curve: no-hint solve rate at 1×/2×/4×/8×, per bucket, one curve per compute channel (parallel pass@1/2/4/8 and sequential budget cuts — see Compute below). Rigor-labeled problems rise with compute; capability-labeled problems stay flat at ~0 across all four budgets in both channels. We never claim "never solves" — we claim flat at four doublings under both ways of spending them, which is falsifiable. The companion stat: a one-line hint at 1× rescues more baseline failures than 8× compute at zero hints, and hint + 8× together clear N/35.

Why this is causal and not circular: the labeling intervention (hint at 1×) and the validating intervention (compute at 8×) are independent manipulations; neither the labels nor the prediction touches the runs that test it.

## Compute = output-token budget

Compute is operationalized as the **total output-token budget of an attempt** (thinking + visible text + tool-call text), enforced by a harness cutoff. Output tokens are the right knob because input tokens are dominated by cache reads and context re-feeding and do not measure model effort; wall-clock confounds with API latency.

- **1× = 200k output tokens** per attempt: a full serious agentic attempt on one problem (a strong single trajectory with self-checking, empirically 100–250k on hard problems).
- **2× / 4× / 8× = 400k / 800k / 1.6M** — 8× is not symbolic: it equals eight full independent attempts' worth of tokens, the entire parallel channel's spend, delivered to one problem.
- **Reasoning effort is fixed at high** (each model's strongest standard setting) in every cell, arm, and channel — effort changes how densely each step thinks, so letting it vary would break "only the token cutoff moves between cells." Per-model config in the appendix.
- **8× is delivered through both canonical channels**, because they fail differently and the capability claim must survive both:
  - **Parallel (diversity):** 8 independent 1× attempts; solved = any attempt passes ground-truth grading. No model-based selector — pass@8 with oracle grading upper-bounds every best-of-n-with-verifier policy, since a verifier can only choose from what sampling produced. The 8 samples also yield the full parallel curve (pass@1/2/4/8, unbiased estimator) at no extra cost.
  - **Sequential (depth):** one work → self-review → revise trajectory per seed, budget cut at 2×/4×/8× for the curve; the model sees its own previous attempt and critique, never an external grade or ground truth.
- A capability gap that survives both — 8 independent tries never find the idea, 8× of self-critique never converges to it — is unclosable by test-time compute in either currency.
- These two channels plus fixed-high effort cover the standard test-time-scaling taxonomy: parallel sampling vs. sequential revision remains the canonical split in the current literature (Agarwal et al. 2025, arXiv:2512.02008; Gu et al. 2026, arXiv:2604.05868), with each channel implemented per its origin method (repeated sampling — Brown et al. 2024; self-refine — Madaan et al. 2023). The remaining family, verifier-guided search, is deliberately absent: no off-the-shelf process verifier exists for proof-level math, its selection component is upper-bounded by oracle-graded pass@8, and building a proof PRM is a method contribution (paper 2), not a compute currency.
- All token spend is logged per attempt and reported per cell; every headline comparison is token-matched (hint + 1× vs. no-hint at equal total tokens; 8 parallel seeds = one sequential 8× trajectory = 1.6M).

## Problems

35 fresh 2026-contest problems models fail or struggle on, spanning hardness types (idea-hard P3/P6-like and grind-hard P2/P4-like: long computation, exhaustive cases), panel-classified from human solutions before any model run. Algebra / combinatorics / number theory only — geometry excluded as a measurement-validity decision (large coordinate derivations make proof-level auditing unreliable under our grading protocol); conclusions restricted accordingly. Post-cutoff, no retrieval corpus, selection criteria frozen before runs.

**Models.** Contamination rule: every model's published training cutoff must predate the earliest contest (Feb 2026); we state each model's cutoff next to each contest date in the appendix.

- **Sub-frontier mains (full 2×2 + both compute channels): Opus 4.8 (cutoff Jan 2026) + GPT-5.5.** Cross-lab replication; all primary statistics are within-model, paired across problems. GPT-5.6 is excluded as contaminated (cutoff postdates the problem set).
- **Frontier (boundary shift): Fable 5 (cutoff Jan 2026)**, no-hint 1× on all 35 (k = 3, same 200k cap, same harness, same grading) + H1/H2 at 1× on its baseline failures only. No saturation sweep — Fable's role is the boundary-shift claim: sub-frontier capability gaps becoming frontier baseline solves shows hint-closable gaps are exactly the ones the next model tier absorbs, making hint response a cheap forecast of scaling. Fable failures that H2 rescues remain hint-closable at the frontier; failures H2 does not rescue feed the case-study section.
- **Open-source replication: reserved for October rebuttal** (discipline rule) — one open-weights reasoning model with a verifiable pre-Feb-2026 cutoff, for permanent reproducibility.

**Ceiling screen:** a problem solved 3/3 at no-hint/1× is ceiling and dropped from analysis (pre-registered). All gating below is within-arm at lower budget (monotonicity: a larger budget with the same harness can only help, since the harness may stop early), so there is no cross-arm selection coupling.

## Hints

- **Authors:** the IMO medalist panel, from problems + official human solutions ONLY, before any model run exists. Hints frozen, committed, hashed, published verbatim in the appendix. A panelist who did not author a problem's hints audits them for strategic leakage and template compliance.
- **H1 — placebo:** restates the problem's objects plus true-but-useless context, zero strategy; length-matched to H2 so a short real hint beating longer padding is a stronger result, not a length confound.
- **H2 — the one-line hint (the "hint" arm of the 2×2):** the key idea as up to 5 tag keywords, each a well-known named technique from a standard source, no steps; tagged at authoring as **analog** (known past technique) vs **novel** (original insight).
- Hint effects are measured against H1, and H1 against no-hint globally (padding may distract; every H2 − H1 effect is partly "useful hint vs. harmful padding").
- No outline tier: two tiers are the minimal set — H2 is the arm, H1 kills the length/attention confound. A deeper outline probe on unrescued problems is rebuttal ammo (October), not core protocol.

## Arms & runs (per problem × model)

Arm slugs below are the exact names used in `config.json`, the harness CLI, and `results/` paths.

| Arm | Cell | Gating | Runs |
|---|---|---|---|
| `baseline` | no-hint 1× | all 35 (ceiling screen) | 3 |
| `baseline-parallel` | no-hint 1×, parallel channel | `baseline` failures only | +5 (8 seeds total with `baseline`) |
| `baseline-sequential` | no-hint 2×, 4×, 8×, sequential channel | `baseline` failures only | 9 |
| `placebo-hint` | H1 (placebo) 1× | non-ceiling | 3 |
| `hint` | H2 1× | non-ceiling | 3 |
| `hint-sequential` | H2 8×, sequential channel | `hint` failures only | 3 |

Worst case 26 runs per problem × model; gating makes the realistic average far lower. No best-of-N-with-verifier arm, no multi-agent arm — paper 2. The no-hint 8× cell of the 2×2 counts as failed only if **both** channels fail. The hint 8× cell is deliberately sequential-only (no hint-parallel arm): it feeds no primary statistic, and the asymmetry is conservative both ways — both channels on no-hint 8× make its predicted **failed** harder to sustain, while sequential-only hint 8× can only undercount hint + 8× solves. Channel-symmetric hint-parallel on surviving failures is October rebuttal ammo, not core protocol.

## Analysis (pre-registered before any 8× run)

- Solve rate per cell from k = 3: **solved = ≥ 2/3, failed = 0/3**. Problems landing at exactly 1/3 in a labeling cell are excluded from buckets (reported separately) but kept in the continuous analysis.
- **Primary test:** among non-ceiling problems, no-hint/8× solve rate is higher for hint-unrescued (provisional rigor) than for hint-rescued (capability) problems; one-sided permutation test, resampling by problem.
- **Secondary (continuous):** hint response R_H = p_H2 − p_H1 at 1×; compute response R_C = p_8× − p_1× no-hint. Coefficient of R_H on R_C negative after conditioning on baseline solve rate; logistic mixed model (model fixed effects, problem random effects), uncertainty bootstrapped by problem.
- **Placebo-high problems** (H1 ≥ 2/3) are reported against baseline only, no mechanism claimed.
- Per-problem 3-seed calls are noisy; all statistics aggregate, paired across problems.

## Grading (validated, not trusted)

- Completeness standard: **7** = complete and rigorous; **6/5** = complete in essence, one small gap with an obvious one-line fix; **0** = anything else, with no other partial credit (a solution missing one of two required bounds scores 0). Every run graded by a frontier-model auditor other than its author, seeing the proof standalone (hint not shown, arm not disclosed): Fable 5 judges the sub-frontier models; Fable-authored runs are judged by GPT-5.6 Sol (cross-lab, so no same-family preference; judge-side contamination is harmless — knowing official solutions only sharpens verification, and the human-validation subset certifies each judge separately).
- **Human validation subset:** 45 unique solutions (30 auditor-passed — false 7s are what corrupt results — 15 auditor-failed), stratified across models × cells, enriched around pivotal cells (the bold predictions). Two medalists grade independently, a third resolves; graders blind to arm. Report precision/recall (positive = 7; precision is load-bearing) with a one-sided lower confidence bound as the acceptance criterion, Cohen's κ, and inter-medalist agreement as the ceiling. Pre-commit: if the precision bound fails, pivotal cells are re-graded by humans.

## Case studies (required section)

One problem per bucket gets a full trajectory read: what the model did with and without the hint, and — for "beyond" problems — where it broke even with the hint in hand (e.g., the observed "right object, wrong invariant" failure: the model builds the hinted construction, attaches the wrong quantity to it, refutes its own misreading, and rationally abandons the correct device). This is what an aggregate table cannot show and what preempts the "your hint just leaked the proof" objection.

## Budget

Worst case ≈ 80× token-units per problem × model (3 baseline + 5 parallel + 42 sequential + 30 hint arms) ≈ 16M output tokens; across 35 problems × 2 sub-frontier models ≈ 1.1B output tokens worst case, realistically ~half after the ceiling screen and gating. Fable adds ≈ 21M (105 × 1× baseline) plus hint cells on its failures only — ~2–3% of total spend at frontier pricing. Auditor grades everything; humans grade 45–60.

**Discipline rule:** nothing runs that doesn't feed the primary test, a figure, or a pre-registered prediction. Rebuttal ammo (extra model, extra seeds, analog-pointer tier) happens in October, not now.
