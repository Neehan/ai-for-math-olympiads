# Strategy-Conditioned Test-Time Scaling

## Question

When a model fails a fresh olympiad proof, can more depth, breadth, or self-generated strategy diversity recover it—or is external strategic information the missing resource?

> **After unaided depth, breadth, and strategy diversification saturate, a short correct strategy can reopen productive inference.**

The claim is about the tested models, problems, protocols, and budgets. The strategy is an oracle diagnostic, not a deployable source of free information.

## Development signal

On the 13-problem Opus 4.8 combinatorics pilot, where solved means at least 2/3 blinded audits score ≥5:

- unaided cumulative coverage is `4/13 → 6/13 → 6/13 → 6/13` at `1×/2×/4×/8×`;
- a frozen ≤25-word correct strategy raises coverage to `9/13` at 1×;
- strategy-conditioned Self-Refine raises it to `10/13 → 12/13 → 13/13` at `2×/4×/8×`.

This is development evidence, not the confirmatory result. Parallel-8 and Uniform Strategy Search-8 are still being completed.

## Frozen experiment

The study is an adaptive screen-and-intervene design:

1. **Baseline.** Run three independent 1× attempts on every problem. Solved = ≥2/3.
2. **Unaided scaling.** Run three 8× Self-Refine trajectories only on baseline failures and audit their exact `2×/4×/8×` prefix cuts. For first-passage coverage, a problem remains solved from the first cut reaching ≥2/3.
3. **Stress controls.** On the cohort still below 2/3 after unaided 8×, run one fresh Parallel-8 bank and one Uniform Strategy Search-8 bank. Parallel uses eight fresh IID 1× attempts. Uniform uses an 80k shared planner to produce up to eight semantically distinct whole-proof plans, followed by exactly eight fresh 190k executors (`1.6M` total). If the planner returns `m<8` plans after merging duplicates, executors are assigned cyclically across them; no filler plans are forced. The planner output is not word-limited. Report Parallel and Uniform separately; never sum them into one compute curve. Uniform is motivated by [TTS-Uniform](https://arxiv.org/abs/2509.17905).
4. **Strategy intervention.** On the same frozen survivor cohort, supply the pre-authored ≤25-word strategy for three 1× attempts. Run three hinted Self-Refine trajectories only where the 1× strategy remains below 2/3.

Strategies are written and audited for every held-out problem before any confirmatory outcomes are inspected, even though they are executed only after the unaided screen. The 13 combinatorics problems are development data; the remaining 22 problems are the untouched confirmation set. Opus is primary, with one GPT-family and one open-weight proof model as replications. Each model defines its own survivor cohort.

## Measurements

- **Correctness:** success = blinded audit score ≥5; any substantive gap scores 0. Repeat at ≥6 and human-check every headline proof.
- **Reliability arms:** report raw `0/3`–`3/3`; primary solved = ≥2/3, unstable = 1/3, failed = 0/3. Report 3/3 as sensitivity.
- **Parallel-8:** one bank of eight fresh IID attempts; report `c/8` and the standard unbiased pass@k estimate for `k∈{1,2,4,8}`. It is a search stress test, not a three-replicate reliability estimate.
- **Uniform-8:** one shared planner, `m∈[1,8]` distinct plans, and exactly eight cyclically assigned executors. Report valid proofs `c/8`, `m`, the assignment map, and how many plans yield at least one valid proof. Analyze it separately from Parallel-8.
- **Strategy:** one frozen audited hint of at most 25 words. It may state the key idea, but not the answer, a substantial derivation, or a proof sketch.
- **Budget:** 1× is at most 200k eligible output tokens, including hidden reasoning, visible text, and tool calls. Over-budget artifacts cannot count.
- **Reporting:** include every late success, exact allocated and realized tokens, first-passing budget, and uncertainty over problems. Problems—not attempts or search branches—are the inferential units.
- **Trace audit:** for a small set of boundary examples, two blinded auditors can classify whether a predeclared decisive idea appeared in recorded unaided outputs. This is supporting evidence, not required for the population-level result and makes no claim about hidden internal thought.

## Main presentation

1. **Gated frontier:** cumulative reliable coverage under unaided `1×/2×/4×/8×`, followed by the labeled strategy intervention and strategy-conditioned `1×/2×/4×/8×`. The phases repeat the budget axis; they are not continuous total compute.
2. **Search stress test:** on the frozen unaided survivor cohort, show per-problem Self-Refine, Parallel-8, Uniform-8, strategy-only, and hinted Self-Refine outcomes.
3. **Confirmation:** report development and untouched held-out results separately; pooled 35-problem results are secondary.

The paper succeeds if held-out data reproduce an unaided plateau across depth, breadth, and self-generated strategy diversification, while the short strategy moves a meaningful fraction of those same failures into a regime where inference succeeds. “Hints help” alone is not the claim.
