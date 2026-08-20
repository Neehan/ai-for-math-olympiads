# Strategy-Conditioned Test-Time Scaling

## Question

When a model fails a fresh olympiad proof, can more depth, breadth, or self-generated strategy diversity recover it—or is external strategic information the missing resource?

> **After unaided depth, breadth, and strategy diversification saturate, a short correct strategy can reopen productive inference.**

The claim is about the tested models, problems, protocols, and budgets. The strategy is an oracle diagnostic, not a deployable source of free information.

## Development signal

On the 13-problem Opus 4.8 combinatorics pilot, where solved means at least 2/3 blinded audits score ≥5:

- unaided cumulative coverage is `4/13 → 6/13 → 6/13 → 6/13` at `1×/2×/4×/8×`;
- Parallel-8 and Uniform-C-8 each rescue `0/7` unaided survivors;
- existing standalone-hint and partially completed hinted Self-Refine artifacts indicate that all seven survivors can execute the supplied strategy, but these development arms were adaptively spliced.

This is development evidence, not the confirmatory result. The primary hinted curve will use hinted Self-Refine's own `1×/2×/4×/8×` checkpoints on all seven survivors; standalone hint is retired from primary plots.

## Frozen experiment

The study is an adaptive screen-and-intervene design:

1. **Baseline.** Run three independent 1× attempts on every problem. Solved = ≥2/3.
2. **Unaided scaling.** Run three 8× Self-Refine trajectories only on baseline failures and audit their exact `2×/4×/8×` prefix cuts. For first-passage coverage, a problem remains solved from the first cut reaching ≥2/3.
3. **Stress controls.** On the same frozen baseline-failure cohort, run one fresh Parallel-8 bank and one Uniform-C-8 bank. Parallel uses eight fresh IID 1× attempts. Uniform-C-8 is a proof-domain adaptation of coarse-grained [TTS-Uniform](https://arxiv.org/abs/2509.17905) without entropy filtering: an 80k shared extractor produces up to eight semantically distinct whole-proof strategies, followed by exactly eight fresh 190k executors (`1.6M` total) allocated cyclically across the deduplicated set. It omits answer-entropy filtering and majority voting, which do not define a proof-level selector here. Report the two controls separately and never sum them into one compute curve.
4. **Strategy intervention.** On the same frozen survivor cohort, run three fresh hinted Self-Refine trajectories from the supplied pre-authored ≤25-word strategy. Audit each trajectory's own exact `1×/2×/4×/8×` prefix cuts. Do not splice standalone-hint attempts into this curve.

Strategies are written and audited for every held-out problem before any confirmatory outcomes are inspected, even though they are executed only after the unaided screen. The 13 combinatorics problems are development data. Five of the remaining 22 problems will be selected before inspection to complete an 18-problem fitting set; the other 17 remain sealed for held-out prediction. Claude and GPT are evaluated under the same protocol, with an open-weight proof model as an additional replication. Each model defines its own survivor cohort and fitted dynamics.

## Measurements

- **Correctness:** success = blinded audit score ≥5; any substantive gap scores 0. Repeat at ≥6 and human-check every headline proof.
- **Reliability arms:** report raw `0/3`–`3/3`; primary solved = ≥2/3, unstable = 1/3, failed = 0/3. Report 3/3 as sensitivity.
- **Parallel-8:** one bank of eight fresh IID attempts; report `c/8` and the standard unbiased pass@k estimate for `k∈{1,2,4,8}`. It is a search stress test, not a three-replicate reliability estimate.
- **Uniform-C-8:** one shared strategy extractor, `m∈[1,8]` distinct plans, and exactly eight cyclically assigned executors. Report valid proofs `c/8`, `m`, the assignment map, and how many plans yield at least one valid proof; do not report pass@`k` for its dependent branches. Analyze it separately from Parallel-8.
- **Strategy:** one frozen audited hint of at most 25 words. It may state the key idea, but not the answer, a substantial derivation, or a proof sketch.
- **States:** label every audited proof artifact symmetrically. Audit score ≥5 gives `S`; missing text is unobserved. For other artifacts, annotate recognition of each step in the frozen outline using its matching reference: 3/3 gives provisional `P`, otherwise `U`. Fit dynamics only from the unaided and oracle-sketch conditions; placebo and search-bank labels are controls and route-discovery diagnostics. Double-annotate U/P decisions and adjudicate materially different candidate routes before fitting the general-strategy model.
- **Budget:** 1× is at most 200k eligible output tokens, including hidden reasoning, visible text, and tool calls. Over-budget artifacts cannot count.
- **Reporting:** include every late success, exact allocated and realized tokens, first-passing budget, and uncertainty over problems. Problems—not attempts or search branches—are the inferential units.
- **Strategy-search audit:** classify each unique Uniform plan as `U/P`, blinded to executor outcome, and grade its assigned proof separately. This directly distinguishes failure to generate a viable strategy from failure to execute one and makes no claim about hidden internal thought.

## Main presentation

1. **Matched scaling curves:** cumulative unaided coverage on all problems and hinted Self-Refine coverage on the frozen unaided survivor cohort, using each arm's own `1×/2×/4×/8×` checkpoints.
2. **Search stress test:** on the frozen baseline-failure cohort, show per-problem Self-Refine, Parallel-8, Uniform-C-8, and hinted Self-Refine outcomes, together with whether each generated plan contains a viable route.
3. **Held-out mechanism test:** fit the shared `U/P/S` dynamics and condition-specific initialization distributions on 18 problems, freeze them, and predict state occupancy and success curves on the 17 held-out problems.

The paper succeeds if held-out data show that models execute short oracle-supplied strategies that substantial unaided depth, breadth, and strategy-diversified search fail to discover, and if the fitted state model predicts where scaling saturates and where strategy reopens it. “Hints help” alone is not the claim.
