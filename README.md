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
2. **Unaided scaling.** Run three 8× Self-Refine trajectories only on baseline failures and audit their own `1×/2×/4×/8×` prefix cuts. For first-passage coverage, a problem remains solved from the first cut reaching ≥2/3.
3. **Stress controls.** On the same frozen baseline-failure cohort, run one fresh Parallel-8 bank and one Uniform-C-8 bank. Parallel uses eight fresh IID 1× attempts. Uniform-C-8 is a proof-domain adaptation of coarse-grained [TTS-Uniform](https://arxiv.org/abs/2509.17905) without entropy filtering: an 80k shared extractor produces up to eight semantically distinct whole-proof strategies, followed by exactly eight fresh 190k executors (`1.6M` total) allocated cyclically across the deduplicated set. It omits answer-entropy filtering and majority voting, which do not define a proof-level selector here. Report the two controls separately and never sum them into one compute curve.
4. **Strategy intervention.** On the same frozen survivor cohort, run three fresh hinted Self-Refine trajectories from the supplied pre-authored ≤25-word strategy. Audit each trajectory's own exact `1×/2×/4×/8×` prefix cuts. Do not splice standalone-hint attempts into this curve.
5. **Dense mechanism audit.** For Claude Opus 4.8 and GPT-5.4 only, recover the last eligible proof at every integer budget from `1×` through `8×` for both Self-Refine conditions, then apply the frozen correctness and route-progress audits. The other models remain in the main empirical comparison but do not enter the fitted state dynamics.

Strategies are written and audited for every held-out problem before any confirmatory outcomes are inspected, even though they are executed only after the unaided screen. The 13 combinatorics problems are development data. Claude, GPT, and DeepSeek V4 Flash are evaluated under the same empirical protocol, with additional open-weight proof models as replications. Each model defines its own survivor cohort; only the two pre-specified strong models enter problem-level cross-validation of the state dynamics.

## Measurements

- **Correctness:** success = blinded audit score ≥5; any substantive gap scores 0. Repeat at ≥6 and human-check every headline proof.
- **Reliability arms:** report raw `0/3`–`3/3`; primary solved = ≥2/3, unstable = 1/3, failed = 0/3. Report 3/3 as sensitivity.
- **Parallel-8:** one bank of eight fresh IID attempts; report `c/8` and the standard unbiased pass@k estimate for `k∈{1,2,4,8}`. It is a search stress test, not a three-replicate reliability estimate.
- **Uniform-C-8:** one shared strategy extractor, `m∈[1,8]` distinct plans, and exactly eight cyclically assigned executors. Report valid proofs `c/8`, `m`, the assignment map, and how many plans yield at least one valid proof; do not report pass@`k` for its dependent branches. Analyze it separately from Parallel-8.
- **Strategy:** one frozen audited hint of at most 25 words. It may state the key idea, but not the answer, a substantial derivation, or a proof sketch.
- **States:** state-audit the matched unaided and oracle-sketch Self-Refine trajectories, plus the Parallel-8 and Uniform-C-8 executor outputs used for route-discovery diagnostics. Audit score ≥5 gives `S`, which is carried forward to later sequential checkpoints; missing text before success is unobserved. For other artifacts, annotate recognition of each step in the frozen outline using its matching reference. Within a trajectory, `P` means the recognized-step count increased from the preceding observed checkpoint (the first artifact is compared with zero), or that all three steps remain recognized while proof execution continues; `U` means an incomplete count stayed flat or decreased. For a search-bank executor, all three steps present means that the frozen route is present. Fit discrete `U/P/S` transition matrices, each with four free probabilities, only from the matched Self-Refine trajectories of Opus 4.8 and GPT-5.4 at every integer `1×` increment. Standalone baseline, hint, placebo, and outline arms receive correctness audits only. Compare shared against condition-specific transition matrices rather than assuming the sketch acts only through initialization.
- **Budget:** 1× is at most 200k eligible output tokens, including hidden reasoning, visible text, and tool calls. Over-budget artifacts cannot count.
- **Reporting:** include every late success, exact allocated and realized tokens, first-passing budget, and uncertainty over problems. Problems—not attempts or search branches—are the inferential units.
- **Strategy-search audit:** classify each unique Uniform plan as `U/P`, blinded to executor outcome, and grade its assigned proof separately. This directly distinguishes failure to generate a viable strategy from failure to execute one and makes no claim about hidden internal thought.

## Main presentation

1. **Matched scaling curves:** cumulative unaided coverage on all problems and hinted Self-Refine coverage on the frozen unaided survivor cohort, using each arm's own `1×/2×/4×/8×` checkpoints.
2. **Search stress test:** on the frozen baseline-failure cohort, show per-problem Self-Refine, Parallel-8, Uniform-C-8, and hinted Self-Refine outcomes, together with whether each generated plan contains a viable route.
3. **Held-out mechanism test:** for Opus 4.8 and GPT-5.4, fit the discrete `U/P/S` dynamics on training problems and early `1×` increments, then predict held-out problems and their `5×`–`8×` state occupancy and success curves. Keep this contribution only if it beats simpler time-only, two-state, route-count, and history-aware predictors.

The paper succeeds if held-out data show that models execute short oracle-supplied strategies that substantial unaided depth, breadth, and strategy-diversified search fail to discover, and if the fitted state model predicts where scaling saturates and where strategy reopens it. “Hints help” alone is not the claim.

## Results backup

Set `HF_TOKEN` in `.env`, then incrementally upload both ignored result trees to the private Hugging Face dataset:

```bash
./scripts/upload_results_to_hf.sh
```

The default destination is `notadib/strategy-ceiling`. Override it with `HF_RESULTS_REPO` and adjust upload concurrency with `HF_UPLOAD_WORKERS`; neither setting is required.
