# Paper 1: When More Inference Fails to Find an Executable Strategy

## Question and claim

When a model fails a fresh olympiad proof, can substantial unaided inference recover it, or does the model fail to discover a strategy it could execute?

We first stress-test unaided inference under sequential depth, parallel breadth, and explicit strategy diversification. Only afterward do we supply a frozen, audited, ≤25-word strategy. The hint never defines which problems receive unaided scaling.

> Some failures survive strong unaided inference—including explicit strategy diversification—yet become solvable once the same model receives a short correct strategy.

This is a discovery–execution gap under specified models, protocols, and 8× budgets. It does not imply that unlimited inference could never discover the strategy.

## Experiment sequence

### 13-problem development pilot

Complete this pilot before freezing the 35-problem confirmation. Baseline, hint, sequential, and Uniform Strategy Search use three independent replicates; each parallel bank comprises the prespecified eight independent 1× attempts.

1. **Baseline 1×:** run all 13 problems.
2. **Define the scaling set:** include every problem that fails baseline in at least one seed.
3. **Sequential 8×:** run the entire scaling set with audited 1×/2×/4×/8× cuts.
4. **Parallel 8×:** run the same scaling set with eight independent 1× attempts.
5. **Define depth–breadth survivors:** problems with no success under either 8× protocol.
6. **Uniform Strategy Search-8:** run every depth–breadth survivor. Inspired by the strategy-extraction and uniform-allocation components of [TTS-Uniform](https://arxiv.org/abs/2509.17905), an 80k shared planner enumerates strategies and eight fresh 190k executors are allocated across them (`80k + 8×190k = 1.6M`). Blinded proof audit measures candidate coverage and a separate trace audit records whether the later oracle strategy's key idea appeared.
7. **Define full-search survivors:** depth–breadth survivors with no successful proof under Uniform Strategy Search-8.
8. **Hint 1×:** now evaluate the frozen ≤25-word strategy on all 13 problems; the primary outcomes are strict witness prevalence among all problems and rescue rate among full-search survivors.
9. **Strategy-conditioned scaling:** among full-search survivors with stable hint failure (0/3), run both hint-sequential and hint-parallel. Problems already rescued by hint 1× need no expensive hint-conditioned arm. These gated arms measure residual rescue, not a full-factorial interaction effect.

All problems satisfying a gate are included. Stable baseline successes are carried forward rather than given unnecessary 8× runs.

**Current status.** Baseline is complete and audited: 39/39 attempts. Eleven problems fail at least once and form the scaling set; the original nine plus `apmo-2026-05` and `china-tst-2026-12`. Baseline-sequential currently has seed 1 for the original nine, leaving 24 attempts: seeds 2–3 for those nine and all three seeds for the two additions. Hint seed 1 is complete; its early execution during development does not alter the survivor-first analysis.

### 35-problem confirmation

After the pilot, freeze the prompts, hints, gates, success threshold, and hypotheses. Repeat the sequence on all 35 post-cutoff algebra, combinatorics, and number-theory problems. Primary claims are within model; GPT-family and open-weight replications test generality, while cross-model comparisons are supporting observational evidence.

## Protocol

- **Success:** blinded audit score ≥5. Scores 7/6/5 mean complete, one obvious one-line omission, or two to three such omissions, each locally checkable without a new idea; any substantive gap is 0. Every headline witness receives independent human validation, and score ≥6 is reported as a sensitivity analysis.
- **Three-seed reporting:** report the raw `0/3`–`3/3` pass count. For summaries, solved = ≥2/3, unstable = 1/3, and failed = 0/3.
- **Strategy:** one frozen, independently audited hint of at most 25 words. It may state the key idea, but not the answer, a substantial intermediate derivation, or a proof sketch.
- **1×:** a strict 200k eligible-output-token tier, including hidden reasoning, visible text, and tool calls. Work stops around 180k, followed when possible by one tool-free response capped at 20k; any phase ending beyond 200k is logged but cannot count.
- **Sequential 8×:** one exact-prefix trajectory with self-review and cumulative budget cuts, but no grade, reference solution, or ground truth.
- **Parallel 8×:** eight independent attempts under the matched baseline or hint prompt, reported as pass@1/2/4/8 with allocated and realized tokens.
- Sequential depth and parallel breadth are analyzed separately; neither is called “unlimited compute.”

## Hypotheses and analyses

1. **Unaided saturation:** rescues are front-loaded, with little additional audited success from 4× to 8× under sequential and parallel inference separately.
2. **Conditional capability beyond the boundary:** a reproducible fraction of full-search survivors is rescued by the ≤25-word strategy at 1×.
3. **Discovery–execution gap:** for the strongest cases, the supplied key idea is absent from all recorded unaided outputs while hint 1× succeeds. If it appears unaided, the case is classified as abandonment or failed execution rather than failed discovery.
4. **Complementarity:** some full-search survivors with stable hint failure pass hint-sequential or hint-parallel, showing that strategic information can make additional inference productive without replacing proof-development work.

Baseline stability and first-passing budget are exploratory diagnostics, not confirmatory predictions.

Report full seed-level transitions, unstable cases, every late rescue, and the mean, median, standard deviation, and maximum realized tokens for each arm. Pivotal proofs and strategy-presence judgments receive independent blinded human review.

The paper succeeds only if held-out results show reproducible short-strategy rescues beyond strong unaided inference. Aggregate gains from hints, reviewers, or stronger models alone are insufficient.
