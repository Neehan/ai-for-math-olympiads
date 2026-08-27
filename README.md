# Strategy Access and Test-Time Scaling

## Research question

When a reasoning model fails a fresh olympiad proof, is the missing resource more inference or access to the right strategy?

Our primary hypothesis is a discovery--execution dissociation:

> A model can execute a concise strategy that substantial inference devoted to finding strategies does not discover.

The claim is finite and operational: it concerns the tested models, problems, inference procedures, and budgets. A frozen oracle sketch is a diagnostic intervention, not a deployable source of free information.

## Contributions

### 1. Strategy access versus inference compute

Compare matched unaided and strategy-conditioned inference. The central cases are problems where depth, independent sampling, and diversified plan search do not produce a viable strategy, while the same model turns a frozen audited sketch of at most 25 words into a valid proof. This is the paper's required contribution.

### 2. Explain the scaling dynamics

Each problem has a human-verified three-step outline of one complete oracle strategy. Retrospective annotations derive `U/P/S` states from changes in recognized mechanisms at every Self-Refine checkpoint. The proposed mechanism is that persistent `U→U` transitions generate acquisition plateaus, while entry into `P` raises the probability of complete strategy acquisition. Keep this contribution only if condition-specific transition matrices predict held-out state occupancy and acquisition coverage better than simpler alternatives. These oracle-dependent states explain the dynamics; they are not deployment-time signals.

### 3. Predict whether more inference is worthwhile

Among unaided trajectories unsolved at 2×, predict whether they produce a valid proof by 8× using only information available without an oracle outline: the problem, generated artifacts, critique history, budget, model, and whether an artifact is missing. Fit and freeze the predictor on the 35-problem benchmark, then test it without refitting on the 22 locked non-geometry problems from the Advanced split of IMO-ProofBench. Compare against model-only and Bayesian base-rate predictors; report precision, recall, calibration, compute saved, and recovered solves at a predeclared operating point.

## Experiment arms

One unit of compute is at most 200k eligible output tokens, including hidden reasoning, visible output, and tool use.

1. **Baseline:** three independent 1× attempts on every problem.
2. **Hint:** fresh 1× attempts with the frozen ≤25-word oracle sketch. This establishes the immediate strategy effect; it is not spliced into sequential curves.
3. **Placebo:** the same wrapper containing the next problem's frozen oracle hint after a lexicographic cyclic shift within domain, controlling for mathematical density and strategy-prompt effects without correct problem--strategy alignment.
4. **Unaided Self-Refine:** three trajectories with checkpoints at every integer budget through 8×.
5. **Hinted Self-Refine:** three fresh trajectories under the same protocol and budgets, with the oracle sketch retained in context.
6. **Parallel-8:** eight independent 1× attempts. Report per-problem `c/8` and pass@\(k\) for \(k\in\{1,2,4,8\}\).
7. **Uniform-C-8:** one shared 80k strategy extractor proposes up to eight deduplicated whole-proof plans; eight fresh 190k executors are assigned cyclically. Report plan coverage and executor outcomes separately; dependent branches are not pass@\(k\).
8. **Proposal/selection diagnostic:** `baseline-uniform-strategy-only` freezes the same extractor's raw proposals without executors. `baseline-uniform-compress` samples three proposals without replacement or outcome filtering under a fixed seed and uses GPT-5.6-sol to compress each to at most 25 words while preserving its route and errors. Problems with fewer than three proposals are excluded and reported. The same three-step strategy audit labels reference-strategy acquisition; before rankings are run, blinded expert review may additionally accept a genuinely valid alternative strategy and records that provenance separately. A deterministic builder then freezes the canonical `strategy_acquired` label and candidate text in `hard_hint_selection.jsonl`. `selection` asks the source model to rank those three sketches plus the oracle sketch with the problem under a tool-free, one-turn, 20k-output-token (0.1×) cap, while `selection-no-problem` uses the identical candidate order and cap without the problem to measure oracle-style leakage. We report whether the top-ranked candidate acquired a verified strategy and, separately, the exact oracle rank. For GPT-5.4 only, `selection-10k` and `selection-40k` repeat the identical task and randomized orders at 0.05× and 0.2× as a budget sensitivity.

The main comparison is unaided versus hinted Self-Refine. Run Baseline, Hint, Placebo, and both Self-Refine conditions on all 35 non-geometry algebra/combinatorics/number-theory problems. Run Parallel-8 and Uniform-C-8 as search-coverage stress tests on the frozen baseline-failure cohort. Then replicate the core comparison on the 22 locked non-geometry problems from the Advanced split of IMO-ProofBench for selected models.

## Auditing and measurement

- Proof success is blinded audit score ≥5/7; report ≥6 and human verification as sensitivities. Problems are inferential units.
- Reliability arms report every `0/3`--`3/3` cell; primary reliable success is ≥2/3.
- Strategy audits record the three human-verified oracle-outline mechanisms and separately adjudicate complete alternative strategies.
- Planner-only and compressed proposals receive the same frozen three-mechanism audit. Any candidate rejected only because it follows a different route is adjudicated by blinded experts before selection; accepted alternatives carry a short adjudication note. Selection cannot run until every displayed generated candidate has a canonical `strategy_acquired` label bound to its exact compressed text.
- `S` begins when all three verified mechanisms appear together or a valid proof is produced, and is thereafter absorbing. Before acquisition, `P` means the incomplete recognized-step count increased from the preceding observed checkpoint; `U` means it stayed flat or decreased. Missing output is `NA`, not `U`, and does not update the comparison point.
- Retain tokens, first-passing budget, late successes, plan assignments, audit agreement, prompts, endpoints, and configuration.

## Current evidence

Current audited curves show the intended separation, but the full balanced experiment and search controls are still being completed:

- Muse Spark 1.2: unaided `8→8/35`, hinted `20→21/35` from 1× to 8×.
- Claude Opus 4.8: unaided `17→23/35`, hinted `24→34/35`.
- GPT-5.4: unaided `17→22/35`, hinted `28→34/35`.
- GPT-5.5: unaided `28→31/35`, hinted `35→35/35`.

Across the currently complete dense annotations, unaided productive checkpoints acquire a complete strategy in the next increment in `9/103` cases, versus `10/792` after unproductive checkpoints. The unproductive self-loop is `0.934`. These are repeated transitions within problems, so inference clusters by problem; the current estimate is exploratory until the frozen analysis is confirmed externally.

Preliminary problem-level validation also favors the three-state explanation over an acquired/not-acquired chain when the oracle annotations are observed, but missingness is informative and a simple `P/U/NA` rule is currently competitive with the full Markov model. The Markov contribution stays only if it predicts held-out state occupancies and acquisition curves better than simpler baselines. The separate no-reference unaided predictor remains to be frozen and tested across datasets.

## Go/no-go

- **Contribution 1 stays** if the dissociation replicates across the completed benchmark, model families, search controls, placebo, and human proof checks.
- **Contribution 2 stays** if three-step oracle-strategy acquisition and the resulting `U/P/S` dynamics predict held-out state occupancies and acquisition curves better than simpler alternatives, with complete alternative strategies adjudicated separately.
- **Contribution 3 stays** only if the frozen no-reference 2× predictor generalizes without refitting to the locked external set and beats model-only and Bayesian baselines. Otherwise report the retrospective state result and drop the operational classifier.

## Results backup

Set `HF_TOKEN` in `.env`, then incrementally upload both ignored result trees to the private Hugging Face dataset:

```bash
./scripts/upload_results_to_hf.sh
```

The default destination is `notadib/strategy-ceiling`; override it with `HF_RESULTS_REPO` if needed.
