# Strategy Access and Test-Time Scaling

## Research question

When a reasoning model fails a fresh olympiad proof, is the missing resource more inference or access to the right strategy?

Our primary hypothesis is a discovery--execution dissociation:

> A model can execute a concise strategy that substantial inference devoted to finding strategies does not discover.

The claim is finite and operational: it concerns the tested models, problems, inference procedures, and budgets. A frozen oracle sketch is a diagnostic intervention, not a deployable source of free information.

## Contributions

### 1. Strategy access versus inference compute

Compare matched unaided and strategy-conditioned inference. The central cases are problems where depth, independent sampling, and diversified plan search do not produce a viable route, while the same model turns a frozen audited sketch of at most 25 words into a valid proof. This is the paper's required contribution.

### 2. Explain the scaling dynamics

Each problem has a human-verified three-step outline of one complete oracle route. Retrospective annotations record which of those three mechanisms appear at every Self-Refine checkpoint. A valid proof is `S`; before success, `P` means the recognized-step count increased from the preceding observed checkpoint, or all three remain recognized during execution, and `U` means an incomplete count stayed flat or decreased. Complete-route acquisition means reaching `S`, recognizing all three oracle steps, or receiving human adjudication for a complete alternative route. These oracle-dependent states explain the dynamics; they are not deployment-time signals.

### 3. Predict whether more inference is worthwhile

Among unaided trajectories unsolved at 2×, predict whether they reach `S` by 8× using only information available without an oracle outline: the problem, generated artifacts, critique history, budget, model, and whether an artifact is missing. Fit and freeze the predictor on the 35-problem benchmark, then test it without refitting on the locked 22-problem non-geometry IMO-Bench hard set. Compare against model-only and Bayesian base-rate predictors; report precision, recall, calibration, compute saved, and recovered solves at a predeclared operating point.

## Experiment arms

One unit of compute is at most 200k eligible output tokens, including hidden reasoning, visible output, and tool use.

1. **Baseline:** three independent 1× attempts on every problem.
2. **Hint:** fresh 1× attempts with the frozen ≤25-word oracle sketch. This establishes the immediate strategy effect; it is not spliced into sequential curves.
3. **Placebo:** the same wrapper containing the next problem's frozen oracle hint after a lexicographic cyclic shift within domain, controlling for mathematical density and strategy-prompt effects without correct problem--strategy alignment.
4. **Unaided Self-Refine:** three trajectories with checkpoints at every integer budget through 8×.
5. **Hinted Self-Refine:** three fresh trajectories under the same protocol and budgets, with the oracle sketch retained in context.
6. **Parallel-8:** eight independent 1× attempts. Report per-problem `c/8` and pass@\(k\) for \(k\in\{1,2,4,8\}\).
7. **Uniform-C-8:** one shared 80k strategy extractor proposes up to eight deduplicated whole-proof plans; eight fresh 190k executors are assigned cyclically. Report plan coverage and executor outcomes separately; dependent branches are not pass@\(k\).

The main comparison is unaided versus hinted Self-Refine. Run Baseline, Hint, Placebo, and both Self-Refine conditions on all 35 algebra/combinatorics/number-theory problems. Run Parallel-8 and Uniform-C-8 as search-coverage stress tests on the frozen baseline-failure cohort. Then replicate the core comparison on locked non-geometry IMO-Bench hard problems for selected models.

## Auditing and measurement

- Proof success is blinded audit score ≥5/7; report ≥6 and human verification as sensitivities. Problems are inferential units.
- Reliability arms report every `0/3`--`3/3` cell; primary reliable success is ≥2/3.
- Route audits record the three human-verified oracle-outline mechanisms and separately adjudicate complete alternative routes.
- `S` begins at the first valid proof. Before success, `P` means the recognized-step count increased from the preceding observed checkpoint or all three steps remain recognized during execution; `U` means an incomplete count stayed flat or decreased. Missing output is `NA`, not `U`, and does not update the comparison point.
- Retain tokens, first-passing budget, late successes, plan assignments, audit agreement, prompts, endpoints, and configuration.

## Current evidence

Current audited curves show the intended separation, but the full balanced experiment and search controls are still being completed:

- Muse Spark 1.2: unaided `8→8/35`, hinted `20→21/35` from 1× to 8×.
- Claude Opus 4.8: unaided `17→23/35`, hinted `24→34/35`.
- GPT-5.4: unaided `17→22/35`, hinted `28→34/35`.
- GPT-5.5: unaided `28→31/35`, hinted `35→35/35`.

The retrospective state signal is already sharp at the 2× landmark. Among currently audited unaided trajectories that remain unsolved, 8/22 productive checkpoints, 15/59 missing checkpoints, and 4/99 unproductive checkpoints reach `S` by 8×. Thus `P` is evidence that further refinement can pay off, while persistent `U` marks a much lower-recovery regime. Because these labels use the oracle outline, this is an explanatory result rather than an operational stopping rule.

Preliminary problem-level validation also favors the three-state explanation over a solved/unsolved chain when the oracle annotations are observed, but missingness is informative and a simple `P/U/NA` rule is currently competitive with the full Markov model. The Markov contribution stays only if it predicts held-out state occupancies and scaling curves better than simpler baselines. The separate no-reference unaided predictor remains to be frozen and tested across datasets.

## Go/no-go

- **Contribution 1 stays** if the dissociation replicates across the completed benchmark, model families, search controls, placebo, and human proof checks.
- **Contribution 2 stays** if three-step oracle-route acquisition and the resulting `U/P/S` dynamics predict held-out state occupancies and scaling curves better than simpler alternatives, with complete alternative routes adjudicated separately.
- **Contribution 3 stays** only if the frozen no-reference 2× predictor generalizes without refitting to the locked external set and beats model-only and Bayesian baselines. Otherwise report the retrospective state result and drop the operational classifier.

## Results backup

Set `HF_TOKEN` in `.env`, then incrementally upload both ignored result trees to the private Hugging Face dataset:

```bash
./scripts/upload_results_to_hf.sh
```

The default destination is `notadib/strategy-ceiling`; override it with `HF_RESULTS_REPO` if needed.
