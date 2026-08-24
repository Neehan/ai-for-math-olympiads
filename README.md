# Strategy Access and Test-Time Scaling

## Research question

When a reasoning model fails a fresh olympiad proof, is the missing resource more inference or access to the right strategy?

Our primary hypothesis is a discovery--execution dissociation:

> A model can execute a concise strategy that substantial inference devoted to finding strategies does not discover.

The claim is finite and operational: it concerns the tested models, problems, inference procedures, and budgets. A frozen oracle sketch is a diagnostic intervention, not a deployable source of free information.

## Contributions

### 1. Strategy access versus inference compute

Compare matched unaided and strategy-conditioned inference. The central cases are problems where depth, independent sampling, and diversified plan search do not produce a viable route, while the same model turns a frozen audited sketch of at most 25 words into a valid proof. This is the paper's required contribution.

### 2. Empirical search-versus-execution decomposition

For viable-route access event \(A\), test \(\Pr(S)=\Pr(A)\Pr(S\mid A)\). Estimate access from blinded annotations of generated plans and proofs, and conditional execution from executors assigned viable plans. The oracle intervention sets access by construction. Allow alternative valid routes: matching one frozen reference route is not equivalent to \(A\).

### 3. Predict whether more inference is worthwhile

Among trajectories unsolved at 2×, use only their observed history and model to predict whether they reach `S` by 8×. Train separate unaided and hinted predictors with problem-grouped cross-validation; compare against model-only and Bayesian base-rate predictors. Keep this contribution only if it improves held-out PR-AUC, calibration, and a predeclared compute-aware operating point.

## Experiment arms

One unit of compute is at most 200k eligible output tokens, including hidden reasoning, visible output, and tool use.

1. **Baseline:** three independent 1× attempts on every problem.
2. **Hint:** fresh 1× attempts with the frozen ≤25-word oracle sketch. This establishes the immediate strategy effect; it is not spliced into sequential curves.
3. **Placebo:** the same intervention wrapper with matched nonstrategic text, controlling for extra text and instruction effects.
4. **Unaided Self-Refine:** three trajectories with checkpoints at every integer budget through 8×.
5. **Hinted Self-Refine:** three fresh trajectories under the same protocol and budgets, with the oracle sketch retained in context.
6. **Parallel-8:** eight independent 1× attempts. Report per-problem `c/8` and pass@\(k\) for \(k\in\{1,2,4,8\}\).
7. **Uniform-C-8:** one shared 80k strategy extractor proposes up to eight deduplicated whole-proof plans; eight fresh 190k executors are assigned cyclically. Report plan coverage and executor outcomes separately; dependent branches are not pass@\(k\).

The main comparison is unaided versus hinted Self-Refine. Run Baseline, Hint, Placebo, and both Self-Refine conditions on all 35 algebra/combinatorics/number-theory problems. Run Parallel-8 and Uniform-C-8 as search-coverage stress tests on the frozen baseline-failure cohort. Then replicate the core comparison on locked non-geometry IMO-Bench hard problems for selected models.

## Auditing and measurement

- Proof success is blinded audit score ≥5/7; report ≥6 and human verification as sensitivities. Problems are inferential units.
- Reliability arms report every `0/3`--`3/3` cell; primary reliable success is ≥2/3.
- Route audits record the three frozen outline ingredients and separately adjudicate alternative viable routes.
- `S` begins at the first valid proof. Before success, `P` means route evidence increased or all three ingredients remain available; otherwise `U`. For analysis, missing output maps to `U` unless all three were already recognized.
- Retain tokens, first-passing budget, late successes, plan assignments, audit agreement, prompts, endpoints, and configuration.

## Current evidence

Current audited curves show the intended separation, but the full balanced experiment and search controls are still being completed:

- Muse Spark 1.2: unaided `8→8/35`, hinted `20→21/35` from 1× to 8×.
- Claude Opus 4.8: unaided `17→23/35`, hinted `24→34/35`.
- GPT-5.4: unaided `17→22/35`, hinted `28→34/35`.
- GPT-5.5: unaided `28→31/35`, hinted `35→35/35`.

The access decomposition is not identified by checkpoint states: complete reference-route recognition is too rarely observed before success. Estimate it from plan-level coverage and execution conditioned on viable generated plans.

The 2× landmark is promising but inconclusive. Among unaided survivors, 27/180 solve by 8×; prediction reaches PR-AUC 0.36 and ROC-AUC 0.75, with about 29% precision and 52% recall. Among hinted survivors, 16/75 solve; ranking reaches PR-AUC 0.53 and ROC-AUC 0.83. By 4× only 10/163 unaided and 5/64 hinted survivors later solve, leaving too few positives.

Bayesian late-success estimates fall from about 15% after an unaided 2× failure to 6% after 4×, and from about 22% to 9% when hinted. These are population stopping priors, not personalized predictions.

## Go/no-go

- **Contribution 1 stays** if the dissociation replicates across the completed benchmark, model families, search controls, placebo, and human proof checks.
- **Contribution 2 stays** if plan-level route access and conditional execution explain observed search success without treating one reference route as exhaustive.
- **Contribution 3 stays** only if the frozen 2× predictor generalizes by problem and beats model-only and Bayesian baselines. Otherwise report saturation descriptively and drop the classifier.

## Results backup

Set `HF_TOKEN` in `.env`, then incrementally upload both ignored result trees to the private Hugging Face dataset:

```bash
./scripts/upload_results_to_hf.sh
```

The default destination is `notadib/strategy-ceiling`; override it with `HF_RESULTS_REPO` if needed.
