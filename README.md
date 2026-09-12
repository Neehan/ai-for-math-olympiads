# When Does Long Thinking Help? Separating Strategy Access from Execution in Mathematical Reasoning

## Research question

Why do apparently useful test-time-compute allocations disagree? We test one operational hypothesis:

> Breadth and depth act on different bottlenecks: breadth can increase access to viable strategies, while depth can improve their execution as rigorous proofs.

These are observable interfaces in an inference procedure, not claims about separate neural modules. All conclusions are finite-budget statements about the tested models, problems, and controllers. The retained selector arms are exploratory and are not part of the primary claim.

## Measurement

- **Strategy proposed:** a planner artifact contains the frozen reference route or a human-adjudicated viable alternative.
- **Strategy acquired:** a proof attempt contains a complete viable route. Every independently accepted proof necessarily counts as acquired, including proofs using alternative routes.
- **Proof solved:** a blinded correctness audit scores the submitted proof at least 5/7.
For incomplete artifacts, the three-step outline of the frozen reference proof supplies a reproducible lower bound on strategy acquisition. Alternative routes used in headline cases are adjudicated separately.

## Experiments

One compute unit is at most 200k eligible output tokens.

| Arm | Allocation | Interpretation |
|---|---|---|
| `baseline` | Three independent 1× proofs | Initial end-to-end capability; cohort screen for secondary analyses |
| `baseline-sequential` | Five Self-Refine trajectories through 8× | End-to-end depth; proposal and execution remain mixed |
| `baseline-sequential-2x` | Twelve independent Self-Refine trajectories through 2× | Three homogeneous observed trials of the $N=4,K=2$ allocation |
| `baseline-sequential-4x` | Six independent Self-Refine trajectories through 4× | Three homogeneous observed trials of the $N=2,K=4$ allocation |
| `baseline-parallel` | Three bank seeds, each with eight independent 1× proofs | End-to-end breadth and eventual strategy access, not proposal alone |
| `baseline-uniform-strategy` | One 80k extractor and eight 190k executors | Explicit plans followed by balanced execution; cross-plan selection is bypassed |
| `baseline-uniform-strategy-only` | The frozen seed-1 planner artifacts without executors | Explicit proposal coverage for the realized Uniform-C bank |
| `selection` | Rank three compressed proposals plus the oracle with the problem, three independent 1× attempts | Exploratory fixed-pool strategy selection |
| `selection-no-problem` | Identical pool, order, model, and 1× protocol without the problem | Exploratory provenance/style-leakage control |
| `hint-sequential` | Five trajectories, each given one frozen ≤25-word oracle strategy and Self-Refine through 8× | Conditional execution after proposal and comparative selection are bypassed |
| `late-baseline-sequential` / `late-hint-sequential` | On an explicitly supplied problem set, fork the same fresh native 3× trajectory and continue for 1× without or with the oracle strategy | Matched estimate of whether accumulated reasoning history attenuates oracle guidance |
| `hint` / `placebo-hint` | Correct or within-domain shifted sketch at 1× | Immediate semantic-information effect and prompt-form control |

Uniform-C extracts `m≤8` strategies. Its eight executors are assigned round-robin, so each strategy receives either `floor(8/m)` or `ceil(8/m)` runs; allocation counts differ by at most one. Report planner coverage and executor outcomes separately. Uniform-C branches are dependent and are never reported as pass@`k`.

## Strategy-access and execution protocol

1. **Proposal.** Audit raw planner strategies before execution. Report reference-route matches and human-adjudicated viable alternatives separately. Planner coverage is the explicit-proposal result.
2. **Eventual access.** Audit Parallel-8 and Uniform-C executor outputs for a complete strategy and a valid proof. These are realistic mixed search procedures, not pure proposal assays.
3. **Execution.** Compare unaided and oracle-conditioned Self-Refine under the same budgets and stopping rule. Because the oracle arm begins with one verified strategy and no competitors, its scaling curve measures conditional proof execution.

Uniform-C planner and executor artifacts are audited independently under the same frozen access rule. A strategy observed only in an executor counts as eventual arm-level access, not as explicit planner proposal.

The retained selector controls are exploratory rather than part of the primary decomposition. On single-reference problems, they freeze one four-sketch candidate set from proposal seed 1, obtain expert candidate labels, and reuse the same set across three independently randomized selector attempts.

## Current GPT-5.4 evidence

On the 23 problems that fail Baseline reliability:

- Unaided Self-Refine reliably solves 10/23.
- The existing seed-1 Parallel-8 bank acquires and proves a strategy at least once on 14/23; this is provisional one-bank evidence, not the planned three-bank primary comparison.
- Uniform-C acquires a strategy on 13/23 and proves 12/23.
- The frozen reference route appears explicitly in 3/23 Uniform-C planner banks; this is a lower bound pending alternative-route adjudication.
- Oracle-conditioned Self-Refine reliably solves 22/23.

Thus current evidence strongly supports an access–execution separation: breadth recovers additional strategies, while externally supplying a verified strategy makes depth productive on nearly every failure. Selection remains unresolved until the 1× three-seed study and candidate-viability review are complete. The archived 20k and 40k runs are pilot data, not the primary selection result.

## Auditing

- Proof success is audit score ≥5/7; report ≥6 and expert adjudication as sensitivities.
- Reliability arms report all `0/3`–`3/3` cells; primary reliable success is ≥2/3.
- Raw planner proposals receive a binary frozen-reference-route audit. Human review separately accepts viable alternatives.
- Sequential and executor artifacts retain proof correctness, recognized route mechanisms, tokens, rounds, first-passing budget, and plan assignment.
- Problems are the inferential units. Parallel branches and repeated executors do not inflate the sample size.

## Scope

Run the complete study on the 35 fresh 2026 non-geometry problems. The primary GPT-5.4 depth-versus-breadth comparison uses three 8× Self-Refine trajectories and three Parallel-8 banks on every problem. Replicate the core unaided-versus-oracle comparison on the 22 non-geometry Advanced IMO-ProofBench problems. Claims are bounded to the tested finite budgets and search procedures.

## Allocation-model report

Recompute the paper's current DE/R-DE predictions and baseline comparisons
directly from the compiled Parallel-8, oracle, and unaided correctness audits:

```bash
uv run python scripts/report_allocation_model.py
```

The default report fits **DE** and **R-DE** using fresh solved counts and oracle
first-completion times only; it does not use strategy-acquisition state audits.
DE uses `alpha = min(1, q / epsilon(1))` when the empirical oracle first-block
rate is positive. If the ratio exceeds one, the constrained joint fit also
adjusts execution. If both first-block counts are zero, acquisition is
unidentified: the report uses alpha=0 and records the alpha=1 sensitivity.
R-DE fits shared Beta/Dirichlet prior parameters within each LLM using six
L-BFGS-B starts, while retaining per-problem parameters. Neither estimator is
claimed to be frequentist unbiased.

N=2 reuses the same full-cohort intervention fit as N=1 and integrates
`2 * E[s] - E[s^2]` over the joint posterior, **not**
`1 - (1 - E[s])^2`. Plain-geometric retains its exact finite-bank estimator.
Linear and OGT are also reported. RMSE/MAE are computed over the aggregate
cumulative count curve (eight checkpoints for N=1, four for N=2); percentage-point
errors divide by the number of evaluated trajectories/pairs. Pooled RMSE is
the square root of mean squared percentage-point errors, weighting LLMs equally.

Use `--profile gpt54-n2` to select a panel or `--json` for complete predictions,
fit diagnostics, cohort metadata, and source-audit SHA-256 hashes. To regenerate
the main prediction plots, N=1/N=2 comparison tables, and R-DE count-error table:

```bash
uv run python scripts/report_allocation_model.py --output-dir paper/img
```

Use a temporary output directory to preview assets without changing the paper.
No ignored `analysis/` files or saved fitted predictions are required. Other
figures and robustness/human-audit tables are not rewritten by this command.

GPT-5.5 uses Parallel-8 seeds 1 and 2 on both datasets (16 branches per problem);
the other LLMs use seeds 1--3 (24 branches). All models use oracle and unaided
seeds 1--3; later seeds are deliberately excluded. The report requires all 35
AOBench and 22 IMO-ProofBench intervention problems, and complete N=1 targets.

The observed two-arm curves pair `baseline-sequential` and
`late-baseline-sequential` only when both have the same problem ID and seed.
Missing required banks, seeds, scores, or checkpoints stop the report rather
than becoming failures. Opus N=2 is explicitly a partial IMO-ProofBench panel;
it is excluded from the N=2 comparison table and pooled error. Proof success is
an audit score of at least 5 by default, irrespective of oracle alignment, and
sequential outcomes are cumulative: a later failing proof cannot undo an earlier
success. Target outcomes are used only for evaluation, after fitting.

The historical solved-or-oracle-acquired U-statistic is available with
`--legacy-acquisition` for reproducing earlier analyses. It is **not** the
default and does not reproduce the current paper.

## Results backup

Set `HF_TOKEN` in `.env`. Pull and safely merge the remote active result trees
before uploading new work:

```bash
./scripts/download_results_from_hf.sh
./scripts/upload_results_to_hf.sh
```

The downloader uses a packed Git clone plus batched Git LFS transfer instead of
per-file Hub downloads, checks generation artifacts at whole-seed granularity,
and aborts before writing if the same seed contains conflicting trajectories.
Use `./scripts/download_results_from_hf.sh --dry-run` to compare without merging.
Existing local files remain authoritative; remote-only files and compiled audit
records are added without mixing conflicting generations.

To upload only the active `results/` and `results-imobench/` trees directly:

```bash
./scripts/upload_results_to_hf.sh
```

The default upload is additive. When a seed has a genuine local/remote
generation conflict and the local generation is authoritative, replace that
seed exactly with:

```bash
./scripts/upload_results_to_hf.sh --replace-seed \
  results/<model>/<arm>/<problem>/seed_<n>
```

Multiple seed paths may follow `--replace-seed`. This mode overwrites changed
files and deletes remote-only files **only inside the explicitly named seed
directories**, preventing stale artifacts from a previous generation from
surviving a local-wins reconciliation.

The default private destination is `notadib/strategy-ceiling`. Root-level `results-archive/` is intentionally ignored by Git and excluded from this uploader.
