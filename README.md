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
| `baseline-sequential` | Three Self-Refine trajectories through 8× | End-to-end depth; proposal and execution remain mixed |
| `baseline-parallel` | Three bank seeds, each with eight independent 1× proofs | End-to-end breadth and eventual strategy access, not proposal alone |
| `baseline-uniform-strategy` | One 80k extractor and eight 190k executors | Explicit plans followed by balanced execution; cross-plan selection is bypassed |
| `baseline-uniform-strategy-only` | The frozen seed-1 planner artifacts without executors | Explicit proposal coverage for the realized Uniform-C bank |
| `selection` | Rank three compressed proposals plus the oracle with the problem, three independent 1× attempts | Exploratory fixed-pool strategy selection |
| `selection-no-problem` | Identical pool, order, model, and 1× protocol without the problem | Exploratory provenance/style-leakage control |
| `hint-sequential` | One frozen ≤25-word oracle strategy followed by Self-Refine through 8× | Conditional execution after proposal and comparative selection are bypassed |
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

## Results backup

Set `HF_TOKEN` in `.env`, then upload only the active `results/` and `results-imobench/` trees:

```bash
./scripts/upload_results_to_hf.sh
```

The default private destination is `notadib/strategy-ceiling`. Root-level `results-archive/` is intentionally ignored by Git and excluded from this uploader.
