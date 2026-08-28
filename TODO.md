# ICLR TODO

## Freeze the claim

- [ ] Center the paper on two observable bottlenecks: **strategy access** and **proof execution**.
- [ ] State the mechanism as `P(proof | policy) = P(strategy accessed | policy) x P(proof | strategy accessed, policy)`.
- [ ] Claim that the framework *reconciles* prior depth-versus-breadth findings; do not claim to establish the mechanism of every prior study.
- [ ] Remove strategy selection from the title, main decomposition, and primary experiments. Keep completed selector runs as exploratory archive only.
- [ ] Define access as a valid proof, all three verified reference mechanisms in their required roles, or a human-adjudicated complete alternative strategy.
- [ ] Bound every conclusion to the tested model, harness, policy, and finite compute cap.

## Primary 35-problem study

- [ ] Complete three unaided and three oracle-conditioned Self-Refine trajectories for all four paper models on all 35 problems.
- [ ] Complete the standalone baseline, shuffled-sketch placebo, and oracle cells.
- [ ] Use GPT-5.4 as the primary breadth model. On all 35 problems, complete three independent Parallel-8 banks; the existing bank is seed 1.
- [ ] Keep one Parallel-8 bank for Muse and Opus as cross-model replication. GPT-5.5 breadth is optional because its failure cohort is small.
- [ ] Keep Uniform-C-8 as a secondary diagnostic with one realized planner bank per evaluated model; do not spend compute obtaining three-bank reliability.
- [ ] Audit every Parallel branch, Uniform-C proposal, and Uniform-C executor for strategy access and proof correctness.

## External confirmation

- [ ] Complete the existing three-seed unaided and oracle-conditioned runs on all 22 non-geometry Advanced IMO-ProofBench problems for GPT-5.4 and Muse Spark~1.2.
- [ ] Freeze each model's external baseline-failure cohort before inspecting the remaining arms.
- [ ] Run one preregistered GPT-5.4 Parallel-8 bank on the external failure cohort. One bank is sufficient here as a problem-level confirmation; report it as one randomized policy realization, not 2/3 reliability.
- [ ] Test the frozen directional predictions: oracle conditioning raises terminal proof coverage, and Parallel-8 raises strategy-access coverage relative to the mean single-trajectory depth rate.

## Expert validation

- [ ] Human-check every breadth-only acquisition, every acquired-but-unsolved case, every oracle-only rescue, and a frozen stratified sample of negative cases.
- [ ] Adjudicate complete alternative strategies so reference-route matching is not treated as the only possible access event.
- [ ] Report automated proof and strategy-audit precision and recall against blinded expert consensus.

## Analysis

- [ ] Treat one 8x Self-Refine trajectory and one Parallel-8 bank as matched-cap policies. Report realized tokens separately because early convergence can make realized compute differ.
- [ ] For the primary all-35 GPT-5.4 comparison, report 0/3--3/3 access and proof outcomes for both depth and Parallel-8, plus the paired >=2/3 contrast.
- [ ] For one-bank replications, compare bank access with the mean of the three depth-seed outcomes; never compare any-of-eight directly with the depth 2/3 count.
- [ ] Report acquisition@k for Parallel-8, per-problem branch frequencies, and breadth-only/depth-only/both/neither overlaps.
- [ ] Report strategy access and valid proof separately. The gap between them is the observed execution-failure count.
- [ ] Use problems as inferential units, exact paired tests where applicable, and problem-cluster bootstrap intervals across models.
- [ ] Label the current pooled breadth estimate as exploratory; reserve confirmatory language for the frozen external analysis.

## Paper

- [ ] Retitle and rewrite around **When Does Long Thinking Help? Separating Strategy Access from Execution in Mathematical Reasoning**.
- [ ] Explain prior results through the two factors: depth can improve conditional execution, breadth can improve access, and either can plateau when its targeted factor is insensitive to more compute.
- [ ] Distinguish the contribution from PlanSearch, TTS-Uniform, USACO hints, and Strategy Executability: we intervene on access and execution within the same model--problem failures under matched caps.
- [ ] Make the main empirical asymmetry explicit: frontier models execute nearly every supplied verified strategy, while unaided depth and breadth still leave strategies unobserved.
- [ ] Keep the discrete state/Markov analysis only if it adds held-out predictive evidence beyond the two-factor decomposition; otherwise move it to the appendix.
- [ ] Rebuild every table and figure from one frozen artifact manifest before submission.
