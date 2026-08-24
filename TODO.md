# ICLR TODO

## Freeze before more full runs

- [ ] Freeze one primary endpoint: paired reliable coverage at the final 8× cap for fresh unaided versus oracle-sketch Self-Refine on each model's baseline-failure cohort.
- [ ] Use each Self-Refine arm's own 1×/2×/4×/8× artifacts for the five-family curves; additionally materialize and audit every integer 1× increment for both Self-Refine conditions of Opus 4.8 and GPT-5.4.
- [ ] Freeze hashes for the 35 problems, sketches, prompts, stopping rules, model endpoints, run order, audit threshold, and analysis code; log realized tokens and rounds.
- [ ] Validate anytime 2×/4× cuts against independent hard-stopped runs on a locked subset; otherwise keep only the final 8× comparison primary.

## Search and content controls

- [ ] Audit and document `baseline-uniform-strategy` as a proof-domain adaptation of **coarse-grained TTS-Uniform without entropy filtering**: one shared 80k-token strategy extractor, uniform allocation of eight fresh 190k-token executors, and no oracle selection.
- [ ] Publish the exact extraction and execution prompts and state every adaptation: whole-proof coverage, semantic deduplication, an eight-strategy cap, proof auditing instead of answer entropy, and no majority-vote aggregation.
- [ ] Run Parallel-8 and the TTS-Uniform-C adaptation on the same frozen baseline-failure cohort; report `c/8` and pass@$k$ only for IID Parallel, and raw executor/strategy yield for the dependent uniform-allocation bank.
- [ ] Freeze a matched mathematical placebo for every problem and, on a locked subset, a same-length sketch with the decisive route clause removed.

## Audits

- [ ] Have two blinded olympiad experts adjudicate every first-passage proof, every final 8× proof, and every proof or plan used in a headline dissociation case.
- [ ] Audit route presence in unaided and TTS-Uniform-C trajectories and adherence in oracle-sketch trajectories; allow valid routes different from the reference solution.
- [ ] Double-label U/P states with a frozen rubric and audit a random failure sample large enough to estimate automated-audit false negatives.

## Analysis and paper

- [ ] Report raw 0/3–3/3 cells, current-checkpoint and cumulative coverage, exact paired uncertainty over problems, and sensitivity at scores ≥6, 7, and 3/3 reliability.
- [ ] Bound every null claim to the tested controller and maximum allocation; never equate early self-convergence with consuming the full 8× budget.
- [ ] Complete the 35-problem, five-family panel and a locked external replication, including at least one open-weight model.
- [x] Freeze the acquisition-state rule: `S` begins at complete 3/3 mechanism recognition or the first audited success and is absorbing; before acquisition, `P` means an incomplete recognized-step count increased and `U` means it stayed flat or decreased; missing artifacts remain unobserved.
- [ ] Fit the discrete U/P/S model on the two strong models, with four free probabilities per transition matrix; train through 4× and predict 5×–8× on held-out problems, compare condition-specific against shared dynamics, and test parameter recovery, time homogeneity, and Markov sufficiency against time-only, two-state, route-count, and history-aware baselines.
- [ ] Keep the state model in the main paper only if its parameters are identifiable and frozen out-of-sample predictions beat the simpler baselines; otherwise move it and the proposition to the appendix.
- [ ] Rebuild every table and TeX figure from an immutable artifact manifest in a clean container.
