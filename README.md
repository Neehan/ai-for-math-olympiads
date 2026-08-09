# Strategy-Conditioned Test-Time Scaling

## Thesis

How much test-time inference is equivalent to knowing what to try? When unaided inference stops producing valid proofs, is the model out of capability, or is it missing a strategy that would make further inference useful?

> **The return to inference is strategy-conditioned: unaided inference can plateau, while a short correct strategy moves the same model into a state where additional inference becomes productive again.**

Strategic information and inference compute are complementary, not interchangeable. This claim is restricted to the tested models, problems, protocols, and budgets. The oracle strategy is a controlled diagnostic intervention, not free deployable information.

## Development signal

On the 13-problem Opus 4.8 combinatorics pilot, with problem solved = at least 2/3 audited attempts scoring ≥5:

- unaided coverage: `4/13 → 6/13 → 6/13 → 6/13` at `1×/2×/4×/8×`;
- supplying the frozen ≤25-word strategy to the seven survivors: `9/13` at 1×;
- further strategy-conditioned inference: provisionally `10/13 → 10/13 → 11/13` at `2×/4×/8×`.

Three missing hinted audits count as failures in this development plot. The result is gated and motivates the thesis; it is not the unbiased interaction estimate. Parallel and Uniform Search are unfinished.

## Frozen experiment design

### 1. Fixed-cohort scaling — primary

1. Freeze prompts, strategies, budgets, audit rules, and analyses.
2. Run baseline 1× with three seeds. Select every problem below 2/3 using baseline only; the pilot cohort has nine problems.
3. On every selected problem, run unaided and strategy-conditioned Self-Refine with three trajectories and audited `1×/2×/4×/8×` exact-prefix cuts.
4. Compare cumulative reliable coverage on the same cohort and denominator. Carry a problem forward within an arm after it reaches 2/3.

The primary question is whether coverage grows more from 1× to 8× after strategy conditioning. Neither final unaided outcomes nor hinted outcomes may define this cohort.

### 2. Unaided-search stress test

Use a predeclared survivor cascade:

1. Continue every problem below 2/3 after unaided Self-Refine 4×—seven in the pilot—to Self-Refine 8×.
2. On that same frozen survivor cohort, independently run Parallel-8 and Uniform Strategy Search-8. Parallel-8 uses eight independent 1× attempts. Uniform uses an 80k shared planner plus eight 190k executors (`1.6M` total), motivated by [TTS-Uniform](https://arxiv.org/abs/2509.17905).

Each search protocol has three replicated banks and is analyzed separately; do not sum their budgets into one inference curve. Because Parallel and Uniform use the identical frozen cohort, compare their rescue rates directly. For survivors of every unaided protocol, freeze an idea rubric, audit the recorded unaided outputs, then report strategy and strategy-plus-inference rescue as a conditional boundary analysis.

### 3. Confirmation and replication

Treat the 13 combinatorics problems as development data. Freeze the design, evaluate the remaining 22 untouched problems, and report both splits separately; pooled results over 35 are secondary. Opus is the main experiment. Replicate the qualitative result with one GPT-family and one open-weight proof model. Each model defines its own baseline-selected and stress-test cohorts; report every available intermediate cut, while reserving expensive search controls for problems passing the frozen gates.

## Protocol

- **Correctness:** attempt success = blinded audit score ≥5; any substantive gap scores 0. Repeat at ≥6 and independently human-check every headline proof.
- **Reliability:** report raw `0/3`–`3/3`; primary solved = ≥2/3, unstable = 1/3, failed = 0/3; report 3/3 as a conservative sensitivity result.
- **Replicates:** any ≥2/3 protocol claim uses three independent replicates. A Parallel bank is the matching frozen 1× attempt plus seven fresh attempts; a Uniform bank is one fresh planner-and-executor replicate.
- **Strategy:** one frozen, audited hint of at most 25 words. It may state the key idea, but not the answer, a substantial intermediate derivation, or a proof sketch.
- **Budget:** 1× is at most 200k eligible output tokens, including hidden reasoning, visible text, and tool calls. Over-budget attempts cannot count.
- **Sequential:** one exact-prefix trajectory with self-review and cumulative cuts, without grades, references, or ground truth.
- **Reporting:** include every late success, first-passing budget, allocated and realized tokens, and token mean, median, standard deviation, and maximum.
- **Trace audit:** for full-search survivors only, freeze a problem-specific decisive-idea rubric before inspection. Two blinded auditors classify the idea as absent, mentioned but not operationalized, developed but abandoned, or executed. This label is required for strict discovery witnesses but not the population-level interaction; claims concern recorded outputs, not unobserved internal thought.

## Figures and success condition

1. **Gated frontier:** unaided plateau → explicit strategy intervention → renewed strategy-conditioned scaling. The two phases repeat the `1×/2×/4×/8×` budget axis and are joined only by the labeled intervention, not as continuous total compute.
2. **Primary interaction:** both scaling curves on the same baseline-selected cohort.
3. **Survivor funnel:** conditional rescue and remaining problems after Self-Refine-8, Parallel-8, Uniform-8, and the strategy intervention.

The paper succeeds only if held-out data show both a reproducible positive strategy–compute interaction and several problems that survive strong unaided search but become solvable under strategy conditioning. The strongest witnesses are below 2/3 under Self-Refine-8, Parallel-8, and Uniform-8, have the decisive idea absent from recorded unaided outputs, but reach at least 2/3 after strategy conditioning. “Hints help” or “stronger models do better” alone is insufficient.
