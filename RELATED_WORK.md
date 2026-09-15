# Related-work research notes

Internal record of what the closest test-time-scaling papers actually establish. These notes are deliberately more detailed than the manuscript's related-work section so future revisions do not rediscover or overclaim prior results.

Last reviewed: 2026-08-30. The papers below were checked beyond their abstracts, including their main methods, experiments, and limitations.

## Novelty guardrails for this project

The following claims are already established and are **not sufficient novelty by themselves**:

- Independent sampling and answer aggregation can outperform one greedy chain.
- Natural-language planning can diversify search and improve downstream implementation.
- Very short correct sketches can substantially improve execution.
- Strategy classes have nonuniform proposal probabilities and different downstream error rates.
- Sequential versus parallel allocation depends on model-relative problem difficulty.
- Long reasoning can help, saturate, loop, or corrupt a terminal answer.
- Some constructed tasks intrinsically require long sequential computation.

Our potentially distinctive claim is narrower:

> On fresh, proof-graded olympiad mathematics, directly separate viable-plan accessibility from conditional proof execution; test whether those quantities predict held-out breadth-versus-depth outcomes; and determine whether this regime differs from easier answer-style mathematics under a matched full-proof protocol.

For multiple viable plans, the exact per-arm decomposition is

\[
s_n(k)=\sum_{z\in\mathcal V_n}\pi_n(z)\varepsilon_n(z,k)
      =\alpha_n\bar\varepsilon_n(k).
\]

An oracle execution curve for one reference plan is not automatically the average execution curve of model-generated viable plans. Any substitution requires an explicit transport assumption or a restriction to the reference route. Likewise, \(1-(1-s)^r\) is oracle any-of-arm coverage, not majority-vote or verifier-selected accuracy.

## Snell et al.: compute-optimal test-time scaling

**Paper:** [Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Parameters for Reasoning](https://openreview.net/pdf?id=4FWAwZtd2n), ICLR 2025.

### Protocol

- 500 MATH test problems and PaLM 2-S* Codey with capability-specific fine-tuning.
- Separately studies PRM-guided search and a learned revision model.
- Revision compute is divided between parallel chains and sequential revisions.
- Selection uses majority voting or a trained outcome verifier.
- Model-relative difficulty is estimated from 2,048 samples per problem and divided into five bins.
- Allocation is selected with two-fold cross-validation by difficulty bin.
- Compute is matched mainly by generated-answer counts rather than exact output tokens.

### Findings

- Easy questions favor sequential revision because initial responses are often already useful.
- Harder questions favor a mixture with more independent search over high-level approaches.
- The hardest bin receives little benefit from either tested method.
- Approximately 38% of already-correct answers become incorrect after revision, making retention/selection important.
- PRM beam search helps at low budgets but can exploit verifier errors at larger budgets; lookahead is often worse.
- Difficulty-adaptive allocation obtains roughly 2--4x compute efficiency over fixed best-of-N policies.

### Boundary relative to our work

This is the closest high-level prior. Its verbal explanation already resembles “on track: revise; wrong approach: restart.” However, scalar correctness-based difficulty conflates plan access and execution. It does not annotate plans, intervene with a verified proof plan, require olympiad-grade proofs, or estimate the two factors separately. Our model matters only if the decomposition predicts allocation beyond scalar pass@1 difficulty.

## PlanSearch

**Paper:** [Planning in Natural Language Improves LLM Search for Code Generation](https://arxiv.org/abs/2409.03733), ICLR 2025.

### Protocol and findings

- Generates observations, combines them into natural-language plans, translates plans through pseudocode, and then writes code.
- Evaluates HumanEval+, MBPP+, and contamination-resistant LiveCodeBench with executable tests.
- Explicitly writes a decomposition of the form \(P(\text{solve}\mid\text{problem},\text{idea})P(\text{idea}\mid\text{problem})\).
- Backtranslates passing code into sketches of different lengths; even roughly 10-token correct sketches substantially improve GPT-4o-mini implementation.
- Samples 25 implementations per idea and finds idea-conditioned success rates concentrated near zero or one.
- Shows that plan/idea diversity predicts gains and includes compute-normalized, plan-depth, subset-size, temperature, and translation ablations.

### Boundary relative to our work

“Short plans unlock execution” is already demonstrated in code. The remaining distinction is proof-grade mathematics without executable tests, conditional execution as a function of very large budget, and held-out allocation prediction across problem regimes.

## TTS-Uniform

**Paper:** [Mitigating Strategy-Selection Bias in Reasoning for More Effective Test-Time Scaling](https://arxiv.org/abs/2509.17905).

### Protocol and findings

- Defines strategy equivalence classes \(R_i\), proposal frequencies \(p(R_i)\), and strategy-specific error rates.
- Extracts strategies, allocates sampling more uniformly across them, and filters unstable paths.
- Evaluates AQuA and AIME 2024/2025; reports stronger gains on AIME and weaker results for GPT-4o-mini.
- “Strategy-selection bias” in this paper means biased proposal frequency over strategy classes, not a separate model choosing among a displayed candidate set.

### Boundary relative to our work

The nonuniform plan-distribution idea is already explicit. TTS-Uniform uses automatically extracted strategies and numeric-answer accuracy. It does not independently identify oracle-conditioned rigorous-proof execution or test a difficulty-regime shift under matched proof grading.

## Self-Consistency

**Paper:** [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171), ICLR 2023.

- Samples up to 40 reasoning paths and majority-votes final answers.
- Evaluates arithmetic, commonsense, and symbolic reasoning benchmarks.
- Substantially outperforms greedy decoding and several search/ranking baselines; gains commonly saturate within 5--10 paths.
- It studies answer aggregation, not plan acquisition, conditional execution budgets, or complete proof validity.

Our any-of-arm formula must not be presented as a model of self-consistency majority voting.

## Inference Scaling Laws

**Paper:** [Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving](https://openreview.net/forum?id=j7DZWSc8qu), ICLR 2025.

- Studies greedy sampling, majority and weighted voting, best-of-N, MCTS, and REBASE on GSM8K and MATH500.
- Derives saturation behavior for voting-based methods and examines compute-optimal model size.
- REBASE helps more on MATH-hard than MATH-easy; GSM8K saturates earlier than MATH.
- Its “depth” is guided tree expansion over comparatively short answers, not a 200k--1.6M-token continued proof trajectory.

The results support task-dependent allocation but do not isolate plan access from proof execution.

## Let Me Think!

**Paper:** [A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones](https://arxiv.org/abs/2505.21825), NeurIPS 2025.

### Contributions

- Gives a conditional-complexity separation for graph connectivity: polynomial-length sequential CoT succeeds where polynomially many constant-length CoTs aggregated by majority cannot substantially beat chance.
- Introduces vertex-query models and bridge-graph instances with sharp sequential-length thresholds.
- Trains transformers on shortest-path, path, and DFS traces; DFS/backtracking training exploits longer CoT more effectively.
- STaR iterations improve bridge-task performance while increasing verified reasoning length.
- Reproduces sequential advantages with several large reasoning models and includes an AIME 2024 grid over sequential length and parallel samples.

### Boundary relative to our work

Its mechanism is intrinsically sequential local exploration, not failure to verbalize a missing high-level plan. In our notation it resembles high plan accessibility with an execution curve that remains near zero below a threshold. Our framework can classify this as an execution-limited regime but does not derive their exponential theorem.

Their LLM experiments use majority voting, whereas their trained-model experiments can use a deterministic evidence verifier. These must not be collapsed into oracle pass@N.

## s1: Simple Test-Time Scaling

**Paper:** [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393).

- Fine-tunes Qwen2.5-32B-Instruct on 1,000 selected Gemini reasoning traces.
- Evaluates AIME24, MATH500, and GPQA Diamond using answer-level scoring.
- Budget forcing terminates at a cap or suppresses stopping and appends “Wait.”
- Reports AIME24 improvement from 50.0% to 56.7%; MATH500 changes only slightly; GPQA improves modestly.
- Additional forced thinking eventually flattens into repetitive loops.
- Rejection sampling for naturally long traces can exhibit inverse scaling because long traces often began incorrectly and backtracked.
- The headline sequential-versus-parallel comparison is not a clean same-model intervention in every figure.

Budget forcing may improve conditional execution, discover a new plan, or corrupt an earlier answer. The paper does not distinguish these mechanisms.

## e3: Learning to Explore

**Paper:** [e3: Learning to Explore Enables Extrapolation of Test-Time Compute for LLMs](https://openreview.net/pdf?id=4oWZ1fBMLN).

- Studies Countdown and multiplication didactics, then trains Qwen3-1.7B on DeepScaleR-derived mathematics and evaluates AIME25/HMMT25.
- Its three ingredients are generation--verification asymmetry, negative RL gradients that increase exploration, and a curriculum coupling difficulty with token budget.
- Removing negative gradients reduces entropy, unique attempts, verification, trace length, and extrapolation.
- Too-short budgets suppress exploration on hard problems; beginning immediately with very long budgets creates optimization problems.
- e3 improves pass@1 and pass@k and extrapolates beyond its training-token budget.

This method intentionally violates fixed plan mass: a long trace generates and tests multiple candidates internally. It is best treated as a trained replanning policy, complementary to our fixed-model diagnosis.

## Mirage of Test-Time Scaling

**Paper:** [Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fc067ac218430c409d6f65403328f740-Abstract-Conference.html), NeurIPS 2025.

- Evaluates DeepSeek-R1 distill models on GSM8K, MATH500, and AIME 2024.
- Compares “Wait” continuation, exact forced-thinking lengths, and minimum lengths through roughly 16K--32K.
- Accuracy often rises initially and later declines; exact-length forcing is especially harmful.
- Longer generated traces correlate with higher final-answer entropy, while merely repeating identical text does not reproduce the effect.
- A Gaussian policy/reward model illustrates how increased variance can first help and later hurt.
- Parallel standard-length samples with majority voting outperform forced continuation in highlighted matched-budget comparisons.

Our cumulative “a proof has appeared” probability cannot decline. To encompass Mirage, use correctness of the terminal retained artifact or add an explicit retention/selection variable. Our proposal--execution model classifies fast execution saturation but does not establish their variance mechanism.

## Strategy Executability and targeted hints

- [Strategy Executability in Mathematical Reasoning](https://arxiv.org/abs/2602.22583) distinguishes strategy appearance from a target model's ability to execute guidance and proposes retrieval/selection using model-relative executability. This is close conceptual prior art and must be credited.
- [Give Me a Hint](https://arxiv.org/abs/2410.05915) studies benefits and harms of mathematical hints.
- [Can Language Models Solve Olympiad Programming?](https://arxiv.org/abs/2404.10952) finds targeted human hints rescue many USACO problems unsolved by the tested automatic methods.

Our oracle plans are therefore a diagnostic intervention, not a practical hint-generation method and not evidence that hints are novel.

## Proposed easier-versus-hard regime test

The clean comparison should change problem difficulty without changing what counts as success:

1. Select fresh, non-geometry AIME-style or easier proof problems with verified reference proofs.
2. Require complete proofs rather than scoring only the numeric answer.
3. Use the same models, harness, token block, stopping rule, proof grader, and frozen plan format as the hard benchmark.
4. Estimate viable-plan access from independent proposal artifacts.
5. Estimate oracle-route execution at \(1\times,2\times,4\times,8\times\).
6. Fit on two independent banks and predict an untouched third bank's plan and proof coverage.
7. Compare allocation prediction from \((\alpha_n,\varepsilon_n(k))\) against scalar baseline pass@1 difficulty.
8. Cluster uncertainty by problem; seeds and branches are repeated measurements, not independent problems.

The target result is not merely “AIME is easier.” It is that answer-style/easier proof tasks and fresh olympiad proofs occupy measurably different proposal--execution regimes, so their compute-optimal allocations differ. The hypothesis fails if the regime distributions overlap or the decomposition does not improve held-out allocation prediction.
