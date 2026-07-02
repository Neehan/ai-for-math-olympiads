# Recognition Is the Wall: Diagnosing and Fixing LLM Failure on Novel Olympiad Math

## 1. Objective

When an LLM fails a hard olympiad proof, is it failing to **retrieve** a known
technique or failing to **reason**? We separate the two with a contamination
contrast — **SEEN** (pre-2025 problems, in training) vs **NOVEL** (2026
contests, outside training) at matched difficulty — instrumented by a crux
corpus that labels the human load-bearing move of each problem.

Claim we aim to support: a fixed fraction of frontier olympiad performance is
retrieval; on provably-novel problems the failure locus moves **upstream** from
execution to idea-selection, and test-time compute does not close that gap.

## 2. Experiment setups

**Data.** SEEN = matched sample of pre-2025 hard problems (in training). NOVEL =
39 hard 2026 problems (outside training). Matched on domain + `difficulty_rating`.

**What SEEN is for — a control, not a power base.** On SEEN the model has seen
the problem, so recognition is near-free: SEEN failures are *not* T1. They are
execution/rigor failures where the memorized solution was lossily compressed and
the model must **re-derive** it (easier than cold, not guaranteed). So SEEN and
NOVEL measure *different* things by design — and that is the point. The paper's
core evidence is the **failure-locus shift**: T1 low on SEEN, high on NOVEL. SEEN
proves the shift is caused by *novelty*, not by difficulty. (We do not lean on
SEEN for sample size; see §4.)

**Knob A — condition ladder** (isolates retrieval vs reasoning). Run on both sets:

| # | Condition | Isolates |
|---|---|---|
| C0 | No KB, no corpus (baseline) | raw ability |
| C1 | + static `knowledge_base.md` | can it pick the right generic technique |
| C2 | + crux corpus (past problems + solutions) | does memory of analogues close the gap |
| C3 | **Oracle crux** — correct technique handed in | given the idea, can it execute |

Key cut: **C3 − C0** (oracle lift). Large on NOVEL + small on SEEN ⇒ seen "skill"
was retrieval; the novel bottleneck is *finding* the idea, not using it.

**Knob B — systems** (isolates test-time compute). Run at fixed condition, both sets:

| System | Adds |
|---|---|
| Single LLM (1 sample) | baseline |
| Best-of-N (report pass@k; no cheap proof verifier) | search |
| Reflection (self-critique loop) | self-correction |
| AutoFyn (verified expert-iteration; **one system under test, not the focus**) | full scaffold ceiling |

Key cut: does the **SEEN–NOVEL gap shrink with compute?** If it plateaus, compute
buys retrieval/search, not reasoning.

**Scope.** Primary = Single LLM across full ladder, both sets. Secondary = systems
sweep at C0, both sets. No full condition×system cross.

## 3. Failure modes

Grade the **first fatal step** of each attempt, tagged to one axis:

| Axis | Failure | Ground truth |
|---|---|---|
| 1. Setup | wrong target / invariant / reformulation | problem statement |
| 2. Technique-selection | didn't find the load-bearing crux | crux corpus (human crux = label) |
| 3. Execution | right crux, botched algebra / casework / bound | human solution |
| 4. Rigor / bluff | ok sketch but unjustified leaps, skipped cases, or **claims solved when not** | reviewer |

**Grading.** LLM-judge over every attempt, **verified by an IMO medalist**; all
attempts + labels released publicly so the grading is auditable, not trusted.
Verification is **blinded** to set (SEEN/NOVEL) and condition to remove
hypothesis bias; report inter-rater agreement on a double-verified subset (this
is where prior work — Proof or Bluff — was criticized).

**Baseline result (AutoFyn + static KB, NOVEL, 39 problems): 24 solved, 15
partial.** Of the 15 failures: **8 technique-selection (T1)**, 4 execution, 3
rigor, **0 setup**. The wall is idea-selection, not execution — and the model
never misreads the problem (setup ≈ 0). This motivates the intervention (§4).
The oracle-crux ablation (C3) turns T1 from a post-hoc label into a causal
test: inject the human crux on the 8 T1 problems — if they solve, the wall was
recognition; if not, it was execution depth masked by recognition.

## 4. Contribution: diagnose → intervene (one paper, two acts)

**Act 1 — diagnose.** The wall on novel hard olympiad math is idea-selection
(T1), not execution: 8/15 baseline failures are T1, 0 are setup (§3). The
oracle-crux ablation (C3) confirms T1 is causal, not a labeling artifact.

**Act 2 — intervene.** Because recognition is the bottleneck, target it:
**parallel-approach search + past-solution corpus retrieval**. Across four
independent systems (Single/BoN, Reflection, Ralph-loop, AutoFyn — each with vs
without the intervention), show a consistent lift, and — the load-bearing figure
— that the lift is **concentrated on T1 problems** while T2/T3 stay stuck. That
ties the fix to the diagnosed axis, not to added compute. Confirmed on IMO 2026
as a **pre-registered, zero-contamination held-out**.

Act 1 justifies the method; Act 2 proves the diagnosis. Neither is a standalone
paper — the arc is the contribution.

Guardrails: (i) per-system ablation (system alone vs +parallel vs +corpus vs
+both) — else the lift can't be attributed; (ii) rule out corpus = near-duplicate
leakage (report retrieval overlap; ablate the analogous problem out); (iii) IMO
2026 (~6 problems) is *confirmation only* — main numbers ride on the 39-problem
novel set.

**Remaining risk (only one left):** single model family. Run ≥2 (one non-Anthropic
frontier) or the claim reads as "about Opus," not "about LLMs."

## 5. Sample size & statistical design

n is small (39 novel; ~6 IMO). Survive it by design, not volume:
- **Paired / within-problem** deltas (C0 vs C3; +intervention vs −), not
  between-group — each problem is its own control, far higher power at fixed n.
- **Effect sizes + CIs**, not p-values — small n is fine when the effect is sharp
  (7/8 T1 flips, not 5/8). Muddy + small dies; sharp + small survives.
- **Every problem a documented case study** — depth as compensation for breadth;
  turn small-n into "we trace every failure," a feature vs noisy 1000-problem sets.
- **Positioning:** a controlled analysis/method study, *not* a benchmark (where
  n=39 is laughable). Pre-empt with a Limitations paragraph before a reviewer does.
