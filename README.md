# Recognition Is the Wall

**Where do frontier LLMs fail on novel olympiad proofs — and can we prove it?**

When a strong model fails a hard, *novel* olympiad problem, the failure is almost never the reasoning or the algebra. It is **recognition**: finding the right approach and the claims that finish the problem. Hand the model those claims and it usually finishes; hand it the low-level proof steps and it *always* finishes. Leave it to find them and it usually cannot.

We show this **causally**, not by hand-labeling. We disclose the reference solution in three nested increments and measure, per problem, the **cheapest disclosure that turns failure into a correct proof**. That point *is* the failure stage — measured by the experiment, never annotated by a grader.

---

## 1. Claim

> **On novel problems, the bottleneck is recognizing what to do, not doing it.**

Two consequences, both testable:
- Reveal the *approach* or the *claims* → the model finishes. (recognition was the wall)
- Reveal the *proof steps* → the model **always** finishes. (execution is not the wall)

That asymmetry — recognition disclosure rescues, execution disclosure is rarely needed — is the entire paper.

---

## 2. The oracle ladder (nested disclosure)

We never label "where the gap is." Instead we give the model progressively more of the reference solution, on a fixed harness, same budget as C0. Each rung **contains** the previous one plus one more layer.

| Rung | What the model is given | The layer it adds |
|---|---|---|
| **C0** | problem statement only | nothing |
| **O-strategy** | + answer + the **approach** (one line: the route that works) | the *region* |
| **O-outline** | + the **claim skeleton** — the lemma *statements* that, if granted, finish the problem (stated, never proved) | the *whats* |
| **O-technique** | + the **proof steps** — the concrete moves that establish those claims | the *hows* |

The three added layers are **strategy → outline → technique**: the approach, the claims, the moves. Each is strictly more of the reference solution than the last.

**The measurement.** For each problem, the **failure stage = the cheapest rung that yields a correct proof.**

| Cheapest rung that works | Failure stage | Reading |
|---|---|---|
| O-strategy | **strategy** | couldn't even find the approach |
| O-outline | **outline** | had the approach, couldn't find the claims |
| O-technique | **execution** | had the claims, couldn't carry out the proof |
| none (even technique fails) | *unsolved* | rare |

This is why the fuzzy "is it strategy or execution?" question — which has no clean hand-drawn answer — **stops mattering**. We don't draw the line; the ladder draws it. If most problems are rescued at strategy/outline and few need technique, recognition is the wall, by construction.

> **Sharp prediction.** O-technique (the proof steps) should rescue *everything* — including problems that only fail at execution. Empirically it does. That is the control that proves the below-the-line work is idea-free: the model can always formalize given the moves.

---

## 3. Data: NOVEL vs SEEN

| Set | What | Role |
|---|---|---|
| **NOVEL** | hard 2026 contest problems (post-training-cutoff) | main evidence |
| **SEEN** | matched pre-2025 problems, same domain + difficulty | control |

On SEEN the model has effectively seen the problem, so recognition is near-free and failures (if any) sit at execution. The result is the **shift**: recognition-stage failures are rare on SEEN, dominant on NOVEL. SEEN shows the shift is caused by *novelty*, not raw difficulty.

---

## 4. Design

**Ladder** (C0 / O-strategy / O-outline / O-technique) × **Sets** (NOVEL main · SEEN control) × **Models** (Opus 4.5 + GPT-5.5, replication — not a "which is better" race).

Primary axis = **Single-LLM across the full ladder**. Each rung is one prompt change on one fixed harness; per-attempt budget is identical everywhere (128 tool-calls, enforced in code), so no rung buys the model more compute — only more of the answer.

Secondary (compute control): re-run C0 under **Best-of-N (N=8, parallel)** and **Ralph (8 iters, sequential)**, matched at 1024 turns. If more compute — parallel or sequential — does **not** move the failure stage off recognition, the wall is recognition, not budget.

---

## 4b. Intervention — attacking the wall

The oracles are cheats: they *hand* the model the approach/claims to prove the wall exists. Act 2 asks whether a **real method** can produce the outline itself, unaided — and how much of the O-outline ceiling it recovers.

**Proof-outliner** — a dedicated agent that, before solving, tries to generate the claim skeleton on its own (case analysis → hypotheses → outline), then hands it to the solver. Its power source is the ablation:

| Variant | Where its ideas come from |
|---|---|
| statement-only | the model's own reasoning |
| **+ static KB** | abstract technique-selection from a fixed knowledge base of methods |
| **+ working corpus** | analogical retrieval from solved problems |
| **+ both** | |

**The headline number:** O-outline is the *ceiling* (claims handed for free); the outliner recovers **X%** of it while finding the claims itself. The KB/corpus split says *which source* closes the gap — abstract method-selection vs analogy to prior solutions.

**Guardrail (required):** corpus results carry a **nearest-neighbor similarity control** — for each rescued problem, report the closest corpus item, so a reviewer can rule out near-duplicate leakage rather than genuine analogical transfer.

---

## 5. Grading & results

### How grading works (validated, not trusted)

Ground truth is the reference solution; the model's self-claim is ignored. The pipeline (full protocol: `audit/rubric.md`):

1. **Per-solution rubrics.** For each problem, IMO medalists read the reference and write the problem-specific conditions for verdict (SOLVED / PARTIAL / FAILED) and stage (strategy / outline / execution). Grading against these concrete conditions — not freehand "is this proof good?" — is what makes an LLM grader reliable.
2. **Two AI graders** (Opus 4.5, GPT-5.5) grade every attempt against its rubric, independently. They check the attempt against medalist-written, problem-specific conditions — a checklist task LLMs do reliably — not freehand "is this proof good?".
3. **Human medalist panel = ground truth on all attempts.** The panel grades everything; the AI grades are validated against it, not the other way round. On top of the full pass, the panel overrides every AI disagreement and audits a random sample of AI *agreements* (catches shared-LLM blind spots).
4. **We report κ** (Cohen's) — AI-grader vs human-panel agreement on both verdict and stage. This is the number that decides whether the instrument is trusted: **κ ≥ 0.7 → the AI grader is valid**; if κ is low, the rubrics are too vague and get tightened until it isn't.
5. **Everything is public** — all attempts, per-solution rubrics, oracle files, both graders' outputs, and panel grades — so grades are auditable, not believed.

### Oracle audit (kills the "inflated ceiling" objection)

Each oracle file is medalist-certified per problem and released as `audit/oracle_audit.jsonl`:
- **O-outline = claim *statements* only** — no proof, construction, or black-box lemma handed over.
- **O-technique** = the moves, with any crux stated as a *target to prove*, never a proven black box.
- **Statements-only check:** we confirm O-outline alone leaves a real gap on most problems (else outline and technique collapse and the middle rung is fake). This is the measured test that the recognition/execution split is real.

### Single-LLM · NOVEL · Opus 4.5

### Single-LLM · NOVEL · Opus 4.5

| Rung | Solved / N | New problems this rung rescues |
|---|---|---|
| C0 | *tbd* | — |
| O-strategy | *tbd* | *tbd* (→ stage = strategy) |
| O-outline | *tbd* | *tbd* (→ stage = outline) |
| O-technique | *tbd* | *tbd* (→ stage = execution) |

**Failure-stage distribution (NOVEL):** strategy *tbd* · outline *tbd* · execution *tbd* · unsolved *tbd*.
**SEEN control:** near-zero recognition-stage failures (expected).

Raw per-problem attempts and grades live in `audit/` (a sibling of `results/`, never model-visible).

### Run status

| Rung | Run | Graded |
|---|---|---|
| C0 | ✓ | ✓ (human) |
| O-strategy | ✓ | claude-judge only — needs human |
| O-outline | **not built** — needs clean `outline.jsonl` + run | — |
| O-technique | ✓ | **not graded** |

---

## 6. Why this is causal, not a re-label

The stage is measured **two independent ways**, and their agreement is the rigor:
- **From the attempt** — the per-solution rubric classifies the C0 attempt (§5).
- **From the intervention** — the cheapest nested rung that rescues the problem (§2). Because the rungs are nested, "cheapest rung that works" is monotone: a problem rescued at outline was, by construction, *not* rescued at strategy.

An **outline**-labeled problem should be rescued by O-outline and not already by O-strategy; an **execution**-labeled one should need O-technique. We report the agreement between the two measures; disagreements are surfaced and resolved by the panel. Neither the hand-label nor the intervention is trusted alone — the double dissociation is what makes recognition-vs-execution falsifiable rather than a definition.

---

## 7. Sample size

n is small (novel set) by design of the domain, not laziness. It survives because:
- **Paired within-problem deltas** (C0 vs each rung) — each problem is its own control; far higher power than between-group at fixed n.
- **Effect sizes, not p-values** — a sharp asymmetry (most problems flip at strategy/outline, almost none need technique) survives small n; a muddy one wouldn't.
- **Every problem is a documented case study.** Depth over breadth.
- Positioned as a **controlled analysis**, not a benchmark.

---

## 8. Contamination control

Runs solve from the statement alone. Enforced, not trusted:

- **One rung per git branch** — a run cannot read another's `results/`, `logs/`, or `.scratch/`.
- **No network** — a Bash hook blocks curl/wget/pip/git-network; every call is logged.
- **Filesystem sandbox** — a hook confines every tool to the per-problem scratch dir.
- **Scratch is committed** — the agent's full working record is auditable, so a reviewer can confirm each problem was solved cold.
- The reference solution is supplied **only at grading time** and is never in the solving repo.

See `src/README.md` for the harness and tool policy.
