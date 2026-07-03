# Recognition Is the Wall

**Diagnosing and fixing LLM failure on novel olympiad math.**

When a frontier LLM fails a hard olympiad proof, *where* does it fail? We show
the bottleneck on **novel** problems is **recognition** — finding the key idea —
not reasoning or computation. We prove it causally with oracle interventions,
then show a real method (a proof-outliner) recovers part of that ceiling.

**Diagnose → intervene, one paper.**

---

## 1. Thesis

On problems outside training, the failure locus moves **upstream** from execution
to idea-selection, and test-time compute (more samples, more refinement) does not
close the gap. A model that is handed the key idea can usually finish; left to
find it, it usually cannot — and, alarmingly, **claims success anyway**.

Two findings:
1. **Recognition is the wall.** Most failures are the model failing to find the
   key idea, with correct problem understanding and intact execution.
2. **Models don't know when they've failed.** The large majority of failed
   attempts assert a complete proof over a real gap ("verified numerically",
   "it is easy to see"). This is why every number here is graded against
   reference solutions, never self-reported.

---

## 2. Failure taxonomy (frozen)

Two **orthogonal** axes. This is fixed; we do not relabel between runs.

### Axis A — Locus: *where the first genuine gap is*

Exclusive and ordered. Apply the decision tree top-down; the first "NO" wins.

```
Did it UNDERSTAND the problem?
  NO  → SETUP
  YES → Did it find the right STRATEGY (the general approach)?
          NO  → RECOGNITION · strategy
          YES → Did it find the CRUX (the load-bearing lemma/step)?
                  NO  → RECOGNITION · crux
                  YES → EXECUTION
```

| Locus | Definition | Test to assign it |
|---|---|---|
| **Setup** | Misread the problem: wrong target, wrong invariant, or solving a *different* question. | Restate the problem correctly — is it now on track? **Yes → setup.** |
| **Recognition · strategy** | Understood the problem, but never found the right general approach (wrong method entirely). | Understood problem but method is doomed and off any known route. |
| **Recognition · crux** | Right approach, but missed the load-bearing lemma / key move. | Right strategy, gap is at the specific hard step. |
| **Execution** | Has the strategy *and* the crux, is filling in the blanks, but gets one wrong (algebra, casework, a bound). | Genuinely attempts every step; error is competence, not omission. |

**Setup ≠ strategy.** Setup = *didn't understand the problem*. Strategy =
*understood it, wrong method*. They sit on opposite sides of the "did it
understand" split and are never merged.

**Recognition = strategy + crux.** It is the dominant bucket and the thesis. It
is split into strategy vs crux **only because each maps to a distinct oracle
rung** (§4). Setup and execution stay coarse — they are the foils, not the story.

### Axis B — Calibration: *did it admit the gap?*

| Tag | Definition |
|---|---|
| **honest** | Solved it, or explicitly flagged its gaps / partial status. |
| **bluff** | Asserted a complete proof while a real gap exists. |

**"Verified numerically" is a bluff, not a locus.** A skipped-then-asserted step
is a calibration failure; its *locus* is wherever the skipped step actually sits
(usually the crux → recognition). Never classify a bluff as "execution" just
because a proof step was hand-waved.

### Grading rules

- Credit **a valid crux, not *the* official crux** — the model may solve a
  different valid way. Grade the model's own route.
- Grade the **first fatal step**, tagged to exactly one locus + one calibration
  tag.
- Every attempt graded against reference solutions, **verified by a human
  medalist**, blinded to set/condition. All attempts + labels released so
  grading is auditable, not trusted. Report inter-rater agreement (κ).

---

## 3. Data: SEEN vs NOVEL

| Set | What | Role |
|---|---|---|
| **NOVEL** | 39 hard 2026 contest problems (outside training) | main evidence |
| **SEEN** | matched pre-2025 problems (in training), same domain + difficulty | control |

**SEEN is a control, not a power base.** On SEEN the model has seen the problem,
so recognition is near-free — SEEN failures are execution/rigor, not recognition.
The core evidence is the **locus shift**: recognition failures low on SEEN, high
on NOVEL. SEEN proves the shift is caused by *novelty*, not difficulty.

---

## 4. Experiments

### Oracle ladder — the causal ceiling (isolates recognition)

Each rung is a **prompt change on a fixed harness**, same turn budget as C0.

| Rung | Given to the model | Isolates |
|---|---|---|
| **C0** | statement only | baseline |
| **Oracle-1a** | answer + high-level strategy ("answer is X; use infinite descent") | strategy-recognition |
| **Oracle-1b** | + the crux, **stated as a target to prove — not a usable black box** | crux-recognition |

Key cut **Oracle-1b − Oracle-1a** = the value of recognizing the *crux* vs the
*strategy*. Large oracle lift on NOVEL + small on SEEN ⇒ the novel bottleneck is
*finding* the idea, not using it.

> The crux must be stated as *what to prove*, never a proven black box — else the
> rung measures execution, not recognition.

### Realistic interventions — the method

| Intervention | What it does | Targets |
|---|---|---|
| **Proof-outliner** | dedicated agent: case analysis → hypotheses → proof outline | recognition |
| ↳ variants | statement-only · **+static KB** · **+past-solution corpus** · **+both** | which source helps |
| **Reviewer** | reviews the proposed execution for errors | execution |

**Keep KB *and* corpus — they probe different mechanisms.** Static KB = abstract
technique-selection; corpus = analogical retrieval from solved problems. The
ablation (statement / +KB / +corpus / +both) *is* the contribution. Corpus
results carry a **nearest-neighbor similarity control** to rule out
near-duplicate leakage. **No intervention for setup** (rare; stated, not fixed).

**Punchline:** how much of the oracle ceiling does the *real* outliner recover?

### Harnesses — compute, matched

Per-attempt budget is **128 tool-calls everywhere** (enforced in code).

| Harness | Total budget | Allocation | Tests |
|---|---|---|---|
| **Single LLM** | 128 | one shot | floor |
| **Best-of-N (N=8)** | 1024 | **parallel** (independent shots) | does resampling find the crux? |
| **Ralph loop (8 iters)** | 1024 | **sequential** (refinement) | does refinement find/fix it? |
| **AutoFyn** | (noted separately) | composed scaffold | full-system ceiling |

BoN and Ralph are **matched at 1024 turns** — the same budget spent *parallel vs
sequential*. If BoN ≈ Ralph on recognition-locus problems, the wall is
recognition, not compute (a strong result). BoN scoring: **pass@8** vs
judge-selected — stated explicitly since Ralph returns one answer.

### Isolation from the compute confound

The core defense is **locus-conditioned lift**: each intervention must lift its
*predicted* locus and not others — outliner → recognition, reviewer → execution.
This double dissociation, not raw budget-matching, ties the fix to the diagnosed
axis rather than to added compute.

### Design matrix

**Conditions** (C0 / Oracle-1a / 1b / outliner ±KB ±corpus / reviewer) ×
**Harnesses** (Single / BoN-8 / Ralph-8 / AutoFyn) ×
**Models** (**Opus 4.5 + GPT-5.5** — replication, same taxonomy & ladder, *not* a
"which is better" comparison) ×
**Sets** (NOVEL main · SEEN control).

Primary = Single LLM across the full ladder, both sets + both models. Secondary =
harness sweep at C0. No full condition×harness cross.

---

## 5. Results

Reference solutions are the ground truth; the model's self-claim is ignored.

### Single LLM · C0 · NOVEL · Opus 4.5 — *provisional pilot (N=39, single-pass LLM judge, not yet human-verified)*

| Locus | count |
|---|---:|
| Setup | *tbd* |
| Recognition · strategy | *tbd* |
| Recognition · crux | *tbd* |
| Execution | *tbd* |
| **Solved** | *tbd* |

| Calibration | count |
|---|---:|
| honest | *tbd* |
| bluff | *tbd* |

> Pilot grading currently shows recognition as the dominant failure locus and a
> high bluff rate, but inter-grader agreement on the *fine* locus split is low
> until the rubric above is applied with multiple judges + human verification
> (§2). Numbers are frozen into this table only after that pass. Raw per-problem
> verdicts live in `audit/` (a sibling of `results/`, never model-visible).

### Full results (filled as runs land)

| Set | Model | Harness | Condition | Solved | Recognition | Execution | Setup | Bluff% |
|---|---|---|---|---|---|---|---|---|
| NOVEL | Opus 4.5 | Single | C0 | | | | | |
| NOVEL | Opus 4.5 | Single | Oracle-1a | | | | | |
| NOVEL | Opus 4.5 | Single | Oracle-1b | | | | | |
| NOVEL | Opus 4.5 | Single | +outliner | | | | | |
| … | | | | | | | | |

---

## 6. Contribution: diagnose → intervene

**Act 1 — diagnose.** The wall on novel hard olympiad math is recognition, not
execution; setup is rare; failures are overwhelmingly bluffed. The oracle ladder
turns "recognition" from a post-hoc label into a **causal** test.

**Act 2 — intervene.** Because recognition is the bottleneck, target it
(outliner + corpus retrieval). Across independent harnesses, show a consistent
lift **concentrated on recognition-locus problems** while execution stays put —
tying the fix to the diagnosed axis, not to added compute. Confirmed on IMO 2026
as a pre-registered, zero-contamination held-out.

Act 1 justifies the method; Act 2 proves the diagnosis. The arc is the paper.

---

## 7. Sample size & statistics

n is small (39 novel; ~6 IMO). Survive it by design, not volume:

- **Paired within-problem deltas** (C0 vs oracle; ±intervention) — each problem
  is its own control; far higher power at fixed n than between-group.
- **Effect sizes + CIs**, not p-values — sharp effects (e.g. 7/8 recognition
  problems flip under oracle) survive small n; muddy ones don't.
- **Every problem a documented case study** — depth compensates for breadth.
- **Positioning:** a controlled analysis/method study, *not* a benchmark.
  Limitations paragraph pre-empts the n=39 objection.

---

## 8. Contamination control

Runs must solve from the statement alone. Enforced, not trusted:

- **One harness per git branch** — the next harness can't read the previous
  one's `results/`, `logs/`, or `.scratch/`.
- **No network** — a Bash hook blocks curl/wget/pip/git-network; every call
  (blocked or not) is in the audit log.
- **Filesystem sandbox** — a hook confines every tool to the per-problem scratch
  dir; access outside is blocked and logged.
- **Scratch is committed** — the agent's full working record is auditable, so a
  reviewer can confirm each problem was solved cold.

See `src/README.md` for the harness implementation and tool policy.
