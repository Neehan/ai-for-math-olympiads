# Grading rubric (operational)

Grade one attempt against the reference solution. Two **orthogonal** axes: **Locus** (where the first genuine gap is) and **Calibration** (did it admit the gap). This is the frozen taxonomy from the top-level README §2, made operational with decision rules, tie-breaks, and worked examples.

The unit graded is the model's **`## Final Solution`** section. Ignore scratch and exploration; grade the final write-up as a referee would.

The reference solution is supplied to the grader **only at grading time** and is never stored in this repository — the solving harness must never have access to it. A reference is either the **official** solution or an **unofficial solution that has been audited and confirmed valid**; both are authoritative ground truth for grading. Treat them identically.

---

## Step 1 — Verdict

- **SOLVED** — a strict olympiad referee awards full marks. Every claim proven; no gap a referee would deduct for. (A *different* valid method than the official still counts — see the "valid crux" rule.)
- **PARTIAL** — genuine, correct progress with the right central idea, but a real gap remains (an unproven key step, a missing case, an unjustified leap). Also use PARTIAL for a multi-part problem where some parts are fully proven and others fail.
- **FAILED** — wrong answer, wrong approach, or only superficial/hand-wave progress.

No credit for confident tone. A confidently-asserted non-proof is FAILED or PARTIAL, never SOLVED.

---

## Step 2 — Locus (WHERE the first genuine gap is)

Exclusive and ordered. Walk the tree top-down; the **first NO wins**. If SOLVED, locus = `none`. **Output the bare value: `setup`, `strategy`, `crux`, `execution`, or `none`** (the "recognition ·" prefix below is just grouping, not the label).

```
Did it UNDERSTAND the problem?                (right target, right claim/answer)
  NO  → SETUP
  YES → Did it find the right STRATEGY?       (the general approach/method)
          NO  → RECOGNITION · strategy
          YES → Did it find the CRUX?         (the load-bearing lemma/key move)
                  NO  → RECOGNITION · crux
                  YES → EXECUTION             (has it all, slips while filling in)
```

### The four loci

| Locus | Assign when… |
|---|---|
| **setup** | The model misread the problem: wrong target, wrong quantity, wrong claimed answer, solved a *different* question, or misstated the constraint. Nothing downstream can be right because it is not the right problem. |
| **recognition · strategy** | The model understood the problem correctly but never found the right general approach — picked a method that cannot work, or flailed between approaches, none of them the right family. |
| **recognition · crux** | The model had the right general approach but never found the specific load-bearing lemma / key move the proof hinges on. The gap is *at* the hard step. |
| **execution** | The model had the strategy **and** the crux and was genuinely filling in the blanks, but made a competence error: arithmetic, a case, an algebra manipulation, a bound. It *attempted* every step. |

### Decisive tests (use these, not gut feel)

- **setup vs strategy:** *"If I restated the problem correctly, would it be on track?"* Yes → misunderstood the problem → **setup**. No, it understood the problem but the method is doomed → **strategy**.
- **strategy vs crux:** *"Is the general approach the one the official (or any valid) solution uses?"* No / a different, non-viable family → **strategy**. Yes, but the specific key lemma/step is missing or unproven → **crux**.
- **crux vs execution:** *"Take the first fatal gap. Is it the load-bearing step, or a routine fill-in?"* **Use the reference solution to decide what is load-bearing** — the key lemma / main idea the reference builds the proof around is the crux; boilerplate around it is routine, not intuition. Gap AT the reference's key idea → **crux** (it never cracked the problem). Gap in a routine step it clearly could do, botched → **execution**.

### Tie-breaks (the boundaries where graders diverge)

1. **Both a missing crux AND a later slip** → grade the **earliest** genuine gap. A missing crux dominates a downstream arithmetic slip → **crux**, not execution.
2. **"Verified numerically / it is easy to see / one checks" on a step** is NOT automatically execution. Ask what the skipped step *is*: if it's the **crux** → **crux** (it hid that it never proved the key thing); if it's a **routine** step it plainly could do → **execution**. The hand-wave decides *calibration* (bluff), not *locus*.
3. **"Sort of the right approach"** — if the approach would need a fundamentally different idea to work, it's **strategy**; if it's the right approach and only the key lemma is missing, it's **crux**.
4. **Right answer, no or bad proof** (common for "find the value" problems): the answer being right does NOT make it execution. If the *justification* never finds the key idea → **crux** (or **strategy**). Locus is about the proof, not the numeric answer.
5. **Correct but different method than official:** grade the model's OWN route (see valid-crux rule). Do not mark "missed key idea" just because it didn't use the official's idea.

---

## Step 3 — Calibration (did it admit the gap?) — orthogonal to locus

| Tag | Assign when… |
|---|---|
| **honest** | SOLVED, **or** the model explicitly flags its gap / states a step as unproven / labels itself partial. |
| **bluff** | The model asserts a complete proof (QED / "thus proved" / closes the argument) while a real gap exists — including hiding the gap behind "verified numerically", "it is easy to see", "one can check", "follows analogously". |
| **truncated** | The attempt was cut off (no `## Final Solution`, or it ends mid-argument): it never got to assert done, so it is not a bluff. |

A SOLVED attempt is always `honest`. A non-SOLVED attempt is `bluff` if it asserts completeness over a gap, `honest` if it owns the gap, `truncated` if it was cut off.

---

## Step 4 — Supporting fields

- **final_answer_correct**: `true` / `false` / `"n/a"` (n/a for pure proof problems with no numeric answer).
- **found_valid_key_idea**: did the model find *a* valid central idea (its own or the official's)? `true` / `partial` / `false`. **Grade a valid crux, not THE official crux.** If the model's route differs but is sound and it found that route's key idea → `true`.

---

## The "valid crux" rule (critical, reduces false "missed idea")

The official solution is ONE valid path. If the model solves it a **different but valid way**, grade *its* path: verdict is SOLVED if its own argument is complete and correct; its locus/idea is judged against *its* approach's key step, not the official's.

Only mark `found_valid_key_idea=false` when the model found **no** viable central idea by any route — not merely a different one than the reference. If you cannot verify the model's alternative route is valid within reasonable effort, mark `partial` and say so in `gap_detail` — do not guess `true`.

---

## Worked examples (anchors)

- **SOLVED / none / honest** — model proves the result rigorously by a valid route (its own or the official's), every step justified. *E.g. a vector-geometry proof reaching the conclusion by a different, fully-justified computation.*
- **PARTIAL / crux / bluff** — right framing and right general approach, correct final answer, but the load-bearing lemma is asserted "by a detailed analysis" without proof, and the write-up closes with QED. Missing lemma → crux; QED over a gap → bluff.
- **FAILED / strategy / bluff** — understood the problem but committed to a method that cannot work (e.g. a counting frame that misses the needed structure), then asserts done. Wrong approach family → strategy; asserted → bluff.
- **FAILED / setup / bluff** — imposed the wrong invariant / solved a different question (e.g. assumed a stronger periodicity than the problem states), producing a wrong answer. Misunderstood problem → setup.
- **FAILED / execution / bluff** — had the right idea and crux, but a genuine algebra/case error produces the wrong result while claiming completeness. Attempted every step, slipped → execution.

---

## Output (one JSON object per attempt)

```json
{
  "problem_id": "...",
  "verdict": "SOLVED|PARTIAL|FAILED",
  "final_answer_correct": true | false | "n/a",
  "found_valid_key_idea": true | false | "partial",
  "locus": "setup|strategy|crux|execution|none",
  "honesty": "honest|bluff|truncated",
  "one_line": "<=25 words: what happened vs a valid solution",
  "gap_detail": "<=50 words: the specific first fatal step, or 'complete' if solved"
}
```

**Valid combinations only.** SOLVED ⟹ `locus=none`, `honesty=honest`, `found_valid_key_idea=true`. A non-SOLVED verdict must have a locus other than `none`. Never emit SOLVED with a bluff, a locus, or a missing idea.

Be adversarial. A missing proof of a key lemma is PARTIAL at best. Grade the first fatal step. When unsure between two loci, apply the decisive test, then the tie-breaks, in order.
