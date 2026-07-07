# Grading protocol

We classify each **baseline (C0) attempt** into a **verdict** (SOLVED / PARTIAL / FAILED) and a **failure stage** (strategy / outline / execution), graded against a **per-solution rubric** written by human IMO medalists. Two AI graders apply that rubric independently; a medalist panel resolves disagreements. The oracle ladder (README §2) then **validates** the stage labels causally — it does not define them.

The unit graded is the model's **`## Final Solution`** section. Ignore scratch; grade the final write-up as a referee would.

The reference solution is given to the grader **only at grading time** and is never in the solving repo. A reference is the **official** solution or an **audited-valid unofficial** one. **No reference ⟹ UNGRADED**, excluded from every count.

---

## 1. Shared scaffold (inherited by every per-solution rubric)

### Verdict

| Verdict | Assign when… |
|---|---|
| **SOLVED** | Every claim proven by a valid route (the model's own or the official's — a different valid method counts). No gap a referee would deduct. |
| **PARTIAL** | Real, verifiable progress (a proven sub-part, one direction of an iff, a valid construction, a correct non-trivial reduction) **and** a real gap remains. |
| **FAILED** | No solid progress: only the answer, only trivial observations, false core claims, or a gap hidden behind "verified numerically / easy to see / one checks". |

### Failure stage (the first thing the attempt was missing)

| Stage | The attempt… |
|---|---|
| **strategy** | never found a viable approach — the route it took cannot reach the answer. |
| **outline** | found the viable approach, but never produced the **claim skeleton** — the lemma *statements* that, granted, finish the problem. |
| **execution** | had the claim skeleton, but failed to **prove** the steps — couldn't prove a lemma it correctly named, or slipped in the algebra/casework. |

SOLVED ⟹ stage = `none`. Otherwise assign the **earliest** stage that is missing (strategy before outline before execution): a missing approach dominates a downstream slip.

**Shared rules**
- Ground truth is the reference, never the model's tone. A confident non-proof is FAILED/PARTIAL, never SOLVED.
- Grade the model's **own route**; only call the approach doomed if *no* viable route was found, not merely a different one than the reference.
- A right final answer with no valid justification is FAILED (or PARTIAL if a real sub-part is proven). Stage is about the proof, not the number.

---

## 2. Per-solution rubric (medalists author this, one per problem)

The shared scaffold is abstract. For each problem, medalists read the reference and instantiate it — writing the **problem-specific conditions** that place an attempt into each stage:

- **strategy conditions** — what the viable approach(es) are; what routes are dead ends.
- **outline conditions** — the specific claim skeleton (the lemma statements) an attempt must reach to clear the outline stage.
- **execution conditions** — the specific steps that must be *proved*; where the known slip points are.

The grader is never asked "where do you feel it failed?" — it checks the attempt against these concrete, pre-written conditions. This is what makes the label reproducible across graders.

---

## 3. Dual AI graders + panel

1. **Opus 4.5** and **GPT-5.5** each grade every attempt against its per-solution rubric, independently, blinded to condition/set.
2. **Agreements stand.** **Disagreements → medalist panel** resolves.
3. Report inter-grader agreement (**κ**) on verdict and on stage.

---

## 4. Ladder validation (causal cross-check — not part of grading)

The rubric assigns the stage from the *attempt*. The oracle ladder confirms it from the *intervention*:

- A **strategy**-labeled problem should be rescued by **O-strategy**.
- An **outline**-labeled problem should be rescued by **O-outline**, and *not already* by O-strategy.
- An **execution**-labeled problem should need **O-technique**.

Where the rescuing rung contradicts the rubric stage, the label is wrong — surfaced, not hidden. This double dissociation between *rubric stage* and *rescuing rung* is the paper's rigor: neither the hand-label nor the intervention is trusted alone.

---

## 5. Output (one JSON object per attempt, per grader)

```json
{
  "problem_id": "...",
  "grader": "opus-4.5 | gpt-5.5 | panel",
  "verdict": "SOLVED|PARTIAL|FAILED|UNGRADED",
  "stage": "strategy|outline|execution|none",
  "final_answer_correct": true | false | "n/a",
  "one_line": "<=25 words: what happened vs a valid solution",
  "gap_detail": "<=50 words: the first fatal gap, or 'complete' if solved"
}
```

Be adversarial. Find the first fatal gap against the per-solution rubric, then read off verdict and stage. When unsure between PARTIAL and FAILED: *is there a proven, non-trivial piece a referee would award marks for?* Yes → PARTIAL, else → FAILED.
