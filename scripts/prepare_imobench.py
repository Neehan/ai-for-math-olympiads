#!/usr/bin/env python3
"""Build the non-geometry IMO-ProofBench Advanced replication files."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import urllib.request
from pathlib import Path


SOURCE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/superhuman/"
    "main/imobench/proofbench_v2.csv"
)
SOURCE_PAGE = (
    "https://github.com/google-deepmind/superhuman/"
    "blob/main/imobench/proofbench_v2.csv"
)
SOURCE_SHA256 = "aa8b813dbd4068137e3d165e5da228f6e0e1cc85a91c37883e1791b954e43af0"
ADVANCED_ID = re.compile(r"PB-Advanced-(\d{3})\Z")
CATEGORY_TO_DOMAIN = {
    "Algebra": "algebra",
    "Combinatorics": "combinatorics",
    "Number theory": "number_theory",
}
HINTS = {
    "PB-Advanced-001": (
        "Shift to $x_m=A_{m-2024}$; bound $A_N=O(\\sqrt{N}\\log N)$, then use "
        "discrete intermediate values for nondecreasing unit-step sequences to make "
        "$m/x_m$ attain all large integers."
    ),
    "PB-Advanced-002": (
        "Put the family on one forward orbit. Remove disjoint cycles, leaving a DAG; "
        "bound their length-lcm from total 120, then backtrace a post-period symmetric difference."
    ),
    "PB-Advanced-004": (
        "Use the maximum-degree-3 dual tree. Cut a maximally balanced edge, then apply "
        "the same edge-cut lemma to its larger component; track all three sizes."
    ),
    "PB-Advanced-006": (
        "Handle constants; otherwise derive $f(0)=0,f(1)=1,f\\circ f=f$ and "
        "$f(x-f(x))=0$. Split by image $\\subseteq\\{-1,0,1\\}$; otherwise distinguish "
        "whether an even preimage of $1$ exists, then propagate fibers."
    ),
    "PB-Advanced-007": (
        "Find a quadratic seed pair by coefficient comparison; repeatedly replace $P$ "
        "with $P\\circ(Q-x-1)$, proving the identity persists while $\\deg P$ multiplies."
    ),
    "PB-Advanced-008": (
        "Use generalized Euler to prove $F_n\\bmod m$ eventually has period "
        "$\\Phi(m)=\\operatorname{lcm}(m,\\varphi(m),\\ldots)$; induct on $c$, "
        "exploiting $\\gcd(c,\\Phi(\\varphi(c)))<c$, and finish by CRT."
    ),
    "PB-Advanced-011": (
        "Force surjectivity using $(x,y)=(c+1,c/f(c+1))$; from $f(a)=f(b)$, compare "
        "substitutions using preimages of $1$ and $1/(ab)$ to prove injectivity; finish with $y=1$."
    ),
    "PB-Advanced-012": (
        "Rule out even $n$ by Fermat descent. For $n=3$, combine $p=c^2+d^2$ "
        "with Jacobi's $p^3$ representations, factor coprime terms into squares, "
        "then descend $x^2+3y^4=z^4$."
    ),
    "PB-Advanced-013": (
        "Add $2^{-(n+1)}$ and induct backward on tail products, repeatedly applying "
        "$\\frac1{1+x}+\\frac1{1+y}\\geq\\frac2{1+\\sqrt{xy}}$ when $xy\\geq1$, "
        "using the ordering to justify applicability."
    ),
    "PB-Advanced-014": (
        "Track parity and residues modulo 4. Align unequal even residues using triple "
        "versus $+2$; for $x,x+4d$, compare repeated-$+2$-then-triple with triple-then-repeated-$+2$."
    ),
    "PB-Advanced-017": (
        "Set $a=d,b=n/d$, so $ab\\mid a^2+b^2+c$. Construct an upper bound; for "
        "smaller $c$, Vieta-descend to $(1,1)$, reverse via "
        "$(x,y)\\mapsto(y,(c+2)y-x)$, and exclude $ab\\equiv6\\pmod7$."
    ),
    "PB-Advanced-018": (
        "Join differently colored neighbors; use a turning $2\\times2$ boundary to "
        "identify critical multiplicity, then force an $n$-cell path. "
        "Conversely, separate $\\sqrt n$-blocks by monochromatic seams."
    ),
    "PB-Advanced-019": (
        "Set $\\alpha=2r=m+\\varepsilon$, $m=\\lfloor\\alpha\\rfloor$; rewrite the "
        "condition as $n\\mid\\sum_{k=1}^n\\lfloor k\\alpha\\rfloor$. Split by "
        "$m$'s parity; induct on $n$, forcing $\\lfloor n\\varepsilon\\rfloor=0$ or "
        "$n-1$ respectively; use all $n$."
    ),
    "PB-Advanced-020": (
        "Rewrite as $\\gcd(x^n+y,y^n+x)$. Stabilization gives "
        "$g\\mid2\\gcd(x,y)$. Write $x=da,y=db$; choosing "
        "$n\\equiv-1\\pmod{\\varphi(d^2ab+1)}$ forces $d^2ab+1$ into the normalized "
        "gcd, which is at most $2$."
    ),
    "PB-Advanced-021": (
        "Classify values by infinite occurrence; prove eventual small-big alternation. "
        "Interpret each returning small value as a frequency-threshold count; bounded "
        "nondecreasing first-repeat gaps force parity-subsequence periodicity."
    ),
    "PB-Advanced-023": (
        "Preselect first-entered cells of successive new rows in distinct columns for "
        "the lower bound. Sweep row 2; branch into flanking or diagonal edge-detour routes."
    ),
    "PB-Advanced-024": (
        "Substitute $y=b-P(a)$ for a two-branch composition law; derive bijectivity and "
        "$P^{-1}(x)=-P(-x)$. Rule out two distinct nonzero values of "
        "$P(x)-P^{-1}(x)$; test $P(x)=2\\lfloor x\\rfloor-x$."
    ),
    "PB-Advanced-025": (
        "For each $1\\leq\\ell\\leq k$, use CRT to write "
        "$n^k\\bmod(2n)^\\ell=c_\\ell n^\\ell$ with $c_\\ell\\geq1$; compare this "
        "remainder with $(d+1)(2n)^{\\ell-1}$, also bounding digit length."
    ),
    "PB-Advanced-026": (
        "Assume roots real; choose $k+1$ roots. Pigeonhole two degree-$k$ divisors "
        "sharing a zero-coefficient position; their gcd has consecutive zero "
        "coefficients, contradicting Rolle after differentiation."
    ),
    "PB-Advanced-027": (
        "Choose $S$ outside $PQ$'s diameter disk, inducing the empty-diameter-disk graph. "
        "Prove connectivity by squared-distance descent; rule out crossing edges via "
        "an obtuse quadrilateral angle."
    ),
    "PB-Advanced-029": (
        "Use $n=2$ for necessity. For the resulting parity, fix $p^e\\Vert n+1$; "
        "reduce $\\binom{n}{i}$ modulo $p^e$ to signed smaller coefficients, group "
        "$p$-blocks, and induct on $n+1$."
    ),
    "PB-Advanced-030": (
        "Fix one person's arcs. Delete Hall-deficient sets and neighbors until a "
        "nonempty matchable remainder; assign it, then induct because each low-value "
        "deletion merges surviving arcs."
    ),
}
OUTLINES = {
    "PB-Advanced-001": [
        "Count $k$-th powers separately to bound $A_N=O(\\sqrt N\\log N)$, making $N/A_N$ unbounded along a suitable subsequence as $N$ grows.",
        "Shift and pad with zeros to $x_m=A_{m-2024}$; this integer sequence has zero-or-one increments and retains unbounded ratios $m/x_m$.",
        "At the first index where $m/x_m$ crosses each large integer, unit increments force equality; translate back to the required divisibility.",
    ],
    "PB-Advanced-002": [
        "View reachability as a semicomplete digraph; contracting mutually reachable cycles yields a vertex whose forward orbit contains the pairwise-comparable family.",
        "Decompose love digraph into disjoint cycles and an acyclic residual; choose $L\\ge120$ divisible by every cycle length to synchronize behavior.",
        "After intersections stabilize, backtrack symmetric differences through the acyclic residual; bound the transient by $240L$ and estimate $L<2^{60}$.",
    ],
    "PB-Advanced-004": [
        "Represent triangles by the dual tree, whose edges are diagonals and whose vertex degrees are at most three.",
        "Use the balanced-edge lemma for subcubic trees to cut once into components sized between $6n-1$ and $12n+1$.",
        "Apply the lemma inside the larger component; integer bounds put all three components between $3n$ and $9n$ triangles.",
    ],
    "PB-Advanced-006": [
        "Handle constant maps first; otherwise strategic substitutions at $x=0,y=0,x=1$ yield $f(0)=0$, $f(1)=1$, idempotence, and the zero identity $f(x-f(x))=0$.",
        "When the image lies in $\\{-1,0,1\\}$, propagate neighboring values to obtain exactly the parity-two and residue-three periodic maps.",
        "Outside this image, use $f(2)$ to exclude even preimages of $1$; eliminate $f(2)=0$, then fiber induction from $f(-1)=-1$ forces identity.",
    ],
    "PB-Advanced-007": [
        "Direct coefficient comparison supplies the explicit quadratic seed $P_0=x^2+5x/2$ and $Q_0=x^2+7x/2+3/2$, satisfying the required polynomial composition identity.",
        "Define $P_{j+1}=P_j\\circ(Q_0-x-1)$; applying the induction hypothesis at $x=Q_0(y)-y-1$ proves every $(P_j,Q_0)$ satisfies exactly the same composition identity.",
        "Each recursion doubles the degree of $P_j$, so choosing a sufficiently large index produces degree at least $2024$.",
    ],
    "PB-Advanced-008": [
        "Prove generalized Euler periodicity for arbitrary bases, then induct on $m$ to make $F_n\\bmod m$ eventually $\\Phi(m)$-periodic.",
        "Use the largest prime factor of $c$ to establish $\\gcd(c,\\Phi(\\varphi(c)))<c$, enabling strong induction on the target modulus.",
        "Choose an inductive solution modulo this gcd; eventual $F_n$-periodicity makes the remaining congruences compatible, so CRT completes the construction.",
    ],
    "PB-Advanced-011": [
        "Substitute $(x,y)=(c+1,c/f(c+1))$ to express each arbitrary positive $c$ as an output of $f$, thereby establishing surjectivity onto $\\mathbb R^+$.",
        "Preimages of $1$ and $1/(ab)$ turn $f(a)=f(b)$ into equal and proportionally equal values at $1+1/a,1+1/b$, forcing $a=b$.",
        "Injectivity applied at $y=1$ gives $f(x)=1/x+c$; surjectivity onto all positive reals forces $c=0$, and direct substitution verifies it.",
    ],
    "PB-Advanced-012": [
        "Apply Fermat's descent for $x^4+y^4=z^2$ to exclude every even exponent $n$, after removing any common prime factors.",
        "For $n=3$, use $p=c^2+d^2$ and Jacobi's two representations of $p^3$; coprime-factor analysis produces a solution to $x^2+3y^4=z^4$.",
        "Factor $(z^2-x)(z^2+x)=3y^4$ in a minimal solution; parity, coprimality, and repeated square extraction yield a smaller solution or contradiction.",
    ],
    "PB-Advanced-013": [
        "Prove the key two-variable compression inequality $1/(1+x)+1/(1+y)\\ge2/(1+\\sqrt{xy})$ for $xy\\ge1$ by directly factoring its difference into nonnegative factors.",
        "Induct backward on $k$ to establish $2^{-(n+1)}+\\sum_{i=k}^n1/b_i\\ge2^{-(k-1)}/(1+(\\prod_{i=k}^na_i)^{2^{k-1}})$, applying the compression lemma carefully at each successive adjoining term.",
        "Ordering and total product one guarantee every product condition; evaluating the final tail-product bound at $k=1$ yields the claim.",
    ],
    "PB-Advanced-014": [
        "Parity is invariant, while both operations swap odd residues modulo four; derive the necessary parity and residue conditions.",
        "For even numbers with different residues, multiply one by three and add two to the other to align modulo four.",
        "For $x,x+4d$, compare $k$ additions then tripling against tripling then $k$ additions; equality follows by choosing $k=3d$.",
    ],
    "PB-Advanced-017": [
        "Set $a=d,b=n/d$ to obtain $ab\\mid a^2+b^2+c$; the pair $(4,19)$ constructs $c=3$, supplying the required upper bound for the minimum.",
        "For $c\\in\\{1,2\\}$, Vieta-jump to smaller positive pairs until $(1,1)$, forcing the invariant quotient to equal $c+2$ under repeated descent.",
        "Reverse descent by $(x,y)\\mapsto(y,(c+2)y-x)$; compute both recurrences modulo $7$ and verify consecutive products there never equal $6$.",
    ],
    "PB-Advanced-018": [
        "Set $m=\\lceil\\sqrt n\\rceil-1$; place regular monochromatic two-by-two barriers at m-grid intersections and extend them into monochromatic seams.",
        "These seams confine every alternating path within one m-by-m block; counting exceptional colors proves $3a(n)\\ge n^2-n-2\\sqrt n-3$.",
        "With multiplicity at most three, differently-colored adjacency components and boundary-turn lemma force grid-diameter $n-1$, yielding a snake and $3a(n)\\le n^2+2$.",
    ],
    "PB-Advanced-019": [
        "Put $\\alpha=2r=m+\\varepsilon$ and rewrite the requirement as $n\\mid\\sum_{k=1}^n\\lfloor k\\alpha\\rfloor$, isolating the integer-part contribution by parity of $m$.",
        "For even $m$, strong induction forces $\\lfloor n\\varepsilon\\rfloor=0$ for every $n$, hence $\\varepsilon=0$; verify the resulting values satisfy every divisibility.",
        "For odd $m$, the same divisibility inductively forces $\\lfloor n\\varepsilon\\rfloor=n-1$ for all $n$, contradicting $0\\le\\varepsilon<1$ in the limit.",
    ],
    "PB-Advanced-020": [
        "Rewrite $a_n$ as $\\gcd(x^n+y,y^n+x)$; if it stabilizes at $g$, comparing consecutive exponents for all large $n$ implies $g\\mid2\\gcd(x,y)$.",
        "Write $x=da,y=db$ with $\\gcd(a,b)=1$; Euler's theorem at $n\\equiv-1\\pmod{\\varphi(d^2ab+1)}$ makes $d^2ab+1$ divide both normalized terms for suitably large exponents.",
        "The stabilized normalized gcd is at most $2$, whereas it is divisible by $d^2ab+1$; deduce $d=a=b=1$ and verify stabilization.",
    ],
    "PB-Advanced-021": [
        "Classify values into recurrent small, exhausted medium, and bounded-frequency big values; prove the sequence eventually alternates small and big.",
        "When a big value precedes small $h$, identify $h$ as the number of recurrent values crossing its frequency threshold.",
        "First-repeat gaps among small values are paritywise nondecreasing and bounded by $2k$; stabilization makes the corresponding gender subsequence eventually periodic.",
    ],
    "PB-Advanced-023": [
        "Against any two-attempt strategy, preselect its first-entered row-two cell and ensuing first-entered row-three cell; distinct columns force both penalties.",
        "First sweep row two to reveal its selected column; when interior, try two flanking routes risking distinct row-three cells.",
        "At an edge, sweep diagonally until the first obstacle, then detour across its row and descend a provably unselected column.",
    ],
    "PB-Advanced-024": [
        "Substitute $y=b-P(a)$ to obtain the two-branch composition law; diagonal specialization gives fixed points and then injectivity with $P(0)=0$.",
        "Specialize the branch law at $y=-P(x)$ to derive $P(-P(x))=-x$, thereby obtaining bijectivity and the inverse relation $P^{-1}(x)=-P(-x)$.",
        "Assume distinct nonzero differences; three cross-applications using inverse preimages force equality or zero, while $P(x)=2\\lfloor x\\rfloor-x$ attains two values.",
    ],
    "PB-Advanced-025": [
        "For each $\\ell\\le k$, CRT expresses $n^k\\bmod(2n)^\\ell=c_\\ell n^\\ell$ with $c_\\ell\\ge1$, thereby encoding precisely the final $\\ell$ base-$2n$ digits.",
        "Compare $c_\\ell n^\\ell$ with $(d+1)(2n)^{\\ell-1}$; for sufficiently large $n$, this forces the $\\ell$-th digit to exceed $d$.",
        "Bound $n^k$ between consecutive powers of $2n$ to obtain exactly $k$ digits, then choose one threshold valid for every $\\ell$.",
    ],
    "PB-Advanced-026": [
        "Assume all roots are real; select any $k+1$ roots, reducing to degree $n=k+1$ while preserving the divisor hypothesis.",
        "Among the $n$ degree-$n-1$ divisors, pigeonhole two sharing an internal zero-coefficient position; their common factor has consecutive zero coefficients.",
        "Differentiate to the missing degree: consecutive zero coefficients make zero a multiple root, contradicting Rolle interlacing for distinct real roots.",
    ],
    "PB-Advanced-027": [
        "Choose $S$ outside $PQ$'s diameter disk; direct similarity makes $AB$ a road exactly when its closed diameter disk is empty.",
        "For connectivity, repeatedly choose a closed-diameter-disk witness in another component; city separation decreases squared distances by at least one.",
        "Crossing edges form a quadrilateral; an angle is at least ninety degrees, putting its vertex in the opposite diameter disk.",
    ],
    "PB-Advanced-029": [
        "Setting $n=2$ forces $3\\mid2+2^k$, establishing first that the universal integrality condition necessarily requires even positive integer $k$.",
        "For each $p^e\\Vert n+1$, prove $\\binom ni\\equiv\\pm\\binom{(n+1)/p-1}{\\lfloor i/p\\rfloor}\\pmod{p^e}$ by separating product factors divisible by $p$ from the unit factors.",
        "Even $k$ removes signs; grouping indices into $p$-blocks gives $S(n)\\equiv pS((n+1)/p-1)\\pmod{p^e}$, and induction supplies divisibility by every $p^e$.",
    ],
    "PB-Advanced-030": [
        "Fix one person's consecutive arcs; connect each person to every arc worth at least one, making the fixed person universal.",
        "Repeatedly delete Hall-deficient people and all neighboring arcs until a nonempty remainder admits a people-saturating matching; allocate its arcs.",
        "For each remaining person, deleting a low-value arc merges adjacent personal arcs with value at least $1+1-1$; apply induction.",
    ],
}


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(
            f"proofbench_v2.csv SHA-256 changed: expected {SOURCE_SHA256}, got {digest}"
        )

    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    advanced = [row for row in rows if ADVANCED_ID.fullmatch(row["Problem ID"])]
    if len(advanced) != 30:
        raise RuntimeError(f"expected 30 PB-Advanced rows, found {len(advanced)}")

    selected = [row for row in advanced if row["Category"] in CATEGORY_TO_DOMAIN]
    selected.sort(key=lambda row: row["Problem ID"])
    if len(selected) != 22:
        raise RuntimeError(f"expected 22 non-geometry Advanced rows, found {len(selected)}")
    selected_ids = {row["Problem ID"].strip() for row in selected}
    if selected_ids != HINTS.keys():
        missing = sorted(selected_ids - HINTS.keys())
        extra = sorted(HINTS.keys() - selected_ids)
        raise RuntimeError(f"hint ID mismatch: missing={missing}, extra={extra}")
    if selected_ids != OUTLINES.keys():
        missing = sorted(selected_ids - OUTLINES.keys())
        extra = sorted(OUTLINES.keys() - selected_ids)
        raise RuntimeError(f"outline ID mismatch: missing={missing}, extra={extra}")

    problems: list[dict[str, object]] = []
    solutions: list[dict[str, object]] = []
    hints: list[dict[str, object]] = []
    outlines: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in selected:
        problem_id = row["Problem ID"].strip()
        statement = row["Problem"].strip()
        solution = row["Solution"].strip()
        domain = CATEGORY_TO_DOMAIN[row["Category"]]
        hint = HINTS[problem_id]
        outline = OUTLINES[problem_id]
        if problem_id in seen:
            raise RuntimeError(f"duplicate problem ID: {problem_id}")
        if not statement or not solution:
            raise RuntimeError(f"missing required content for {problem_id}")
        if not 1 <= len(hint.split()) <= 25:
            raise RuntimeError(
                f"{problem_id}: hint must contain 1-25 whitespace-delimited words"
            )
        if len(outline) != 3:
            raise RuntimeError(f"{problem_id}: outline must contain exactly 3 steps")
        for index, step in enumerate(outline, start=1):
            word_count = len(step.split())
            if not 17 <= word_count <= 20:
                raise RuntimeError(
                    f"{problem_id}: outline step {index} has {word_count} words; "
                    "expected 17-20"
                )
        seen.add(problem_id)

        problems.append(
            {
                "problem_id": problem_id,
                "statement": statement,
                "domain": domain,
            }
        )
        solutions.append(
            {
                "problem_id": problem_id,
                "statement": statement,
                "reference_solutions": [
                    {
                        "type": "imobench",
                        "route_id": "hard_hint",
                        "solution": solution,
                        "source_url": SOURCE_PAGE,
                        "note": "Imported from IMO-ProofBench v2.",
                    }
                ],
            }
        )
        hints.append(
            {
                "problem_id": problem_id,
                "domain": domain,
                "hint": hint,
            }
        )
        outlines.append(
            {
                "problem_id": problem_id,
                "steps": [{"step": step} for step in outline],
                "note": "audited",
            }
        )

    output_dir = Path(__file__).resolve().parents[1] / "local_data"
    _write_jsonl(output_dir / "imobench_problems.jsonl", problems)
    _write_jsonl(output_dir / "imobench_solutions.jsonl", solutions)
    _write_jsonl(output_dir / "imobench_hints.jsonl", hints)
    _write_jsonl(output_dir / "imobench_outlines.jsonl", outlines)
    print(
        "Wrote 22 IMO-ProofBench Advanced problems "
        "(8 algebra, 8 combinatorics, 6 number theory)."
    )


if __name__ == "__main__":
    main()
