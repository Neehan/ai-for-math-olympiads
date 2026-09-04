#!/usr/bin/env python3
"""Build the AIME 2026 answer-graded dataset files.

AIME is a short-answer contest: every problem has a single integer answer in
0-999 and no published proof.  The harness therefore grades this dataset by
final-answer equivalence (prompts/audit_answer.md) rather than by proof
review, and no oracle hint, outline, or selection artifacts exist for it.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


SOURCE_URL = "https://huggingface.co/datasets/math-ai/aime26/resolve/main/aime2026.jsonl"
SOURCE_PAGE = "https://huggingface.co/datasets/math-ai/aime26"
SOURCE_SHA256 = "52822957957a3f577d1e9706c36a66a8108a3f99b6aff424cfb72dff0094a9ee"
EXPECTED_PROBLEMS = 30
# AIME answers are exactly the integers 0-999; anything else is a broken row.
ANSWER_MIN = 0
ANSWER_MAX = 999
# AIME publishes no per-problem subject label, and the harness only uses domain
# for CLI filtering, so the whole contest forms one filterable group.
DOMAIN = "aime"
TASK = "answer_only"
PROBLEM_ID_FORMAT = "aime-2026-{index:02d}"


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
            f"aime2026.jsonl SHA-256 changed: expected {SOURCE_SHA256}, got {digest}"
        )

    rows = [
        json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()
    ]
    if len(rows) != EXPECTED_PROBLEMS:
        raise RuntimeError(f"expected {EXPECTED_PROBLEMS} rows, found {len(rows)}")
    # The source orders problems by its own 1..30 id; sort on it so the frozen
    # problem_id assignment cannot depend on file order.
    rows.sort(key=lambda row: int(row["id"]))
    if [int(row["id"]) for row in rows] != list(range(1, EXPECTED_PROBLEMS + 1)):
        raise RuntimeError("source ids are not exactly 1..30")

    problems: list[dict[str, object]] = []
    solutions: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        problem_id = PROBLEM_ID_FORMAT.format(index=index)
        statement = str(row["problem"]).strip()
        answer = str(row["answer"]).strip()
        if not statement:
            raise RuntimeError(f"{problem_id}: empty statement")
        if not answer.isdigit() or not ANSWER_MIN <= int(answer) <= ANSWER_MAX:
            raise RuntimeError(f"{problem_id}: answer {answer!r} is not an AIME answer")
        # Strip the leading zeros a mirror may carry, so the judge compares the
        # same canonical integer the contest publishes.
        answer = str(int(answer))

        problems.append(
            {
                "problem_id": problem_id,
                "statement": statement,
                "domain": DOMAIN,
                "task": TASK,
            }
        )
        solutions.append(
            {
                "problem_id": problem_id,
                "statement": statement,
                "answer": answer,
                "source_url": SOURCE_PAGE,
                "note": "Imported from math-ai/aime26; answer-graded, no reference proof.",
            }
        )

    output_dir = Path(__file__).resolve().parents[1] / "local_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "aime26_problems.jsonl", problems)
    _write_jsonl(output_dir / "aime26_solutions.jsonl", solutions)
    print(f"Wrote {len(problems)} AIME 2026 problems and answers to {output_dir}.")


if __name__ == "__main__":
    main()
