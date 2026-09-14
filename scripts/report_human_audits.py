#!/usr/bin/env python3
"""Compare human CSVs with the frozen judgments in their audit packets.

Print aggregate and per-auditor counts; do not modify data or paper files.
Score-matrix rows are automated scores and columns are human scores.
"""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORES = (0, 5, 6, 7)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def boolean(value):
    if value.strip().lower() not in {"true", "false"}:
        raise ValueError(f"Invalid step label: {value!r}")
    return value.strip().lower() == "true"


def collect(auditor, folder, prefix):
    metadata = read_csv(ROOT / "audit_package" / folder / "metadata.csv")
    human = read_csv(ROOT / "local_data" / f"auditor_{auditor}_scores.csv")
    index = {row["item_id"]: row for row in metadata}
    assert len(index) == len(metadata) == len(human) == 25
    rows = []
    for row in human:
        item = f"{prefix}{int(row['Problem ID']):02d}"
        reference = index[item]
        automatic_score = int(reference["existing_score"])
        human_score = int(row["Solution Score"])
        assert automatic_score in SCORES and human_score in SCORES
        rows.append(dict(
            item=item, path=reference["solution_path"],
            auto=automatic_score, human=human_score,
            auto_steps=[boolean(reference[f"existing_step{k}_present"]) for k in (1, 2, 3)],
            human_steps=[boolean(row[f"Step {k} Present"]) for k in (1, 2, 3)],
        ))
    assert {row["item"] for row in rows} == set(index)
    return rows


def binary(pairs):
    counts = Counter(pairs)  # (automated prediction, human reference)
    tp, fp = counts[True, True], counts[True, False]
    fn, tn = counts[False, True], counts[False, False]
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, agreement=tp + tn,
                labels=tp + fp + fn + tn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)


def summarize(rows):
    matrix = [[sum(r["auto"] == a and r["human"] == h for r in rows)
               for h in SCORES] for a in SCORES]
    differences = Counter(r["human"] - r["auto"] for r in rows)
    return dict(
        proofs=len(rows),
        solved=binary((r["auto"] >= 5, r["human"] >= 5) for r in rows),
        steps=binary(pair for r in rows for pair in zip(r["auto_steps"], r["human_steps"])),
        score_order=SCORES, score_counts=matrix,
        score_row_percent=[[100 * c / sum(row) if sum(row) else None for c in row] for row in matrix],
        exact_scores=differences[0],
        human_minus_automated=dict(sorted(differences.items())),
    )


def main():
    a = collect("a", "adib", "A")
    b = collect("b", "thanic", "T")
    assert len({r["path"] for r in a + b}) == 50, "Packets overlap"
    report = {"auditor_a": summarize(a), "auditor_b": summarize(b),
              "combined": summarize(a + b)}
    assert report["combined"]["solved"]["labels"] == 50
    assert report["combined"]["steps"]["labels"] == 150
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
