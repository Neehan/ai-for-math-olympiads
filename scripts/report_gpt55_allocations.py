#!/usr/bin/env python3
"""Read-only, equal-budget GPT-5.5 N=1/2/4 comparison for the paper.

Use all 57 problems, three allocation trials per problem, and score >=5.
N=4 groups are fixed by seed index, not selected by outcome. Bootstrap
problems within dataset, retaining all seeds and allocations together.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_joint_allocation import proof_curve


def load(base, arm, blocks, seeds):
    directory = base / arm
    records = [json.loads(line) for line in (directory / "audit.jsonl").read_text().splitlines() if line.strip()]
    compiled = {(r["problem_id"], r["seed"]): r for r in records}
    assert len(compiled) == len(records), "Duplicate compiled audit identities"
    curves = {}
    for path in sorted(directory.glob("*/seed_*/audit.json")):
        record = json.loads(path.read_text())
        key = record["problem_id"], record["seed"]
        if key[1] not in seeds:
            continue
        assert record == compiled[key], f"Stale compiled audit: {path}"
        curves[key] = proof_curve(record, blocks, 5)
        for k in range(1, blocks + 1):
            audit = record if k == blocks else record["budget_cuts"][f"{k}x"]
            proof = path.parent / ("solution.md" if k == blocks else f"solution_{k}x.md")
            content = proof.read_text() if proof.exists() else ""
            if content.strip():
                assert hashlib.sha256(content.encode()).hexdigest() == audit["solution_sha256"], str(proof)
            else:
                assert audit["audit_score"] < 5, f"Passing audit without proof: {proof}"
    return curves


def collect():
    rows = []
    for dataset, expected in [("results", 35), ("results-imobench", 22)]:
        base = ROOT / dataset / "litellm-gpt-5.5"
        long = load(base, "baseline-sequential", 8, range(1, 4))
        late = load(base, "late-baseline-sequential", 4, range(1, 4))
        short = load(base, "baseline-sequential-2x", 2, range(1, 13))
        problems = sorted({p for p, _ in long})
        assert len(problems) == expected
        assert set(long) == set(late) == {(p, s) for p in problems for s in range(1, 4)}
        assert set(short) == {(p, s) for p in problems for s in range(1, 13)}
        for problem in problems:
            outcomes = []
            for trial in range(1, 4):
                group = range(4 * trial - 3, 4 * trial + 1)
                outcomes.append([long[problem, trial][-1], max(long[problem, trial][3], late[problem, trial][-1]), max(short[problem, seed][-1] for seed in group)])
            rows.append(dict(dataset=dataset, problem=problem, outcomes=outcomes))
    return rows


def summarize(rows):
    values = np.array([r["outcomes"] for r in rows])
    return dict(problems=len(rows), trials=len(rows) * 3,
                solved=values.sum(axis=(0, 1)).tolist(),
                percent=(100 * values.mean(axis=(0, 1))).tolist())


def main():
    rows = collect()
    rng = np.random.default_rng(20260914)
    totals = np.zeros((30000, 3))
    datasets = {}
    for dataset in ("results", "results-imobench"):
        subset = [r for r in rows if r["dataset"] == dataset]
        datasets[dataset] = summarize(subset)
        values = np.array([r["outcomes"] for r in subset]).sum(axis=1)
        indices = rng.integers(0, len(subset), (30000, len(subset)))
        totals += values[indices].sum(axis=1)
    rates = 100 * totals / (3 * len(rows))
    intervals = {f"N={a} minus N={b}": np.quantile(rates[:, i] - rates[:, j], [.025, .975]).tolist()
                 for a, b, i, j in [(2, 1, 1, 0), (2, 4, 1, 2), (4, 1, 2, 0)]}
    print(json.dumps(dict(allocations=[1, 2, 4], datasets=datasets,
                          combined=summarize(rows), bootstrap_replicates=30000,
                          bootstrap_seed=20260914, intervals_pp=intervals), indent=2))


if __name__ == "__main__":
    main()
