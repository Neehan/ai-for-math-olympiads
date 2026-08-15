#!/usr/bin/env python3
"""Recoverably archive sequential results that stopped before the round floor.

The whole seed directory moves, so its completion marker, generated artifacts,
and any stale audit verdict cannot be mixed with the replacement trajectory.
Compiled arm audits move too; the next audit command rebuilds them from the
remaining and replacement per-seed audit records.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.constants import SEQUENTIAL_MIN_ROUNDS_BEFORE_CONVERGENCE


SEQUENTIAL_ARMS = frozenset({"baseline-sequential", "hint-sequential"})


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _affected(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for meta_path in sorted(results_root.glob("*/*/*/seed_*/meta.json")):
        relative = meta_path.relative_to(results_root)
        if len(relative.parts) != 5 or relative.parts[1] not in SEQUENTIAL_ARMS:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        labels = meta.get("phase_labels", [])
        if not isinstance(labels, list):
            raise ValueError(f"Malformed phase_labels in {meta_path}")
        rounds = sum(label == "critique" for label in labels)
        if (
            meta.get("termination_reason") != "self_converged"
            or rounds >= SEQUENTIAL_MIN_ROUNDS_BEFORE_CONVERGENCE
        ):
            continue
        seed_dir = meta_path.parent
        if seed_dir.is_symlink():
            raise ValueError(f"Refusing symlinked seed directory: {seed_dir}")
        rows.append(
            {
                "model": relative.parts[0],
                "arm": relative.parts[1],
                "problem_id": relative.parts[2],
                "seed": int(relative.parts[3].removeprefix("seed_")),
                "completed_critique_rounds": rounds,
                "additional_rounds_to_floor": (
                    SEQUENTIAL_MIN_ROUNDS_BEFORE_CONVERGENCE - rounds
                ),
                "had_audit": (seed_dir / "audit.json").is_file(),
                "source": seed_dir.as_posix(),
                "relative_seed_dir": seed_dir.relative_to(results_root).as_posix(),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    archive_root = args.archive_root.resolve()
    rows = _affected(results_root)
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise SystemExit(
            f"refusing migration: expected {args.expected_count} affected seeds, "
            f"found {len(rows)}"
        )

    summary: dict[str, object] = {
        "minimum_critique_rounds": SEQUENTIAL_MIN_ROUNDS_BEFORE_CONVERGENCE,
        "affected_seed_count": len(rows),
        "audited_seed_count": sum(bool(row["had_audit"]) for row in rows),
        "additional_rounds_to_floor": sum(
            int(row["additional_rounds_to_floor"]) for row in rows
        ),
        "status": "dry_run",
        "moved_seed_count": 0,
        "seeds": rows,
        "compiled_audits": [],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.apply:
        return
    if archive_root.exists():
        raise SystemExit(f"refusing existing archive root: {archive_root}")

    compiled_audits = sorted(
        {
            results_root / str(row["model"]) / str(row["arm"]) / "audit.jsonl"
            for row in rows
        }
    )
    for row in rows:
        source = Path(str(row["source"]))
        target = archive_root / "results" / str(row["relative_seed_dir"])
        if not source.is_dir():
            raise SystemExit(f"source disappeared during preflight: {source}")
        if target.exists():
            raise SystemExit(f"archive collision during preflight: {target}")
    for source in compiled_audits:
        target = archive_root / "compiled-audits" / source.relative_to(results_root)
        if target.exists():
            raise SystemExit(f"archive collision during preflight: {target}")

    archive_root.mkdir(parents=True)
    manifest_path = archive_root / "manifest.json"
    summary["status"] = "moving"
    summary["compiled_audits"] = [
        source.relative_to(results_root).as_posix()
        for source in compiled_audits
        if source.is_file()
    ]
    _atomic_json(manifest_path, summary)

    for row in rows:
        source = Path(str(row["source"]))
        target = archive_root / "results" / str(row["relative_seed_dir"])
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        summary["moved_seed_count"] = int(summary["moved_seed_count"]) + 1
        _atomic_json(manifest_path, summary)

    for source in compiled_audits:
        if not source.is_file():
            continue
        target = archive_root / "compiled-audits" / source.relative_to(results_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)

    summary["status"] = "complete"
    _atomic_json(manifest_path, summary)
    print(f"archived {len(rows)} seed directories under {archive_root}")


if __name__ == "__main__":
    main()
