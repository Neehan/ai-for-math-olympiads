#!/usr/bin/env python3
"""Safely merge a cloned Hugging Face results snapshot into the workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path


RESULT_ROOTS = ("results", "results-imobench")
DERIVED_SEED_FILES = frozenset({"audit.json", "state_audit.json"})
COMPILED_FILES = frozenset({"audit.jsonl", "state_audit.jsonl"})


def _seed_dirs(base: Path) -> dict[str, Path]:
    seeds: dict[str, Path] = {}
    for root_name in RESULT_ROOTS:
        root = base / root_name
        if not root.exists():
            continue
        for path in root.glob("*/*/*/seed_*"):
            if path.is_dir():
                seeds[path.relative_to(base).as_posix()] = path
    return seeds


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generation_manifest(seed_dir: Path) -> dict[str, str]:
    """Hash generation artifacts while excluding judge-derived files."""
    manifest: dict[str, str] = {}
    for path in seed_dir.rglob("*"):
        if not path.is_file() or path.name in DERIVED_SEED_FILES:
            continue
        relative = path.relative_to(seed_dir)
        if "audit_scratch" in relative.parts or path.name == ".DS_Store":
            continue
        manifest[relative.as_posix()] = _sha256(path)
    return manifest


def _record_key(record: dict[str, object]) -> tuple[str, ...]:
    fields = (
        "problem_id",
        "arm",
        "seed",
        "parallel_run",
        "run",
        "strategy_index",
        "candidate_index",
        "uniform_strategy_bank_seed",
        "uniform_strategy_index",
        "uniform_strategy_run",
    )
    return tuple(str(record.get(field, "")) for field in fields)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def _atomic_write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _merge_compiled_jsonl(source: Path, destination: Path) -> int:
    remote = {_record_key(record): record for record in _load_jsonl(source)}
    local = {_record_key(record): record for record in _load_jsonl(destination)}
    remote.update(local)  # Existing local verdicts remain authoritative.
    records = [remote[key] for key in sorted(remote)]
    _atomic_write_jsonl(destination, records)
    return len(records)


def merge_snapshot(source: Path, destination: Path, *, dry_run: bool) -> int:
    remote_seeds = _seed_dirs(source)
    local_seeds = _seed_dirs(destination)
    conflicts: list[tuple[str, list[str]]] = []
    identical = 0
    local_only = len(set(local_seeds) - set(remote_seeds))
    remote_only = len(set(remote_seeds) - set(local_seeds))
    remote_superset = 0
    local_superset = 0
    compatible_diverged = 0

    for key in sorted(set(remote_seeds) & set(local_seeds)):
        local_manifest = _generation_manifest(local_seeds[key])
        remote_manifest = _generation_manifest(remote_seeds[key])
        mismatches = sorted(
            path
            for path in set(local_manifest) & set(remote_manifest)
            if local_manifest[path] != remote_manifest[path]
        )
        if mismatches:
            conflicts.append((key, mismatches))
            continue
        local_files = set(local_manifest)
        remote_files = set(remote_manifest)
        if local_files == remote_files:
            identical += 1
        elif local_files < remote_files:
            remote_superset += 1
        elif remote_files < local_files:
            local_superset += 1
        else:
            compatible_diverged += 1

    print(
        "Seed comparison: "
        f"{identical} identical, {remote_only} remote-only, "
        f"{local_only} local-only, {remote_superset} remote supersets, "
        f"{local_superset} local supersets, "
        f"{compatible_diverged} compatible divergent supersets."
    )
    if conflicts:
        print("Generation conflicts; no files were merged:")
        for seed, paths in conflicts:
            preview = ", ".join(paths[:5])
            suffix = " ..." if len(paths) > 5 else ""
            print(f"  {seed}: {preview}{suffix}")
        return 3
    if dry_run:
        print("Dry run complete; no files were changed.")
        return 0

    copied = 0
    compiled_sources: list[tuple[Path, Path]] = []
    for root_name in RESULT_ROOTS:
        remote_root = source / root_name
        if not remote_root.exists():
            continue
        for remote_path in remote_root.rglob("*"):
            if not remote_path.is_file() or remote_path.name == ".DS_Store":
                continue
            relative = remote_path.relative_to(source)
            local_path = destination / relative
            if remote_path.name in COMPILED_FILES:
                compiled_sources.append((remote_path, local_path))
                continue
            if local_path.exists():
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote_path, local_path)
            copied += 1

    compiled = 0
    for remote_path, local_path in compiled_sources:
        _merge_compiled_jsonl(remote_path, local_path)
        compiled += 1
    print(f"Merged {copied} missing files and reconciled {compiled} compiled JSONLs.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        merge_snapshot(
            args.source.resolve(), args.destination.resolve(), dry_run=args.dry_run
        )
    )


if __name__ == "__main__":
    main()
