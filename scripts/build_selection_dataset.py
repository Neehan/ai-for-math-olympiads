"""Aggregate reviewed compression artifacts into hard_hint_selection.jsonl."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import (
    COMPRESSED_STRATEGIES_FILENAME,
    META_FILENAME,
    SEED_STATE_AUDIT_FILENAME,
    UNIFORM_COMPRESS_SAMPLE_SEED,
    UNIFORM_STRATEGIES_FILENAME,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def _canonical_hints(path: Path) -> dict[str, str]:
    hints: dict[str, str] = {}
    for record in _read_jsonl(path):
        problem_id = record.get("problem_id")
        hint = record.get("hint")
        if not isinstance(problem_id, str) or not isinstance(hint, str) or not hint.strip():
            raise ValueError(f"Malformed hint record in {path}")
        if problem_id in hints:
            raise ValueError(f"Duplicate hint record for {problem_id}")
        hints[problem_id] = hint.strip()
    return hints


def build_records(
    results_root: Path,
    hints_path: Path,
) -> list[dict[str, object]]:
    """Aggregate every compressed bank with labels inherited from raw audits."""
    hints = _canonical_hints(hints_path)
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    pattern = f"*/baseline-uniform-compress/*/seed_*/{COMPRESSED_STRATEGIES_FILENAME}"
    for artifact_path in sorted(results_root.glob(pattern)):
        artifact = _read_json(artifact_path)
        meta_path = artifact_path.with_name(META_FILENAME)
        if not meta_path.is_file():
            raise FileNotFoundError(f"Compression artifact lacks meta.json: {artifact_path}")
        meta = _read_json(meta_path)
        source_model = artifact.get("source_model")
        problem_id = artifact.get("problem_id")
        seed = meta.get("seed")
        oracle = artifact.get("oracle_strategy")
        generated = artifact.get("generated_strategies")
        if (
            not isinstance(source_model, str)
            or not isinstance(problem_id, str)
            or type(seed) is not int
            or seed != 1
        ):
            raise ValueError(
                f"Selection candidates must come from proposal seed 1: {artifact_path}"
            )
        key = (source_model, problem_id)
        if key in seen:
            raise ValueError(
                f"Duplicate compression artifact for {source_model}/{problem_id}"
            )
        seen.add(key)
        if meta.get("model") != source_model or meta.get("problem_id") != problem_id:
            raise ValueError(f"Compression artifact/meta identity mismatch: {artifact_path}")
        if meta.get("mode") != "uniform_compress":
            raise ValueError(f"Unexpected compression mode: {meta_path}")

        planner_dir = (
            artifact_path.parents[3]
            / "baseline-uniform-strategy-only"
            / problem_id
            / f"seed_{seed}"
        )
        planner_path = planner_dir / UNIFORM_STRATEGIES_FILENAME
        planner_meta_path = planner_dir / META_FILENAME
        state_path = planner_dir / SEED_STATE_AUDIT_FILENAME
        if not planner_path.is_file() or not planner_meta_path.is_file():
            raise FileNotFoundError(
                f"Compression artifact lacks its raw planner bank: {artifact_path}"
            )
        if not state_path.is_file():
            raise FileNotFoundError(
                f"Raw planner bank lacks state_audit.json: {planner_dir}"
            )
        planner = _read_json(planner_path)
        planner_meta = _read_json(planner_meta_path)
        state = _read_json(state_path)
        raw_strategies = planner.get("strategies")
        if not isinstance(raw_strategies, list) or not all(
            isinstance(value, str) and value.strip() for value in raw_strategies
        ):
            raise ValueError(f"Malformed raw planner bank: {planner_path}")
        if (
            planner_meta.get("problem_id") != problem_id
            or planner_meta.get("mode") != "uniform_strategy_only"
            or planner_meta.get("model") != source_model
            or planner_meta.get("seed") != seed
        ):
            raise ValueError(f"Raw planner/meta identity mismatch: {planner_dir}")
        if (
            state.get("problem_id") != problem_id
            or state.get("arm") != "baseline-uniform-strategy-only"
            or state.get("solver_model") != source_model
            or state.get("seed") != seed
        ):
            raise ValueError(f"Raw planner state-audit identity mismatch: {state_path}")
        if meta.get("sample_seed") != UNIFORM_COMPRESS_SAMPLE_SEED:
            raise ValueError(f"Unexpected sampling seed: {meta_path}")
        sampled_meta = meta.get("sampled_raw_strategy_indices")
        if (
            not isinstance(sampled_meta, list)
            or len(sampled_meta) != 3
            or not all(type(value) is int and value >= 1 for value in sampled_meta)
            or len(set(sampled_meta)) != 3
        ):
            raise ValueError(f"Malformed sampled strategy indices: {meta_path}")
        if not isinstance(oracle, str) or oracle.strip() != hints.get(problem_id):
            raise ValueError(f"Oracle sketch drift for {source_model}/{problem_id}")
        if len(oracle.split()) > 25:
            raise ValueError(f"Oracle sketch exceeds 25 words for {problem_id}")
        oracle_digest = hashlib.sha256(oracle.strip().encode("utf-8")).hexdigest()
        if (
            meta.get("oracle_strategy_sha256") != oracle_digest
            or meta.get("oracle_word_count") != len(oracle.split())
        ):
            raise ValueError(f"Frozen oracle/meta mismatch: {artifact_path}")
        if not isinstance(generated, list) or len(generated) != 3:
            raise ValueError(f"Expected exactly three compressed strategies: {artifact_path}")
        normalized: list[dict[str, object]] = []
        raw_indices: set[int] = set()
        candidate_ids: set[str] = set()
        state_records = state.get("strategies")
        if (
            not isinstance(state_records, list)
            or len(state_records) != len(raw_strategies)
        ):
            raise ValueError(f"Raw planner/state-audit count mismatch: {state_path}")
        for index, candidate in enumerate(generated, 1):
            if not isinstance(candidate, dict):
                raise ValueError(f"Malformed generated strategy: {artifact_path}")
            candidate_id = candidate.get("candidate_id")
            strategy = candidate.get("strategy")
            raw_strategy = candidate.get("raw_strategy")
            raw_index = candidate.get("raw_strategy_index")
            if (
                not isinstance(candidate_id, str)
                or not candidate_id.strip()
                or candidate_id == "oracle"
                or not isinstance(strategy, str)
                or not strategy.strip()
                or not isinstance(raw_strategy, str)
                or not raw_strategy.strip()
                or type(raw_index) is not int
                or raw_index < 1
            ):
                raise ValueError(
                    f"Malformed generated strategy {index}: {artifact_path}"
                )
            if not 18 <= len(strategy.split()) <= 25:
                raise ValueError(
                    f"Generated strategy {index} must contain 18--25 words: "
                    f"{artifact_path}"
                )
            if raw_index in raw_indices:
                raise ValueError(f"Repeated sampled raw strategy: {artifact_path}")
            if candidate_id in candidate_ids:
                raise ValueError(f"Repeated candidate_id: {artifact_path}")
            raw_indices.add(raw_index)
            candidate_ids.add(candidate_id)
            if raw_index > len(raw_strategies):
                raise ValueError(f"Raw strategy index is out of range: {artifact_path}")
            frozen_raw = str(raw_strategies[raw_index - 1]).strip()
            if raw_strategy.strip() != frozen_raw:
                raise ValueError(
                    f"Compression/raw planner text mismatch at candidate {index}: "
                    f"{artifact_path}"
                )
            state_record = state_records[raw_index - 1]
            digest = hashlib.sha256(frozen_raw.encode("utf-8")).hexdigest()
            oracle_match = (
                state_record.get("oracle_strategy_match")
                if isinstance(state_record, dict)
                else None
            )
            if (
                not isinstance(state_record, dict)
                or state_record.get("strategy_index") != raw_index
                or state_record.get("candidate_id") != f"strategy_{raw_index}"
                or state_record.get("strategy_sha256") != digest
                or not isinstance(oracle_match, bool)
            ):
                raise ValueError(
                    f"Raw strategy/state-audit mismatch at candidate {index}: "
                    f"{artifact_path}"
                )
            normalized.append(
                {
                    "candidate_id": candidate_id,
                    "strategy": strategy.strip(),
                    "oracle_strategy_match": oracle_match,
                    "raw_strategy_index": raw_index,
                }
            )
        if [candidate["raw_strategy_index"] for candidate in normalized] != sampled_meta:
            raise ValueError(f"Artifact/meta sampled-index mismatch: {artifact_path}")
        records.append(
            {
                "problem_id": problem_id,
                "source_model": source_model,
                "proposal_seed": seed,
                "oracle_strategy": oracle.strip(),
                "sample_seed": UNIFORM_COMPRESS_SAMPLE_SEED,
                "generated_strategies": normalized,
            }
        )
    return sorted(
        records,
        key=lambda record: (
            str(record["source_model"]),
            str(record["problem_id"]),
        ),
    )


def _atomic_write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--hints", type=Path, default=Path("local_data/hard_hints.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local_data/hard_hint_selection.jsonl"),
    )
    args = parser.parse_args()
    records = build_records(args.results_root, args.hints)
    if not records:
        raise SystemExit("No completed compression artifacts were found")
    _atomic_write_jsonl(args.output, records)
    print(f"wrote {len(records)} audited selection records to {args.output}")


if __name__ == "__main__":
    main()
