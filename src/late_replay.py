"""Canonical 3x baseline replay used by matched late interventions."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zstandard

from src.constants import (
    LOGS_FILENAME,
    META_FILENAME,
    PHASE_CRITIQUE,
    PHASE_PLAN,
    PHASE_PLAN_WRAP_UP,
    RESULTS_ROOT,
    SEED_AUDIT_FILENAME,
)
from src.models import ExperimentConfig, Problem

BASELINE_SOURCE_ARM = "baseline-sequential"
SOURCE_CUTOFF_UNITS = 3


@dataclass(frozen=True)
class LateReplaySource:
    """One audited, unsolved baseline prefix and its deterministic replay."""

    history: str
    provenance: dict[str, object]


def source_root(config: ExperimentConfig) -> Path:
    """Directory containing staged baseline sources for the active model."""
    return RESULTS_ROOT / config.model_dirname / BASELINE_SOURCE_ARM


def source_output_dir(
    config: ExperimentConfig, problem_id: str, seed: int
) -> Path:
    return source_root(config) / problem_id / f"seed_{seed}"


def _load_log(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    compressed = path.read_bytes()
    text = zstandard.ZstdDecompressor().decompress(compressed).decode("utf-8")
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError("source phase log is empty or malformed")
    return compressed, records


def _completed_prefix(
    records: list[dict[str, Any]], cutoff_tokens: int
) -> list[dict[str, Any]]:
    """All complete phase exchanges ending no later than the 3x cutoff."""
    prefix: list[dict[str, Any]] = []
    previous_cumulative = -1
    for record in records:
        cumulative = record.get("cumulative_output_tokens")
        if isinstance(cumulative, bool) or not isinstance(cumulative, int):
            raise ValueError("source phase has malformed cumulative token count")
        if cumulative < previous_cumulative:
            raise ValueError("source phase token counts are not monotone")
        previous_cumulative = cumulative
        if cumulative > cutoff_tokens:
            continue
        if record.get("budget_exhausted") is not False:
            continue
        prompt = record.get("prompt")
        text = record.get("text")
        tools = record.get("tool_calls", [])
        if not isinstance(prompt, str) or not isinstance(text, str):
            raise ValueError("source phase prompt/response is malformed")
        if not isinstance(tools, list):
            raise ValueError("source phase tool ledger is malformed")
        prefix.append(record)
    return prefix


def _proof_at_cut(prefix: list[dict[str, Any]]) -> str | None:
    excluded = {PHASE_CRITIQUE, PHASE_PLAN, PHASE_PLAN_WRAP_UP}
    for record in reversed(prefix):
        text = str(record.get("text", ""))
        if record.get("label") not in excluded and text.strip():
            return text
    return None


def _canonical_history(prefix: list[dict[str, Any]]) -> str:
    """Serialize every logged conversational field, omitting operations metadata."""
    phases: list[dict[str, object]] = []
    for index, record in enumerate(prefix, start=1):
        tool_calls: list[dict[str, object]] = []
        for raw_call in record.get("tool_calls", []):
            if not isinstance(raw_call, dict):
                raise ValueError("source tool call is malformed")
            tool_calls.append(
                {
                    "name": str(raw_call.get("name", "")),
                    "input": raw_call.get("input", {}),
                    "result": str(raw_call.get("result", "")),
                    "is_error": bool(raw_call.get("is_error", False)),
                }
            )
        phases.append(
            {
                "phase": index,
                "label": str(record.get("label", "")),
                "cumulative_output_tokens": int(record["cumulative_output_tokens"]),
                "user_prompt": str(record["prompt"]),
                "assistant_response": str(record["text"]),
                "tool_interactions": tool_calls,
            }
        )
    return json.dumps(phases, ensure_ascii=False, separators=(",", ":"))


def load_late_replay_source(
    config: ExperimentConfig, problem: Problem, seed: int
) -> tuple[LateReplaySource | None, str]:
    """Load an exact eligible source; return a mechanical exclusion reason."""
    output_dir = source_output_dir(config, problem.problem_id, seed)
    required = [META_FILENAME, LOGS_FILENAME, SEED_AUDIT_FILENAME]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        return None, f"missing baseline source files: {', '.join(missing)}"

    meta = json.loads((output_dir / META_FILENAME).read_text(encoding="utf-8"))
    expected = {
        "problem_id": problem.problem_id,
        "arm": BASELINE_SOURCE_ARM,
        "mode": "sequential",
        "hint": "none",
        "model": config.model,
        "seed": seed,
        "budget_output_tokens": 8 * config.unit_output_tokens,
    }
    mismatches = {
        key: {"expected": value, "actual": meta.get(key)}
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatches:
        raise ValueError(f"baseline source metadata mismatch: {mismatches}")

    compressed_log, records = _load_log(output_dir / LOGS_FILENAME)
    cutoff_tokens = SOURCE_CUTOFF_UNITS * config.unit_output_tokens
    prefix = _completed_prefix(records, cutoff_tokens)
    proof_text = _proof_at_cut(prefix)
    if proof_text is None:
        return None, "no complete baseline write-up was logged within 3x"

    audit = json.loads(
        (output_dir / SEED_AUDIT_FILENAME).read_text(encoding="utf-8")
    )
    audit_expected = {
        "problem_id": problem.problem_id,
        "arm": BASELINE_SOURCE_ARM,
        "seed": seed,
        "solver_model": config.model,
    }
    audit_mismatches = {
        key: {"expected": value, "actual": audit.get(key)}
        for key, value in audit_expected.items()
        if audit.get(key) != value
    }
    if audit_mismatches:
        raise ValueError(f"baseline source audit mismatch: {audit_mismatches}")
    cuts = audit.get("budget_cuts")
    cut = cuts.get("3x") if isinstance(cuts, dict) else None
    if not isinstance(cut, dict) or not isinstance(cut.get("audit_score"), int):
        return None, "baseline source has no audited 3x checkpoint"
    score = int(cut["audit_score"])
    proof_artifact = proof_text.strip() + "\n"
    proof_digest = hashlib.sha256(proof_artifact.encode("utf-8")).hexdigest()
    if cut.get("solution_sha256") != proof_digest:
        raise ValueError("baseline 3x audit is not bound to the replayed proof")
    if score >= 5:
        return None, f"baseline source already solved by 3x (score {score})"

    history = _canonical_history(prefix)
    source_path = output_dir.relative_to(RESULTS_ROOT).as_posix()
    provenance: dict[str, object] = {
        "source_arm": BASELINE_SOURCE_ARM,
        "source_result_path": source_path,
        "source_cutoff_units": SOURCE_CUTOFF_UNITS,
        "source_cutoff_output_tokens": cutoff_tokens,
        "source_completed_phase_count": len(prefix),
        "source_last_completed_phase_tokens": int(
            prefix[-1]["cumulative_output_tokens"]
        ),
        "source_self_terminated_before_cutoff": (
            meta.get("termination_reason") == "self_converged"
            and max(int(record["cumulative_output_tokens"]) for record in records)
            < cutoff_tokens
        ),
        "source_log_sha256": hashlib.sha256(compressed_log).hexdigest(),
        "source_replay_sha256": hashlib.sha256(history.encode("utf-8")).hexdigest(),
        "source_3x_solution_sha256": proof_digest,
        "source_3x_audit_score": score,
        "source_3x_audit_model": cut.get("audit_model", audit.get("audit_model")),
        "replay_protocol": "completed_phase_log_through_3x_v1",
    }
    return LateReplaySource(history=history, provenance=provenance), "eligible"


def remove_staged_sources(config: ExperimentConfig) -> None:
    """Erase staged source artifacts before any tool-enabled solver starts."""
    root = source_root(config)
    if root.exists():
        shutil.rmtree(root)
