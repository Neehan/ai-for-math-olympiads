"""Matched native-session intervention after an exact unaided 3x prefix."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import fork_session

from src.checkpoint import phase_from_record, phase_record, protocol_fingerprint
from src.constants import (
    CHECKPOINT_ROOT_DEFAULT,
    CHECKPOINT_ROOT_ENV,
    SESSION_STATE_SUBDIR,
)
from src.models import ExperimentConfig, PhaseResult, Problem
from src.solver import agent_settings_path

LATE_BASELINE_ARM = "late-baseline-sequential"
PREFIX_UNITS = 3
PREFIX_SCHEMA_VERSION = 2

_FORK_LOCK = threading.Lock()


@dataclass(frozen=True)
class LatePrefixSource:
    """Retained native 3x transcript/scratch state for one treatment branch."""

    workspace: Path
    scratch_name: str
    session_id: str
    phases: list[PhaseResult]
    provenance: dict[str, object]


def _checkpoint_root() -> Path:
    return Path(os.environ.get(CHECKPOINT_ROOT_ENV, str(CHECKPOINT_ROOT_DEFAULT)))


def _source_identity(
    config: ExperimentConfig, problem: Problem, seed: int
) -> dict[str, object]:
    return {
        "schema_version": PREFIX_SCHEMA_VERSION,
        "model": config.model,
        "effort": config.effort,
        "problem_id": problem.problem_id,
        "problem_statement": problem.statement,
        "seed": seed,
        "prefix_units": PREFIX_UNITS,
        "unit_output_tokens": config.unit_output_tokens,
        "wrap_up_reserve_tokens": config.wrap_up_reserve_tokens,
        "max_turns_per_phase": config.max_turns_per_phase,
        "protocol_fingerprint": protocol_fingerprint(
            agent_settings_path(config.model)
        ),
    }


def prefix_source_dir(
    config: ExperimentConfig, problem: Problem, seed: int
) -> Path:
    canonical = json.dumps(
        _source_identity(config, problem, seed),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return _checkpoint_root() / "late-prefixes" / digest[:24]


def _proof_at_prefix(phases: list[PhaseResult]) -> str | None:
    for phase in reversed(phases):
        if (
            phase.label != "critique"
            and not phase.budget_exhausted
            and phase.text.strip()
        ):
            return phase.text.strip() + "\n"
    return None


def save_prefix_source(
    config: ExperimentConfig,
    problem: Problem,
    seed: int,
    scratch_path: Path,
    session_id: str,
    phases: list[PhaseResult],
    output_tokens_spent: int,
) -> LatePrefixSource:
    """Atomically retain the exact native transcript and filesystem at 3x."""
    source_dir = prefix_source_dir(config, problem, seed)
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    proof = _proof_at_prefix(phases)
    identity = _source_identity(config, problem, seed)
    manifest: dict[str, object] = {
        **identity,
        "source_arm": LATE_BASELINE_ARM,
        "scratch_name": scratch_path.name,
        "source_session_id": session_id,
        "prefix_budget_output_tokens": PREFIX_UNITS * config.unit_output_tokens,
        "prefix_output_tokens_spent": output_tokens_spent,
        "prefix_phase_count": len(phases),
        "prefix_has_gradeable_proof": proof is not None,
        "prefix_solution_sha256": (
            hashlib.sha256(proof.encode("utf-8")).hexdigest()
            if proof is not None
            else None
        ),
        "fork_protocol": "native_transcript_and_scratch_snapshot_v1",
    }
    temp = source_dir.with_name(f"{source_dir.name}.tmp-{uuid.uuid4().hex[:8]}")
    shutil.copytree(scratch_path, temp / "workspace")
    (temp / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (temp / "prefix_phases.json").write_text(
        json.dumps(
            [phase_record(phase) for phase in phases],
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    if source_dir.exists():
        shutil.rmtree(source_dir)
    os.replace(temp, source_dir)
    return LatePrefixSource(
        workspace=source_dir / "workspace",
        scratch_name=scratch_path.name,
        session_id=session_id,
        phases=list(phases),
        provenance=manifest,
    )


def load_prefix_source(
    config: ExperimentConfig, problem: Problem, seed: int
) -> tuple[LatePrefixSource | None, str]:
    """Load one retained prefix selected by the caller's problem list."""
    source_dir = prefix_source_dir(config, problem, seed)
    manifest_path = source_dir / "manifest.json"
    phases_path = source_dir / "prefix_phases.json"
    workspace = source_dir / "workspace"
    if (
        not manifest_path.is_file()
        or not phases_path.is_file()
        or not workspace.is_dir()
    ):
        return None, "matching retained late-baseline 3x session is absent"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "prefix_has_gradeable_proof" not in manifest:
        manifest["prefix_has_gradeable_proof"] = isinstance(
            manifest.get("prefix_solution_sha256"), str
        )
    has_gradeable_proof = manifest.get("prefix_has_gradeable_proof")
    proof_sha256 = manifest.get("prefix_solution_sha256")
    if not isinstance(has_gradeable_proof, bool) or has_gradeable_proof != isinstance(
        proof_sha256, str
    ):
        raise ValueError("Retained prefix proof metadata is malformed")
    expected = _source_identity(config, problem, seed)
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Retained prefix identity mismatch: {mismatches}")
    scratch_name = manifest.get("scratch_name")
    session_id = manifest.get("source_session_id")
    if not isinstance(scratch_name, str) or not isinstance(session_id, str):
        raise ValueError("Retained prefix manifest is malformed")
    raw_phases = json.loads(phases_path.read_text(encoding="utf-8"))
    if not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError("Retained prefix phase ledger is malformed")
    phases = [
        phase_from_record(record)
        for record in raw_phases
        if isinstance(record, dict)
    ]
    if len(phases) != len(raw_phases):
        raise ValueError("Retained prefix phase ledger is malformed")

    provenance = {
        key: value
        for key, value in manifest.items()
        if key not in {"problem_statement"}
    }
    return (
        LatePrefixSource(
            workspace=workspace,
            scratch_name=scratch_name,
            session_id=session_id,
            phases=phases,
            provenance=provenance,
        ),
        "eligible",
    )


def fork_native_session(scratch_path: Path, session_id: str) -> str:
    """Fork a retained Claude transcript while preserving its exact cwd."""
    config_dir = scratch_path / SESSION_STATE_SUBDIR
    if not config_dir.is_dir():
        raise FileNotFoundError(
            f"Retained native transcript directory is absent: {config_dir}"
        )
    # The public SDK mutation API resolves storage through this process-level
    # variable. Serialize the tiny local file operation so concurrent seeds
    # cannot observe one another's config root.
    with _FORK_LOCK:
        previous = os.environ.get("CLAUDE_CONFIG_DIR")
        os.environ["CLAUDE_CONFIG_DIR"] = str(config_dir)
        try:
            result = fork_session(session_id, directory=str(scratch_path))
        finally:
            if previous is None:
                os.environ.pop("CLAUDE_CONFIG_DIR", None)
            else:
                os.environ["CLAUDE_CONFIG_DIR"] = previous
    return result.session_id
