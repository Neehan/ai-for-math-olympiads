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
    META_FILENAME,
    RESULTS_ROOT,
    SEED_AUDIT_FILENAME,
    SESSION_STATE_SUBDIR,
)
from src.models import ExperimentConfig, PhaseResult, Problem
from src.solver import agent_settings_path

BASELINE_SOURCE_ARM = "baseline-sequential"
LATE_BASELINE_ARM = "late-baseline-sequential"
PREFIX_UNITS = 3
PASS_THRESHOLD = 5
PREFIX_SCHEMA_VERSION = 1
DELETE_STAGED_SOURCES_ENV = "HARNESS_DELETE_LATE_SOURCE_RESULTS_AFTER_LOAD"

_FORK_LOCK = threading.Lock()


@dataclass(frozen=True)
class LateProblemEligibility:
    """Frozen problem-level inclusion decision from the original 3x runs."""

    provenance: dict[str, object]


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


def _audit_path(
    config: ExperimentConfig, arm: str, problem_id: str, seed: int
) -> Path:
    return (
        RESULTS_ROOT
        / config.model_dirname
        / arm
        / problem_id
        / f"seed_{seed}"
        / SEED_AUDIT_FILENAME
    )


def _cut_score(
    path: Path,
    *,
    problem_id: str,
    arm: str,
    seed: int,
    model: str,
) -> tuple[int, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "problem_id": problem_id,
        "arm": arm,
        "seed": seed,
        "solver_model": model,
    }
    mismatches = {
        key: {"expected": value, "actual": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Late-source audit identity mismatch: {mismatches}")
    cuts = record.get("budget_cuts")
    cut = cuts.get("3x") if isinstance(cuts, dict) else None
    if not isinstance(cut, dict) or not isinstance(cut.get("audit_score"), int):
        raise ValueError(f"{path} has no audited 3x checkpoint")
    return int(cut["audit_score"]), cut


def problem_eligibility(
    config: ExperimentConfig, problem: Problem
) -> tuple[LateProblemEligibility | None, str]:
    """Include a problem iff fewer than two original seeds pass at 3x."""
    scores: dict[str, int] = {}
    audit_models: dict[str, str | None] = {}
    try:
        for seed in (1, 2, 3):
            path = _audit_path(
                config, BASELINE_SOURCE_ARM, problem.problem_id, seed
            )
            score, cut = _cut_score(
                path,
                problem_id=problem.problem_id,
                arm=BASELINE_SOURCE_ARM,
                seed=seed,
                model=config.model,
            )
            scores[str(seed)] = score
            raw_model = cut.get("audit_model")
            audit_models[str(seed)] = str(raw_model) if raw_model else None
    except (FileNotFoundError, ValueError) as error:
        return None, str(error)

    pass_count = sum(score >= PASS_THRESHOLD for score in scores.values())
    if pass_count >= 2:
        return None, f"original baseline passed {pass_count}/3 seeds by 3x"
    return (
        LateProblemEligibility(
            provenance={
                "eligibility_source_arm": BASELINE_SOURCE_ARM,
                "eligibility_checkpoint": "3x",
                "eligibility_rule": "fewer_than_2_of_3_pass_at_3x",
                "eligibility_pass_threshold": PASS_THRESHOLD,
                "eligibility_scores": scores,
                "eligibility_audit_models": audit_models,
                "eligibility_pass_count": pass_count,
            }
        ),
        "eligible",
    )


def remove_staged_result_sources(config: ExperimentConfig, arm: str) -> None:
    """Delete staged audit dependencies before any tool-enabled solver starts."""
    root = RESULTS_ROOT / config.model_dirname / arm
    if root.exists():
        shutil.rmtree(root)


def _proof_at_prefix(phases: list[PhaseResult]) -> str:
    for phase in reversed(phases):
        if phase.label != "critique" and not phase.budget_exhausted and phase.text.strip():
            return phase.text.strip() + "\n"
    raise ValueError("The completed 3x prefix has no gradeable proof")


def save_prefix_source(
    config: ExperimentConfig,
    problem: Problem,
    seed: int,
    scratch_path: Path,
    session_id: str,
    phases: list[PhaseResult],
    output_tokens_spent: int,
    eligibility: LateProblemEligibility,
) -> LatePrefixSource:
    """Atomically retain the exact native transcript and filesystem at 3x."""
    source_dir = prefix_source_dir(config, problem, seed)
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    proof = _proof_at_prefix(phases)
    identity = _source_identity(config, problem, seed)
    manifest: dict[str, object] = {
        **identity,
        **eligibility.provenance,
        "source_arm": LATE_BASELINE_ARM,
        "scratch_name": scratch_path.name,
        "source_session_id": session_id,
        "prefix_budget_output_tokens": PREFIX_UNITS * config.unit_output_tokens,
        "prefix_output_tokens_spent": output_tokens_spent,
        "prefix_phase_count": len(phases),
        "prefix_solution_sha256": hashlib.sha256(proof.encode("utf-8")).hexdigest(),
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
    """Load every retained prefix in the frozen hard-problem intervention set."""
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
