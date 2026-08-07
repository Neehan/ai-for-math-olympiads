"""Disk I/O: load problems, lay out per-seed result dirs, write attempt outputs.

Per-seed output layout (nothing mixed):
    results/<model>/<arm>/<problem_id>/seed_<k>/
        logs.jsonl.zst   one JSON line per phase: prompt, text, tool calls, usage
        solution.md      the graded final write-up (last proof phase)
        scratch/         copy of the proof executor's scratch dir
        plan_scratch/    IdeaSearch only: copy of the isolated planner scratch
        meta.json        attempt metadata; written LAST = completion marker
"""

import json
import os
import shutil
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import zstandard

from src.constants import (
    ARM_AUDIT_FILENAME,
    AUDIT_SCRATCH_SUBDIR,
    FETCH_TIMEOUT_SECONDS,
    HINTS_FILE_ENV,
    HINTS_URL,
    LOGS_FILENAME,
    META_FILENAME,
    MODE_IDEASEARCH,
    MODE_SEQUENTIAL,
    OUTLINES_FILE_ENV,
    OUTLINES_URL,
    PHASE_CRITIQUE,
    PHASE_PLAN,
    PHASE_PLAN_WRAP_UP,
    PLAN_SCRATCH_SUBDIR,
    PROBLEMS_FILE_ENV,
    PROBLEMS_URL,
    RESULTS_ROOT,
    SCRATCH_SUBDIR,
    SESSION_STATE_SUBDIR,
    SEED_AUDIT_FILENAME,
    SOLUTION_CUT_FILENAME_FORMAT,
    SOLUTION_FILENAME,
    ZSTD_LEVEL,
)
from src.models import ArmConfig, ExperimentConfig, PhaseResult, Problem


def _fetch_jsonl(env_name: str, url: str) -> list[dict[str, Any]]:
    """Load one jsonl data source into memory, leaving no trace on disk.

    If the env var points at a prefetched file (Docker: downloaded by the
    entrypoint before the firewall closed), it is read and DELETED immediately,
    so the agent can never see it. Otherwise the URL is fetched directly with
    stdlib urllib — no hf_hub machinery, no disk cache.
    """
    path_value = os.environ.get(env_name)
    if path_value is not None:
        path = Path(path_value)
        text = path.read_text(encoding="utf-8")
        path.unlink()
    else:
        with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_SECONDS) as response:
            text = response.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _outline_text(steps: list[dict[str, Any]]) -> str:
    """Render an outline's steps as a numbered list (the h3 hint text)."""
    return "\n".join(f"{i}. {step['step']}" for i, step in enumerate(steps, start=1))


def load_problems() -> list[Problem]:
    """Fetch problems + hints + outlines and join them by problem_id.

    Only problem_id, statement, and domain are kept from the problems file —
    contest-identifying metadata is dropped at the door. Hint ladder:
    h1 = placebo (the hints file's 'placebo' field; None until authored, so
    placebo arms fail fast), h2 = the frozen one-sentence strategy hint from
    the hints file's scalar 'hint' field, h3 = strategy outline (numbered
    steps; used by the outline arms).
    """
    hints_by_id = {
        r["problem_id"]: r for r in _fetch_jsonl(HINTS_FILE_ENV, HINTS_URL)
    }
    steps_by_id = {
        r["problem_id"]: r["steps"]
        for r in _fetch_jsonl(OUTLINES_FILE_ENV, OUTLINES_URL)
    }
    problems: list[Problem] = []
    for record in _fetch_jsonl(PROBLEMS_FILE_ENV, PROBLEMS_URL):
        problem_id = record["problem_id"]
        hints = hints_by_id.get(problem_id, {})
        placebo = hints.get("placebo")
        hint = hints.get("hint")
        if hint is not None and not isinstance(hint, str):
            raise TypeError(
                f"Hint record for {problem_id!r} has non-string 'hint' field"
            )
        steps = steps_by_id.get(problem_id)
        problems.append(
            Problem(
                problem_id=problem_id,
                statement=record["statement"],
                domain=record["domain"],
                hint_h1=str(placebo) if placebo else None,
                hint_h2=hint if hint else None,
                hint_h3=_outline_text(steps) if steps else None,
            )
        )
    return problems


def seed_output_dir(
    config: ExperimentConfig, arm: ArmConfig, problem_id: str, seed: int
) -> Path:
    """Result directory for one (model, arm, problem, seed) attempt."""
    return RESULTS_ROOT / config.model_dirname / arm.name / problem_id / f"seed_{seed}"


def seed_done(output_dir: Path) -> bool:
    """True if the attempt completed (meta.json is written last)."""
    return (output_dir / META_FILENAME).exists()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically: temp file in the same dir, then os.replace."""
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _phase_record(phase: PhaseResult) -> dict[str, object]:
    """One phase as a JSON-serializable log record (full, untruncated)."""
    return {
        "label": phase.label,
        "prompt": phase.prompt,
        "text": phase.text,
        "output_tokens": phase.output_tokens,
        "cumulative_output_tokens": phase.cumulative_output_tokens,
        "num_turns": phase.num_turns,
        "duration_ms": phase.duration_ms,
        "total_cost_usd": phase.total_cost_usd,
        "is_error": phase.is_error,
        "stop_reason": phase.stop_reason,
        "budget_exhausted": phase.budget_exhausted,
        "session_reconnect_count": len(phase.reconnects),
        "session_reconnects": [asdict(event) for event in phase.reconnects],
        "tool_calls": [
            {
                "name": c.name,
                "input": c.tool_input,
                "result": c.result,
                "is_error": c.is_error,
            }
            for c in phase.tool_calls
        ],
        "process_resume_count": phase.process_resume_count,
        "discarded_output_text": phase.discarded_output_text,
        "discarded_tool_calls": [
            {
                "name": c.name,
                "input": c.tool_input,
                "result": c.result,
                "is_error": c.is_error,
            }
            for c in phase.discarded_tool_calls
        ],
    }


def final_solution_text(phases: list[PhaseResult]) -> str:
    """The graded write-up: last COMPLETE proof-producing phase's text.

    Same convention as the budget-cut snapshots — an interrupted phase's text
    is partial commentary, not a final message, and grading it would let the
    full-budget point score below a lower cut (artifactual non-monotonicity).
    Planner and critique phases are never gradeable. Falls back to the last
    proof-producing phase only when no such phase completed cleanly.
    """
    excluded = {PHASE_CRITIQUE, PHASE_PLAN, PHASE_PLAN_WRAP_UP}
    for phase in reversed(phases):
        if (
            phase.label not in excluded
            and not phase.budget_exhausted
            and phase.text.strip()
        ):
            return phase.text
    for phase in reversed(phases):
        if phase.label not in excluded and phase.text.strip():
            return phase.text
    raise ValueError("Attempt has no nonempty solve/revise phase to grade")


def budget_cut_multipliers(budget_units: int) -> list[int]:
    """Powers of two strictly below the full budget (e.g. 8 -> [1, 2, 4]).

    These are the saturation-curve cuts of a sequential trajectory; the full
    budget itself is graded from solution.md.
    """
    cuts: list[int] = []
    multiplier = 1
    while multiplier < budget_units:
        cuts.append(multiplier)
        multiplier *= 2
    return cuts


def cut_solution_path(output_dir: Path, multiplier: int) -> Path:
    """Path of the snapshot graded as 'the solution at <multiplier>x budget'."""
    return output_dir / SOLUTION_CUT_FILENAME_FORMAT.format(multiplier=multiplier)


def _phase_at_cut(
    phases: list[PhaseResult], threshold_tokens: int
) -> tuple[int, PhaseResult] | None:
    """Last COMPLETE non-critique phase within a cumulative token threshold.

    This is exactly what a run hard-stopped at the threshold would have been
    graded on: its last fully-emitted write-up. Interrupted (budget_exhausted)
    phases are never a cut snapshot; None means the trajectory produced no
    complete write-up within this budget.
    """
    found: tuple[int, PhaseResult] | None = None
    for index, phase in enumerate(phases):
        if (
            phase.label == PHASE_CRITIQUE
            or phase.budget_exhausted
            or not phase.text.strip()
        ):
            continue
        if phase.cumulative_output_tokens <= threshold_tokens:
            found = (index, phase)
    return found


def write_seed_outputs(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    budget_tokens: int,
    phases: list[PhaseResult],
    scratch_path: Path,
    plan_scratch_path: Path | None = None,
    termination_reason: str | None = None,
    provider_session_ids: dict[str, str] | None = None,
) -> Path:
    """Write logs.jsonl.zst, solution.md, scratch/ copy, then meta.json (marker).

    meta.json is written last so an interrupted write never masquerades as a
    completed attempt on resume. Returns the seed output dir.
    """
    output_dir = seed_output_dir(config, arm, problem.problem_id, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # default=str so a non-serializable tool_input value can never crash the
    # audit-log write and lose the whole attempt's record.
    log_lines = "\n".join(
        json.dumps(_phase_record(p), ensure_ascii=False, default=str) for p in phases
    )
    compressed = zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(
        (log_lines + "\n").encode("utf-8")
    )
    _atomic_write_bytes(output_dir / LOGS_FILENAME, compressed)

    try:
        solution_text = final_solution_text(phases).strip()
    except ValueError:
        # Producing no gradeable response within the budget is an experimental
        # failure, not an infrastructure exception.  Persist an empty artifact
        # so the audit stage can assign the pre-registered score 0 without
        # giving the solver another stochastic attempt.
        solution_text = ""
    _atomic_write_bytes(
        output_dir / SOLUTION_FILENAME,
        ((solution_text + "\n") if solution_text else "").encode("utf-8"),
    )

    # Sequential arms: snapshot the solution at each lower budget cut so the
    # saturation curve's 2x/4x points can be audited as standalone proofs.
    budget_cuts: dict[str, object] = {}
    if arm.mode == MODE_SEQUENTIAL:
        for multiplier in budget_cut_multipliers(arm.budget_units):
            found = _phase_at_cut(phases, multiplier * config.unit_output_tokens)
            key = f"{multiplier}x"
            if found is None:
                budget_cuts[key] = None
                # A stale snapshot from an earlier aborted write must not
                # survive to be graded as this attempt's cut.
                cut_solution_path(output_dir, multiplier).unlink(missing_ok=True)
                continue
            index, phase = found
            _atomic_write_bytes(
                cut_solution_path(output_dir, multiplier),
                (phase.text.strip() + "\n").encode("utf-8"),
            )
            budget_cuts[key] = {
                "phase_index": index,
                "phase_label": phase.label,
                "cumulative_output_tokens": phase.cumulative_output_tokens,
            }

    scratch_copy = output_dir / SCRATCH_SUBDIR
    if scratch_copy.exists():
        shutil.rmtree(scratch_copy)
    shutil.copytree(
        scratch_path,
        scratch_copy,
        ignore=shutil.ignore_patterns(SESSION_STATE_SUBDIR),
    )

    plan_scratch_copy = output_dir / PLAN_SCRATCH_SUBDIR
    if plan_scratch_copy.exists():
        shutil.rmtree(plan_scratch_copy)
    if plan_scratch_path is not None:
        shutil.copytree(
            plan_scratch_path,
            plan_scratch_copy,
            ignore=shutil.ignore_patterns(SESSION_STATE_SUBDIR),
        )

    process_resume_count = sum(p.process_resume_count for p in phases)
    meta = {
        "problem_id": problem.problem_id,
        "arm": arm.name,
        "mode": arm.mode,
        "hint": arm.hint,
        "model": config.model,
        "effort": config.effort,
        "seed": seed,
        "scratch_dir_name": scratch_path.name,
        "budget_output_tokens": budget_tokens,
        "output_tokens_spent": sum(p.output_tokens for p in phases),
        "budget_exhausted": any(p.budget_exhausted for p in phases),
        "gradeable_solution_emitted": bool(solution_text),
        "num_phases": len(phases),
        "phase_labels": [p.label for p in phases],
        "total_cost_usd": sum(p.total_cost_usd for p in phases),
        "session_reconnect_count": sum(len(p.reconnects) for p in phases),
        "session_reconnects": [
            asdict(event) for phase in phases for event in phase.reconnects
        ],
        "budget_cuts": budget_cuts,
        "process_resume_count": process_resume_count,
        "token_accounting_status": (
            "provider_reported_complete"
            if process_resume_count == 0
            else "process_recovered_unreported_suffix_possible"
        ),
    }
    if provider_session_ids:
        meta["provider_session_ids"] = dict(sorted(provider_session_ids.items()))
    if plan_scratch_path is not None:
        meta["plan_scratch_dir_name"] = plan_scratch_path.name
    if termination_reason is not None:
        meta["termination_reason"] = termination_reason
    if arm.mode == MODE_IDEASEARCH:
        plan_labels = {PHASE_PLAN, PHASE_PLAN_WRAP_UP}
        meta.update(
            {
                "ideasearch_plan_budget_output_tokens": config.ideasearch_plan_tokens,
                "ideasearch_proof_budget_output_tokens": (
                    config.unit_output_tokens - config.ideasearch_plan_tokens
                ),
                "ideasearch_plan_output_tokens_spent": sum(
                    p.output_tokens for p in phases if p.label in plan_labels
                ),
                "ideasearch_proof_output_tokens_spent": sum(
                    p.output_tokens for p in phases if p.label not in plan_labels
                ),
            }
        )
    _atomic_write_bytes(
        output_dir / META_FILENAME,
        (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return output_dir


def seed_solution_text(output_dir: Path) -> str:
    """Read a completed attempt's graded write-up (solution.md)."""
    return (output_dir / SOLUTION_FILENAME).read_text(encoding="utf-8")


def seed_audited(output_dir: Path) -> bool:
    """True if this attempt already has a judge verdict (resumable audits)."""
    return (output_dir / SEED_AUDIT_FILENAME).exists()


def write_seed_audit(output_dir: Path, record: dict[str, object]) -> None:
    """Write one attempt's judge verdict (audit.json) atomically."""
    _atomic_write_bytes(
        output_dir / SEED_AUDIT_FILENAME,
        (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def archive_audit_scratches(
    output_dir: Path, scratch_paths: dict[str, Path]
) -> None:
    """Archive each isolated judge call's visible scratch beside the attempt.

    Keeps the audit auditable: any computation the judge ran while grading is
    preserved under audit_scratch/. The live checkpoint workspace is retained
    until audit.json is durable, so a crash during copying can retry without
    destroying either the transcript or an earlier complete archive.
    """
    destination_root = output_dir / AUDIT_SCRATCH_SUBDIR
    if destination_root.exists():
        shutil.rmtree(destination_root)
    visible = {
        role: scratch
        for role, scratch in scratch_paths.items()
        if any(entry.name != SESSION_STATE_SUBDIR for entry in scratch.iterdir())
    }
    if not visible:
        return
    destination_root.mkdir()
    for role, scratch in sorted(visible.items()):
        shutil.copytree(
            scratch,
            destination_root / role,
            ignore=shutil.ignore_patterns(SESSION_STATE_SUBDIR),
        )


def compile_arm_audit(config: ExperimentConfig, arm: ArmConfig) -> tuple[Path, int]:
    """Compile EVERY seed verdict on disk into results/<model>/<arm>/audit.jsonl.

    Scans the arm's whole results tree rather than any CLI problem filter, so
    a re-audit of one problem can never truncate the compiled file down to
    that subset. One JSON line per audited (problem, seed), sorted. Returns
    the file path and the number of records written.
    """
    arm_root = RESULTS_ROOT / config.model_dirname / arm.name
    records: list[dict[str, object]] = []
    for audit_file in sorted(arm_root.glob(f"*/seed_*/{SEED_AUDIT_FILENAME}")):
        records.append(json.loads(audit_file.read_text(encoding="utf-8")))
    path = arm_root / ARM_AUDIT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    _atomic_write_bytes(path, (lines + "\n").encode("utf-8"))
    return path, len(records)
