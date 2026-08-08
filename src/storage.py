"""Disk I/O: load problems, lay out per-seed result dirs, write attempt outputs.

Per-seed output layout (nothing mixed):
    results/<model>/<arm>/<problem_id>/seed_<k>/
        logs.jsonl.zst   one JSON line per phase: prompt, text, tool calls, usage
        solution.md      the graded final write-up (last proof phase)
        scratch/         copy of the proof executor's scratch dir
        plan_scratch/    Uniform Strategy planner scratch
        strategies.json  parsed strategy set and branch allocation
        branch_<k>/      one fresh proof executor's normal attempt artifacts
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
    MODE_SEQUENTIAL,
    MODE_UNIFORM_STRATEGY,
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
    UNIFORM_BRANCH_DIR_FORMAT,
    UNIFORM_STRATEGIES_FILENAME,
    ZSTD_LEVEL,
)
from src.models import ArmConfig, ExperimentConfig, PhaseResult, Problem
from src.solver import provider_transport_policy, session_recovery_policy

_PROVIDER_USAGE_TOKEN_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


def _provider_usage_totals(phases: list[PhaseResult]) -> dict[str, int]:
    """Sum standard SDK usage counters while retaining raw usage in logs."""
    totals: dict[str, int] = {}
    for phase in phases:
        for field in _PROVIDER_USAGE_TOKEN_FIELDS:
            value = phase.provider_usage.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            totals[field] = totals.get(field, 0) + int(value)
    return totals


def _merge_provider_usage_totals(destination: dict[str, int], source: object) -> None:
    """Add a child attempt's standard provider counters into a bank total."""
    if not isinstance(source, dict):
        return
    for field in _PROVIDER_USAGE_TOKEN_FIELDS:
        value = source.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        destination[field] = destination.get(field, 0) + int(value)


def _token_accounting_status(
    phases: list[PhaseResult], process_resume_count: int
) -> str:
    """Conservatively disclose paths that can lose an unreported token suffix."""
    transport_recovered = any(
        event.reason == "transport" for phase in phases for event in phase.reconnects
    )
    if process_resume_count and transport_recovered:
        return "process_and_transport_recovered_unreported_suffix_possible"
    if process_resume_count:
        return "process_recovered_unreported_suffix_possible"
    if transport_recovered:
        return "transport_recovered_unreported_suffix_possible"
    return "provider_reported_complete"


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
    hints_by_id = {r["problem_id"]: r for r in _fetch_jsonl(HINTS_FILE_ENV, HINTS_URL)}
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


def uniform_branch_output_dir(bank_dir: Path, branch: int) -> Path:
    """Result directory for one executor inside a Uniform Strategy bank."""
    if branch < 1:
        raise ValueError("Uniform Strategy branch numbers start at 1")
    return bank_dir / UNIFORM_BRANCH_DIR_FORMAT.format(branch=branch)


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
        "provider_usage": phase.provider_usage,
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


def final_solution_text(phases: list[PhaseResult], budget_tokens: int) -> str:
    """Last complete proof-producing phase ending within the hard tier.

    Interrupted or over-budget phases remain in the logs but are ineligible:
    no response produced after the tier may affect the reported success.
    Planner and critique phases are never gradeable.
    """
    excluded = {PHASE_CRITIQUE, PHASE_PLAN, PHASE_PLAN_WRAP_UP}
    for phase in reversed(phases):
        if (
            phase.label not in excluded
            and not phase.budget_exhausted
            and phase.cumulative_output_tokens <= budget_tokens
            and phase.text.strip()
        ):
            return phase.text
    raise ValueError("Attempt has no complete within-budget solve/revise phase")


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
    output_dir_override: Path | None = None,
    meta_extra: dict[str, object] | None = None,
) -> Path:
    """Write logs.jsonl.zst, solution.md, scratch/ copy, then meta.json (marker).

    meta.json is written last so an interrupted write never masquerades as a
    completed attempt on resume. Returns the seed output dir.
    """
    output_dir = output_dir_override or seed_output_dir(
        config, arm, problem.problem_id, seed
    )
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
        solution_text = final_solution_text(phases, budget_tokens).strip()
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
    output_tokens_spent = sum(p.output_tokens for p in phases)
    meta = {
        "problem_id": problem.problem_id,
        "arm": arm.name,
        "mode": arm.mode,
        "hint": arm.hint,
        "model": config.model,
        "provider_transport_policy": provider_transport_policy(config.model),
        "session_recovery_policy": session_recovery_policy(),
        "effort": config.effort,
        "seed": seed,
        "scratch_dir_name": scratch_path.name,
        "budget_output_tokens": budget_tokens,
        "output_tokens_spent": output_tokens_spent,
        "output_tokens_over_budget": max(0, output_tokens_spent - budget_tokens),
        "within_budget_artifact_emitted": bool(solution_text),
        "budget_exhausted": any(p.budget_exhausted for p in phases),
        "gradeable_solution_emitted": bool(solution_text),
        "num_phases": len(phases),
        "phase_labels": [p.label for p in phases],
        "total_cost_usd": sum(p.total_cost_usd for p in phases),
        "provider_usage_totals": _provider_usage_totals(phases),
        "session_reconnect_count": sum(len(p.reconnects) for p in phases),
        "session_reconnects": [
            asdict(event) for phase in phases for event in phase.reconnects
        ],
        "budget_cuts": budget_cuts,
        "process_resume_count": process_resume_count,
        "token_accounting_status": _token_accounting_status(
            phases, process_resume_count
        ),
    }
    if provider_session_ids:
        meta["provider_session_ids"] = dict(sorted(provider_session_ids.items()))
    if plan_scratch_path is not None:
        meta["plan_scratch_dir_name"] = plan_scratch_path.name
    if termination_reason is not None:
        meta["termination_reason"] = termination_reason
    if meta_extra:
        meta.update(meta_extra)
    _atomic_write_bytes(
        output_dir / META_FILENAME,
        (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return output_dir


def write_uniform_strategy_plan_artifacts(
    bank_dir: Path,
    phases: list[PhaseResult],
    strategies: list[str],
    assignments: list[int],
    plan_scratch_path: Path,
) -> None:
    """Persist the shared planner output after every executor has finished.

    The bank-level meta.json remains reserved as the completion marker and is
    written only after every branch has completed.
    """
    bank_dir.mkdir(parents=True, exist_ok=True)
    log_lines = "\n".join(
        json.dumps(_phase_record(p), ensure_ascii=False, default=str) for p in phases
    )
    compressed = zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(
        (log_lines + "\n").encode("utf-8")
    )
    _atomic_write_bytes(bank_dir / LOGS_FILENAME, compressed)
    _atomic_write_bytes(
        bank_dir / UNIFORM_STRATEGIES_FILENAME,
        (
            json.dumps(
                {"strategies": strategies, "branch_strategy_indices": assignments},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    )
    destination = bank_dir / PLAN_SCRATCH_SUBDIR
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        plan_scratch_path,
        destination,
        ignore=shutil.ignore_patterns(SESSION_STATE_SUBDIR),
    )


def write_uniform_strategy_bank_meta(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    bank_dir: Path,
    plan_phases: list[PhaseResult],
    strategy_count: int,
    assignments: list[int],
    executor_budget: int,
    provider_session_ids: dict[str, str],
) -> None:
    """Write the bank completion marker after all executor metas are durable."""
    if len(assignments) != config.uniform_strategy_branches:
        raise ValueError("Uniform Strategy bank must assign every executor branch")
    if strategy_count < 1 or any(
        index < 1 or index > strategy_count for index in assignments
    ):
        raise ValueError("Uniform Strategy bank contains an invalid strategy index")
    branch_metas = []
    for branch in range(1, config.uniform_strategy_branches + 1):
        path = uniform_branch_output_dir(bank_dir, branch) / META_FILENAME
        branch_metas.append(json.loads(path.read_text(encoding="utf-8")))
    plan_spent = sum(phase.output_tokens for phase in plan_phases)
    branch_spent = sum(int(meta["output_tokens_spent"]) for meta in branch_metas)
    total_budget = config.budget_tokens(arm)
    allocated = (
        config.uniform_strategy_plan_tokens
        + config.uniform_strategy_branches * executor_budget
    )
    if allocated != total_budget:
        raise AssertionError(
            f"Uniform Strategy allocation {allocated} != bank budget {total_budget}"
        )
    process_resume_count = sum(p.process_resume_count for p in plan_phases) + sum(
        int(meta.get("process_resume_count", 0)) for meta in branch_metas
    )
    plan_reconnects = [
        event for phase in plan_phases for event in phase.reconnects
    ]
    branch_reconnects = [
        event
        for meta in branch_metas
        for event in meta.get("session_reconnects", [])
        if isinstance(event, dict)
    ]
    output_tokens_spent = plan_spent + branch_spent
    provider_usage_totals = _provider_usage_totals(plan_phases)
    for branch_meta in branch_metas:
        _merge_provider_usage_totals(
            provider_usage_totals, branch_meta.get("provider_usage_totals")
        )
    meta = {
        "problem_id": problem.problem_id,
        "arm": arm.name,
        "mode": MODE_UNIFORM_STRATEGY,
        "hint": arm.hint,
        "model": config.model,
        "provider_transport_policy": provider_transport_policy(config.model),
        "session_recovery_policy": session_recovery_policy(),
        "effort": config.effort,
        "seed": seed,
        "budget_output_tokens": total_budget,
        "output_tokens_spent": output_tokens_spent,
        "output_tokens_over_budget": max(0, output_tokens_spent - total_budget),
        "provider_usage_totals": provider_usage_totals,
        "uniform_strategy_plan_budget_output_tokens": (
            config.uniform_strategy_plan_tokens
        ),
        "uniform_strategy_plan_output_tokens_spent": plan_spent,
        "uniform_strategy_executor_count": config.uniform_strategy_branches,
        "uniform_strategy_executor_budget_output_tokens_each": executor_budget,
        "uniform_strategy_executor_output_tokens_spent": branch_spent,
        "strategy_count": strategy_count,
        "branch_strategy_indices": assignments,
        "provider_session_ids": dict(sorted(provider_session_ids.items())),
        "branch_provider_session_ids": {
            str(i): meta.get("provider_session_ids", {})
            for i, meta in enumerate(branch_metas, start=1)
        },
        "process_resume_count": process_resume_count,
        "session_reconnect_count": len(plan_reconnects) + len(branch_reconnects),
        "session_reconnects": [
            *[asdict(event) for event in plan_reconnects],
            *branch_reconnects,
        ],
        "token_accounting_status": (
            "process_and_transport_recovered_unreported_suffix_possible"
            if process_resume_count
            and (
                any(event.reason == "transport" for event in plan_reconnects)
                or any(event.get("reason") == "transport" for event in branch_reconnects)
            )
            else "process_recovered_unreported_suffix_possible"
            if process_resume_count
            else "transport_recovered_unreported_suffix_possible"
            if (
                any(event.reason == "transport" for event in plan_reconnects)
                or any(event.get("reason") == "transport" for event in branch_reconnects)
            )
            else "provider_reported_complete"
        ),
        "gradeable_solution_emitted": any(
            bool(meta.get("gradeable_solution_emitted")) for meta in branch_metas
        ),
    }
    # run.sh distinguishes newly completed staged attempts from pre-seeded
    # resume markers by requiring solution.md beside the last-written meta.json.
    # The bank is graded from branch_<k>/solution.md; this top-level file is an
    # explicit manifest, never a proof candidate.
    _atomic_write_bytes(
        bank_dir / SOLUTION_FILENAME,
        (
            "Uniform Strategy Search bank. Grade the executor candidates in "
            "branch_1 through branch_"
            f"{config.uniform_strategy_branches}; this file is not a candidate proof.\n"
        ).encode("utf-8"),
    )
    _atomic_write_bytes(
        bank_dir / META_FILENAME,
        (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def write_uniform_strategy_planner_failure(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    bank_dir: Path,
    plan_phases: list[PhaseResult],
    provider_session_ids: dict[str, str],
    reason: str,
) -> None:
    """Complete a bank as a scoreable failure when no eligible plan exists."""
    plan_spent = sum(phase.output_tokens for phase in plan_phases)
    total_budget = config.budget_tokens(arm)
    meta = {
        "problem_id": problem.problem_id,
        "arm": arm.name,
        "mode": MODE_UNIFORM_STRATEGY,
        "hint": arm.hint,
        "model": config.model,
        "provider_transport_policy": provider_transport_policy(config.model),
        "session_recovery_policy": session_recovery_policy(),
        "effort": config.effort,
        "seed": seed,
        "budget_output_tokens": total_budget,
        "output_tokens_spent": plan_spent,
        "output_tokens_over_budget": max(
            0, plan_spent - config.uniform_strategy_plan_tokens
        ),
        "provider_usage_totals": _provider_usage_totals(plan_phases),
        "uniform_strategy_plan_budget_output_tokens": (
            config.uniform_strategy_plan_tokens
        ),
        "uniform_strategy_plan_output_tokens_spent": plan_spent,
        "uniform_strategy_executor_count": 0,
        "strategy_count": 0,
        "branch_strategy_indices": [],
        "provider_session_ids": dict(sorted(provider_session_ids.items())),
        "session_reconnect_count": sum(
            len(phase.reconnects) for phase in plan_phases
        ),
        "session_reconnects": [
            asdict(event) for phase in plan_phases for event in phase.reconnects
        ],
        "process_resume_count": sum(
            phase.process_resume_count for phase in plan_phases
        ),
        "token_accounting_status": _token_accounting_status(
            plan_phases,
            sum(phase.process_resume_count for phase in plan_phases),
        ),
        "planner_failure": reason,
        "gradeable_solution_emitted": False,
        "within_budget_artifact_emitted": False,
    }
    _atomic_write_bytes(bank_dir / SOLUTION_FILENAME, b"")
    _atomic_write_bytes(
        bank_dir / META_FILENAME,
        (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


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


def archive_audit_scratches(output_dir: Path, scratch_paths: dict[str, Path]) -> None:
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
