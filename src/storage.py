"""Disk I/O: load problems, lay out per-seed result dirs, write attempt outputs.

Per-seed output layout (nothing mixed):
    results/<model>/<arm>/<problem_id>/seed_<k>/
        logs.jsonl.zst   one JSON line per phase: prompt, text, tool calls, usage
        solution.md      the graded final write-up (last non-critique phase)
        scratch/         copy of the agent's scratch dir (its created files)
        meta.json        attempt metadata; written LAST = completion marker
"""

import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import zstandard

from src.constants import (
    ARM_AUDIT_FILENAME,
    FETCH_TIMEOUT_SECONDS,
    HINTS_FILE_ENV,
    HINTS_URL,
    LOGS_FILENAME,
    META_FILENAME,
    MODE_SEQUENTIAL,
    OUTLINES_FILE_ENV,
    OUTLINES_URL,
    PHASE_CRITIQUE,
    PROBLEMS_FILE_ENV,
    PROBLEMS_URL,
    RESULTS_ROOT,
    SCRATCH_ROOT,
    SCRATCH_SUBDIR,
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
    """Render an outline's steps as a numbered list (the H2 hint text)."""
    return "\n".join(f"{i}. {step['step']}" for i, step in enumerate(steps, start=1))


def load_problems() -> list[Problem]:
    """Fetch problems + hints + outlines and join them by problem_id.

    Only problem_id and statement are kept from the problems file — contest-
    identifying metadata is dropped at the door. H1 = technique tags
    (comma-joined), H2 = solution outline (numbered steps).
    """
    tags_by_id = {
        r["problem_id"]: r["tags"] for r in _fetch_jsonl(HINTS_FILE_ENV, HINTS_URL)
    }
    steps_by_id = {
        r["problem_id"]: r["steps"]
        for r in _fetch_jsonl(OUTLINES_FILE_ENV, OUTLINES_URL)
    }
    problems: list[Problem] = []
    for record in _fetch_jsonl(PROBLEMS_FILE_ENV, PROBLEMS_URL):
        problem_id = record["problem_id"]
        tags = tags_by_id.get(problem_id)
        steps = steps_by_id.get(problem_id)
        problems.append(
            Problem(
                problem_id=problem_id,
                statement=record["statement"],
                domain=record["domain"],
                hint_h1=", ".join(tags) if tags else None,
                hint_h2=_outline_text(steps) if steps else None,
            )
        )
    return problems


def seed_output_dir(
    config: ExperimentConfig, arm: ArmConfig, problem_id: str, seed: int
) -> Path:
    """Result directory for one (model, arm, problem, seed) attempt."""
    return RESULTS_ROOT / config.model / arm.name / problem_id / f"seed_{seed}"


def seed_done(output_dir: Path) -> bool:
    """True if the attempt completed (meta.json is written last)."""
    return (output_dir / META_FILENAME).exists()


def fresh_scratch_dir(arm: ArmConfig, problem_id: str, seed: int) -> Path:
    """Create an EMPTY per-attempt scratch dir for the agent's file work.

    Any pre-existing contents are stale leftovers of a killed run of this same
    attempt (completed attempts are skipped on resume and never reach here);
    clearing on entry guarantees every attempt provably starts cold.
    """
    path = SCRATCH_ROOT / arm.name / problem_id / f"seed_{seed}"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically: temp file in the same dir, then os.replace."""
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


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
        "tool_calls": [
            {
                "name": c.name,
                "input": c.tool_input,
                "result": c.result,
                "is_error": c.is_error,
            }
            for c in phase.tool_calls
        ],
    }


def final_solution_text(phases: list[PhaseResult]) -> str:
    """The graded write-up: text of the last non-critique phase."""
    for phase in reversed(phases):
        if phase.label != PHASE_CRITIQUE:
            return phase.text
    raise ValueError("Attempt has no solve/revise phase to grade")


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
        if phase.label == PHASE_CRITIQUE or phase.budget_exhausted:
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

    _atomic_write_bytes(
        output_dir / SOLUTION_FILENAME,
        (final_solution_text(phases).strip() + "\n").encode("utf-8"),
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
    shutil.copytree(scratch_path, scratch_copy)

    meta = {
        "problem_id": problem.problem_id,
        "arm": arm.name,
        "mode": arm.mode,
        "hint": arm.hint,
        "model": config.model,
        "effort": config.effort,
        "seed": seed,
        "budget_output_tokens": budget_tokens,
        "output_tokens_spent": phases[-1].cumulative_output_tokens if phases else 0,
        "budget_exhausted": any(p.budget_exhausted for p in phases),
        "num_phases": len(phases),
        "phase_labels": [p.label for p in phases],
        "total_cost_usd": sum(p.total_cost_usd for p in phases),
        "budget_cuts": budget_cuts,
    }
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


def compile_arm_audit(
    config: ExperimentConfig, arm: ArmConfig, problems: list[Problem]
) -> tuple[Path, int]:
    """Compile every seed verdict into results/<model>/<arm>/audit.jsonl.

    One JSON line per audited (problem, seed), in problem order then seed
    order. Returns the file path and the number of records written.
    """
    records: list[dict[str, object]] = []
    for problem in problems:
        for seed in arm.seeds:
            output_dir = seed_output_dir(config, arm, problem.problem_id, seed)
            if seed_audited(output_dir):
                records.append(
                    json.loads(
                        (output_dir / SEED_AUDIT_FILENAME).read_text(encoding="utf-8")
                    )
                )
    path = RESULTS_ROOT / config.model / arm.name / ARM_AUDIT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    _atomic_write_bytes(path, (lines + "\n").encode("utf-8"))
    return path, len(records)
