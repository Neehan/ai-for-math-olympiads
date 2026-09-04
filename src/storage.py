"""Disk I/O: load problems, lay out per-seed result dirs, write attempt outputs.

Per-seed output layout (nothing mixed):
    results/<model>/<arm>/<problem_id>/seed_<k>/
        logs.jsonl.zst   one JSON line per phase: prompt, text, tool calls, usage
        solution.md      the graded final write-up (last proof phase)
        scratch/         copy of the proof executor's scratch dir
        plan_scratch/    Uniform Strategy planner scratch
        strategies.json  parsed strategy set and executor-run allocation
        run_<kk>/        one fresh bank member's artifacts
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
    ARM_STATE_AUDIT_FILENAME,
    AUDIT_SCRATCH_SUBDIR,
    DATASET_ANSWER_GRADED,
    DATASET_HAS_STRATEGY_ARTIFACTS,
    DATASET_NAME,
    FETCH_TIMEOUT_SECONDS,
    HINTS_FILE_ENV,
    HINTS_URL,
    LOGS_FILENAME,
    META_FILENAME,
    MODE_PARALLEL,
    MODE_SEQUENTIAL,
    MODE_UNIFORM_STRATEGY,
    MODE_UNIFORM_STRATEGY_ONLY,
    OUTLINES_FILE_ENV,
    OUTLINES_URL,
    PARALLEL_BANK_PROTOCOL,
    PHASE_CRITIQUE,
    PHASE_PLAN,
    PHASE_PLAN_WRAP_UP,
    PLAN_SCRATCH_SUBDIR,
    PROBLEMS_FILE_ENV,
    PROBLEMS_URL,
    RESULTS_ROOT,
    SCRATCH_SUBDIR,
    SELECTION_FILE_ENV,
    SELECTION_URL,
    SESSION_STATE_SUBDIR,
    SEED_AUDIT_FILENAME,
    SEED_STATE_AUDIT_FILENAME,
    SEQUENTIAL_MIN_ROUNDS_BEFORE_CONVERGENCE,
    SEQUENTIAL_NO_GAP_STREAK_TO_STOP,
    SOLUTIONS_FILE_ENV,
    SOLUTIONS_URL,
    SOLUTION_CUT_FILENAME_FORMAT,
    SOLUTION_FILENAME,
    BANK_RUN_DIR_FORMAT,
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
    """Describe eligible-output accounting without invalidating recovery.

    The experiment budget counts output delivered into the persisted transcript
    (stream usage plus completed per-query Result usage). Provider-side work
    that produced no transcript-visible output is transport overhead, not an
    experimental token. Recovery provenance remains available separately in
    ``process_resume_count`` and ``session_reconnects``.
    """
    transport_recovered = any(
        event.reason == "transport" for phase in phases for event in phase.reconnects
    )
    if process_resume_count or transport_recovered:
        return "recovered_eligible_output_accounted"
    return "provider_reported_complete"


def _fetch_jsonl(env_name: str, url: str | None) -> list[dict[str, Any]]:
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
    elif url is None:
        raise ValueError(f"Dataset {DATASET_NAME!r} publishes no {env_name} source")
    else:
        with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_SECONDS) as response:
            text = response.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _outline_text(steps: list[dict[str, Any]]) -> str:
    """Render an outline's steps as a numbered list (the h3 hint text)."""
    return "\n".join(f"{i}. {step['step']}" for i, step in enumerate(steps, start=1))


def _domain_shifted_placebos(
    problem_records: list[dict[str, Any]], hints_by_id: dict[str, dict[str, Any]]
) -> dict[str, str]:
    """Assign the next problem's frozen hint within each sorted domain.

    This is a deterministic cyclic derangement: every hint is used exactly
    once as a placebo in its own domain, and no problem receives its own hint.
    """
    ids_by_domain: dict[str, list[str]] = {}
    for record in problem_records:
        problem_id = record.get("problem_id")
        domain = record.get("domain")
        if not isinstance(problem_id, str) or not isinstance(domain, str):
            raise TypeError("Every problem needs string problem_id and domain fields")
        ids_by_domain.setdefault(domain, []).append(problem_id)

    placebos: dict[str, str] = {}
    for domain, unsorted_ids in ids_by_domain.items():
        problem_ids = sorted(unsorted_ids)
        if len(problem_ids) != len(set(problem_ids)):
            raise ValueError(f"Duplicate problem_id in domain {domain!r}")
        if len(problem_ids) < 2:
            raise ValueError(
                f"Domain {domain!r} needs at least two problems for a placebo shift"
            )
        for index, problem_id in enumerate(problem_ids):
            source_id = problem_ids[(index + 1) % len(problem_ids)]
            source_hint = hints_by_id.get(source_id, {}).get("hint")
            if not isinstance(source_hint, str) or not source_hint.strip():
                raise ValueError(
                    f"Placebo source {source_id!r} has no frozen strategy hint"
                )
            placebos[problem_id] = source_hint
    return placebos


def _load_problems_without_hints() -> list[Problem]:
    """Load an answer-graded dataset, which publishes no strategy artifacts.

    Every hint tier is absent rather than empty, so the arms that need one are
    refused by their entrypoint instead of silently running unhinted.
    """
    return [
        Problem(
            problem_id=record["problem_id"],
            statement=record["statement"],
            domain=record["domain"],
            hint_h1=None,
            hint_h2=None,
            hint_h3=None,
        )
        for record in _fetch_jsonl(PROBLEMS_FILE_ENV, PROBLEMS_URL)
    ]


def load_problems() -> list[Problem]:
    """Fetch problems + hints + outlines and join them by problem_id.

    Only problem_id, statement, and domain are kept from the problems file —
    contest-identifying metadata is dropped at the door. Hint ladder:
    h1 = deterministic within-domain cyclic shift of the frozen h2 hints,
    h2 = the frozen one-sentence strategy hint from the hints file's scalar
    'hint' field, h3 = strategy outline (numbered steps; used by outline arms).
    A dataset that publishes no proofs has none of these tiers.
    """
    if not DATASET_HAS_STRATEGY_ARTIFACTS:
        return _load_problems_without_hints()
    hints_by_id = {r["problem_id"]: r for r in _fetch_jsonl(HINTS_FILE_ENV, HINTS_URL)}
    steps_by_id = {
        r["problem_id"]: r["steps"]
        for r in _fetch_jsonl(OUTLINES_FILE_ENV, OUTLINES_URL)
    }
    problem_records = _fetch_jsonl(PROBLEMS_FILE_ENV, PROBLEMS_URL)
    placebos_by_id = _domain_shifted_placebos(problem_records, hints_by_id)
    problems: list[Problem] = []
    for record in problem_records:
        problem_id = record["problem_id"]
        hints = hints_by_id.get(problem_id, {})
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
                hint_h1=placebos_by_id[problem_id],
                hint_h2=hint if hint else None,
                hint_h3=_outline_text(steps) if steps else None,
            )
        )
    return problems


def load_selection_candidates(
    source_model: str,
) -> dict[str, dict[str, Any]]:
    """Load one frozen proposal-seed-1 candidate set per problem."""
    selected: dict[str, dict[str, Any]] = {}
    for record in _fetch_jsonl(SELECTION_FILE_ENV, SELECTION_URL):
        if record.get("source_model") != source_model:
            continue
        problem_id = record.get("problem_id")
        proposal_seed = record.get("proposal_seed")
        oracle = record.get("oracle_strategy")
        generated = record.get("generated_strategies")
        if (
            not isinstance(problem_id, str)
            or proposal_seed != 1
            or not isinstance(oracle, str)
            or not oracle.strip()
            or not isinstance(generated, list)
            or len(generated) != 3
        ):
            raise ValueError("Malformed hard-hint selection record")
        if len(oracle.split()) > 25:
            raise ValueError(f"{problem_id}: oracle selection sketch exceeds 25 words")
        normalized: list[dict[str, object]] = []
        for index, item in enumerate(generated, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"{problem_id}: malformed generated strategy")
            strategy = item.get("strategy")
            oracle_strategy_match = item.get("oracle_strategy_match")
            candidate_id = item.get("candidate_id", f"generated_{index}")
            if not isinstance(strategy, str) or not strategy.strip():
                raise ValueError(f"{problem_id}: empty generated strategy")
            if not 18 <= len(strategy.split()) <= 25:
                raise ValueError(
                    f"{problem_id}: generated strategy {index} must contain "
                    "18--25 words"
                )
            if not isinstance(oracle_strategy_match, bool):
                raise ValueError(
                    f"{problem_id}: generated strategy {index} lacks frozen "
                    "oracle_strategy_match label"
                )
            if (
                not isinstance(candidate_id, str)
                or not candidate_id.strip()
                or candidate_id == "oracle"
            ):
                raise ValueError(
                    f"{problem_id}: generated strategy {index} has invalid candidate_id"
                )
            normalized.append(
                {
                    "candidate_id": candidate_id.strip(),
                    "strategy": strategy.strip(),
                    "oracle_strategy_match": oracle_strategy_match,
                }
            )
        candidate_ids = [str(item["candidate_id"]) for item in normalized]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"{problem_id}: duplicate generated candidate_id")
        if problem_id in selected:
            raise ValueError(
                f"Duplicate hard-hint selection record for {source_model}/{problem_id}"
            )
        selected[problem_id] = {
            "problem_id": problem_id,
            "source_model": source_model,
            "proposal_seed": 1,
            "oracle_strategy": oracle.strip(),
            "generated_strategies": normalized,
        }
    return selected


def write_auxiliary_result(
    output_dir: Path,
    artifact_name: str,
    artifact: dict[str, object],
    meta: dict[str, object],
    *,
    audit: dict[str, object] | None = None,
) -> None:
    """Durably write a non-proof strategy artifact and its completion marker."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(
        output_dir / artifact_name,
        (json.dumps(artifact, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    if audit is not None:
        write_seed_audit(output_dir, audit)
    _atomic_write_bytes(
        output_dir / SOLUTION_FILENAME,
        b"Auxiliary strategy experiment; no proof artifact.\n",
    )
    _atomic_write_bytes(
        output_dir / META_FILENAME,
        (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def _outline_reference(record: dict[str, Any]) -> str:
    """Select index 0 after verifying its explicit outline-matching marker."""
    problem_id = record.get("problem_id")
    references = record.get("reference_solutions")
    if not isinstance(problem_id, str) or not isinstance(references, list):
        raise ValueError("Malformed hard-solutions record")
    aligned = [
        reference
        for reference in references
        if isinstance(reference, dict) and reference.get("route_id") == "hard_hint"
    ]
    if len(aligned) != 1:
        raise ValueError(
            f"{problem_id}: expected exactly one reference with route_id='hard_hint'; "
            f"found {len(aligned)}"
        )
    first = references[0] if references else None
    if not isinstance(first, dict) or first.get("route_id") != "hard_hint":
        raise ValueError(
            f"{problem_id}: the hard_hint reference must be reference_solutions[0]"
        )
    solution = first.get("solution")
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError(f"{problem_id}: outline reference solution is empty")
    return solution.strip()


def _answer_reference(record: dict[str, Any]) -> str:
    """Read the published answer an answer-graded dataset grades against."""
    problem_id = record.get("problem_id")
    answer = record.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{problem_id}: no published answer")
    return answer.strip()


def _correctness_reference(record: dict[str, Any]) -> str:
    """Return whichever ground truth this dataset's judge grades against."""
    if DATASET_ANSWER_GRADED:
        return _answer_reference(record)
    return _outline_reference(record)


def load_audit_references() -> dict[str, tuple[str, str]]:
    """Load problem statements and the fixed index-0 correctness reference."""
    references: dict[str, tuple[str, str]] = {}
    for record in _fetch_jsonl(SOLUTIONS_FILE_ENV, SOLUTIONS_URL):
        problem_id = record.get("problem_id")
        statement = record.get("statement")
        if not isinstance(problem_id, str) or not isinstance(statement, str):
            raise ValueError("Malformed hard-solutions identity")
        if problem_id in references:
            raise ValueError(f"Duplicate hard-solutions problem_id: {problem_id}")
        references[problem_id] = (statement.strip(), _correctness_reference(record))
    return references


def load_state_audit_references() -> dict[str, tuple[str, str]]:
    """Load problem statements and explicitly outline-matching full solutions.

    Generation never receives this source. Returning the stored statement lets
    the caller verify the problem-id join before constructing any prompt.
    """
    references: dict[str, tuple[str, str]] = {}
    for record in _fetch_jsonl(SOLUTIONS_FILE_ENV, SOLUTIONS_URL):
        problem_id = record.get("problem_id")
        statement = record.get("statement")
        if not isinstance(problem_id, str) or not isinstance(statement, str):
            raise ValueError("Malformed hard-solutions identity")
        if problem_id in references:
            raise ValueError(f"Duplicate hard-solutions problem_id: {problem_id}")
        references[problem_id] = (statement.strip(), _outline_reference(record))
    return references


def seed_output_dir(
    config: ExperimentConfig, arm: ArmConfig, problem_id: str, seed: int
) -> Path:
    """Result directory for one (model, arm, problem, seed) attempt."""
    return RESULTS_ROOT / config.model_dirname / arm.name / problem_id / f"seed_{seed}"


def bank_run_output_dir(bank_dir: Path, run: int) -> Path:
    """Result directory for one prespecified member of an 8-run bank."""
    if not 1 <= run <= 8:
        raise ValueError("Bank run numbers must be in 1..8")
    return bank_dir / BANK_RUN_DIR_FORMAT.format(run=run)


def _validate_bank_member_meta(
    meta: dict[str, Any], expected: dict[str, object], label: str
) -> None:
    """Reject stale/cross-bank metadata before it can certify a bank."""
    mismatches = {
        key: {"expected": value, "actual": meta.get(key)}
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} metadata identity mismatch: {mismatches}")


def seed_done(output_dir: Path) -> bool:
    """True if the attempt completed (meta.json is written last)."""
    return (output_dir / META_FILENAME).exists()


def parallel_bank_done(output_dir: Path) -> bool:
    """True only when the current fresh-eight bank marker is present.

    Generation containers intentionally receive metadata but not prior
    solutions, so the last-written bank marker is the resumability certificate.
    Candidate artifacts are checked again before aggregation and audit.
    """
    meta_path = output_dir / META_FILENAME
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if (
        meta.get("parallel_bank_protocol") != PARALLEL_BANK_PROTOCOL
        or meta.get("parallel_run_count") != 8
    ):
        return False
    return all(
        (bank_run_output_dir(output_dir, run) / META_FILENAME).exists()
        for run in range(1, 9)
    )


def uniform_strategy_bank_done(output_dir: Path) -> bool:
    """True only for a complete Uniform-C bank or a recorded planner failure."""
    meta_path = output_dir / META_FILENAME
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if meta.get("mode") != MODE_UNIFORM_STRATEGY:
        return False
    executor_count = meta.get("uniform_strategy_executor_count")
    strategy_count = meta.get("strategy_count")
    assignments = meta.get("run_strategy_indices")
    if executor_count == 0:
        return (
            strategy_count == 0
            and assignments == []
            and isinstance(meta.get("planner_failure"), str)
        )
    if (
        executor_count != 8
        or not isinstance(strategy_count, int)
        or strategy_count < 1
        or not isinstance(assignments, list)
        or len(assignments) != 8
        or any(
            not isinstance(index, int) or index < 1 or index > strategy_count
            for index in assignments
        )
    ):
        return False
    return all(
        (bank_run_output_dir(output_dir, run) / META_FILENAME).exists()
        for run in range(1, 9)
    )


def uniform_strategy_only_done(output_dir: Path) -> bool:
    """True when a planner-only bank has a valid frozen strategy artifact."""
    meta_path = output_dir / META_FILENAME
    strategies_path = output_dir / UNIFORM_STRATEGIES_FILENAME
    if (
        not meta_path.exists()
        or not strategies_path.exists()
        or not (output_dir / SOLUTION_FILENAME).exists()
    ):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        artifact = json.loads(strategies_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    strategies = artifact.get("strategies")
    return (
        meta.get("mode") == MODE_UNIFORM_STRATEGY_ONLY
        and isinstance(strategies, list)
        and len(strategies) == meta.get("strategy_count")
        and all(isinstance(strategy, str) and strategy.strip() for strategy in strategies)
        and (bool(strategies) or isinstance(meta.get("planner_failure"), str))
    )


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


def all_budget_cut_multipliers(budget_units: int) -> list[int]:
    """Every integer multiplier strictly below the full sequential budget."""
    return list(range(1, budget_units))


def cut_solution_path(output_dir: Path, multiplier: int) -> Path:
    """Path of the snapshot graded as 'the solution at <multiplier>x budget'."""
    return output_dir / SOLUTION_CUT_FILENAME_FORMAT.format(multiplier=multiplier)


def materialize_budget_cut_snapshots(
    config: ExperimentConfig,
    arm: ArmConfig,
    output_dir: Path,
    multipliers: list[int],
) -> None:
    """Recover requested sequential snapshots from the immutable phase log.

    Generation historically materialized only the 1x/2x/4x headline cuts.
    The compressed log nevertheless retains every complete phase and its
    cumulative token count, so dense mechanism audits can recover the exact
    same hard-cut artifact at each integer multiplier without rerunning the
    solver. Existing snapshots are verified rather than silently replaced.
    """
    if arm.mode != MODE_SEQUENTIAL:
        raise ValueError("Budget-cut snapshots exist only for sequential arms")
    invalid = [m for m in multipliers if not 1 <= m < arm.budget_units]
    if invalid:
        raise ValueError(f"Invalid budget-cut multipliers: {invalid}")

    log_path = output_dir / LOGS_FILENAME
    with log_path.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            log_text = reader.read().decode("utf-8")
    records = [json.loads(line) for line in log_text.splitlines() if line.strip()]

    for multiplier in multipliers:
        threshold = multiplier * config.unit_output_tokens
        selected: dict[str, Any] | None = None
        for record in records:
            label = record.get("label")
            text = record.get("text")
            cumulative = record.get("cumulative_output_tokens")
            exhausted = record.get("budget_exhausted")
            if (
                label == PHASE_CRITIQUE
                or not isinstance(text, str)
                or not text.strip()
                or isinstance(cumulative, bool)
                or not isinstance(cumulative, int)
                or exhausted is not False
            ):
                continue
            if cumulative <= threshold:
                selected = record

        path = cut_solution_path(output_dir, multiplier)
        if selected is None:
            path.unlink(missing_ok=True)
            continue
        expected = (str(selected["text"]).strip() + "\n").encode("utf-8")
        if path.exists():
            if path.read_bytes() != expected:
                raise ValueError(
                    f"{path} disagrees with the immutable phase log"
                )
            continue
        _atomic_write_bytes(path, expected)


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
    if arm.mode == MODE_SEQUENTIAL:
        meta["completed_critique_rounds"] = sum(
            phase.label == PHASE_CRITIQUE for phase in phases
        )
        meta["sequential_stopping_policy"] = {
            "minimum_critique_rounds": SEQUENTIAL_MIN_ROUNDS_BEFORE_CONVERGENCE,
            "consecutive_no_gap_critiques": SEQUENTIAL_NO_GAP_STREAK_TO_STOP,
        }
    if meta_extra:
        meta.update(meta_extra)
    _atomic_write_bytes(
        output_dir / META_FILENAME,
        (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return output_dir


def write_parallel_bank_meta(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    bank_seed: int,
    bank_dir: Path,
) -> None:
    """Write one fresh-IID 8-run Parallel bank after all members finish."""
    if arm.mode != MODE_PARALLEL or arm.budget_units != 8:
        raise ValueError("Parallel bank must use mode='parallel' and budget_units=8")
    run_metas: list[dict[str, Any]] = []
    for run in range(1, 9):
        path = bank_run_output_dir(bank_dir, run) / META_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"Parallel run_{run:02d} is incomplete: {path}")
        run_meta = json.loads(path.read_text(encoding="utf-8"))
        _validate_bank_member_meta(
            run_meta,
            {
                "problem_id": problem.problem_id,
                "arm": arm.name,
                "mode": MODE_PARALLEL,
                "model": config.model,
                "seed": bank_seed,
                "budget_output_tokens": config.unit_output_tokens,
                "parallel_bank_seed": bank_seed,
                "parallel_run": run,
                "parallel_run_budget_output_tokens": config.unit_output_tokens,
            },
            f"Parallel run_{run:02d}",
        )
        run_metas.append(run_meta)

    output_tokens_spent = sum(
        int(meta.get("output_tokens_spent", 0)) for meta in run_metas
    )
    total_budget = config.budget_tokens(arm)
    reconnects = [
        event
        for meta in run_metas
        for event in meta.get("session_reconnects", [])
        if isinstance(event, dict)
    ]
    process_resume_count = sum(
        int(meta.get("process_resume_count", 0)) for meta in run_metas
    )
    provider_usage_totals: dict[str, int] = {}
    for run_meta in run_metas:
        _merge_provider_usage_totals(
            provider_usage_totals, run_meta.get("provider_usage_totals")
        )
    transport_recovered = any(
        event.get("reason") == "transport" for event in reconnects
    )
    accounting_status = (
        "recovered_eligible_output_accounted"
        if process_resume_count or transport_recovered
        else "provider_reported_complete"
    )
    run_records = []
    for run, run_meta in enumerate(run_metas, start=1):
        run_records.append(
            {
                "run": run,
                "local_result_path": bank_run_output_dir(bank_dir, run)
                .relative_to(RESULTS_ROOT)
                .as_posix(),
                "output_tokens_spent": int(run_meta.get("output_tokens_spent", 0)),
                "gradeable_solution_emitted": bool(
                    run_meta.get("gradeable_solution_emitted")
                ),
                "process_resume_count": int(
                    run_meta.get("process_resume_count", 0)
                ),
                "session_reconnect_count": int(
                    run_meta.get("session_reconnect_count", 0)
                ),
            }
        )
    meta = {
        "problem_id": problem.problem_id,
        "arm": arm.name,
        "mode": MODE_PARALLEL,
        "hint": arm.hint,
        "model": config.model,
        "provider_transport_policy": provider_transport_policy(config.model),
        "session_recovery_policy": session_recovery_policy(),
        "effort": config.effort,
        "seed": bank_seed,
        "budget_output_tokens": total_budget,
        "output_tokens_spent": output_tokens_spent,
        "output_tokens_over_budget": max(0, output_tokens_spent - total_budget),
        "provider_usage_totals": provider_usage_totals,
        "parallel_bank_protocol": PARALLEL_BANK_PROTOCOL,
        "parallel_run_count": 8,
        "parallel_run_budget_output_tokens_each": config.unit_output_tokens,
        "runs": run_records,
        "process_resume_count": process_resume_count,
        "session_reconnect_count": len(reconnects),
        "session_reconnects": reconnects,
        "token_accounting_status": accounting_status,
        "gradeable_solution_emitted": any(
            bool(meta.get("gradeable_solution_emitted")) for meta in run_metas
        ),
    }
    _atomic_write_bytes(
        bank_dir / SOLUTION_FILENAME,
        b"Fresh-IID Parallel-8 bank; grade run_01 through run_08.\n",
    )
    _atomic_write_bytes(
        bank_dir / META_FILENAME,
        (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def write_uniform_strategy_plan_artifacts(
    bank_dir: Path,
    phases: list[PhaseResult],
    strategies: list[str],
    assignments: list[int],
    plan_scratch_path: Path,
) -> None:
    """Persist the shared planner output before optional executor generation.

    For a full Uniform-C bank, top-level meta.json remains reserved as the
    completion marker and is written only after every executor finishes. A
    planner-only bank writes its own marker immediately after these artifacts.
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
                {"strategies": strategies, "run_strategy_indices": assignments},
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
        raise ValueError("Uniform Strategy bank must assign every executor run")
    if strategy_count < 1 or any(
        index < 1 or index > strategy_count for index in assignments
    ):
        raise ValueError("Uniform Strategy bank contains an invalid strategy index")
    run_metas: list[dict[str, Any]] = []
    for run in range(1, config.uniform_strategy_branches + 1):
        path = bank_run_output_dir(bank_dir, run) / META_FILENAME
        run_meta = json.loads(path.read_text(encoding="utf-8"))
        _validate_bank_member_meta(
            run_meta,
            {
                "problem_id": problem.problem_id,
                "arm": arm.name,
                "mode": MODE_UNIFORM_STRATEGY,
                "model": config.model,
                "seed": seed,
                "budget_output_tokens": executor_budget,
                "uniform_strategy_bank_seed": seed,
                "uniform_strategy_run": run,
                "uniform_strategy_executor_budget": executor_budget,
            },
            f"Uniform Strategy run_{run:02d}",
        )
        run_metas.append(run_meta)
    plan_spent = sum(phase.output_tokens for phase in plan_phases)
    run_spent = sum(int(meta["output_tokens_spent"]) for meta in run_metas)
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
        int(meta.get("process_resume_count", 0)) for meta in run_metas
    )
    plan_reconnects = [
        event for phase in plan_phases for event in phase.reconnects
    ]
    run_reconnects = [
        event
        for meta in run_metas
        for event in meta.get("session_reconnects", [])
        if isinstance(event, dict)
    ]
    output_tokens_spent = plan_spent + run_spent
    provider_usage_totals = _provider_usage_totals(plan_phases)
    for run_meta in run_metas:
        _merge_provider_usage_totals(
            provider_usage_totals, run_meta.get("provider_usage_totals")
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
        "uniform_strategy_executor_output_tokens_spent": run_spent,
        "strategy_count": strategy_count,
        "run_strategy_indices": assignments,
        "provider_session_ids": dict(sorted(provider_session_ids.items())),
        "run_provider_session_ids": {
            str(i): meta.get("provider_session_ids", {})
            for i, meta in enumerate(run_metas, start=1)
        },
        "process_resume_count": process_resume_count,
        "session_reconnect_count": len(plan_reconnects) + len(run_reconnects),
        "session_reconnects": [
            *[asdict(event) for event in plan_reconnects],
            *run_reconnects,
        ],
        "token_accounting_status": (
            "recovered_eligible_output_accounted"
            if process_resume_count
            or (
                any(event.reason == "transport" for event in plan_reconnects)
                or any(event.get("reason") == "transport" for event in run_reconnects)
            )
            else "provider_reported_complete"
        ),
        "gradeable_solution_emitted": any(
            bool(meta.get("gradeable_solution_emitted")) for meta in run_metas
        ),
    }
    # run.sh distinguishes newly completed staged attempts from pre-seeded
    # resume markers by requiring solution.md beside the last-written meta.json.
    # The bank is graded from run_<kk>/solution.md; this top-level file is an
    # explicit manifest, never a proof candidate.
    _atomic_write_bytes(
        bank_dir / SOLUTION_FILENAME,
        (
            "Uniform Strategy Search bank. Grade the executor candidates in "
            "run_01 through run_"
            f"{config.uniform_strategy_branches:02d}; this file is not a candidate proof.\n"
        ).encode("utf-8"),
    )
    _atomic_write_bytes(
        bank_dir / META_FILENAME,
        (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def write_uniform_strategy_only_meta(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    bank_dir: Path,
    plan_phases: list[PhaseResult],
    strategy_count: int,
    provider_session_ids: dict[str, str],
    *,
    reused_from: str | None = None,
    planner_failure: str | None = None,
) -> None:
    """Write the completion marker for a planner-only strategy bank."""
    if strategy_count < 0:
        raise ValueError("Planner-only strategy count cannot be negative")
    if strategy_count == 0 and not planner_failure:
        raise ValueError("Empty planner-only bank requires a failure reason")
    plan_spent = sum(phase.output_tokens for phase in plan_phases)
    meta: dict[str, object] = {
        "problem_id": problem.problem_id,
        "arm": arm.name,
        "mode": MODE_UNIFORM_STRATEGY_ONLY,
        "hint": arm.hint,
        "model": config.model,
        "effort": config.effort,
        "seed": seed,
        "budget_output_tokens": config.uniform_strategy_plan_tokens,
        "output_tokens_spent": plan_spent,
        "uniform_strategy_plan_budget_output_tokens": config.uniform_strategy_plan_tokens,
        "uniform_strategy_plan_output_tokens_spent": plan_spent,
        "strategy_count": strategy_count,
        "provider_session_ids": dict(sorted(provider_session_ids.items())),
        "gradeable_solution_emitted": False,
    }
    if reused_from is not None:
        meta["reused_from"] = reused_from
    if planner_failure is not None:
        meta["planner_failure"] = planner_failure
    _atomic_write_bytes(
        bank_dir / SOLUTION_FILENAME,
        b"Planner-only Uniform-C strategy bank; no proof artifact.\n",
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
        "run_strategy_indices": [],
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


def parallel_bank_audited(output_dir: Path) -> bool:
    """True only for a bank audit produced under the current fresh-eight protocol."""
    path = output_dir / SEED_AUDIT_FILENAME
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return record.get("parallel_bank_protocol") == PARALLEL_BANK_PROTOCOL


def write_seed_audit(output_dir: Path, record: dict[str, object]) -> None:
    """Write one attempt's judge verdict (audit.json) atomically."""
    _atomic_write_bytes(
        output_dir / SEED_AUDIT_FILENAME,
        (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def write_seed_state_audit(state_dir: Path, record: dict[str, object]) -> None:
    """Write one compact route-state annotation atomically."""
    state_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(
        state_dir / SEED_STATE_AUDIT_FILENAME,
        (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def archive_audit_scratches(
    output_dir: Path,
    scratch_paths: dict[str, Path],
    *,
    preserve_existing: bool = False,
) -> None:
    """Archive each isolated judge call's visible scratch beside the attempt.

    Keeps the audit auditable: any computation the judge ran while grading is
    preserved under audit_scratch/. The live checkpoint workspace is retained
    until audit.json is durable, so a crash during copying can retry without
    destroying either the transcript or an earlier complete archive.
    """
    destination_root = output_dir / AUDIT_SCRATCH_SUBDIR
    if destination_root.exists() and not preserve_existing:
        shutil.rmtree(destination_root)
    visible = {
        role: scratch
        for role, scratch in scratch_paths.items()
        if any(entry.name != SESSION_STATE_SUBDIR for entry in scratch.iterdir())
    }
    if not visible:
        return
    destination_root.mkdir(exist_ok=True)
    for role, scratch in sorted(visible.items()):
        destination = destination_root / role
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(
            scratch,
            destination,
            ignore=shutil.ignore_patterns(SESSION_STATE_SUBDIR),
        )


def compile_arm_audit(
    config: ExperimentConfig,
    arm: ArmConfig,
    *,
    results_root: Path | None = None,
) -> tuple[Path, int]:
    """Compile every configured-seed verdict into the arm's audit.jsonl.

    Scans the arm's whole results tree rather than any CLI problem filter, so
    a re-audit of one problem can never truncate the compiled file down to
    that subset. Results from retired/out-of-config seed layouts are never
    silently mixed into a current analysis. One line per audited configured
    (problem, seed), sorted. Returns the path and record count.
    """
    root = RESULTS_ROOT if results_root is None else results_root
    arm_root = root / config.model_dirname / arm.name
    records: list[dict[str, object]] = []
    for audit_file in sorted(arm_root.glob(f"*/seed_*/{SEED_AUDIT_FILENAME}")):
        seed_name = audit_file.parent.name
        try:
            seed = int(seed_name.removeprefix("seed_"))
        except ValueError:
            continue
        if seed not in arm.seeds:
            continue
        record = json.loads(audit_file.read_text(encoding="utf-8"))
        if (
            arm.mode == MODE_PARALLEL
            and record.get("parallel_bank_protocol") != PARALLEL_BANK_PROTOCOL
        ):
            continue
        records.append(record)
    path = arm_root / ARM_AUDIT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    _atomic_write_bytes(path, (lines + "\n").encode("utf-8"))
    return path, len(records)


def compile_arm_state_audit(
    config: ExperimentConfig, arm: ArmConfig
) -> tuple[Path, int]:
    """Compile every configured-seed proof-artifact state without truncation."""
    arm_root = RESULTS_ROOT / config.model_dirname / arm.name
    records: list[dict[str, object]] = []
    state_files = list(
        arm_root.glob(
            (
                f"*/seed_*/run_*/{SEED_STATE_AUDIT_FILENAME}"
                if arm.mode in {MODE_PARALLEL, MODE_UNIFORM_STRATEGY}
                else f"*/seed_*/{SEED_STATE_AUDIT_FILENAME}"
            )
        )
    )
    for state_file in sorted(state_files):
        seed_dir = (
            state_file.parent.parent
            if state_file.parent.name.startswith("run_")
            else state_file.parent
        )
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        if seed not in arm.seeds:
            continue
        records.append(json.loads(state_file.read_text(encoding="utf-8")))
    path = arm_root / ARM_STATE_AUDIT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    _atomic_write_bytes(path, (lines + "\n").encode("utf-8") if lines else b"")
    return path, len(records)
