"""Entrypoint: run ONE experiment arm over the problem set (resumable).

Usage:
    python -m src.run --arm baseline
    python -m src.run --arm hint --problems usamo-2026-3,china-2026-5

An attempt = one (problem, seed) pair. Mode 'single' runs one solve phase;
mode 'sequential' runs solve -> (critique -> revise)*; each mode 'parallel'
seed forms one bank of eight fresh independent 1x attempts;
mode 'uniform_strategy' runs one shared planner followed by eight fresh proof
executors; auxiliary modes freeze/compress/rank strategy proposals. Completed attempts (meta.json present) are skipped, so an
interrupted run resumes cleanly.
"""

import argparse
import dataclasses
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any

import anyio

from src.checkpoint import (
    AttemptCheckpoint,
    phase_record,
    progress_tool_calls,
    protocol_fingerprint,
    tool_calls_from_records,
)
from src.concurrency import run_all
from src.config import load_config, override_models
from src.constants import (
    CONFIG_PATH,
    DEFAULT_UNIFORM_COMPRESS_MODEL,
    HINT_H1,
    HINT_H2,
    HINT_H3,
    HINT_NONE,
    LATE_CONTINUATION_PROMPT_FILE,
    LOG_FORMAT,
    LOG_LEVEL,
    META_FILENAME,
    MODE_PARALLEL,
    MODE_SEQUENTIAL,
    MODE_UNIFORM_STRATEGY,
    MODE_UNIFORM_STRATEGY_ONLY,
    MODE_UNIFORM_COMPRESS,
    MODE_SELECTION,
    MODE_SELECTION_NO_PROBLEM,
    NO_GENUINE_GAP_MARKER,
    PHASE_CRITIQUE,
    PHASE_REVISE,
    PHASE_PLAN,
    PHASE_PLAN_WRAP_UP,
    PHASE_SOLVE,
    PHASE_WRAP_UP,
    RESULTS_ROOT,
    RUN_REFERENCE_FILENAME,
    SELECTION_FILENAME,
    SELECTION_PROMPT_FILE,
    SELECTION_WRAP_PROMPT_FILE,
    SEQUENTIAL_MIN_ROUNDS_BEFORE_CONVERGENCE,
    SEQUENTIAL_NO_GAP_STREAK_TO_STOP,
)
from src.models import (
    ArmConfig,
    ExperimentConfig,
    PhaseResult,
    Problem,
    arm_checkpoint_identity,
)
from src.late_intervention import (
    LATE_BASELINE_ARM,
    PREFIX_UNITS,
    LatePrefixSource,
    fork_native_session,
    load_prefix_source,
    save_prefix_source,
)
from src.openrouter_routing import route_for
from src.prompts import (
    critique_prompt,
    late_continuation_prompt,
    revise_prompt,
    selection_no_problem_prompt,
    selection_prompt,
    selection_wrap_prompt,
    task_prompt,
    uniform_strategy_execute_prompt,
    uniform_strategy_plan_prompt,
    uniform_strategy_plan_wrap_up_prompt,
    wrap_up_prompt,
)
from src.solver import (
    BudgetTracker,
    ResumableClaudeSession,
    STOP_BUDGET_EXHAUSTED,
    agent_runtime_policy,
    agent_settings_path,
    build_options,
    process_recovery_prompt,
    provider_transport_policy,
    run_phase,
    token_env_name,
    uses_litellm,
    uses_meta,
    uses_vllm,
)
from src.storage import (
    bank_run_output_dir,
    load_problems,
    load_selection_candidates,
    parallel_bank_done,
    seed_done,
    seed_output_dir,
    uniform_strategy_bank_done,
    uniform_strategy_only_done,
    write_parallel_bank_meta,
    write_auxiliary_result,
    write_uniform_strategy_bank_meta,
    write_uniform_strategy_plan_artifacts,
    write_uniform_strategy_planner_failure,
    write_uniform_strategy_only_meta,
    write_seed_outputs,
)
from src.strategy_experiments import (
    _selection_candidates,
    compress_uniform_strategies,
    compression_source_status,
)
from src.token_pool import TokenPool

log = logging.getLogger("run")

LATE_HINT_ARM_NAME = "late-hint-sequential"
LATE_BASELINE_ARM_NAME = LATE_BASELINE_ARM
LATE_INTERVENTION_ARMS = frozenset({LATE_BASELINE_ARM_NAME, LATE_HINT_ARM_NAME})
LATE_INTERVENTION_PROTOCOL_VERSION = 2


async def _checkpointed_phase(
    checkpoint: AttemptCheckpoint,
    role: str,
    client: ResumableClaudeSession,
    tracker: BudgetTracker,
    prompt: str,
    label: str,
    stop_at_tokens: int,
) -> PhaseResult:
    """Run or recover one phase with streamed accounting committed durably."""
    active = checkpoint.active(role)
    process_recovery = active is not None
    if active is None:
        if tracker.spent >= stop_at_tokens:
            raise RuntimeError(
                f"Refusing to start {label!r} at exhausted cutoff "
                f"{tracker.spent}/{stop_at_tokens}"
            )
        active = checkpoint.begin_phase(role, label, prompt, stop_at_tokens, tracker)
    else:
        if active.get("label") != label:
            raise ValueError(
                f"Checkpoint expected phase {active.get('label')!r}, not {label!r}"
            )
        stop_at = int(active["stop_at_tokens"])
        if tracker.spent >= stop_at:
            progress = active.get("progress", {})
            if not isinstance(progress, dict):
                raise TypeError("Checkpoint phase progress is corrupt")
            text_parts = progress.get("text_parts", [])
            text = "\n".join(str(part) for part in text_parts)
            reconnect_start = int(active.get("reconnect_start", 0))
            reconnects = checkpoint.reconnects(role)[reconnect_start:]
            phase_tokens = tracker.finish_phase(None)
            phase = PhaseResult(
                label=label,
                prompt=str(active["prompt"]),
                text=text,
                output_tokens=phase_tokens,
                cumulative_output_tokens=tracker.spent,
                num_turns=0,
                duration_ms=0,
                total_cost_usd=0.0,
                is_error=False,
                stop_reason=STOP_BUDGET_EXHAUSTED,
                budget_exhausted=True,
                tool_calls=progress_tool_calls(progress),
                reconnects=reconnects,
                process_resume_count=int(active.get("process_resume_count", 0)) + 1,
                discarded_output_text=str(active.get("discarded_output_text", "")),
                discarded_tool_calls=tool_calls_from_records(
                    active.get("discarded_tool_calls", [])
                ),
            )
            checkpoint.finish_phase(
                role,
                phase,
                tracker,
                client.session_id,
                client.reconnect_events,
            )
            log.warning(
                "Finalized killed %s phase at its token cutoff without "
                "issuing an over-budget recovery query",
                label,
            )
            return phase
        active = checkpoint.prepare_process_resume(role)
        log.warning(
            "Resuming killed %s phase for role %s (%d/%d reported tokens)",
            label,
            role,
            tracker.spent,
            tracker.budget_tokens,
        )

    original_prompt = str(active["prompt"])
    stop_at = int(active["stop_at_tokens"])
    raw_text_block_keys = active.get("discarded_text_block_keys", [])
    if not isinstance(raw_text_block_keys, list):
        raise TypeError("Checkpoint discarded text-block keys are corrupt")
    raw_message_ids = active.get("discarded_message_ids", [])
    if not isinstance(raw_message_ids, list):
        raise TypeError("Checkpoint discarded message ids are corrupt")

    def save_progress(progress: dict[str, object]) -> None:
        checkpoint.save_progress(
            role,
            tracker,
            client.session_id,
            client.reconnect_events,
            progress,
        )

    def finish(phase: PhaseResult) -> None:
        checkpoint.finish_phase(
            role,
            phase,
            tracker,
            client.session_id,
            client.reconnect_events,
        )

    try:
        return await run_phase(
            client,
            original_prompt,
            label,
            tracker,
            stop_at,
            query_prompt=(
                process_recovery_prompt(original_prompt)
                if process_recovery
                else original_prompt
            ),
            process_resume_count=int(active.get("process_resume_count", 0)),
            discarded_output_text=str(active.get("discarded_output_text", "")),
            discarded_tool_calls=tool_calls_from_records(
                active.get("discarded_tool_calls", [])
            ),
            discarded_text_block_keys=[str(key) for key in raw_text_block_keys],
            discarded_message_ids=[str(value) for value in raw_message_ids],
            reconnect_start=int(active.get("reconnect_start", 0)),
            on_progress=save_progress,
            on_complete=finish,
        )
    finally:
        # Also persist reconnects when every bounded transport retry fails
        # before another stream event arrives.  A later invocation can still
        # resume the stable transcript instead of resetting paid work.
        checkpoint.save_session(role, client.session_id, client.reconnect_events)


async def _run_strict_wrap_phase(
    config: ExperimentConfig,
    checkpoint: AttemptCheckpoint,
    role: str,
    tracker: BudgetTracker,
    phases: list[PhaseResult],
    scratch_path: Path,
    pool: TokenPool,
    label: str,
    prompt: str,
    response_cap: int,
) -> None:
    """Resume once for a tool-free, one-turn final response.

    The provider task envelope remains at least 20k, while the explicit
    per-response cap is the predeclared wrap reserve. Artifact selection later
    rejects any phase whose cumulative usage finishes beyond the hard tier.
    """
    active = checkpoint.active(role)
    active_is_wrap = active is not None and active.get("label") == label
    if not active_is_wrap and not (
        tracker.soft_exhausted
        and not tracker.exhausted
        and not any(phase.label == label for phase in phases)
    ):
        return
    async with ResumableClaudeSession(
        pool,
        lambda token, session_id, resume_id, stderr: build_options(
            config,
            str(scratch_path),
            response_cap,
            token,
            stderr,
            session_id=session_id,
            resume_session_id=resume_id,
            max_output_tokens_per_response=response_cap,
            max_turns=1,
            tools_enabled=False,
        ),
        session_id=checkpoint.session_id(role),
        reconnects=checkpoint.reconnects(role),
    ) as client:
        checkpoint.save_session(role, client.session_id, client.reconnect_events)
        wrapped = await _checkpointed_phase(
            checkpoint,
            role,
            client,
            tracker,
            str(active["prompt"]) if active_is_wrap and active is not None else prompt,
            label,
            (
                int(active["stop_at_tokens"])
                if active_is_wrap and active is not None
                else tracker.budget_tokens
            ),
        )
        phases.append(wrapped)


_SELECTION_RANKING_RE = re.compile(
    r"<ranking>\s*([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\s*</ranking>",
    re.IGNORECASE,
)
_SELECTION_REASON_RE = re.compile(
    r"<reason>\s*(.*?)\s*</reason>", re.IGNORECASE | re.DOTALL
)


def _parse_selection_decision(text: str) -> dict[str, object] | None:
    """Parse the final tagged ranking; malformed or incomplete output is absent."""
    rankings = list(_SELECTION_RANKING_RE.finditer(text))
    reasons = list(_SELECTION_REASON_RE.finditer(text))
    if not rankings or not reasons:
        return None
    ranking = [int(value) for value in rankings[-1].groups()]
    reason = reasons[-1].group(1).strip()
    if sorted(ranking) != [1, 2, 3, 4] or not reason:
        return None
    return {"ranking": ranking, "reason": reason}


async def run_selection(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    pool: TokenPool,
    frozen_record: dict[str, Any],
    *,
    include_problem: bool,
) -> None:
    """Rank four strategies under the standard work/reserve budget controller."""
    candidates, oracle_position = _selection_candidates(
        problem, frozen_record, seed, config.model
    )
    budget_tokens = config.budget_tokens(arm)
    reserve_tokens = config.wrap_up_reserve_tokens
    if reserve_tokens <= 0 or reserve_tokens >= budget_tokens:
        raise ValueError(
            f"Selection budget {budget_tokens} must exceed its positive "
            f"wrap-up reserve {reserve_tokens}"
        )
    candidate_texts = [str(candidate["strategy"]) for candidate in candidates]
    prompt = (
        selection_prompt(
            problem, candidate_texts, budget_tokens, reserve_tokens
        )
        if include_problem
        else selection_no_problem_prompt(
            candidate_texts, budget_tokens, reserve_tokens
        )
    )
    output_dir = seed_output_dir(config, arm, problem.problem_id, seed)
    checkpoint = AttemptCheckpoint(
        {
            "stage": arm.mode,
            "source_model": config.model,
            "problem_id": problem.problem_id,
            **(
                {"problem_statement": problem.statement}
                if include_problem
                else {}
            ),
            "seed": seed,
            "include_problem": include_problem,
            "budget_output_tokens": budget_tokens,
            "working_output_tokens": budget_tokens - reserve_tokens,
            "wrap_up_reserve_tokens": reserve_tokens,
            "tools_enabled_during_work": True,
            "tools_enabled_during_wrap": False,
            "max_turns_per_phase": config.max_turns_per_phase,
            "agent_runtime_policy": agent_runtime_policy(config.model),
            "selection_protocol_fingerprint": protocol_fingerprint(
                agent_settings_path(config.model),
                (SELECTION_PROMPT_FILE, SELECTION_WRAP_PROMPT_FILE),
            ),
            # Bind resumption to the displayed order without persisting the
            # hidden oracle/provenance labels in the tool-visible checkpoint.
            "candidate_strategy_sha256s": [
                hashlib.sha256(text.encode("utf-8")).hexdigest()
                for text in candidate_texts
            ],
        }
    )
    role = "selection"
    try:
        scratch_path = checkpoint.scratch_dir(role)
        phases = checkpoint.phases(role)
        tracker = checkpoint.tracker(role, budget_tokens, reserve_tokens)
        active = checkpoint.active(role)
        active_is_wrap = (
            active is not None and active.get("label") == "selection_wrap"
        )
        needs_work_session = not active_is_wrap and (
            active is not None or not phases
        )
        if needs_work_session:
            async with ResumableClaudeSession(
                pool,
                lambda token, session_id, resume_id, stderr: build_options(
                    config,
                    str(scratch_path),
                    max(1, tracker.remaining),
                    token,
                    stderr,
                    session_id=session_id,
                    resume_session_id=resume_id,
                    tools_enabled=True,
                ),
                session_id=checkpoint.session_id(role),
                reconnects=checkpoint.reconnects(role),
            ) as client:
                checkpoint.save_session(
                    role, client.session_id, client.reconnect_events
                )
                phases.append(
                    await _checkpointed_phase(
                        checkpoint,
                        role,
                        client,
                        tracker,
                        prompt,
                        "selection_work",
                        tracker.soft_limit_tokens,
                    )
                )

        await _run_strict_wrap_phase(
            config,
            checkpoint,
            role,
            tracker,
            phases,
            scratch_path,
            pool,
            "selection_wrap",
            selection_wrap_prompt(tracker.remaining),
            reserve_tokens,
        )

        completed_verdicts = [
            parsed
            for phase in phases
            if (
                not phase.budget_exhausted
                and phase.cumulative_output_tokens <= budget_tokens
            )
            for parsed in [_parse_selection_decision(phase.text)]
            if parsed is not None
        ]
        verdict = completed_verdicts[-1] if completed_verdicts else {}
        ranking_value = verdict.get("ranking")
        valid_ranking = (
            isinstance(ranking_value, list)
            and all(type(position) is int for position in ranking_value)
            and sorted(ranking_value) == [1, 2, 3, 4]
        )
        ranking = (
            [int(position) for position in ranking_value]
            if valid_ranking and isinstance(ranking_value, list)
            else []
        )
        ranked_candidates = [candidates[position - 1] for position in ranking]
        oracle_rank = ranking.index(oracle_position) + 1 if ranking else None
        top = ranked_candidates[0] if ranked_candidates else None
        decision_status = "selected" if valid_ranking else "no_decision"
        terminal_stop_reason = phases[-1].stop_reason if phases else "no_response"
        artifact: dict[str, object] = {
            "problem_id": problem.problem_id,
            "source_model": config.model,
            "seed": seed,
            "include_problem": include_problem,
            "decision_status": decision_status,
            "candidates": candidates,
            "ranking_positions": ranking,
            "ranked_candidate_ids": [
                candidate["candidate_id"] for candidate in ranked_candidates
            ],
            "reason": str(
                verdict.get(
                    "reason",
                    f"No valid within-budget ranking returned ({terminal_stop_reason}).",
                )
            ),
            "stop_reason": terminal_stop_reason,
            "phases": [phase_record(phase) for phase in phases],
        }
        audit: dict[str, object] = {
            "problem_id": problem.problem_id,
            "arm": arm.name,
            "seed": seed,
            "source_model": config.model,
            "decision_status": decision_status,
            "oracle_rank": oracle_rank,
            "oracle_top1": oracle_rank == 1,
            "top_candidate_id": top["candidate_id"] if top is not None else None,
        }
        if include_problem:
            audit["oracle_strategy_match_top1"] = bool(
                top is not None and top["oracle_strategy_match"]
            )
        meta: dict[str, object] = {
            "problem_id": problem.problem_id,
            "arm": arm.name,
            "mode": arm.mode,
            "model": config.model,
            "seed": seed,
            "candidate_count": 4,
            "budget_output_tokens": budget_tokens,
            "working_output_tokens": budget_tokens - reserve_tokens,
            "wrap_up_reserve_tokens": reserve_tokens,
            "output_tokens_spent": tracker.spent,
            "budget_eligible": tracker.spent <= budget_tokens,
            "tools_enabled_during_work": True,
            "tools_enabled_during_wrap": False,
            "max_turns_per_phase": config.max_turns_per_phase,
            "oracle_position": oracle_position,
            "decision_status": decision_status,
            "provider_session_ids": checkpoint.session_ids(),
            "gradeable_solution_emitted": False,
        }
        checkpoint.prepare_completion(
            (output_dir / "meta.json").relative_to(RESULTS_ROOT).as_posix()
        )
        write_auxiliary_result(
            output_dir, SELECTION_FILENAME, artifact, meta, audit=audit
        )
        checkpoint.complete()
    finally:
        checkpoint.close()


def _planner_text(phases: list[PhaseResult], budget_tokens: int) -> str:
    """Last complete planner response finishing within its hard allocation."""
    for phase in reversed(phases):
        if (
            not phase.budget_exhausted
            and phase.cumulative_output_tokens <= budget_tokens
            and phase.text.strip()
        ):
            return phase.text
    raise RuntimeError(
        "Uniform Strategy planner produced no complete within-budget strategy text"
    )


_STRATEGY_TAG = re.compile(r"<strategy>(.*?)</strategy>", re.DOTALL | re.IGNORECASE)


def _proposed_strategies(
    phases: list[PhaseResult], maximum: int, budget_tokens: int
) -> list[str]:
    """Parse and deduplicate the planner's final tagged strategy set.

    If the final response is substantive but imperfectly formatted, it becomes
    one strategy rather than triggering another paid planner attempt.
    """
    text = _planner_text(phases, budget_tokens)
    parsed = [match.strip() for match in _STRATEGY_TAG.findall(text) if match.strip()]
    if not parsed:
        marker = "## Strategy Set"
        fallback = text.split(marker, 1)[-1].strip() if marker in text else text.strip()
        parsed = [fallback]
    unique: list[str] = []
    seen: set[str] = set()
    for strategy in parsed:
        normalized = " ".join(strategy.split()).casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(strategy)
        if len(unique) == maximum:
            break
    if not unique:
        raise RuntimeError("Uniform Strategy planner produced no usable strategy")
    return unique


def _critique_reports_no_gap(text: str) -> bool:
    """True only when a critique contains the exact standalone no-gap verdict.

    Markdown emphasis and a final period are ignored. Merely discussing or
    quoting the marker inside a longer sentence cannot trigger early stopping.
    """
    for line in text.splitlines():
        normalized = line.strip().strip("*_` ").rstrip(".").strip()
        if normalized == NO_GENUINE_GAP_MARKER:
            return True
    return False


async def _solve_parallel_run(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    bank_seed: int,
    run: int,
    pool: TokenPool,
) -> None:
    """Run one fresh 1x member (run_01..run_08) of a Parallel bank."""
    if not 1 <= run <= 8:
        raise ValueError("Fresh Parallel runs must be numbered 01..08")
    bank_dir = seed_output_dir(config, arm, problem.problem_id, bank_seed)
    output_dir = bank_run_output_dir(bank_dir, run)
    if seed_done(output_dir):
        return
    identity = run_checkpoint_identity(config, arm, problem, bank_seed)
    identity.update(
        {
            "parallel_bank_seed": bank_seed,
            "parallel_run": run,
            "parallel_run_budget": config.unit_output_tokens,
        }
    )
    checkpoint = AttemptCheckpoint(identity)
    try:
        scratch_path = checkpoint.scratch_dir("main")
        phases = checkpoint.phases("main")
        tracker = checkpoint.tracker(
            "main", config.unit_output_tokens, config.wrap_up_reserve_tokens
        )
        active_main = checkpoint.active("main")
        active_is_wrap = (
            active_main is not None and active_main.get("label") == PHASE_WRAP_UP
        )
        needs_work_session = not active_is_wrap and (
            active_main is not None or not phases
        )
        if needs_work_session:
            async with ResumableClaudeSession(
                pool,
                lambda token, session_id, resume_id, stderr: build_options(
                    config,
                    str(scratch_path),
                    max(1, tracker.remaining),
                    token,
                    stderr,
                    session_id=session_id,
                    resume_session_id=resume_id,
                ),
                session_id=checkpoint.session_id("main"),
                reconnects=checkpoint.reconnects("main"),
            ) as executor:
                checkpoint.save_session(
                    "main", executor.session_id, executor.reconnect_events
                )
                if not phases:
                    phases.append(
                        await _checkpointed_phase(
                            checkpoint,
                            "main",
                            executor,
                            tracker,
                            task_prompt(
                                problem,
                                hint_for(problem, arm),
                                str(scratch_path),
                                config.unit_output_tokens,
                            ),
                            PHASE_SOLVE,
                            tracker.soft_limit_tokens,
                        )
                    )
        await _run_strict_wrap_phase(
            config,
            checkpoint,
            "main",
            tracker,
            phases,
            scratch_path,
            pool,
            PHASE_WRAP_UP,
            wrap_up_prompt(tracker.remaining),
            config.wrap_up_reserve_tokens,
        )
        checkpoint.prepare_completion(
            (output_dir / META_FILENAME).relative_to(RESULTS_ROOT).as_posix()
        )
        write_seed_outputs(
            config,
            arm,
            problem,
            bank_seed,
            config.unit_output_tokens,
            phases,
            scratch_path,
            provider_session_ids=checkpoint.session_ids(),
            output_dir_override=output_dir,
            meta_extra={
                "parallel_bank_seed": bank_seed,
                "parallel_run": run,
                "parallel_run_budget_output_tokens": config.unit_output_tokens,
            },
        )
        log.info(
            "%s/%s bank seed %d run_%02d done (%d/%d tokens)",
            arm.name,
            problem.problem_id,
            bank_seed,
            run,
            tracker.spent,
            config.unit_output_tokens,
        )
        checkpoint.complete()
    finally:
        checkpoint.close()


async def solve_parallel_bank(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    bank_seed: int,
    pool: TokenPool,
) -> None:
    """Build one 8x bank from eight fresh independent 1x attempts."""
    bank_dir = seed_output_dir(config, arm, problem.problem_id, bank_seed)
    # A pre-protocol bank may have placed a baseline pointer in run_01. It is
    # not evidence and must not coexist with the new fresh run_01 artifacts.
    (bank_run_output_dir(bank_dir, 1) / RUN_REFERENCE_FILENAME).unlink(
        missing_ok=True
    )
    tasks = []
    for run in range(1, 9):
        if seed_done(bank_run_output_dir(bank_dir, run)):
            continue
        tasks.append(
            lambda r=run: _solve_parallel_run(
                config, arm, problem, bank_seed, r, pool
            )
        )
    await run_all(tasks, min(config.max_concurrency, 8))
    write_parallel_bank_meta(
        config,
        arm,
        problem,
        bank_seed,
        bank_dir,
    )
    log.info(
        "%s/%s bank seed %d done (8 fresh IID runs) -> %s",
        arm.name,
        problem.problem_id,
        bank_seed,
        bank_dir,
    )


async def _solve_uniform_strategy_run(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    bank_seed: int,
    run: int,
    strategy_index: int,
    strategy: str,
    executor_budget: int,
    pool: TokenPool,
) -> None:
    """Run one fresh executor under one shared-bank strategy."""
    bank_dir = seed_output_dir(config, arm, problem.problem_id, bank_seed)
    output_dir = bank_run_output_dir(bank_dir, run)
    if seed_done(output_dir):
        return
    identity = run_checkpoint_identity(config, arm, problem, bank_seed)
    identity.update(
        {
            "uniform_strategy_run": run,
            "uniform_strategy_index": strategy_index,
            "uniform_strategy_text": strategy,
            "uniform_strategy_executor_budget": executor_budget,
        }
    )
    checkpoint = AttemptCheckpoint(identity)
    try:
        scratch_path = checkpoint.scratch_dir("main")
        phases = checkpoint.phases("main")
        tracker = checkpoint.tracker(
            "main", executor_budget, config.wrap_up_reserve_tokens
        )
        active_main = checkpoint.active("main")
        active_is_wrap = (
            active_main is not None and active_main.get("label") == PHASE_WRAP_UP
        )
        needs_work_session = not active_is_wrap and (
            active_main is not None or not phases
        )
        if needs_work_session:
            async with ResumableClaudeSession(
                pool,
                lambda token, session_id, resume_id, stderr: build_options(
                    config,
                    str(scratch_path),
                    max(1, tracker.remaining),
                    token,
                    stderr,
                    session_id=session_id,
                    resume_session_id=resume_id,
                ),
                session_id=checkpoint.session_id("main"),
                reconnects=checkpoint.reconnects("main"),
            ) as executor:
                checkpoint.save_session(
                    "main", executor.session_id, executor.reconnect_events
                )
                if not phases:
                    solve = await _checkpointed_phase(
                        checkpoint,
                        "main",
                        executor,
                        tracker,
                        uniform_strategy_execute_prompt(
                            problem,
                            strategy,
                            str(scratch_path),
                            executor_budget,
                        ),
                        PHASE_SOLVE,
                        tracker.soft_limit_tokens,
                    )
                    phases.append(solve)
        await _run_strict_wrap_phase(
            config,
            checkpoint,
            "main",
            tracker,
            phases,
            scratch_path,
            pool,
            PHASE_WRAP_UP,
            wrap_up_prompt(tracker.remaining),
            config.wrap_up_reserve_tokens,
        )
        checkpoint.prepare_completion(
            (output_dir / META_FILENAME).relative_to(RESULTS_ROOT).as_posix()
        )
        write_seed_outputs(
            config,
            arm,
            problem,
            bank_seed,
            executor_budget,
            phases,
            scratch_path,
            provider_session_ids=checkpoint.session_ids(),
            output_dir_override=output_dir,
            meta_extra={
                "uniform_strategy_bank_seed": bank_seed,
                "uniform_strategy_run": run,
                "uniform_strategy_index": strategy_index,
                "uniform_strategy_executor_budget": executor_budget,
            },
        )
        log.info(
            "%s/%s seed %d run_%02d done (%d/%d tokens)",
            arm.name,
            problem.problem_id,
            bank_seed,
            run,
            tracker.spent,
            executor_budget,
        )
        checkpoint.complete()
    finally:
        checkpoint.close()


async def solve_uniform_strategy_bank(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    pool: TokenPool,
    checkpoint: AttemptCheckpoint,
) -> None:
    """Run one shared strategy planner and eight fresh proof executors."""
    plan_budget = config.uniform_strategy_plan_tokens
    total_budget = config.budget_tokens(arm)
    executor_budget = (
        0
        if arm.mode == MODE_UNIFORM_STRATEGY_ONLY
        else (total_budget - plan_budget) // config.uniform_strategy_branches
    )
    bank_dir = seed_output_dir(config, arm, problem.problem_id, seed)
    plan_scratch_path = checkpoint.scratch_dir("plan")
    plan_phases = checkpoint.phases("plan")
    existing_runs = [
        run
        for run in range(1, config.uniform_strategy_branches + 1)
        if seed_done(bank_run_output_dir(bank_dir, run))
    ]
    if existing_runs and not plan_phases:
        raise RuntimeError(
            "Uniform-C bank has completed executors but no resumable planner "
            "checkpoint; refusing to mix them with a newly generated strategy "
            "set. Recoverably archive this seed directory and rerun the bank "
            "from zero."
        )
    plan_tracker = checkpoint.tracker(
        "plan", plan_budget, config.uniform_strategy_plan_wrap_up_reserve_tokens
    )
    active_plan = checkpoint.active("plan")
    active_is_plan_wrap = (
        active_plan is not None and active_plan.get("label") == PHASE_PLAN_WRAP_UP
    )
    plan_needs_work_session = not active_is_plan_wrap and (
        active_plan is not None or not plan_phases
    )
    if plan_needs_work_session:
        async with ResumableClaudeSession(
            pool,
            lambda token, session_id, resume_id, stderr: build_options(
                config,
                str(plan_scratch_path),
                max(1, plan_tracker.remaining),
                token,
                stderr,
                session_id=session_id,
                resume_session_id=resume_id,
            ),
            session_id=checkpoint.session_id("plan"),
            reconnects=checkpoint.reconnects("plan"),
        ) as planner:
            checkpoint.save_session(
                "plan", planner.session_id, planner.reconnect_events
            )
            if not plan_phases:
                plan = await _checkpointed_phase(
                    checkpoint,
                    "plan",
                    planner,
                    plan_tracker,
                    uniform_strategy_plan_prompt(
                        problem,
                        str(plan_scratch_path),
                        plan_budget,
                        config.uniform_strategy_plan_wrap_up_reserve_tokens,
                        config.uniform_strategy_branches,
                    ),
                    PHASE_PLAN,
                    plan_tracker.soft_limit_tokens,
                )
                plan_phases.append(plan)
    await _run_strict_wrap_phase(
        config,
        checkpoint,
        "plan",
        plan_tracker,
        plan_phases,
        plan_scratch_path,
        pool,
        PHASE_PLAN_WRAP_UP,
        uniform_strategy_plan_wrap_up_prompt(
            plan_tracker.remaining,
            config.uniform_strategy_branches,
        ),
        config.uniform_strategy_plan_wrap_up_reserve_tokens,
    )

    try:
        strategies = _proposed_strategies(
            plan_phases,
            config.uniform_strategy_branches,
            plan_budget,
        )
    except RuntimeError as error:
        write_uniform_strategy_plan_artifacts(
            bank_dir,
            plan_phases,
            [],
            [],
            plan_scratch_path,
        )
        if arm.mode == MODE_UNIFORM_STRATEGY_ONLY:
            checkpoint.prepare_completion(
                (bank_dir / META_FILENAME).relative_to(RESULTS_ROOT).as_posix()
            )
            write_uniform_strategy_only_meta(
                config,
                arm,
                problem,
                seed,
                bank_dir,
                plan_phases,
                0,
                checkpoint.session_ids(),
                planner_failure=str(error),
            )
            checkpoint.complete()
            log.warning(
                "%s/%s seed %d planner-only bank completed with no eligible "
                "strategy: %s",
                arm.name,
                problem.problem_id,
                seed,
                error,
            )
            return
        checkpoint.prepare_completion(
            (bank_dir / META_FILENAME).relative_to(RESULTS_ROOT).as_posix()
        )
        write_uniform_strategy_planner_failure(
            config,
            arm,
            problem,
            seed,
            bank_dir,
            plan_phases,
            checkpoint.session_ids(),
            str(error),
        )
        log.warning(
            "%s/%s seed %d bank failed before execution: %s",
            arm.name,
            problem.problem_id,
            seed,
            error,
        )
        checkpoint.complete()
        return
    assignments = [
        (run - 1) % len(strategies) + 1
        for run in range(1, config.uniform_strategy_branches + 1)
    ]
    log.info(
        "%s/%s seed %d: %d strategies planned (%d/%d tokens)",
        arm.name,
        problem.problem_id,
        seed,
        len(strategies),
        plan_tracker.spent,
        plan_budget,
    )
    if arm.mode == MODE_UNIFORM_STRATEGY_ONLY:
        write_uniform_strategy_plan_artifacts(
            bank_dir,
            plan_phases,
            strategies,
            [],
            plan_scratch_path,
        )
        checkpoint.prepare_completion(
            (bank_dir / META_FILENAME).relative_to(RESULTS_ROOT).as_posix()
        )
        write_uniform_strategy_only_meta(
            config,
            arm,
            problem,
            seed,
            bank_dir,
            plan_phases,
            len(strategies),
            checkpoint.session_ids(),
        )
        checkpoint.complete()
        log.info(
            "%s/%s seed %d planner-only bank done (%d strategies) -> %s",
            arm.name,
            problem.problem_id,
            seed,
            len(strategies),
            bank_dir,
        )
        return
    tasks = []
    for run, strategy_index in enumerate(assignments, start=1):
        output_dir = bank_run_output_dir(bank_dir, run)
        if seed_done(output_dir):
            continue
        strategy = strategies[strategy_index - 1]
        tasks.append(
            lambda r=run, i=strategy_index, s=strategy: (
                _solve_uniform_strategy_run(
                    config,
                    arm,
                    problem,
                    seed,
                    r,
                    i,
                    s,
                    executor_budget,
                    pool,
                )
            )
        )
    await run_all(tasks, min(config.max_concurrency, config.uniform_strategy_branches))
    # Do not expose the shared strategy set in the mounted result tree while
    # executors are live; each run receives only its assigned strategy.
    write_uniform_strategy_plan_artifacts(
        bank_dir,
        plan_phases,
        strategies,
        assignments,
        plan_scratch_path,
    )
    missing_runs = [
        run
        for run in range(1, config.uniform_strategy_branches + 1)
        if not seed_done(bank_run_output_dir(bank_dir, run))
    ]
    if missing_runs:
        log.warning(
            "%s/%s seed %d bank remains incomplete; missing executor run(s): %s",
            arm.name,
            problem.problem_id,
            seed,
            ", ".join(f"{run:02d}" for run in missing_runs),
        )
        return
    checkpoint.prepare_completion(
        (bank_dir / META_FILENAME).relative_to(RESULTS_ROOT).as_posix()
    )
    write_uniform_strategy_bank_meta(
        config,
        arm,
        problem,
        seed,
        bank_dir,
        plan_phases,
        len(strategies),
        assignments,
        executor_budget,
        checkpoint.session_ids(),
    )
    log.info(
        "%s/%s seed %d bank done (%d strategies, %d executors) -> %s",
        arm.name,
        problem.problem_id,
        seed,
        len(strategies),
        config.uniform_strategy_branches,
        bank_dir,
    )
    checkpoint.complete()


def hint_for(problem: Problem, arm: ArmConfig) -> str | None:
    """The hint text this arm injects for this problem (None for no-hint).

    Fails fast for a missing tier before an attempt can spend any tokens.
    """
    if arm.hint == HINT_NONE:
        return None
    by_tier = {
        HINT_H1: problem.hint_h1,
        HINT_H2: problem.hint_h2,
        HINT_H3: problem.hint_h3,
    }
    hint = by_tier[arm.hint]
    if hint is None:
        raise ValueError(
            f"Arm '{arm.name}' needs hint tier '{arm.hint}' but problem "
            f"'{problem.problem_id}' has no such hint in the dataset"
        )
    return hint


def run_checkpoint_identity(
    config: ExperimentConfig, arm: ArmConfig, problem: Problem, seed: int
) -> dict[str, object]:
    """Canonical identity shared by normal runs and audited legacy migration."""
    identity: dict[str, object] = {
        "stage": "run",
        "model": config.model,
        "effort": config.effort,
        "arm": arm_checkpoint_identity(arm),
        "problem_id": problem.problem_id,
        "problem_statement": problem.statement,
        "hint": hint_for(problem, arm),
        "seed": seed,
        "unit_output_tokens": config.unit_output_tokens,
        "wrap_up_reserve_tokens": config.wrap_up_reserve_tokens,
        "uniform_strategy_plan_tokens": config.uniform_strategy_plan_tokens,
        "uniform_strategy_plan_wrap_up_reserve_tokens": (
            config.uniform_strategy_plan_wrap_up_reserve_tokens
        ),
        "uniform_strategy_branches": config.uniform_strategy_branches,
        "max_turns_per_phase": config.max_turns_per_phase,
        "protocol_fingerprint": protocol_fingerprint(
            agent_settings_path(config.model),
            (
                (LATE_CONTINUATION_PROMPT_FILE,)
                if arm.name in LATE_INTERVENTION_ARMS
                else ()
            ),
        ),
    }
    # Local proxy/server routes, Meta, and frozen OpenRouter aliases have
    # explicit transport controls. Ordinary Anthropic/OpenRouter ids retain
    # their original checkpoint identities.
    if (
        uses_litellm(config.model)
        or uses_meta(config.model)
        or uses_vllm(config.model)
        or route_for(config.model) is not None
    ):
        identity["provider_transport_policy"] = provider_transport_policy(config.model)
    if uses_vllm(config.model):
        identity["agent_runtime_policy"] = agent_runtime_policy(config.model)
    if arm.name in LATE_INTERVENTION_ARMS:
        identity["late_intervention_protocol_version"] = (
            LATE_INTERVENTION_PROTOCOL_VERSION
        )
    return identity


def _sequential_self_converged(round_num: int, no_gap_streak: int) -> bool:
    """Whether the pre-registered sequential stopping rule is satisfied."""
    return (
        round_num >= SEQUENTIAL_MIN_ROUNDS_BEFORE_CONVERGENCE
        and no_gap_streak >= SEQUENTIAL_NO_GAP_STREAK_TO_STOP
    )


async def _run_late_sequential_role(
    config: ExperimentConfig,
    checkpoint: AttemptCheckpoint,
    role: str,
    scratch_path: Path,
    pool: TokenPool,
    budget_tokens: int,
    initial_prompt: str,
    initial_label: str,
    *,
    allow_self_convergence: bool,
) -> tuple[list[PhaseResult], BudgetTracker, str]:
    """Run one resumable sequential leg of the matched late experiment."""
    phases = checkpoint.phases(role)
    tracker = checkpoint.tracker(
        role, budget_tokens, config.wrap_up_reserve_tokens
    )
    termination_reason = "token_limit"

    async with ResumableClaudeSession(
        pool,
        lambda token, session_id, resume_id, stderr: build_options(
            config,
            str(scratch_path),
            max(1, tracker.remaining),
            token,
            stderr,
            session_id=session_id,
            resume_session_id=resume_id,
        ),
        session_id=checkpoint.session_id(role),
        reconnects=checkpoint.reconnects(role),
    ) as client:
        checkpoint.save_session(role, client.session_id, client.reconnect_events)
        if not phases:
            phase = await _checkpointed_phase(
                checkpoint,
                role,
                client,
                tracker,
                initial_prompt,
                initial_label,
                tracker.soft_limit_tokens,
            )
            phases.append(phase)

        active = checkpoint.active(role)
        if active is not None and active.get("label") in {
            PHASE_SOLVE,
            PHASE_CRITIQUE,
            PHASE_REVISE,
        }:
            recovered = await _checkpointed_phase(
                checkpoint,
                role,
                client,
                tracker,
                str(active["prompt"]),
                str(active["label"]),
                int(active["stop_at_tokens"]),
            )
            phases.append(recovered)

        no_gap_streak = 0
        for phase in phases:
            if phase.label == PHASE_CRITIQUE:
                if not phase.budget_exhausted and _critique_reports_no_gap(phase.text):
                    no_gap_streak += 1
                else:
                    no_gap_streak = 0
        round_num = sum(phase.label == PHASE_CRITIQUE for phase in phases)

        while not tracker.soft_exhausted and not (
            allow_self_convergence
            and _sequential_self_converged(round_num, no_gap_streak)
        ):
            if phases[-1].label in {PHASE_SOLVE, PHASE_REVISE}:
                round_num += 1
                critique = await _checkpointed_phase(
                    checkpoint,
                    role,
                    client,
                    tracker,
                    critique_prompt(),
                    PHASE_CRITIQUE,
                    tracker.soft_limit_tokens,
                )
                phases.append(critique)
                if not critique.budget_exhausted and _critique_reports_no_gap(
                    critique.text
                ):
                    no_gap_streak += 1
                else:
                    no_gap_streak = 0
                if (
                    allow_self_convergence
                    and _sequential_self_converged(round_num, no_gap_streak)
                ) or tracker.soft_exhausted:
                    break

            if phases[-1].label == PHASE_CRITIQUE:
                revise = await _checkpointed_phase(
                    checkpoint,
                    role,
                    client,
                    tracker,
                    revise_prompt(),
                    PHASE_REVISE,
                    tracker.soft_limit_tokens,
                )
                phases.append(revise)

        termination_reason = (
            "self_converged"
            if allow_self_convergence
            and _sequential_self_converged(round_num, no_gap_streak)
            else "token_limit"
        )

    await _run_strict_wrap_phase(
        config,
        checkpoint,
        role,
        tracker,
        phases,
        scratch_path,
        pool,
        PHASE_WRAP_UP,
        wrap_up_prompt(tracker.remaining),
        config.wrap_up_reserve_tokens,
    )
    return phases, tracker, termination_reason


def _offset_phases(phases: list[PhaseResult], offset: int) -> list[PhaseResult]:
    return [
        dataclasses.replace(
            phase,
            cumulative_output_tokens=phase.cumulative_output_tokens + offset,
        )
        for phase in phases
    ]


async def solve_late_baseline_seed(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    pool: TokenPool,
) -> None:
    """Create a native 3x prefix, retain it, then run the no-hint 4th block."""
    checkpoint_identity = run_checkpoint_identity(config, arm, problem, seed)
    checkpoint = AttemptCheckpoint(checkpoint_identity)
    try:
        prefix_scratch = checkpoint.scratch_dir("prefix")
        prefix_budget = PREFIX_UNITS * config.unit_output_tokens
        prefix_phases, prefix_tracker, _ = await _run_late_sequential_role(
            config,
            checkpoint,
            "prefix",
            prefix_scratch,
            pool,
            prefix_budget,
            task_prompt(problem, None, str(prefix_scratch), prefix_budget),
            PHASE_SOLVE,
            allow_self_convergence=True,
        )
        prefix_session_id = checkpoint.session_id("prefix")
        if prefix_session_id is None:
            raise RuntimeError("Completed late prefix has no native session id")
        source, _ = load_prefix_source(config, problem, seed)
        if source is None:
            source = save_prefix_source(
                config,
                problem,
                seed,
                prefix_scratch,
                prefix_session_id,
                prefix_phases,
                prefix_tracker.spent,
            )
        elif source.session_id != prefix_session_id:
            raise ValueError(
                "Retained prefix belongs to a different native session; "
                "refusing to mix reruns"
            )

        control_session_id = checkpoint.session_id("control")
        if control_session_id is None:
            control_session_id = fork_native_session(
                prefix_scratch, prefix_session_id
            )
            checkpoint.save_session("control", control_session_id, [])
        branch_budget = config.unit_output_tokens
        control_phases, control_tracker, termination_reason = (
            await _run_late_sequential_role(
                config,
                checkpoint,
                "control",
                prefix_scratch,
                pool,
                branch_budget,
                late_continuation_prompt(
                    None, str(prefix_scratch), branch_budget
                ),
                PHASE_REVISE,
                allow_self_convergence=True,
            )
        )
        phases = [
            *prefix_phases,
            *_offset_phases(control_phases, prefix_tracker.spent),
        ]
        expected_output_dir = seed_output_dir(
            config, arm, problem.problem_id, seed
        )
        checkpoint.prepare_completion(
            (expected_output_dir / META_FILENAME).relative_to(RESULTS_ROOT).as_posix()
        )
        output_dir = write_seed_outputs(
            config,
            arm,
            problem,
            seed,
            config.budget_tokens(arm),
            phases,
            prefix_scratch,
            termination_reason=termination_reason,
            provider_session_ids=checkpoint.session_ids(),
            meta_extra={
                "intervention": "native_3x_prefix_then_no_hint_1x",
                "prefix_budget_units": PREFIX_UNITS,
                "prefix_output_tokens_spent": prefix_tracker.spent,
                "continuation_budget_units": 1,
                "continuation_output_tokens_spent": control_tracker.spent,
                "native_prefix_source_retained": True,
                "native_prefix_session_id": source.session_id,
                "control_fork_session_id": control_session_id,
                "late_intervention_protocol": (
                    "matched_native_3x_sibling_forks_v1"
                ),
            },
        )
        log.info(
            "%s/%s seed %d done -> %s",
            arm.name,
            problem.problem_id,
            seed,
            output_dir,
        )
        checkpoint.complete()
    finally:
        checkpoint.close()


async def solve_late_hint_seed(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    pool: TokenPool,
    source: LatePrefixSource,
) -> None:
    """Fork the retained 3x native state and spend one block with h2."""
    checkpoint_identity = run_checkpoint_identity(config, arm, problem, seed)
    checkpoint_identity["late_prefix_source"] = source.provenance
    checkpoint = AttemptCheckpoint(checkpoint_identity)
    try:
        scratch_path = checkpoint.restore_scratch_dir(
            "main", source.scratch_name, source.workspace
        )
        branch_session_id = checkpoint.session_id("main")
        if branch_session_id is None:
            branch_session_id = fork_native_session(scratch_path, source.session_id)
            checkpoint.save_session("main", branch_session_id, [])
        branch_budget = config.unit_output_tokens
        branch_phases, tracker, termination_reason = await _run_late_sequential_role(
            config,
            checkpoint,
            "main",
            scratch_path,
            pool,
            branch_budget,
            late_continuation_prompt(
                hint_for(problem, arm), str(scratch_path), branch_budget
            ),
            PHASE_REVISE,
            allow_self_convergence=True,
        )
        raw_prefix_spent = source.provenance.get("prefix_output_tokens_spent")
        if isinstance(raw_prefix_spent, bool) or not isinstance(
            raw_prefix_spent, int
        ):
            raise ValueError("Retained prefix token count is malformed")
        prefix_spent = raw_prefix_spent
        phases = [
            *source.phases,
            *_offset_phases(branch_phases, prefix_spent),
        ]
        budget_tokens = config.budget_tokens(arm)
        expected_output_dir = seed_output_dir(
            config, arm, problem.problem_id, seed
        )
        checkpoint.prepare_completion(
            (expected_output_dir / META_FILENAME).relative_to(RESULTS_ROOT).as_posix()
        )
        output_dir = write_seed_outputs(
            config,
            arm,
            problem,
            seed,
            budget_tokens,
            phases,
            scratch_path,
            termination_reason=termination_reason,
            provider_session_ids=checkpoint.session_ids(),
            meta_extra={
                **source.provenance,
                "intervention": "oracle_h2_after_native_unaided_3x",
                "continuation_budget_units": 1,
                "continuation_output_tokens_spent": tracker.spent,
                "nominal_cumulative_budget_units": arm.budget_units,
                "hint_fork_session_id": branch_session_id,
                "late_intervention_protocol": (
                    "matched_native_3x_sibling_forks_v1"
                ),
            },
        )
        log.info(
            "%s/%s seed %d done -> %s",
            arm.name,
            problem.problem_id,
            seed,
            output_dir,
        )
        checkpoint.complete()
    finally:
        checkpoint.close()


async def solve_seed(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    pool: TokenPool,
    *,
    worker_model: str | None = None,
    all_problems: list[Problem] | None = None,
    selection_record: dict[str, object] | None = None,
    late_prefix_source: LatePrefixSource | None = None,
) -> None:
    """Run one attempt (one seed) of one problem under one arm; write outputs.

    Working phases (solve, critique, revise) run against the soft limit
    (budget minus the wrap-up reserve). If the soft limit is reached, a final
    wrap-up phase tells the model how many tokens remain and to write down
    what it has; only that phase may spend into the hard budget.
    """
    if arm.name == LATE_BASELINE_ARM_NAME:
        if late_prefix_source is not None:
            raise ValueError("Late baseline does not accept a retained prefix")
        await solve_late_baseline_seed(config, arm, problem, seed, pool)
        return
    if arm.name == LATE_HINT_ARM_NAME:
        if late_prefix_source is None:
            raise ValueError("Late hint requires its retained native 3x source only")
        await solve_late_hint_seed(
            config, arm, problem, seed, pool, late_prefix_source
        )
        return
    if late_prefix_source is not None:
        raise ValueError("Late intervention inputs are valid only for late arms")

    if arm.mode == MODE_PARALLEL:
        await solve_parallel_bank(config, arm, problem, seed, pool)
        return

    if arm.mode == MODE_UNIFORM_COMPRESS:
        if all_problems is None or worker_model is None:
            raise ValueError("Compression requires the frozen examples and worker model")
        await compress_uniform_strategies(
            config, arm, problem, seed, pool, all_problems, worker_model
        )
        return

    if arm.mode in {MODE_SELECTION, MODE_SELECTION_NO_PROBLEM}:
        if selection_record is None:
            raise ValueError("Selection requires a frozen candidate record")
        await run_selection(
            config,
            arm,
            problem,
            seed,
            pool,
            selection_record,
            include_problem=arm.mode == MODE_SELECTION,
        )
        return

    checkpoint_identity = run_checkpoint_identity(config, arm, problem, seed)
    checkpoint = AttemptCheckpoint(checkpoint_identity)
    try:
        if arm.mode in {MODE_UNIFORM_STRATEGY, MODE_UNIFORM_STRATEGY_ONLY}:
            await solve_uniform_strategy_bank(
                config, arm, problem, seed, pool, checkpoint
            )
            return

        scratch_path = checkpoint.scratch_dir("main")
        budget_tokens = config.budget_tokens(arm)
        phases = checkpoint.phases("main")
        tracker = checkpoint.tracker(
            "main", budget_tokens, config.wrap_up_reserve_tokens
        )
        termination_reason: str | None = None

        async with ResumableClaudeSession(
            pool,
            lambda token, session_id, resume_id, stderr: build_options(
                config,
                str(scratch_path),
                max(1, tracker.remaining),
                token,
                stderr,
                session_id=session_id,
                resume_session_id=resume_id,
            ),
            session_id=checkpoint.session_id("main"),
            reconnects=checkpoint.reconnects("main"),
        ) as client:
            checkpoint.save_session("main", client.session_id, client.reconnect_events)
            if not phases:
                initial_prompt = task_prompt(
                    problem,
                    hint_for(problem, arm),
                    str(scratch_path),
                    budget_tokens,
                )
                solve = await _checkpointed_phase(
                    checkpoint,
                    "main",
                    client,
                    tracker,
                    initial_prompt,
                    PHASE_SOLVE,
                    tracker.soft_limit_tokens,
                )
                phases.append(solve)
                log.info(
                    "%s/%s seed %d: solve done (%d/%d tokens)",
                    arm.name,
                    problem.problem_id,
                    seed,
                    tracker.spent,
                    budget_tokens,
                )

            active_main = checkpoint.active("main")
            if active_main is not None and active_main.get("label") in {
                PHASE_CRITIQUE,
                PHASE_REVISE,
            }:
                recovered_phase = await _checkpointed_phase(
                    checkpoint,
                    "main",
                    client,
                    tracker,
                    str(active_main["prompt"]),
                    str(active_main["label"]),
                    int(active_main["stop_at_tokens"]),
                )
                phases.append(recovered_phase)

            if arm.mode == MODE_SEQUENTIAL and not any(
                phase.label == PHASE_WRAP_UP for phase in phases
            ):
                no_gap_streak = 0
                for phase in phases:
                    if phase.label == PHASE_CRITIQUE:
                        if not phase.budget_exhausted and _critique_reports_no_gap(
                            phase.text
                        ):
                            no_gap_streak += 1
                        else:
                            no_gap_streak = 0
                round_num = sum(phase.label == PHASE_CRITIQUE for phase in phases)

                while (
                    not tracker.soft_exhausted
                    and not _sequential_self_converged(round_num, no_gap_streak)
                ):
                    last_label = phases[-1].label
                    if last_label in {PHASE_SOLVE, PHASE_REVISE}:
                        round_num += 1
                        critique = await _checkpointed_phase(
                            checkpoint,
                            "main",
                            client,
                            tracker,
                            critique_prompt(),
                            PHASE_CRITIQUE,
                            tracker.soft_limit_tokens,
                        )
                        phases.append(critique)
                        if not critique.budget_exhausted and _critique_reports_no_gap(
                            critique.text
                        ):
                            no_gap_streak += 1
                        else:
                            no_gap_streak = 0
                        if _sequential_self_converged(round_num, no_gap_streak):
                            log.info(
                                "%s/%s seed %d: stopping after round %d and %d "
                                "consecutive no-gap critiques (%d/%d tokens)",
                                arm.name,
                                problem.problem_id,
                                seed,
                                round_num,
                                no_gap_streak,
                                tracker.spent,
                                budget_tokens,
                            )
                            break
                        if tracker.soft_exhausted:
                            break

                    if phases[-1].label == PHASE_CRITIQUE:
                        revise = await _checkpointed_phase(
                            checkpoint,
                            "main",
                            client,
                            tracker,
                            revise_prompt(),
                            PHASE_REVISE,
                            tracker.soft_limit_tokens,
                        )
                        phases.append(revise)
                        log.info(
                            "%s/%s seed %d: round %d done (%d/%d tokens)",
                            arm.name,
                            problem.problem_id,
                            seed,
                            round_num,
                            tracker.spent,
                            budget_tokens,
                        )

                termination_reason = (
                    "self_converged"
                    if _sequential_self_converged(round_num, no_gap_streak)
                    else "token_limit"
                )

        await _run_strict_wrap_phase(
            config,
            checkpoint,
            "main",
            tracker,
            phases,
            scratch_path,
            pool,
            PHASE_WRAP_UP,
            wrap_up_prompt(tracker.remaining),
            config.wrap_up_reserve_tokens,
        )

        # If the process died after committing wrap-up but before writing the
        # final result, reconstruct the non-token controller field as well.
        if arm.mode == MODE_SEQUENTIAL and termination_reason is None:
            recovered_no_gap_streak = 0
            for phase in phases:
                if phase.label == PHASE_CRITIQUE:
                    if not phase.budget_exhausted and _critique_reports_no_gap(
                        phase.text
                    ):
                        recovered_no_gap_streak += 1
                    else:
                        recovered_no_gap_streak = 0
            termination_reason = (
                "self_converged"
                if _sequential_self_converged(
                    sum(phase.label == PHASE_CRITIQUE for phase in phases),
                    recovered_no_gap_streak,
                )
                else "token_limit"
            )

        expected_output_dir = seed_output_dir(config, arm, problem.problem_id, seed)
        checkpoint.prepare_completion(
            (expected_output_dir / META_FILENAME).relative_to(RESULTS_ROOT).as_posix()
        )
        output_dir = write_seed_outputs(
            config,
            arm,
            problem,
            seed,
            budget_tokens,
            phases,
            scratch_path,
            termination_reason=termination_reason,
            provider_session_ids=checkpoint.session_ids(),
        )
        log.info(
            "%s/%s seed %d done -> %s",
            arm.name,
            problem.problem_id,
            seed,
            output_dir,
        )
        checkpoint.complete()
    finally:
        checkpoint.close()


def select_seeds(arm: ArmConfig, seeds_csv: str | None) -> list[int]:
    """Seed subset for this invocation (pilot runs); must be the arm's seeds."""
    if seeds_csv is None:
        return arm.seeds
    wanted = [int(s) for s in seeds_csv.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in arm.seeds]
    if not wanted or unknown:
        raise ValueError(f"Seeds {unknown} not in arm '{arm.name}' seeds {arm.seeds}")
    return wanted


def select_problems(
    problems: list[Problem], ids_csv: str | None, domain: str | None
) -> list[Problem]:
    """Filter problems by id list and/or domain; fail loud on unknown values."""
    if domain is not None:
        domains = {p.domain for p in problems}
        if domain not in domains:
            raise ValueError(
                f"Unknown domain '{domain}'; dataset has {sorted(domains)}"
            )
        problems = [p for p in problems if p.domain == domain]
    if ids_csv is None:
        return problems
    wanted = [pid.strip() for pid in ids_csv.split(",") if pid.strip()]
    by_id = {p.problem_id: p for p in problems}
    unknown = [pid for pid in wanted if pid not in by_id]
    if unknown:
        raise ValueError(f"Unknown problem ids (after domain filter): {unknown}")
    return [by_id[pid] for pid in wanted]


async def main() -> None:
    """Parse args, then run every pending (problem, seed) attempt of one arm."""
    parser = argparse.ArgumentParser(description="Run one experiment arm.")
    parser.add_argument("--arm", required=True, help="Arm name from config.json")
    parser.add_argument(
        "--problems", default=None, help="Comma-separated problem ids (default: all)"
    )
    parser.add_argument(
        "--domain", default=None, help="Only problems in this domain (default: all)"
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed subset for pilot runs (default: the arm's seeds)",
    )
    parser.add_argument(
        "--model", default=None, help="Solver model override (default: config.json)"
    )
    parser.add_argument(
        "--audit-model",
        default=None,
        help="Judge model override (default: config.json); must differ from the solver",
    )
    parser.add_argument(
        "--worker-model",
        default=None,
        help=(
            "Compression worker override (default: litellm/gpt-5.6-sol). "
            "Selection arms always use the source model."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    config = override_models(load_config(CONFIG_PATH), args.model, args.audit_model)
    log.info("Solver model: %s", config.model)
    if args.arm not in config.arms:
        raise SystemExit(
            f"Unknown arm '{args.arm}'; config defines {sorted(config.arms)}"
        )
    arm = config.arms[args.arm]
    seeds = select_seeds(arm, args.seeds)
    all_problems = load_problems()
    problems = select_problems(all_problems, args.problems, args.domain)
    # Fail fast BEFORE spending tokens if any selected problem lacks the hint.
    for problem in problems:
        hint_for(problem, arm)

    worker_model: str | None = args.worker_model
    if arm.mode == MODE_UNIFORM_COMPRESS:
        worker_model = worker_model or DEFAULT_UNIFORM_COMPRESS_MODEL
    elif arm.mode in {MODE_SELECTION, MODE_SELECTION_NO_PROBLEM}:
        if worker_model is not None:
            raise SystemExit(
                "Selection arms must use the source model; omit --worker-model"
            )
    elif worker_model is not None:
        raise SystemExit("--worker-model is valid only for the compression arm")

    selection_records: dict[str, dict[str, object]] = {}
    if arm.mode in {MODE_SELECTION, MODE_SELECTION_NO_PROBLEM}:
        selection_records = load_selection_candidates(config.model)

    def generation_done(selected_arm: ArmConfig, output_dir: Path) -> bool:
        if selected_arm.mode == MODE_PARALLEL:
            return parallel_bank_done(output_dir)
        if selected_arm.mode == MODE_UNIFORM_STRATEGY:
            return uniform_strategy_bank_done(output_dir)
        if selected_arm.mode == MODE_UNIFORM_STRATEGY_ONLY:
            return uniform_strategy_only_done(output_dir)
        return seed_done(output_dir)

    candidate_pairs = [
        (problem, seed)
        for problem in problems
        for seed in seeds
        if not generation_done(
            arm, seed_output_dir(config, arm, problem.problem_id, seed)
        )
    ]
    skipped_count = 0
    late_prefix_sources: dict[tuple[str, int], LatePrefixSource] = {}
    if arm.name == LATE_HINT_ARM_NAME:
        pending = []
        skipped = []
        for problem, seed in candidate_pairs:
            source, reason = load_prefix_source(config, problem, seed)
            if source is None:
                skipped.append(f"{problem.problem_id}/seed_{seed} ({reason})")
                continue
            late_prefix_sources[(problem.problem_id, seed)] = source
            pending.append((problem, seed))
        if skipped:
            skipped_count = len(skipped)
            log.warning(
                "Late hint skipped %d trajectory/trajectories without a retained "
                "native 3x prefix: %s",
                len(skipped),
                ", ".join(skipped),
            )
    elif arm.name == LATE_BASELINE_ARM_NAME:
        pending = candidate_pairs
    elif arm.mode == MODE_UNIFORM_COMPRESS:
        pending = []
        skipped: list[str] = []
        for problem, seed in candidate_pairs:
            eligible, reason = compression_source_status(
                config, problem.problem_id, seed
            )
            if eligible:
                pending.append((problem, seed))
            else:
                skipped.append(f"{problem.problem_id}/seed_{seed} ({reason})")
        if skipped:
            skipped_count = len(skipped)
            log.warning(
                "Compression skipped %d attempt(s) without at least three available "
                "planner proposals: %s",
                len(skipped),
                ", ".join(skipped),
            )
    elif arm.mode in {MODE_SELECTION, MODE_SELECTION_NO_PROBLEM}:
        pending = []
        skipped = []
        for problem, seed in candidate_pairs:
            if problem.problem_id in selection_records:
                pending.append((problem, seed))
            else:
                skipped.append(f"{problem.problem_id}/seed_{seed}")
        if skipped:
            skipped_count = len(skipped)
            log.warning(
                "Selection skipped %d attempt(s) without a frozen candidate set: %s",
                len(skipped),
                ", ".join(skipped),
            )
    else:
        pending = candidate_pairs
    total = len(problems) * len(seeds)
    completed_count = total - len(candidate_pairs)
    if skipped_count:
        log.info(
            "Arm %s: %d attempts to run, %d already done, %d unavailable skipped",
            arm.name,
            len(pending),
            completed_count,
            skipped_count,
        )
    else:
        log.info(
            "Arm %s: %d attempts to run, %d already done",
            arm.name,
            len(pending),
            completed_count,
        )
    active_model = worker_model or config.model
    pool = TokenPool.from_env(token_env_name(active_model))
    tasks = [
        lambda p=problem, s=seed: solve_seed(
            config,
            arm,
            p,
            s,
            pool,
            worker_model=worker_model,
            all_problems=all_problems,
            selection_record=selection_records.get(p.problem_id),
            late_prefix_source=late_prefix_sources.get((p.problem_id, s)),
        )
        for problem, seed in pending
    ]
    # Parallel and Uniform Strategy banks launch their own executor calls.
    # Running one bank controller at a time preserves the global cap of eight.
    outer_limit = (
        1
        if arm.mode in {
            MODE_PARALLEL,
            MODE_UNIFORM_STRATEGY,
            MODE_UNIFORM_STRATEGY_ONLY,
        }
        else config.max_concurrency
    )
    await run_all(tasks, outer_limit)


if __name__ == "__main__":
    anyio.run(main)
