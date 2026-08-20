"""Entrypoint: run ONE experiment arm over the problem set (resumable).

Usage:
    python -m src.run --arm baseline
    python -m src.run --arm hint --problems usamo-2026-3,china-2026-5

An attempt = one (problem, seed) pair. Mode 'single' runs one solve phase;
mode 'sequential' runs solve -> (critique -> revise)*; mode 'parallel' forms
one bank of eight fresh independent 1x attempts;
mode 'uniform_strategy' runs one shared planner followed by eight fresh proof
executors. Completed attempts (meta.json present) are skipped, so an
interrupted run resumes cleanly.
"""

import argparse
import dataclasses
import logging
import re
from pathlib import Path

import anyio

from src.checkpoint import (
    AttemptCheckpoint,
    progress_tool_calls,
    protocol_fingerprint,
    tool_calls_from_records,
)
from src.concurrency import run_all
from src.config import load_config, override_models
from src.constants import (
    CONFIG_PATH,
    HINT_H1,
    HINT_H2,
    HINT_H3,
    HINT_NONE,
    LOG_FORMAT,
    LOG_LEVEL,
    META_FILENAME,
    MODE_PARALLEL,
    MODE_SEQUENTIAL,
    MODE_UNIFORM_STRATEGY,
    NO_GENUINE_GAP_MARKER,
    PHASE_CRITIQUE,
    PHASE_REVISE,
    PHASE_PLAN,
    PHASE_PLAN_WRAP_UP,
    PHASE_SOLVE,
    PHASE_WRAP_UP,
    RESULTS_ROOT,
    RUN_REFERENCE_FILENAME,
    SEQUENTIAL_MIN_ROUNDS_BEFORE_CONVERGENCE,
    SEQUENTIAL_NO_GAP_STREAK_TO_STOP,
)
from src.models import ArmConfig, ExperimentConfig, PhaseResult, Problem
from src.openrouter_routing import route_for
from src.prompts import (
    critique_prompt,
    revise_prompt,
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
    parallel_bank_done,
    seed_done,
    seed_output_dir,
    write_parallel_bank_meta,
    write_uniform_strategy_bank_meta,
    write_uniform_strategy_plan_artifacts,
    write_uniform_strategy_planner_failure,
    write_seed_outputs,
)
from src.token_pool import TokenPool

log = logging.getLogger("run")


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
    executor_budget = (total_budget - plan_budget) // config.uniform_strategy_branches
    bank_dir = seed_output_dir(config, arm, problem.problem_id, seed)
    plan_scratch_path = checkpoint.scratch_dir("plan")
    plan_phases = checkpoint.phases("plan")
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
        detail = " (h1/placebo is not authored yet)" if arm.hint == HINT_H1 else ""
        raise ValueError(
            f"Arm '{arm.name}' needs hint tier '{arm.hint}' but problem "
            f"'{problem.problem_id}' has no such hint in the dataset{detail}"
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
        "arm": dataclasses.asdict(arm),
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
            agent_settings_path(config.model)
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
    return identity


def _sequential_self_converged(round_num: int, no_gap_streak: int) -> bool:
    """Whether the pre-registered sequential stopping rule is satisfied."""
    return (
        round_num >= SEQUENTIAL_MIN_ROUNDS_BEFORE_CONVERGENCE
        and no_gap_streak >= SEQUENTIAL_NO_GAP_STREAK_TO_STOP
    )


async def solve_seed(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    pool: TokenPool,
) -> None:
    """Run one attempt (one seed) of one problem under one arm; write outputs.

    Working phases (solve, critique, revise) run against the soft limit
    (budget minus the wrap-up reserve). If the soft limit is reached, a final
    wrap-up phase tells the model how many tokens remain and to write down
    what it has; only that phase may spend into the hard budget.
    """
    if arm.mode == MODE_PARALLEL:
        await solve_parallel_bank(config, arm, problem, seed, pool)
        return

    checkpoint = AttemptCheckpoint(run_checkpoint_identity(config, arm, problem, seed))
    try:
        if arm.mode == MODE_UNIFORM_STRATEGY:
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
                solve = await _checkpointed_phase(
                    checkpoint,
                    "main",
                    client,
                    tracker,
                    task_prompt(
                        problem,
                        hint_for(problem, arm),
                        str(scratch_path),
                        budget_tokens,
                    ),
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
    problems = select_problems(load_problems(), args.problems, args.domain)
    # Fail fast BEFORE spending tokens if any selected problem lacks the hint.
    for problem in problems:
        hint_for(problem, arm)

    def generation_done(selected_arm: ArmConfig, output_dir: Path) -> bool:
        return (
            parallel_bank_done(output_dir)
            if selected_arm.mode == MODE_PARALLEL
            else seed_done(output_dir)
        )

    pending = [
        (problem, seed)
        for problem in problems
        for seed in seeds
        if not generation_done(
            arm, seed_output_dir(config, arm, problem.problem_id, seed)
        )
    ]
    total = len(problems) * len(seeds)
    log.info(
        "Arm %s: %d attempts to run, %d already done",
        arm.name,
        len(pending),
        total - len(pending),
    )
    pool = TokenPool.from_env(token_env_name(config.model))
    tasks = [
        lambda p=problem, s=seed: solve_seed(config, arm, p, s, pool)
        for problem, seed in pending
    ]
    # Parallel and Uniform Strategy banks launch their own executor calls.
    # Running one bank controller at a time preserves the global cap of eight.
    outer_limit = (
        1
        if arm.mode in {MODE_PARALLEL, MODE_UNIFORM_STRATEGY}
        else config.max_concurrency
    )
    await run_all(tasks, outer_limit)


if __name__ == "__main__":
    anyio.run(main)
