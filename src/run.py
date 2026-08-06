"""Entrypoint: run ONE experiment arm over the problem set (resumable).

Usage:
    python -m src.run --arm baseline
    python -m src.run --arm hint --problems usamo-2026-3,china-2026-5

An attempt = one (problem, seed) pair. Mode 'single' runs one solve phase;
mode 'sequential' runs solve -> (critique -> revise)*; mode 'ideasearch'
runs a fresh planner followed by a fresh proof executor. Completed attempts
(meta.json present) are skipped, so an interrupted run resumes cleanly.
"""

import argparse
import dataclasses
import logging

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
    MODE_IDEASEARCH,
    MODE_SEQUENTIAL,
    NO_GENUINE_GAP_MARKER,
    PHASE_CRITIQUE,
    PHASE_REVISE,
    PHASE_PLAN,
    PHASE_PLAN_WRAP_UP,
    PHASE_SOLVE,
    PHASE_WRAP_UP,
    RESULTS_ROOT,
    SEQUENTIAL_NO_GAP_STREAK_TO_STOP,
)
from src.models import ArmConfig, ExperimentConfig, PhaseResult, Problem
from src.prompts import (
    critique_prompt,
    ideasearch_execute_prompt,
    ideasearch_plan_prompt,
    ideasearch_plan_wrap_up_prompt,
    revise_prompt,
    task_prompt,
    wrap_up_prompt,
)
from src.solver import (
    BudgetTracker,
    ResumableClaudeSession,
    STOP_BUDGET_EXHAUSTED,
    build_options,
    process_recovery_prompt,
    run_phase,
    token_env_name,
)
from src.storage import (
    load_problems,
    seed_done,
    seed_output_dir,
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
        active = checkpoint.begin_phase(
            role, label, prompt, stop_at_tokens, tracker
        )
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
                process_resume_count=int(active.get("process_resume_count", 0))
                + 1,
                discarded_output_text=str(
                    active.get("discarded_output_text", "")
                ),
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
        reconnect_start=int(active.get("reconnect_start", 0)),
        on_progress=save_progress,
        on_complete=finish,
    )


def _proposed_strategy_text(phases: list[PhaseResult]) -> str:
    """Last complete planner response, falling back to interrupted text."""
    for phase in reversed(phases):
        if not phase.budget_exhausted and phase.text.strip():
            return phase.text
    for phase in reversed(phases):
        if phase.text.strip():
            return phase.text
    raise RuntimeError("IdeaSearch planner produced no strategy text")


def _offset_cumulative_tokens(
    phases: list[PhaseResult], offset: int
) -> list[PhaseResult]:
    """Express a fresh executor's token counts in the branch-wide coordinate."""
    return [
        dataclasses.replace(
            phase,
            cumulative_output_tokens=phase.cumulative_output_tokens + offset,
        )
        for phase in phases
    ]


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


async def solve_ideasearch_seed(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    pool: TokenPool,
    checkpoint: AttemptCheckpoint,
) -> None:
    """Run one isolated IdeaSearch branch: planner (20k) -> executor (180k).

    The two agents use fresh SDK sessions and distinct scratch directories.
    Unused planner tokens are not transferred to the executor, so every branch
    has the same pre-registered caps and never exceeds one 200k token-unit.
    """
    plan_budget = config.ideasearch_plan_tokens
    proof_budget = config.unit_output_tokens - plan_budget
    plan_scratch_path = checkpoint.scratch_dir("plan")
    proof_scratch_path = checkpoint.scratch_dir("proof")

    # Load phase ledgers first: this reconciles a process death between the
    # durable phase commit and the controller-state advance.
    plan_phases = checkpoint.phases("plan")
    plan_tracker = checkpoint.tracker(
        "plan", plan_budget, config.ideasearch_plan_wrap_up_reserve_tokens
    )
    plan_has_wrap = any(
        phase.label == PHASE_PLAN_WRAP_UP for phase in plan_phases
    )
    plan_needs_session = (
        checkpoint.active("plan") is not None
        or not plan_phases
        or (
            plan_tracker.soft_exhausted
            and not plan_tracker.exhausted
            and not plan_has_wrap
        )
    )
    if plan_needs_session:
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
                    ideasearch_plan_prompt(
                        problem,
                        str(plan_scratch_path),
                        plan_budget,
                        config.ideasearch_plan_wrap_up_reserve_tokens,
                    ),
                    PHASE_PLAN,
                    plan_tracker.soft_limit_tokens,
                )
                plan_phases.append(plan)
            active_plan = checkpoint.active("plan")
            active_is_plan_wrap = (
                active_plan is not None
                and active_plan.get("label") == PHASE_PLAN_WRAP_UP
            )
            if active_is_plan_wrap or (
                plan_tracker.soft_exhausted
                and not plan_tracker.exhausted
                and not any(
                    phase.label == PHASE_PLAN_WRAP_UP for phase in plan_phases
                )
            ):
                plan_wrap = await _checkpointed_phase(
                    checkpoint,
                    "plan",
                    planner,
                    plan_tracker,
                    (
                        str(active_plan["prompt"])
                        if active_is_plan_wrap and active_plan is not None
                        else ideasearch_plan_wrap_up_prompt(plan_tracker.remaining)
                    ),
                    PHASE_PLAN_WRAP_UP,
                    (
                        int(active_plan["stop_at_tokens"])
                        if active_is_plan_wrap and active_plan is not None
                        else plan_tracker.budget_tokens
                    ),
                )
                plan_phases.append(plan_wrap)

    proposed_strategy = _proposed_strategy_text(plan_phases)
    log.info(
        "%s/%s seed %d: plan done (%d/%d tokens)",
        arm.name,
        problem.problem_id,
        seed,
        plan_tracker.spent,
        plan_budget,
    )

    proof_phases = checkpoint.phases("proof")
    proof_tracker = checkpoint.tracker(
        "proof", proof_budget, config.wrap_up_reserve_tokens
    )
    proof_has_wrap = any(phase.label == PHASE_WRAP_UP for phase in proof_phases)
    proof_needs_session = (
        checkpoint.active("proof") is not None
        or not proof_phases
        or (
            proof_tracker.soft_exhausted
            and not proof_tracker.exhausted
            and not proof_has_wrap
        )
    )
    if proof_needs_session:
        async with ResumableClaudeSession(
            pool,
            lambda token, session_id, resume_id, stderr: build_options(
                config,
                str(proof_scratch_path),
                max(1, proof_tracker.remaining),
                token,
                stderr,
                session_id=session_id,
                resume_session_id=resume_id,
            ),
            session_id=checkpoint.session_id("proof"),
            reconnects=checkpoint.reconnects("proof"),
        ) as executor:
            checkpoint.save_session(
                "proof", executor.session_id, executor.reconnect_events
            )
            if not proof_phases:
                solve = await _checkpointed_phase(
                    checkpoint,
                    "proof",
                    executor,
                    proof_tracker,
                    ideasearch_execute_prompt(
                        problem,
                        proposed_strategy,
                        str(proof_scratch_path),
                        proof_budget,
                    ),
                    PHASE_SOLVE,
                    proof_tracker.soft_limit_tokens,
                )
                proof_phases.append(solve)
            active_proof = checkpoint.active("proof")
            active_is_wrap = (
                active_proof is not None
                and active_proof.get("label") == PHASE_WRAP_UP
            )
            if active_is_wrap or (
                proof_tracker.soft_exhausted
                and not proof_tracker.exhausted
                and not any(phase.label == PHASE_WRAP_UP for phase in proof_phases)
            ):
                wrap_up = await _checkpointed_phase(
                    checkpoint,
                    "proof",
                    executor,
                    proof_tracker,
                    (
                        str(active_proof["prompt"])
                        if active_is_wrap and active_proof is not None
                        else wrap_up_prompt(proof_tracker.remaining)
                    ),
                    PHASE_WRAP_UP,
                    (
                        int(active_proof["stop_at_tokens"])
                        if active_is_wrap and active_proof is not None
                        else proof_tracker.budget_tokens
                    ),
                )
                proof_phases.append(wrap_up)

    phases = plan_phases + _offset_cumulative_tokens(
        proof_phases, plan_tracker.spent
    )
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
        config.unit_output_tokens,
        phases,
        proof_scratch_path,
        plan_scratch_path=plan_scratch_path,
        provider_session_ids=checkpoint.session_ids(),
    )
    log.info(
        "%s/%s seed %d done (%d/%d tokens) -> %s",
        arm.name,
        problem.problem_id,
        seed,
        plan_tracker.spent + proof_tracker.spent,
        config.unit_output_tokens,
        output_dir,
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
        detail = (
            " (h1/placebo is not authored yet)" if arm.hint == HINT_H1 else ""
        )
        raise ValueError(
            f"Arm '{arm.name}' needs hint tier '{arm.hint}' but problem "
            f"'{problem.problem_id}' has no such hint in the dataset{detail}"
        )
    return hint


def run_checkpoint_identity(
    config: ExperimentConfig, arm: ArmConfig, problem: Problem, seed: int
) -> dict[str, object]:
    """Canonical identity shared by normal runs and audited legacy migration."""
    return {
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
        "ideasearch_plan_tokens": config.ideasearch_plan_tokens,
        "ideasearch_plan_wrap_up_reserve_tokens": (
            config.ideasearch_plan_wrap_up_reserve_tokens
        ),
        "max_turns_per_phase": config.max_turns_per_phase,
        "protocol_fingerprint": protocol_fingerprint(),
    }


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
    checkpoint = AttemptCheckpoint(
        run_checkpoint_identity(config, arm, problem, seed)
    )
    try:
        if arm.mode == MODE_IDEASEARCH:
            await solve_ideasearch_seed(
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
            checkpoint.save_session(
                "main", client.session_id, client.reconnect_events
            )
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
                        if (
                            not phase.budget_exhausted
                            and _critique_reports_no_gap(phase.text)
                        ):
                            no_gap_streak += 1
                        else:
                            no_gap_streak = 0
                round_num = sum(
                    phase.label == PHASE_CRITIQUE for phase in phases
                )

                while (
                    not tracker.soft_exhausted
                    and no_gap_streak < SEQUENTIAL_NO_GAP_STREAK_TO_STOP
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
                        if (
                            not critique.budget_exhausted
                            and _critique_reports_no_gap(critique.text)
                        ):
                            no_gap_streak += 1
                        else:
                            no_gap_streak = 0
                        if (
                            no_gap_streak
                            >= SEQUENTIAL_NO_GAP_STREAK_TO_STOP
                        ):
                            log.info(
                                "%s/%s seed %d: stopping after %d consecutive "
                                "no-gap critiques (%d/%d tokens)",
                                arm.name,
                                problem.problem_id,
                                seed,
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
                    if no_gap_streak >= SEQUENTIAL_NO_GAP_STREAK_TO_STOP
                    else "token_limit"
                )

            active_main = checkpoint.active("main")
            active_is_wrap = (
                active_main is not None
                and active_main.get("label") == PHASE_WRAP_UP
            )
            if active_is_wrap or (
                tracker.soft_exhausted
                and not tracker.exhausted
                and not any(phase.label == PHASE_WRAP_UP for phase in phases)
            ):
                wrap_up = await _checkpointed_phase(
                    checkpoint,
                    "main",
                    client,
                    tracker,
                    (
                        str(active_main["prompt"])
                        if active_is_wrap and active_main is not None
                        else wrap_up_prompt(tracker.remaining)
                    ),
                    PHASE_WRAP_UP,
                    (
                        int(active_main["stop_at_tokens"])
                        if active_is_wrap and active_main is not None
                        else tracker.budget_tokens
                    ),
                )
                phases.append(wrap_up)

        # If the process died after committing wrap-up but before writing the
        # final result, reconstruct the non-token controller field as well.
        if arm.mode == MODE_SEQUENTIAL and termination_reason is None:
            recovered_no_gap_streak = 0
            for phase in phases:
                if phase.label == PHASE_CRITIQUE:
                    if (
                        not phase.budget_exhausted
                        and _critique_reports_no_gap(phase.text)
                    ):
                        recovered_no_gap_streak += 1
                    else:
                        recovered_no_gap_streak = 0
            termination_reason = (
                "self_converged"
                if recovered_no_gap_streak >= SEQUENTIAL_NO_GAP_STREAK_TO_STOP
                else "token_limit"
            )

        expected_output_dir = seed_output_dir(
            config, arm, problem.problem_id, seed
        )
        checkpoint.prepare_completion(
            (expected_output_dir / META_FILENAME)
            .relative_to(RESULTS_ROOT)
            .as_posix()
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
            raise ValueError(f"Unknown domain '{domain}'; dataset has {sorted(domains)}")
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

    pending = [
        (problem, seed)
        for problem in problems
        for seed in seeds
        if not seed_done(seed_output_dir(config, arm, problem.problem_id, seed))
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
    await run_all(tasks, config.max_concurrency)


if __name__ == "__main__":
    anyio.run(main)
