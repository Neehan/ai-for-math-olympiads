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

from claude_agent_sdk import ClaudeSDKClient

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
    MODE_IDEASEARCH,
    MODE_SEQUENTIAL,
    PHASE_CRITIQUE,
    PHASE_REVISE,
    PHASE_PLAN,
    PHASE_PLAN_WRAP_UP,
    PHASE_SOLVE,
    PHASE_WRAP_UP,
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
    StderrTail,
    build_options,
    run_phase,
    run_resumable,
    spend_limit_guard,
    token_env_name,
)
from src.storage import (
    clear_scratch_root,
    fresh_scratch_dir,
    load_problems,
    seed_done,
    seed_output_dir,
    write_seed_outputs,
)
from src.token_pool import TokenPool

log = logging.getLogger("run")


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


async def solve_ideasearch_seed(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    oauth_token: str,
) -> None:
    """Run one isolated IdeaSearch branch: planner (20k) -> executor (180k).

    The two agents use fresh SDK sessions and distinct scratch directories.
    Unused planner tokens are not transferred to the executor, so every branch
    has the same pre-registered caps and never exceeds one 200k token-unit.
    """
    plan_budget = config.ideasearch_plan_tokens
    proof_budget = config.unit_output_tokens - plan_budget
    plan_scratch_path = fresh_scratch_dir()
    proof_scratch_path = fresh_scratch_dir()

    plan_tracker = BudgetTracker(
        plan_budget, config.ideasearch_plan_wrap_up_reserve_tokens
    )
    plan_stderr = StderrTail()
    plan_options = build_options(
        config, str(plan_scratch_path), plan_budget, oauth_token, plan_stderr
    )
    plan_phases: list[PhaseResult] = []
    async with (
        spend_limit_guard(plan_stderr),
        ClaudeSDKClient(options=plan_options) as planner,
    ):
        plan = await run_phase(
            planner,
            ideasearch_plan_prompt(
                problem, str(plan_scratch_path), plan_budget
            ),
            PHASE_PLAN,
            plan_tracker,
            plan_tracker.soft_limit_tokens,
        )
        plan_phases.append(plan)
        if plan_tracker.soft_exhausted:
            plan_wrap = await run_phase(
                planner,
                ideasearch_plan_wrap_up_prompt(plan_tracker.remaining),
                PHASE_PLAN_WRAP_UP,
                plan_tracker,
                plan_tracker.budget_tokens,
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

    proof_tracker = BudgetTracker(proof_budget, config.wrap_up_reserve_tokens)
    proof_stderr = StderrTail()
    proof_options = build_options(
        config, str(proof_scratch_path), proof_budget, oauth_token, proof_stderr
    )
    proof_phases: list[PhaseResult] = []
    async with (
        spend_limit_guard(proof_stderr),
        ClaudeSDKClient(options=proof_options) as executor,
    ):
        solve = await run_phase(
            executor,
            ideasearch_execute_prompt(
                problem,
                proposed_strategy,
                str(proof_scratch_path),
                proof_budget,
            ),
            PHASE_SOLVE,
            proof_tracker,
            proof_tracker.soft_limit_tokens,
        )
        proof_phases.append(solve)
        if proof_tracker.soft_exhausted:
            wrap_up = await run_phase(
                executor,
                wrap_up_prompt(proof_tracker.remaining),
                PHASE_WRAP_UP,
                proof_tracker,
                proof_tracker.budget_tokens,
            )
            proof_phases.append(wrap_up)

    phases = plan_phases + _offset_cumulative_tokens(
        proof_phases, plan_tracker.spent
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


async def solve_seed(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    oauth_token: str,
) -> None:
    """Run one attempt (one seed) of one problem under one arm; write outputs.

    Working phases (solve, critique, revise) run against the soft limit
    (budget minus the wrap-up reserve). If the soft limit is reached, a final
    wrap-up phase tells the model how many tokens remain and to write down
    what it has; only that phase may spend into the hard budget.
    """
    if arm.mode == MODE_IDEASEARCH:
        await solve_ideasearch_seed(config, arm, problem, seed, oauth_token)
        return

    scratch_path = fresh_scratch_dir()
    budget_tokens = config.budget_tokens(arm)
    tracker = BudgetTracker(budget_tokens, config.wrap_up_reserve_tokens)
    stderr_tail = StderrTail()
    options = build_options(
        config, str(scratch_path), budget_tokens, oauth_token, stderr_tail
    )
    phases: list[PhaseResult] = []

    async with spend_limit_guard(stderr_tail), ClaudeSDKClient(options=options) as client:
        solve = await run_phase(
            client,
            task_prompt(problem, hint_for(problem, arm), str(scratch_path), budget_tokens),
            PHASE_SOLVE,
            tracker,
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

        if arm.mode == MODE_SEQUENTIAL:
            round_num = 0
            while not tracker.soft_exhausted and round_num < config.sequential_max_rounds:
                round_num += 1
                critique = await run_phase(
                    client,
                    critique_prompt(),
                    PHASE_CRITIQUE,
                    tracker,
                    tracker.soft_limit_tokens,
                )
                phases.append(critique)
                if tracker.soft_exhausted:
                    break
                revise = await run_phase(
                    client,
                    revise_prompt(),
                    PHASE_REVISE,
                    tracker,
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

        if tracker.soft_exhausted:
            wrap_up = await run_phase(
                client,
                wrap_up_prompt(tracker.remaining),
                PHASE_WRAP_UP,
                tracker,
                tracker.budget_tokens,
            )
            phases.append(wrap_up)

    output_dir = write_seed_outputs(
        config, arm, problem, seed, budget_tokens, phases, scratch_path
    )
    log.info("%s/%s seed %d done -> %s", arm.name, problem.problem_id, seed, output_dir)


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
    clear_scratch_root()
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
        lambda p=problem, s=seed: run_resumable(
            pool, lambda token: solve_seed(config, arm, p, s, token)
        )
        for problem, seed in pending
    ]
    await run_all(tasks, config.max_concurrency)


if __name__ == "__main__":
    anyio.run(main)
