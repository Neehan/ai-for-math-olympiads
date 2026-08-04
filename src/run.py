"""Entrypoint: run ONE experiment arm over the problem set (resumable).

Usage:
    python -m src.run --arm baseline
    python -m src.run --arm hint --problems usamo-2026-3,china-2026-5

An attempt = one (problem, seed) pair. Mode 'single' runs one solve phase;
mode 'sequential' runs solve -> (critique -> revise)* until the attempt's
output-token budget is exhausted. Completed attempts (meta.json present) are
skipped, so an interrupted run resumes cleanly.
"""

import argparse
import logging

import anyio

from claude_agent_sdk import ClaudeSDKClient

from src.concurrency import run_all
from src.config import load_config
from src.constants import (
    CONFIG_PATH,
    HINT_H1,
    HINT_NONE,
    LOG_FORMAT,
    LOG_LEVEL,
    MODE_SEQUENTIAL,
    PHASE_CRITIQUE,
    PHASE_REVISE,
    PHASE_SOLVE,
)
from src.models import ArmConfig, ExperimentConfig, PhaseResult, Problem
from src.prompts import critique_prompt, revise_prompt, task_prompt
from src.solver import BudgetTracker, build_options, run_phase, run_resumable
from src.storage import (
    fresh_scratch_dir,
    load_problems,
    seed_done,
    seed_output_dir,
    write_seed_outputs,
)
from src.token_pool import TokenPool

log = logging.getLogger("run")


def hint_for(problem: Problem, arm: ArmConfig) -> str | None:
    """The hint text this arm injects for this problem (None for no-hint)."""
    if arm.hint == HINT_NONE:
        return None
    hint = problem.hint_h1 if arm.hint == HINT_H1 else problem.hint_h2
    if hint is None:
        raise ValueError(
            f"Arm '{arm.name}' needs hint '{arm.hint}' but problem "
            f"'{problem.problem_id}' has none in problems.jsonl"
        )
    return hint


async def solve_seed(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    oauth_token: str,
) -> None:
    """Run one attempt (one seed) of one problem under one arm; write outputs."""
    scratch_path = fresh_scratch_dir(arm, problem.problem_id, seed)
    budget_tokens = config.budget_tokens(arm)
    tracker = BudgetTracker(budget_tokens)
    options = build_options(config, str(scratch_path), budget_tokens, oauth_token)
    phases: list[PhaseResult] = []

    async with ClaudeSDKClient(options=options) as client:
        solve = await run_phase(
            client,
            task_prompt(problem, hint_for(problem, arm), str(scratch_path), budget_tokens),
            PHASE_SOLVE,
            tracker,
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
            while not tracker.exhausted:
                round_num += 1
                critique = await run_phase(
                    client, critique_prompt(), PHASE_CRITIQUE, tracker
                )
                phases.append(critique)
                if tracker.exhausted:
                    break
                revise = await run_phase(client, revise_prompt(), PHASE_REVISE, tracker)
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

    output_dir = write_seed_outputs(
        config, arm, problem, seed, budget_tokens, phases, scratch_path
    )
    log.info("%s/%s seed %d done -> %s", arm.name, problem.problem_id, seed, output_dir)


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
    args = parser.parse_args()

    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    config = load_config(CONFIG_PATH)
    if args.arm not in config.arms:
        raise SystemExit(
            f"Unknown arm '{args.arm}'; config defines {sorted(config.arms)}"
        )
    arm = config.arms[args.arm]
    problems = select_problems(load_problems(), args.problems, args.domain)
    # Fail fast BEFORE spending tokens if any selected problem lacks the hint.
    for problem in problems:
        hint_for(problem, arm)

    pending = [
        (problem, seed)
        for problem in problems
        for seed in arm.seeds
        if not seed_done(seed_output_dir(config, arm, problem.problem_id, seed))
    ]
    total = len(problems) * len(arm.seeds)
    log.info(
        "Arm %s: %d attempts to run, %d already done",
        arm.name,
        len(pending),
        total - len(pending),
    )
    pool = TokenPool.from_env()
    tasks = [
        lambda p=problem, s=seed: run_resumable(
            pool, lambda token: solve_seed(config, arm, p, s, token)
        )
        for problem, seed in pending
    ]
    await run_all(tasks, config.max_concurrency)


if __name__ == "__main__":
    anyio.run(main)
