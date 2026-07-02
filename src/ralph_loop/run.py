"""Ralph-loop harness: iterative self-refinement in one persistent session.

One client session per problem: the agent produces an initial solution, then is
re-prompted to critique and improve its own work for a fixed number of
iterations, keeping full context across the loop. Every iteration is recorded so
the refinement trajectory is auditable; the last iteration is the final answer.

Usage:
    python -m src.ralph_loop.run
"""

import anyio

from claude_agent_sdk import ClaudeSDKClient

from src.shared.concurrency import run_all
from src.shared.constants import (
    MAX_TURNS_PER_ATTEMPT,
    MODEL,
    RALPH_ITERATIONS,
    RALPH_LOOP_DIR,
)
from src.shared.io_utils import (
    load_problems,
    result_exists,
    scratch_dir,
    write_problem_run,
)
from src.shared.logging_setup import configure_logging, get_logger
from src.shared.models import Problem, ProblemRun
from src.shared.prompts import ralph_refine_prompt, task_prompt
from src.shared.solver import build_options, run_attempt

log = get_logger(RALPH_LOOP_DIR)


async def solve_problem(problem: Problem) -> None:
    """Run the refinement loop on one problem and write one markdown file."""
    cwd = str(scratch_dir(RALPH_LOOP_DIR, problem.problem_id))
    options = build_options(cwd=cwd, max_turns=MAX_TURNS_PER_ATTEMPT)
    run = ProblemRun(
        problem_id=problem.problem_id,
        harness=RALPH_LOOP_DIR,
        model=MODEL,
    )
    labels: list[str] = []
    async with ClaudeSDKClient(options=options) as client:
        first = await run_attempt(
            client, task_prompt(problem, cwd, MAX_TURNS_PER_ATTEMPT)
        )
        run.attempts.append(first)
        labels.append("Iteration 1 (initial)")
        log.info("%s iteration 1/%d", problem.problem_id, RALPH_ITERATIONS)

        for iteration in range(2, RALPH_ITERATIONS + 1):
            refined = await run_attempt(
                client,
                ralph_refine_prompt(
                    iteration, RALPH_ITERATIONS, MAX_TURNS_PER_ATTEMPT
                ),
            )
            run.attempts.append(refined)
            labels.append(f"Iteration {iteration} (refine)")
            log.info(
                "%s iteration %d/%d", problem.problem_id, iteration, RALPH_ITERATIONS
            )

    path = write_problem_run(run, problem, attempt_labels=labels)
    log.info("%s done -> %s", problem.problem_id, path)


async def main() -> None:
    """Run the Ralph-loop harness over all problems (resumable, concurrent).

    Different problems run in parallel; iterations within a problem stay
    sequential (each refines the previous one).
    """
    configure_logging()
    problems = load_problems()
    pending = [p for p in problems if not result_exists(RALPH_LOOP_DIR, p.problem_id)]
    skipped = len(problems) - len(pending)
    log.info("%d problems to run, %d already done", len(pending), skipped)
    tasks = [lambda p=p: solve_problem(p) for p in pending]
    await run_all(tasks)


if __name__ == "__main__":
    anyio.run(main)
