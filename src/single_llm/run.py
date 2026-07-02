"""Single-LLM harness: one attempt per problem.

Usage:
    python -m src.single_llm.run
"""

import anyio

from claude_agent_sdk import ClaudeSDKClient

from src.shared.concurrency import run_all
from src.shared.constants import MAX_TURNS_PER_ATTEMPT, MODEL, SINGLE_LLM_DIR
from src.shared.io_utils import (
    load_problems,
    result_exists,
    scratch_dir,
    write_problem_run,
)
from src.shared.logging_setup import configure_logging, get_logger
from src.shared.models import Problem, ProblemRun
from src.shared.prompts import task_prompt
from src.shared.solver import build_options, run_attempt

log = get_logger(SINGLE_LLM_DIR)


async def solve_problem(problem: Problem) -> None:
    """Run one attempt on one problem and write its markdown file."""
    cwd = str(scratch_dir(SINGLE_LLM_DIR, problem.problem_id))
    options = build_options(cwd=cwd, max_turns=MAX_TURNS_PER_ATTEMPT)
    run = ProblemRun(
        problem_id=problem.problem_id,
        harness=SINGLE_LLM_DIR,
        model=MODEL,
    )
    async with ClaudeSDKClient(options=options) as client:
        attempt = await run_attempt(
            client, task_prompt(problem, cwd, MAX_TURNS_PER_ATTEMPT)
        )
        run.attempts.append(attempt)
    path = write_problem_run(run, problem, attempt_labels=["Attempt 1"])
    log.info("%s done -> %s", problem.problem_id, path)


async def main() -> None:
    """Run the single-LLM harness over all problems (resumable, concurrent)."""
    configure_logging()
    problems = load_problems()
    pending = [p for p in problems if not result_exists(SINGLE_LLM_DIR, p.problem_id)]
    skipped = len(problems) - len(pending)
    log.info("%d problems to run, %d already done", len(pending), skipped)
    tasks = [lambda p=p: solve_problem(p) for p in pending]
    await run_all(tasks)


if __name__ == "__main__":
    anyio.run(main)
