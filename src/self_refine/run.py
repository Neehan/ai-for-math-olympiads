"""Self-Refine harness: generate -> self-critique -> revise, in one session.

The cheap reflection baseline (Self-Refine, Madaan et al. 2023). One client
session per problem, same model throughout, three phases per round:

  1. generate  - solve from the statement (identical to the single-LLM C0 solve)
  2. critique  - the model, as an adversarial grader, finds the first genuine gap
  3. revise    - the model rewrites the solution acting on that critique

Feedback (critique) is deliberately separated from refinement (revise): unlike
the Ralph loop's single combined "critique-and-fix" step, Self-Refine produces a
standalone critique first, then acts on it. With SELF_REFINE_ROUNDS = 1 this is
2 full agentic attempts + 1 critique per problem (~2x the single-LLM budget),
4x cheaper than Best-of-N / Ralph (8x). Ralph owns the many-round refinement
axis; this harness is the fixed one-round control.

Every phase is recorded as its own attempt so the full generate/critique/revise
trajectory is auditable; the final revise phase is the graded answer.

Usage:
    python -m src.self_refine.run
"""

import anyio

from claude_agent_sdk import ClaudeSDKClient

from src.shared.concurrency import run_all
from src.shared.constants import (
    MAX_TURNS_PER_ATTEMPT,
    MODEL,
    SELF_REFINE_DIR,
    SELF_REFINE_ROUNDS,
)
from src.shared.io_utils import (
    load_problems,
    result_exists,
    scratch_dir,
    write_problem_run,
)
from src.shared.logging_setup import configure_logging, get_logger
from src.shared.models import Problem, ProblemRun
from src.shared.prompts import (
    self_refine_critique_prompt,
    self_refine_revise_prompt,
    task_prompt,
)
from src.shared.solver import build_options, run_attempt, run_resumable

log = get_logger(SELF_REFINE_DIR)


async def solve_problem(problem: Problem) -> None:
    """Run generate -> (critique -> revise) x rounds on one problem; write it.

    All phases share one persistent session, so the critique sees the generated
    solution and the revise sees the critique, exactly as Self-Refine requires.
    """
    cwd = str(scratch_dir(SELF_REFINE_DIR, problem.problem_id))
    options = build_options(cwd=cwd, max_turns=MAX_TURNS_PER_ATTEMPT)
    run = ProblemRun(
        problem_id=problem.problem_id,
        harness=SELF_REFINE_DIR,
        model=MODEL,
    )
    labels: list[str] = []
    async with ClaudeSDKClient(options=options) as client:
        initial = await run_attempt(
            client, task_prompt(problem, cwd, MAX_TURNS_PER_ATTEMPT)
        )
        run.attempts.append(initial)
        labels.append("Generate (initial)")
        log.info("%s generate", problem.problem_id)

        for round_num in range(1, SELF_REFINE_ROUNDS + 1):
            critique = await run_attempt(
                client, self_refine_critique_prompt(MAX_TURNS_PER_ATTEMPT)
            )
            run.attempts.append(critique)
            labels.append(f"Critique (round {round_num})")
            log.info(
                "%s critique %d/%d", problem.problem_id, round_num, SELF_REFINE_ROUNDS
            )

            revise = await run_attempt(
                client, self_refine_revise_prompt(MAX_TURNS_PER_ATTEMPT)
            )
            run.attempts.append(revise)
            labels.append(f"Revise (round {round_num})")
            log.info(
                "%s revise %d/%d", problem.problem_id, round_num, SELF_REFINE_ROUNDS
            )

    path = write_problem_run(run, problem, attempt_labels=labels)
    log.info("%s done -> %s", problem.problem_id, path)


async def main() -> None:
    """Run the Self-Refine harness over all problems (resumable, concurrent).

    Different problems run in parallel; the phases within a problem stay
    sequential (critique reads the generation, revise reads the critique).
    """
    configure_logging()
    problems = load_problems()
    pending = [
        p for p in problems if not result_exists(SELF_REFINE_DIR, p.problem_id)
    ]
    skipped = len(problems) - len(pending)
    log.info("%d problems to run, %d already done", len(pending), skipped)
    tasks = [lambda p=p: run_resumable(lambda: solve_problem(p)) for p in pending]
    await run_all(tasks)


if __name__ == "__main__":
    anyio.run(main)
