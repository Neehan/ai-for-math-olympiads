"""Best-of-N harness: N independent attempts per problem.

No proof verifier and no selector — all N attempts are stored so pass@k and
selection are decided later by the LLM-judge (per the paper's design).

Every (problem, sample) pair is an independent task; all pairs share the global
concurrency limit so exactly MAX_CONCURRENCY agent sessions stay busy across
problem and sample boundaries. A problem's markdown file is written once all of
its samples finish, with samples ordered by index.

Usage:
    python -m src.best_of_n.run
"""

import anyio

from claude_agent_sdk import ClaudeSDKClient

from src.shared.concurrency import run_all
from src.shared.constants import (
    BEST_OF_N_DIR,
    MAX_TURNS_PER_ATTEMPT,
    MODEL,
    N_SAMPLES,
)
from src.shared.io_utils import (
    load_problems,
    result_exists,
    scratch_dir,
    write_problem_run,
)
from src.shared.logging_setup import configure_logging, get_logger
from src.shared.models import AttemptResult, Problem, ProblemRun
from src.shared.prompts import task_prompt
from src.shared.solver import build_options, run_attempt

log = get_logger(BEST_OF_N_DIR)


async def run_sample(
    problem: Problem, sample_index: int, sink: dict[int, AttemptResult]
) -> None:
    """Run one independent sample and store it by index for stable ordering."""
    cwd = str(
        scratch_dir(BEST_OF_N_DIR, f"{problem.problem_id}/sample_{sample_index + 1}")
    )
    options = build_options(cwd=cwd, max_turns=MAX_TURNS_PER_ATTEMPT)
    async with ClaudeSDKClient(options=options) as client:
        attempt = await run_attempt(
            client, task_prompt(problem, cwd, MAX_TURNS_PER_ATTEMPT)
        )
        sink[sample_index] = attempt
    log.info(
        "%s sample %d/%d done", problem.problem_id, sample_index + 1, N_SAMPLES
    )


def _write_problem(problem: Problem, sink: dict[int, AttemptResult]) -> None:
    """Write a problem's files only if all N samples succeeded.

    If any sample failed (its index is missing from the sink), the problem is
    left unwritten and logged, so a resumable rerun retries it in full rather
    than committing a partial (fewer-than-N) result.
    """
    if len(sink) != N_SAMPLES:
        missing = [i + 1 for i in range(N_SAMPLES) if i not in sink]
        log.warning(
            "%s incomplete (%d/%d samples); missing %s. Not writing; will retry "
            "on next run.",
            problem.problem_id,
            len(sink),
            N_SAMPLES,
            missing,
        )
        return
    run = ProblemRun(problem_id=problem.problem_id, harness=BEST_OF_N_DIR, model=MODEL)
    labels: list[str] = []
    for i in range(N_SAMPLES):
        run.attempts.append(sink[i])
        labels.append(f"Sample {i + 1}")
    path = write_problem_run(run, problem, attempt_labels=labels)
    log.info("%s done -> %s", problem.problem_id, path)


async def main() -> None:
    """Run the best-of-N harness over all problems (resumable, concurrent).

    Every (problem, sample) pair is one task and all pairs share a single
    global concurrency limit, so exactly MAX_CONCURRENCY agent sessions stay
    busy across problem and sample boundaries.
    """
    configure_logging()
    problems = load_problems()
    pending = [p for p in problems if not result_exists(BEST_OF_N_DIR, p.problem_id)]
    skipped = len(problems) - len(pending)
    log.info("%d problems to run, %d already done", len(pending), skipped)

    sinks: dict[str, dict[int, AttemptResult]] = {p.problem_id: {} for p in pending}
    tasks = [
        lambda p=p, i=i: run_sample(p, i, sinks[p.problem_id])
        for p in pending
        for i in range(N_SAMPLES)
    ]
    await run_all(tasks)

    for problem in pending:
        _write_problem(problem, sinks[problem.problem_id])


if __name__ == "__main__":
    anyio.run(main)
