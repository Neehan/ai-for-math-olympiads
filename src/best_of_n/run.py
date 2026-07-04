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

from dataclasses import dataclass, field

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
from src.shared.solver import build_options, run_attempt, run_resumable

log = get_logger(BEST_OF_N_DIR)


@dataclass
class ProblemSink:
    """Per-problem collector for concurrent samples, with a completion lock.

    Runtime-only state (holds an anyio.Lock), so it lives here rather than in
    models.py. Samples of the same problem race to fill `attempts`; the lock
    serializes the "am I the Nth sample?" check so exactly one sample triggers
    the write.
    """

    lock: anyio.Lock = field(default_factory=anyio.Lock)
    attempts: dict[int, AttemptResult] = field(default_factory=dict)


async def run_sample(
    problem: Problem, sample_index: int, sink: ProblemSink
) -> None:
    """Run one independent sample; write the problem once its Nth sample lands.

    The result is stored by index for stable ordering, then — under the sink's
    per-problem lock — this sample checks whether all N samples are now present.
    The one that completes the set writes the problem immediately, so a crash
    mid-run leaves every already-finished problem safely on disk (resumable
    reruns skip them). A sample that raises never reaches the sink, so the count
    never hits N and the problem is left unwritten and retried in full.
    """
    cwd = str(
        scratch_dir(BEST_OF_N_DIR, f"{problem.problem_id}/sample_{sample_index + 1}")
    )
    options = build_options(cwd=cwd, max_turns=MAX_TURNS_PER_ATTEMPT)
    async with ClaudeSDKClient(options=options) as client:
        attempt = await run_attempt(
            client, task_prompt(problem, cwd, MAX_TURNS_PER_ATTEMPT)
        )
        async with sink.lock:
            sink.attempts[sample_index] = attempt
            log.info(
                "%s sample %d/%d done", problem.problem_id, sample_index + 1, N_SAMPLES
            )
            if len(sink.attempts) == N_SAMPLES:
                _write_problem(problem, sink.attempts)


def _write_problem(problem: Problem, attempts: dict[int, AttemptResult]) -> None:
    """Write a problem's files once all N samples have succeeded.

    Called under the problem's sink lock exactly when the sample dict reaches N
    entries, so it always writes a complete (N-sample) result.
    """
    run = ProblemRun(problem_id=problem.problem_id, harness=BEST_OF_N_DIR, model=MODEL)
    labels: list[str] = []
    for i in range(N_SAMPLES):
        run.attempts.append(attempts[i])
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

    sinks: dict[str, ProblemSink] = {p.problem_id: ProblemSink() for p in pending}
    tasks = [
        lambda p=p, i=i: run_resumable(lambda: run_sample(p, i, sinks[p.problem_id]))
        for p in pending
        for i in range(N_SAMPLES)
    ]
    await run_all(tasks)

    # Each problem is written by its Nth successful sample (in run_sample), so a
    # crash mid-run leaves completed problems on disk. After the batch drains,
    # log any problem still short of N — a failed sample left it unwritten; a
    # resumable rerun retries it in full.
    for problem in pending:
        sink = sinks[problem.problem_id]
        if len(sink.attempts) != N_SAMPLES:
            missing = [i + 1 for i in range(N_SAMPLES) if i not in sink.attempts]
            log.warning(
                "%s incomplete (%d/%d samples); missing %s. Not written; will "
                "retry on next run.",
                problem.problem_id,
                len(sink.attempts),
                N_SAMPLES,
                missing,
            )


if __name__ == "__main__":
    anyio.run(main)
