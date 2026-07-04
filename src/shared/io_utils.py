"""Loading problems and writing per-problem markdown result files."""

import json
import os
from pathlib import Path

from src.shared.constants import (
    LOGS_ROOT,
    PROBLEMS_PATH,
    RESULTS_ROOT,
    SCRATCH_ROOT,
    TOOL_LOG_TRUNCATE,
)
from src.shared.models import AttemptResult, Problem, ProblemRun, ToolCall


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text to path atomically: write a temp file, then os.replace it.

    os.replace is atomic on the same filesystem, so a reader (e.g. result_exists
    on resume) never sees a partially-written file — the path either has the old
    content or the complete new content, never a truncation from a crash mid-write.
    """
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _truncate(text: str) -> str:
    """Truncate long text for the tool-call log, marking omission."""
    if len(text) <= TOOL_LOG_TRUNCATE:
        return text
    return text[:TOOL_LOG_TRUNCATE] + f"... [{len(text) - TOOL_LOG_TRUNCATE} more chars]"


def _tool_calls_block(tool_calls: list[ToolCall]) -> str:
    """Render the full tool-call log for one attempt (audit trail)."""
    if not tool_calls:
        return "### Tool calls\n\n_No tools were used._\n"
    lines = ["### Tool calls\n"]
    for i, call in enumerate(tool_calls, start=1):
        input_json = _truncate(json.dumps(call.tool_input, ensure_ascii=False))
        lines.append(
            f"{i}. **{call.name}** (is_error={call.is_error})\n"
            f"   - input: `{input_json}`\n"
            f"   - result: `{_truncate(call.result.replace(chr(10), ' '))}`\n"
        )
    return "\n".join(lines)


def load_problems() -> list[Problem]:
    """Load all problems from problems.jsonl."""
    problems: list[Problem] = []
    with open(PROBLEMS_PATH, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            problems.append(
                Problem(
                    problem_id=record["problem_id"],
                    statement=record["statement"],
                    country=record["country"],
                    source=record["source"],
                    url=record["url"],
                    year=record["year"],
                    domain=record["domain"],
                    difficulty_rating=record["difficulty_rating"],
                    difficulty_level=record["difficulty_level"],
                    task=record["task"],
                    answer_type=record["answer_type"],
                )
            )
    return problems


def scratch_dir(harness_dir: str, problem_id: str) -> Path:
    """Per-problem scratch working dir for the agent's filesystem/Bash tools."""
    path = SCRATCH_ROOT / harness_dir / problem_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_path(harness_dir: str, problem_id: str) -> Path:
    """Path of the markdown result file for a problem under a harness dir."""
    return RESULTS_ROOT / harness_dir / f"{problem_id}.md"


def result_exists(harness_dir: str, problem_id: str) -> bool:
    """Return True if a result file already exists (for resumable runs)."""
    return result_path(harness_dir, problem_id).exists()


def _attempt_section(index: int, label: str, attempt: AttemptResult) -> str:
    """Render one attempt as a markdown section."""
    header = f"## {label}" if label else f"## Attempt {index + 1}"
    tool_names = sorted({c.name for c in attempt.tool_calls})
    meta = (
        f"- turns: {attempt.num_turns}\n"
        f"- duration_ms: {attempt.duration_ms}\n"
        f"- cost_usd: {attempt.total_cost_usd:.4f}\n"
        f"- stop_reason: {attempt.stop_reason}\n"
        f"- is_error: {attempt.is_error}\n"
        f"- num_tool_calls: {len(attempt.tool_calls)}\n"
        f"- tools_used: {tool_names}\n"
    )
    return (
        f"{header}\n\n{meta}\n{_tool_calls_block(attempt.tool_calls)}\n"
        f"### Response\n\n{attempt.text.strip()}\n"
    )


def _write_full_log(run: ProblemRun, problem: Problem, labels: list[str]) -> Path:
    """Write the full, untruncated per-attempt log to logs/<harness>/<id>.jsonl.

    One JSON line per attempt with every tool call (name, full input, full
    result). This is the audit trail: it proves exactly which tools ran.
    """
    path = LOGS_ROOT / run.harness / f"{problem.problem_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, attempt in enumerate(run.attempts):
        record = {
            "problem_id": problem.problem_id,
            "harness": run.harness,
            "model": run.model,
            "attempt_index": i,
            "label": labels[i] if i < len(labels) else "",
            "num_turns": attempt.num_turns,
            "duration_ms": attempt.duration_ms,
            "total_cost_usd": attempt.total_cost_usd,
            "is_error": attempt.is_error,
            "stop_reason": attempt.stop_reason,
            "text": attempt.text,
            "tool_calls": [
                {
                    "name": c.name,
                    "input": c.tool_input,
                    "result": c.result,
                    "is_error": c.is_error,
                }
                for c in attempt.tool_calls
            ],
        }
        # default=str so a non-JSON-serializable tool_input value can never
        # crash the audit-log write and lose the whole run's record.
        lines.append(json.dumps(record, ensure_ascii=False, default=str))
    _atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def write_problem_run(
    run: ProblemRun,
    problem: Problem,
    attempt_labels: list[str],
) -> Path:
    """Write the markdown result file and the full JSONL audit log for a run.

    attempt_labels[i] labels attempts[i]; use "" for the default numbering.
    Returns the markdown path.
    """
    _write_full_log(run, problem, attempt_labels)
    path = result_path(run.harness, problem.problem_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    total_cost = sum(a.total_cost_usd for a in run.attempts)
    front = (
        f"# {problem.problem_id}\n\n"
        f"- harness: {run.harness}\n"
        f"- model: {run.model}\n"
        f"- domain: {problem.domain}\n"
        f"- difficulty_rating: {problem.difficulty_rating}\n"
        f"- difficulty_level: {problem.difficulty_level}\n"
        f"- task: {problem.task}\n"
        f"- answer_type: {problem.answer_type}\n"
        f"- num_attempts: {len(run.attempts)}\n"
        f"- total_cost_usd: {total_cost:.4f}\n\n"
        f"## Problem statement\n\n{problem.statement.strip()}\n"
    )

    sections = [
        _attempt_section(
            i,
            attempt_labels[i] if i < len(attempt_labels) else "",
            attempt,
        )
        for i, attempt in enumerate(run.attempts)
    ]

    _atomic_write_text(path, front + "\n" + "\n".join(sections))
    return path
