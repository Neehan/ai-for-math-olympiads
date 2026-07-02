"""Data models for problems and attempt results."""

from dataclasses import dataclass, field


@dataclass
class Problem:
    """A single olympiad problem loaded from problems.jsonl."""

    problem_id: str
    statement: str
    country: str
    source: str
    url: str
    year: int
    domain: str
    difficulty_rating: int
    difficulty_level: str
    task: str
    answer_type: str


@dataclass
class ToolCall:
    """One tool invocation by the agent, with its result.

    Logged for every attempt so a run can be audited: it proves exactly which
    tools the agent used (e.g. that it never invoked WebSearch or a network
    Bash command).
    """

    name: str
    tool_input: dict[str, object]
    result: str
    is_error: bool


@dataclass
class AttemptResult:
    """The outcome of one agent attempt at one problem."""

    text: str
    num_turns: int
    duration_ms: int
    total_cost_usd: float
    is_error: bool
    stop_reason: str
    tool_calls: list["ToolCall"]


@dataclass
class ProblemRun:
    """The full record of a harness run on one problem (one or more attempts)."""

    problem_id: str
    harness: str
    model: str
    attempts: list[AttemptResult] = field(default_factory=list)
