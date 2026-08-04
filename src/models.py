"""Data models for problems, arm/experiment config, and phase results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    """A single olympiad problem, fetched from the dataset URLs into memory.

    Deliberately minimal: only the statement ever reaches the model, and no
    contest-identifying metadata (country, source, url, year) is held at all.
    domain (algebra/combinatorics/number theory) is kept for CLI filtering
    only and never enters a prompt. Hint ladder, joined by problem_id:
    hint_h1 = placebo (None until authored), hint_h2 = technique tags,
    hint_h3 = solution outline. Arms fail fast if their tier is missing for a
    selected problem.
    """

    problem_id: str
    statement: str
    domain: str
    hint_h1: str | None
    hint_h2: str | None
    hint_h3: str | None


@dataclass(frozen=True)
class ArmConfig:
    """One experiment arm from config.json (see README arm table)."""

    name: str
    hint: str
    mode: str
    budget_units: int
    seeds: list[int]


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated top-level experiment configuration from config.json."""

    model: str
    audit_model: str
    effort: str
    unit_output_tokens: int
    wrap_up_reserve_tokens: int
    max_turns_per_phase: int
    sequential_max_rounds: int
    audit_max_turns: int
    max_concurrency: int
    arms: dict[str, ArmConfig]

    def budget_tokens(self, arm: ArmConfig) -> int:
        """Total output-token budget for one attempt of this arm."""
        return self.unit_output_tokens * arm.budget_units

    @property
    def model_dirname(self) -> str:
        """Filesystem-safe model name for results paths ('/' becomes '-')."""
        return self.model.replace("/", "-")


class RateLimitExhausted(Exception):
    """Raised when the account's usage limit is hit (RateLimitInfo 'rejected').

    Carries resets_at (unix seconds) so the caller can wait until the limit
    resets and resume.
    """

    def __init__(self, resets_at: int) -> None:
        """Store the reset time and build a human-readable message."""
        self.resets_at = resets_at
        super().__init__(f"Rate limit exhausted; resets at unix {resets_at}")


@dataclass
class ToolCall:
    """One tool invocation by the agent, with its full untruncated result.

    Logged for every phase so a run can be audited: it proves exactly which
    tools the agent used and that nothing left the sandbox.
    """

    name: str
    tool_input: dict[str, object]
    result: str
    is_error: bool


@dataclass
class PhaseResult:
    """The outcome of one prompt->response phase within an attempt."""

    label: str
    prompt: str
    text: str
    output_tokens: int
    cumulative_output_tokens: int
    num_turns: int
    duration_ms: int
    total_cost_usd: float
    is_error: bool
    stop_reason: str
    budget_exhausted: bool
    tool_calls: list[ToolCall]
