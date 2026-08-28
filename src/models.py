"""Data models for problems, arm/experiment config, and phase results."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Problem:
    """A single olympiad problem, fetched from the dataset URLs into memory.

    Deliberately minimal: only the statement ever reaches the model, and no
    contest-identifying metadata (country, source, url, year) is held at all.
    domain (algebra/combinatorics/number theory) is kept for CLI filtering
    only and never enters a prompt. Hint ladder, joined by problem_id:
    hint_h1 = deterministic within-domain cyclic shift of the h2 hints,
    hint_h2 = frozen one-sentence strategy hint, hint_h3 = solution outline.
    Arms fail fast if their tier is missing for a selected problem.
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


def arm_checkpoint_identity(arm: ArmConfig) -> dict[str, object]:
    """Return the attempt protocol without Parallel's replication count.

    Parallel's configured seed list controls which independent banks the host
    launches; it does not change the protocol inside any bank. Canonicalizing
    that field preserves paid seed-1 checkpoints created before the study
    expanded from one bank to three.
    """
    return {
        "name": arm.name,
        "hint": arm.hint,
        "mode": arm.mode,
        "budget_units": arm.budget_units,
        "seeds": [1] if arm.mode == "parallel" else list(arm.seeds),
    }


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated top-level experiment configuration from config.json."""

    model: str
    audit_model: str
    effort: str
    unit_output_tokens: int
    wrap_up_reserve_tokens: int
    uniform_strategy_plan_tokens: int
    uniform_strategy_plan_wrap_up_reserve_tokens: int
    uniform_strategy_branches: int
    max_turns_per_phase: int
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


class TokenSpendLimit(Exception):
    """Raised when a token's org spend limit is hit (CLI dies at startup).

    Unlike a rate limit there is no reset to wait for — the token is removed
    from rotation and the live conversation resumes on the next token.
    """


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


@dataclass(frozen=True)
class ReconnectEvent:
    """One operational provider-session recovery, with no secret material."""

    reason: str
    resets_at: int | None
    from_credential: str
    to_credential: str


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
    reconnects: list[ReconnectEvent]
    # Raw per-query SDK usage (input, cached input, output, and any
    # provider-specific details) is retained so subscription credits can be
    # reconstructed independently of the SDK's dollar estimate.
    provider_usage: dict[str, object] = field(default_factory=dict)
    # A killed process may be resumed from the same provider transcript.  Its
    # incomplete response prefix is retained for auditing but is not spliced
    # into the replacement complete response used as the phase artifact.
    process_resume_count: int = 0
    discarded_output_text: str = ""
    discarded_tool_calls: list[ToolCall] = field(default_factory=list)
