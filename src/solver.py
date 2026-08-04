"""One agent phase: build SDK options, stream the response, enforce the budget.

Compute is operationalized as the attempt's total output-token budget (see
README). Enforcement is two-layered: the API-side task_budget tells the model
its remaining budget so it paces itself, and the harness hard-cuts by
interrupting the session the moment cumulative output tokens exceed the budget.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    RateLimitEvent,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from src.constants import (
    AGENT_SETTINGS_PATH,
    ALLOWED_TOOLS,
    DISALLOWED_TOOLS,
    OAUTH_TOKEN_ENV,
    PERMISSION_MODE,
)
from src.models import (
    ExperimentConfig,
    PhaseResult,
    RateLimitExhausted,
    ToolCall,
)
from src.prompts import system_prompt
from src.token_pool import TokenPool

log = logging.getLogger("solver")

T = TypeVar("T")

STOP_BUDGET_EXHAUSTED: str = "budget_exhausted"


class BudgetTracker:
    """Per-attempt session accounting: output tokens, cost, and turn deltas.

    All phases of one attempt share one tracker (and one SDK session). The
    soft limit (budget minus the wrap-up reserve) is where working phases are
    interrupted so a final wrap-up phase can spend the reserve writing the
    solution down; the hard budget is the absolute cutoff.
    """

    def __init__(self, budget_tokens: int, wrap_up_reserve_tokens: int) -> None:
        """budget_tokens is the attempt's total output-token budget."""
        self.budget_tokens = budget_tokens
        self.soft_limit_tokens = budget_tokens - wrap_up_reserve_tokens
        self.spent = 0
        self._seen_message_ids: set[str] = set()
        self._prev_session_cost = 0.0
        self._prev_session_turns = 0

    def add(self, message_id: str | None, usage: dict[str, object] | None) -> None:
        """Accumulate one assistant message's output tokens (deduped by id)."""
        if usage is None:
            return
        if message_id is not None:
            if message_id in self._seen_message_ids:
                return
            self._seen_message_ids.add(message_id)
        self.spent += int(str(usage.get("output_tokens", 0)))

    @property
    def exhausted(self) -> bool:
        """True once the attempt has spent its full output-token budget."""
        return self.spent >= self.budget_tokens

    @property
    def soft_exhausted(self) -> bool:
        """True once only the wrap-up reserve remains."""
        return self.spent >= self.soft_limit_tokens

    @property
    def remaining(self) -> int:
        """Output tokens left before the hard budget."""
        return max(0, self.budget_tokens - self.spent)

    def phase_cost_delta(self, session_cost_usd: float) -> float:
        """This phase's cost from the session-cumulative figure the SDK reports."""
        delta = max(0.0, session_cost_usd - self._prev_session_cost)
        self._prev_session_cost = max(self._prev_session_cost, session_cost_usd)
        return delta

    def phase_turns_delta(self, session_turns: int) -> int:
        """This phase's turn count from the session-cumulative figure."""
        delta = max(0, session_turns - self._prev_session_turns)
        self._prev_session_turns = max(self._prev_session_turns, session_turns)
        return delta


def build_options(
    config: ExperimentConfig, scratch_dir: str, budget_tokens: int, oauth_token: str
) -> ClaudeAgentOptions:
    """Construct agent options for one attempt.

    Tool policy comes from agent_settings.json (deny list) plus
    disallowed_tools; network isolation comes from the container firewall.
    oauth_token is the pool-assigned key for this attempt's session.

    NOTE: user/project settings are excluded via extra_args — the SDK drops
    a falsy setting_sources=[] instead of sending it, so the flag must be
    passed explicitly with an empty value.
    """
    return ClaudeAgentOptions(
        model=config.model,
        env={OAUTH_TOKEN_ENV: oauth_token},
        effort=config.effort,  # type: ignore[arg-type]
        system_prompt=system_prompt(),
        allowed_tools=list(ALLOWED_TOOLS),
        disallowed_tools=list(DISALLOWED_TOOLS),
        settings=str(AGENT_SETTINGS_PATH),
        extra_args={"setting-sources": ""},
        permission_mode=PERMISSION_MODE,
        max_turns=config.max_turns_per_phase,
        task_budget={"total": budget_tokens},
        cwd=scratch_dir,
    )


def _collect_text(messages: list[AssistantMessage]) -> str:
    """Concatenate all assistant text blocks in order."""
    parts: list[str] = []
    for message in messages:
        for block in message.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
    return "\n".join(parts)


def _stringify_result(content: object) -> str:
    """Flatten a ToolResultBlock's content (str or list of blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _collect_tool_calls(
    tool_uses: dict[str, ToolUseBlock],
    tool_results: dict[str, ToolResultBlock],
) -> list[ToolCall]:
    """Pair each tool-use block with its result block by tool_use_id."""
    calls: list[ToolCall] = []
    for use_id, use in tool_uses.items():
        result = tool_results.get(use_id)
        if result is None:
            # No result block: the call did not complete (interrupted, denied
            # without a result, or truncated at the turn limit). Record it as an
            # error so the audit log never shows an incomplete call as success.
            calls.append(
                ToolCall(
                    name=use.name,
                    tool_input=dict(use.input),
                    result="<no result returned>",
                    is_error=True,
                )
            )
            continue
        calls.append(
            ToolCall(
                name=use.name,
                tool_input=dict(use.input),
                result=_stringify_result(result.content),
                is_error=bool(result.is_error),
            )
        )
    return calls


async def run_phase(
    client: ClaudeSDKClient,
    prompt: str,
    label: str,
    tracker: BudgetTracker,
    stop_at_tokens: int,
) -> PhaseResult:
    """Send one prompt on an existing session and capture the full response.

    Every assistant message's output tokens are added to the shared tracker;
    the moment cumulative spend reaches stop_at_tokens (the soft limit for
    working phases, the hard budget for the wrap-up phase) the session is
    interrupted and the phase is marked budget_exhausted. Fails loud if no
    ResultMessage arrives or the result is an API error.
    """
    await client.query(prompt)

    assistant_messages: list[AssistantMessage] = []
    tool_uses: dict[str, ToolUseBlock] = {}
    tool_results: dict[str, ToolResultBlock] = {}
    result_message: ResultMessage | None = None
    rate_limit_reset: int | None = None
    spent_before = tracker.spent
    interrupted = False

    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            assistant_messages.append(message)
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    tool_uses[block.id] = block
            tracker.add(message.message_id, message.usage)
            if tracker.spent >= stop_at_tokens and not interrupted:
                interrupted = True
                log.info(
                    "%s: token cutoff reached (%d/%d, stop at %d); interrupting",
                    label,
                    tracker.spent,
                    tracker.budget_tokens,
                    stop_at_tokens,
                )
                await client.interrupt()
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    tool_results[block.tool_use_id] = block
        elif isinstance(message, ResultMessage):
            result_message = message
        elif isinstance(message, RateLimitEvent):
            info = message.rate_limit_info
            if info.status == "rejected":
                rate_limit_reset = info.resets_at if info.resets_at is not None else 0

    if rate_limit_reset is not None:
        raise RateLimitExhausted(rate_limit_reset)

    if result_message is None:
        raise RuntimeError(f"Agent produced no ResultMessage for phase '{label}'")

    # An API-error result (auth failure, server error, max-turns error, ...)
    # must never be written as a completed attempt: fail the task so the
    # resumable rerun retries it. A deliberate interrupt is not an error.
    if result_message.is_error and not interrupted:
        raise RuntimeError(
            f"Phase '{label}' failed: subtype={result_message.subtype} "
            f"errors={result_message.errors} result={result_message.result!r}"
        )

    text = _collect_text(assistant_messages)
    if result_message.result is not None and not interrupted:
        text = result_message.result

    cost = result_message.total_cost_usd
    stop_reason = (
        STOP_BUDGET_EXHAUSTED
        if interrupted
        else (result_message.stop_reason or "end_turn")
    )
    return PhaseResult(
        label=label,
        prompt=prompt,
        text=text,
        output_tokens=tracker.spent - spent_before,
        cumulative_output_tokens=tracker.spent,
        num_turns=tracker.phase_turns_delta(result_message.num_turns),
        duration_ms=result_message.duration_ms,
        total_cost_usd=tracker.phase_cost_delta(cost if cost is not None else 0.0),
        is_error=result_message.is_error,
        stop_reason=stop_reason,
        budget_exhausted=interrupted,
        tool_calls=_collect_tool_calls(tool_uses, tool_results),
    )


async def run_resumable(pool: TokenPool, factory: Callable[[str], Awaitable[T]]) -> T:
    """Run an attempt factory with pool tokens, rotating on rate limits.

    The factory receives the OAuth token for its session. On RateLimitExhausted
    the token is put on cooldown and the attempt restarts from scratch with the
    next available token; the pool only sleeps when every token is cooling, so
    a rate limit never kills the run.
    """
    while True:
        token = await pool.acquire()
        try:
            return await factory(token)
        except RateLimitExhausted as exhausted:
            await pool.mark_rate_limited(token, exhausted.resets_at)
