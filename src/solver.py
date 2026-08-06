"""One agent phase: build SDK options, stream the response, enforce the budget.

Compute is operationalized as the attempt's total output-token budget (see
README). Enforcement is two-layered: the API-side task_budget tells the model
its remaining budget so it paces itself, and the harness hard-cuts by
interrupting the session the moment cumulative output tokens exceed the budget.
"""

import logging
import os
import uuid
from collections.abc import AsyncIterator, Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    RateLimitEvent,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from src.constants import (
    AGENT_SETTINGS_PATH,
    ALLOWED_TOOLS,
    CLAUDE_CONFIG_DIR_ENV,
    CLI_PATH_ENV,
    DISALLOWED_TOOLS,
    ANTHROPIC_API_KEY_ENV,
    ANTHROPIC_AUTH_TOKEN_ENV,
    ANTHROPIC_BASE_URL_ENV,
    MAX_OUTPUT_TOKENS_ENV,
    MAX_OUTPUT_TOKENS_PER_RESPONSE,
    OAUTH_TOKEN_ENV,
    OPENROUTER_BASE_URL,
    OPENROUTER_KEY_ENV,
    PERMISSION_MODE,
    PROCESS_RECOVERY_PROMPT,
    SESSION_RECOVERY_PROMPT,
    SESSION_STATE_SUBDIR,
    SPEND_LIMIT_MARKERS,
)
from src.models import (
    ExperimentConfig,
    PhaseResult,
    ReconnectEvent,
    TokenSpendLimit,
    ToolCall,
)
from src.prompts import system_prompt
from src.token_pool import TokenPool

log = logging.getLogger("solver")

STOP_BUDGET_EXHAUSTED: str = "budget_exhausted"


def process_recovery_prompt(pending_prompt: str) -> str:
    """Self-contained restart prompt, safe even if the first query never landed."""
    return f"{PROCESS_RECOVERY_PROMPT}\n\nPending request:\n{pending_prompt}"


class StderrTail:
    """Collects the CLI's stderr lines for one session.

    The SDK surfaces a fatal CLI exit as a generic exception; the real cause
    (e.g. an org spend limit) only appears on stderr, so every session records
    it via the options' stderr callback.
    """

    def __init__(self) -> None:
        """Start with no captured lines."""
        self.lines: list[str] = []

    def __call__(self, line: str) -> None:
        """SDK stderr callback: record one line."""
        self.lines.append(line)

    def raise_if_spend_limit(self) -> None:
        """Re-raise a generic SDK failure as TokenSpendLimit when stderr shows it."""
        text = "\n".join(self.lines).lower()
        if any(marker in text for marker in SPEND_LIMIT_MARKERS):
            raise TokenSpendLimit("\n".join(self.lines[-2:]))


OptionsFactory = Callable[
    [str, str | None, str | None, StderrTail], ClaudeAgentOptions
]


class ResumableClaudeSession:
    """One Claude conversation that can rotate credentials without restarting.

    The CLI transcript is local and identified by an explicit UUID. If the
    provider rejects a live query, the exhausted credential is cooled (or
    disabled for a spend limit), the next available credential is selected, and
    a new CLI process resumes the same transcript. The caller's phase objects,
    scratch directory, and BudgetTracker stay alive across that transition.
    """

    def __init__(
        self,
        pool: TokenPool,
        options_factory: OptionsFactory,
        *,
        session_id: str | None = None,
        reconnects: list[ReconnectEvent] | None = None,
    ) -> None:
        self._pool = pool
        self._options_factory = options_factory
        self._session_id = session_id or str(uuid.uuid4())
        self._resume_on_enter = session_id is not None
        self._client: ClaudeSDKClient | None = None
        self._token: str | None = None
        self._stderr_tail: StderrTail | None = None
        self._reconnects: list[ReconnectEvent] = list(reconnects or [])
        self._connection_id: str | None = None
        self._pending_prompt: str | None = None

    @property
    def session_id(self) -> str:
        """Stable provider conversation UUID, safe to persist (not a secret)."""
        return self._session_id

    @property
    def connection_id(self) -> str:
        """Opaque id for the current CLI process's cumulative cost counter."""
        if self._connection_id is None:
            raise RuntimeError("Session has no open CLI connection")
        return self._connection_id

    @property
    def reconnect_count(self) -> int:
        """Number of successful credential transitions in this conversation."""
        return len(self._reconnects)

    @property
    def reconnect_events(self) -> list[ReconnectEvent]:
        """Return an immutable-by-convention snapshot for result logging."""
        return list(self._reconnects)

    async def __aenter__(self) -> "ResumableClaudeSession":
        if self._resume_on_enter:
            log.warning(
                "Resuming checkpointed session %s after process restart",
                self._session_id,
            )
        await self._open(resume=self._resume_on_enter)
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        await self._close(suppress_errors=exc_type is not None)
        return False

    async def _open(self, *, resume: bool) -> None:
        """Open a CLI process, skipping credentials already at spend limit."""
        while True:
            token = await self._pool.acquire()
            stderr_tail = StderrTail()
            options = self._options_factory(
                token,
                None if resume else self._session_id,
                self._session_id if resume else None,
                stderr_tail,
            )
            client = ClaudeSDKClient(options=options)
            try:
                await client.connect()
            except Exception:
                try:
                    stderr_tail.raise_if_spend_limit()
                except TokenSpendLimit:
                    await self._pool.mark_dead(token)
                    await self._pool.release(token)
                    log.warning(
                        "%s was unusable at session startup; trying another "
                        "credential",
                        self._pool.credential_label(token),
                    )
                    continue
                await self._pool.release(token)
                raise
            self._token = token
            self._stderr_tail = stderr_tail
            self._client = client
            self._connection_id = str(uuid.uuid4())
            return

    async def _close(self, *, suppress_errors: bool) -> None:
        """Close the active CLI; release is a pool-policy compatibility hook."""
        client, token = self._client, self._token
        self._client = None
        self._token = None
        self._stderr_tail = None
        try:
            if client is not None:
                await client.disconnect()
        except Exception:
            if not suppress_errors:
                raise
            log.debug("Ignoring expected CLI shutdown error during recovery")
        finally:
            if token is not None:
                await self._pool.release(token)

    async def _recover(self, reason: str, resets_at: int | None) -> None:
        """Rotate credentials and resume this exact local transcript."""
        old_token = self._token
        if old_token is None:
            raise RuntimeError("Cannot recover a session with no leased credential")
        old_label = self._pool.credential_label(old_token)
        if reason == "rate_limit":
            await self._pool.mark_rate_limited(old_token, resets_at or 0)
        elif reason == "spend_limit":
            await self._pool.mark_dead(old_token)
        else:
            raise ValueError(f"Unknown recovery reason: {reason}")

        await self._close(suppress_errors=True)
        await self._open(resume=True)
        if self._token is None:
            raise RuntimeError("Recovery opened no replacement credential")
        new_label = self._pool.credential_label(self._token)
        event = ReconnectEvent(
            reason=reason,
            resets_at=resets_at,
            from_credential=old_label,
            to_credential=new_label,
        )
        self._reconnects.append(event)
        log.warning(
            "Session %s resumed after %s (%s -> %s); preserving accumulated "
            "output-token accounting",
            self._session_id,
            reason,
            old_label,
            new_label,
        )

    async def query(self, prompt: str) -> None:
        """Send a normal experiment prompt to the active conversation."""
        if self._client is None:
            raise RuntimeError("Session is not open")
        self._pending_prompt = prompt
        await self._client.query(prompt)

    async def receive_response(self) -> AsyncIterator[object]:
        """Stream one response, transparently reconnecting after quota rejection."""
        while True:
            if self._client is None or self._stderr_tail is None:
                raise RuntimeError("Session is not open")
            rate_limit_reset: int | None = None
            spend_limited = False
            try:
                async for message in self._client.receive_response():
                    message_session_id = getattr(message, "session_id", None)
                    if message_session_id:
                        self._session_id = str(message_session_id)
                    if isinstance(message, RateLimitEvent):
                        info = message.rate_limit_info
                        if info.status == "rejected":
                            rate_limit_reset = (
                                info.resets_at if info.resets_at is not None else 0
                            )
                        continue
                    # Once rejected, drain the dying CLI without exposing its
                    # error ResultMessage as a completed experimental phase.
                    if rate_limit_reset is None:
                        yield message
            except Exception:
                if rate_limit_reset is None:
                    try:
                        self._stderr_tail.raise_if_spend_limit()
                    except TokenSpendLimit:
                        spend_limited = True
                    if not spend_limited:
                        raise

            if rate_limit_reset is not None:
                await self._recover("rate_limit", rate_limit_reset)
            elif spend_limited:
                await self._recover("spend_limit", None)
            else:
                return

            if self._client is None:
                raise RuntimeError("Recovered session has no active client")
            recovery = SESSION_RECOVERY_PROMPT
            if self._pending_prompt:
                recovery = f"{recovery}\n\nPending request:\n{self._pending_prompt}"
            await self._client.query(recovery)

    async def interrupt(self) -> None:
        """Interrupt the currently active CLI process at the output cutoff."""
        if self._client is None:
            raise RuntimeError("Session is not open")
        await self._client.interrupt()


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
        self._completed_phases_tokens = 0
        # Persist maxima for the whole conversation: a resumed CLI may replay
        # an earlier message, including one from a completed phase.
        self._message_tokens: dict[str, int] = {}
        self._current_phase_streamed_tokens = 0
        self._prev_session_cost = 0.0
        self._prev_connection_id: str | None = None

    def add(self, message_id: str | None, usage: dict[str, object] | None) -> None:
        """Accumulate the current phase's streamed usage (max snapshot per id).

        Usage snapshots for one API message (message_start, then the real
        count in message_delta) share one id and grow — track the max, never
        the first (undercounts) nor the sum (double counts).
        """
        if usage is None:
            return
        tokens = int(str(usage.get("output_tokens", 0)))
        if message_id is None:
            self._current_phase_streamed_tokens += tokens
            self.spent += tokens
            return
        previous = self._message_tokens.get(message_id, 0)
        if tokens > previous:
            delta = tokens - previous
            self._current_phase_streamed_tokens += delta
            self.spent += delta
            self._message_tokens[message_id] = tokens

    def finish_phase(self, result_output_tokens: int | None) -> int:
        """Close a phase with the authoritative per-query result usage.

        Streamed deltas enforce the cutoff mid-phase; the ResultMessage's
        per-query output_tokens is exact for completed turns but excludes an
        interrupt-aborted final turn, so never go BELOW the streamed count —
        otherwise a soft-limit interrupt could un-trip soft_exhausted and
        skip the wrap-up phase. Returns the phase's token count.
        """
        streamed = self._current_phase_streamed_tokens
        phase_tokens = max(result_output_tokens or 0, streamed)
        self._completed_phases_tokens += phase_tokens
        self._current_phase_streamed_tokens = 0
        self.spent = self._completed_phases_tokens
        return phase_tokens

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

    def phase_cost_delta(
        self, session_cost_usd: float, connection_id: str
    ) -> float:
        """This phase's cost from a CLI-process-cumulative cost figure.

        The CLI counter accumulates across queries in one live process but
        resets when a transcript is resumed by a new process.  Track that
        boundary explicitly; comparing numeric values cannot detect a reset
        when the first resumed phase happens to cost more than the old total.
        """
        if self._prev_connection_id != connection_id:
            delta = max(0.0, session_cost_usd)
        else:
            delta = max(0.0, session_cost_usd - self._prev_session_cost)
        self._prev_session_cost = max(0.0, session_cost_usd)
        self._prev_connection_id = connection_id
        return delta

    def snapshot(self) -> dict[str, object]:
        """Return every counter needed for exact cross-process restoration."""
        return {
            "budget_tokens": self.budget_tokens,
            "soft_limit_tokens": self.soft_limit_tokens,
            "spent": self.spent,
            "completed_phases_tokens": self._completed_phases_tokens,
            "message_tokens": dict(self._message_tokens),
            "current_phase_streamed_tokens": self._current_phase_streamed_tokens,
            "prev_session_cost": self._prev_session_cost,
            "prev_connection_id": self._prev_connection_id,
        }

    @classmethod
    def restore(
        cls,
        snapshot: dict[str, object],
        budget_tokens: int,
        wrap_up_reserve_tokens: int,
    ) -> "BudgetTracker":
        """Restore a tracker, rejecting a checkpoint for a different budget."""
        tracker = cls(budget_tokens, wrap_up_reserve_tokens)
        expected_soft = budget_tokens - wrap_up_reserve_tokens
        if int(str(snapshot["budget_tokens"])) != budget_tokens or int(
            str(snapshot["soft_limit_tokens"])
        ) != expected_soft:
            raise ValueError("Checkpoint token budget does not match this invocation")
        tracker.spent = int(str(snapshot["spent"]))
        tracker._completed_phases_tokens = int(
            str(snapshot["completed_phases_tokens"])
        )
        raw_messages = snapshot.get("message_tokens", {})
        if not isinstance(raw_messages, dict):
            raise TypeError("Checkpoint message_tokens must be an object")
        tracker._message_tokens = {
            str(message_id): int(tokens)
            for message_id, tokens in raw_messages.items()
        }
        tracker._current_phase_streamed_tokens = int(
            str(snapshot["current_phase_streamed_tokens"])
        )
        tracker._prev_session_cost = float(str(snapshot["prev_session_cost"]))
        raw_connection_id = snapshot.get("prev_connection_id")
        tracker._prev_connection_id = (
            str(raw_connection_id) if raw_connection_id else None
        )
        if tracker.spent != (
            tracker._completed_phases_tokens
            + tracker._current_phase_streamed_tokens
        ):
            raise ValueError("Checkpoint BudgetTracker counters are inconsistent")
        return tracker


def uses_openrouter(model: str) -> bool:
    """True for 'vendor/model' ids, which route through OpenRouter."""
    return "/" in model


def token_env_name(model: str) -> str:
    """Name of the env var family holding this model's provider API keys."""
    return OPENROUTER_KEY_ENV if uses_openrouter(model) else OAUTH_TOKEN_ENV


def provider_env(model: str, api_key: str) -> dict[str, str]:
    """Per-session env: provider auth plus the raised per-response output cap."""
    cap = {MAX_OUTPUT_TOKENS_ENV: str(MAX_OUTPUT_TOKENS_PER_RESPONSE)}
    if uses_openrouter(model):
        return {
            ANTHROPIC_BASE_URL_ENV: OPENROUTER_BASE_URL,
            ANTHROPIC_AUTH_TOKEN_ENV: api_key,
            ANTHROPIC_API_KEY_ENV: "",
            **cap,
        }
    return {OAUTH_TOKEN_ENV: api_key, **cap}


def isolated_session_env(model: str, api_key: str, scratch_dir: str) -> dict[str, str]:
    """Provider auth plus a transcript store private to this attempt."""
    return {
        **provider_env(model, api_key),
        CLAUDE_CONFIG_DIR_ENV: os.path.join(scratch_dir, SESSION_STATE_SUBDIR),
    }


def build_options(
    config: ExperimentConfig,
    scratch_dir: str,
    budget_tokens: int,
    oauth_token: str,
    stderr_tail: StderrTail,
    *,
    session_id: str | None = None,
    resume_session_id: str | None = None,
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
        cli_path=os.environ.get(CLI_PATH_ENV),
        env=isolated_session_env(config.model, oauth_token, scratch_dir),
        effort=config.effort,  # type: ignore[arg-type]
        stderr=stderr_tail,
        system_prompt=system_prompt(),
        allowed_tools=list(ALLOWED_TOOLS),
        disallowed_tools=list(DISALLOWED_TOOLS),
        settings=str(AGENT_SETTINGS_PATH),
        extra_args={"setting-sources": ""},
        permission_mode=PERMISSION_MODE,
        max_turns=config.max_turns_per_phase,
        task_budget={"total": budget_tokens},
        include_partial_messages=True,
        cwd=scratch_dir,
        session_id=session_id,
        resume=resume_session_id,
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
    client: ResumableClaudeSession,
    prompt: str,
    label: str,
    tracker: BudgetTracker,
    stop_at_tokens: int,
    *,
    query_prompt: str | None = None,
    process_resume_count: int = 0,
    discarded_output_text: str = "",
    discarded_tool_calls: list[ToolCall] | None = None,
    reconnect_start: int | None = None,
    on_progress: Callable[[dict[str, object]], None] | None = None,
    on_complete: Callable[[PhaseResult], None] | None = None,
) -> PhaseResult:
    """Send one prompt on an existing session and capture the full response.

    Every assistant message's output tokens are added to the shared tracker;
    the moment cumulative spend reaches stop_at_tokens (the soft limit for
    working phases, the hard budget for the wrap-up phase) the session is
    interrupted and the phase is marked budget_exhausted. Fails loud if no
    ResultMessage arrives or the result is an API error.
    """
    await client.query(query_prompt if query_prompt is not None else prompt)
    if reconnect_start is None:
        reconnect_start = client.reconnect_count

    assistant_messages: list[AssistantMessage] = []
    seen_assistant_ids: set[str] = set()
    tool_uses: dict[str, ToolUseBlock] = {}
    tool_results: dict[str, ToolResultBlock] = {}
    result_message: ResultMessage | None = None
    interrupted = False
    interrupted_connections: set[str] = set()

    current_stream_id: str | None = None
    streamed_text: dict[str, list[str]] = {}

    def progress_record() -> dict[str, object]:
        """Serializable prefix retained if the local process is killed."""
        return {
            "text_parts": [
                *(
                    block.text
                    for message in assistant_messages
                    for block in message.content
                    if isinstance(block, TextBlock)
                ),
                *(
                    "".join(parts)
                    for parts in streamed_text.values()
                    if parts
                ),
            ],
            "seen_assistant_ids": sorted(seen_assistant_ids),
            "current_stream_id": current_stream_id,
            "tool_uses": {
                use_id: {"name": use.name, "input": dict(use.input)}
                for use_id, use in tool_uses.items()
            },
            "tool_results": {
                use_id: {
                    "result": _stringify_result(result.content),
                    "is_error": bool(result.is_error),
                }
                for use_id, result in tool_results.items()
            },
        }

    def save_progress() -> None:
        if on_progress is not None:
            on_progress(progress_record())

    async def interrupt_if_over_budget() -> None:
        nonlocal interrupted
        connection_id = client.connection_id
        if (
            tracker.spent >= stop_at_tokens
            and connection_id not in interrupted_connections
        ):
            interrupted = True
            interrupted_connections.add(connection_id)
            log.info(
                "%s: token cutoff reached (%d/%d, stop at %d); interrupting",
                label,
                tracker.spent,
                tracker.budget_tokens,
                stop_at_tokens,
            )
            await client.interrupt()

    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            # API message ids are stable if a resumed CLI replays the tail;
            # SDK envelope UUIDs need not be.
            assistant_id = message.message_id or message.uuid
            if assistant_id is None or assistant_id not in seen_assistant_ids:
                assistant_messages.append(message)
                if assistant_id is not None:
                    seen_assistant_ids.add(assistant_id)
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        tool_uses[block.id] = block
            if assistant_id is not None:
                streamed_text.pop(assistant_id, None)
            if current_stream_id is not None:
                streamed_text.pop(current_stream_id, None)
            # SDK assistant envelopes should carry the API message id. If one
            # does not, reuse the most recent stream id so its initial usage
            # snapshot cannot be double-counted against message_delta.
            tracker.add(message.message_id or current_stream_id, message.usage)
            save_progress()
            await interrupt_if_over_budget()
        elif isinstance(message, StreamEvent):
            # The real per-message output count arrives in message_delta;
            # AssistantMessage.usage only carries the initial tiny snapshot.
            event_type = message.event.get("type")
            if event_type == "message_start":
                current_stream_id = message.event.get("message", {}).get("id")
                if current_stream_id is not None:
                    streamed_text.setdefault(current_stream_id, [])
                save_progress()
                # If a quota handoff happened after the previous CLI was
                # interrupted, stop the replacement before it can continue
                # materially beyond the same attempt cutoff.
                await interrupt_if_over_budget()
            elif event_type == "message_delta":
                tracker.add(current_stream_id, message.event.get("usage"))
                save_progress()
                await interrupt_if_over_budget()
            elif event_type == "content_block_delta":
                delta = message.event.get("delta", {})
                if (
                    current_stream_id is not None
                    and isinstance(delta, dict)
                    and delta.get("type") == "text_delta"
                ):
                    streamed_text.setdefault(current_stream_id, []).append(
                        str(delta.get("text", ""))
                    )
                    save_progress()
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    tool_results[block.tool_use_id] = block
            save_progress()
        elif isinstance(message, ResultMessage):
            result_message = message

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

    reconnects = client.reconnect_events[reconnect_start:]
    text = _collect_text(assistant_messages)
    if result_message.result is not None and not interrupted and not reconnects:
        text = result_message.result

    result_usage = result_message.usage or {}
    result_tokens = result_usage.get("output_tokens")
    phase_tokens = tracker.finish_phase(
        int(str(result_tokens)) if result_tokens is not None else None
    )

    cost = result_message.total_cost_usd
    stop_reason = (
        STOP_BUDGET_EXHAUSTED
        if interrupted
        else (result_message.stop_reason or "end_turn")
    )
    phase = PhaseResult(
        label=label,
        prompt=prompt,
        text=text,
        output_tokens=phase_tokens,
        cumulative_output_tokens=tracker.spent,
        # ResultMessage.num_turns is per query (unlike total_cost_usd, which
        # accumulates across queries in one live CLI process).
        num_turns=result_message.num_turns,
        duration_ms=result_message.duration_ms,
        total_cost_usd=tracker.phase_cost_delta(
            cost if cost is not None else 0.0, client.connection_id
        ),
        is_error=result_message.is_error,
        stop_reason=stop_reason,
        budget_exhausted=interrupted,
        tool_calls=_collect_tool_calls(tool_uses, tool_results),
        reconnects=reconnects,
        process_resume_count=process_resume_count,
        discarded_output_text=discarded_output_text,
        discarded_tool_calls=list(discarded_tool_calls or []),
    )
    if on_complete is not None:
        on_complete(phase)
    return phase
