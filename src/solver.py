"""One agent phase: build SDK options, stream the response, enforce the budget.

Compute is operationalized as the attempt's total output-token budget (see
README). Enforcement is two-layered: the API-side task_budget tells the model
its remaining budget so it paces itself, and the harness hard-cuts by
interrupting the session the moment cumulative output tokens exceed the budget.
"""

import hashlib
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator, Callable
from functools import cache
from pathlib import Path

import anyio
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
    AUTO_COMPACT_WINDOW,
    CLAUDE_CONFIG_DIR_ENV,
    CLI_PATH_ENV,
    DISALLOWED_TOOLS,
    GPT_5_4_MINI_AUTO_COMPACT_WINDOW,
    ANTHROPIC_API_KEY_ENV,
    ANTHROPIC_AUTH_TOKEN_ENV,
    ANTHROPIC_BASE_URL_ENV,
    CLAUDE_API_TIMEOUT_ENV,
    CLAUDE_API_TIMEOUT_MS,
    CLAUDE_DISABLE_NONSTREAMING_FALLBACK_ENV,
    CLAUDE_ENABLE_STREAM_WATCHDOG_ENV,
    CLAUDE_MAX_API_RETRIES,
    CLAUDE_MAX_API_RETRIES_ENV,
    CLAUDE_STREAM_IDLE_TIMEOUT_ENV,
    CLAUDE_STREAM_IDLE_TIMEOUT_MS,
    MAX_OUTPUT_TOKENS_ENV,
    MAX_OUTPUT_TOKENS_PER_RESPONSE,
    LITELLM_API_KEY_ENV,
    LITELLM_BASE_URL_ENV,
    LITELLM_MODEL_PREFIX,
    OAUTH_TOKEN_ENV,
    OPENROUTER_BASE_URL,
    OPENROUTER_KEY_ENV,
    OPENROUTER_PROXY_URL_ENV,
    PERMISSION_MODE,
    PROVIDER_MIN_TASK_BUDGET_TOKENS,
    PROCESS_RECOVERY_PROMPT,
    SESSION_RECOVERY_PROMPT,
    SESSION_STATE_SUBDIR,
    SPEND_LIMIT_MARKERS,
    TRANSPORT_RECOVERY_BASE_DELAY_SECONDS,
    TRANSPORT_RECOVERY_MAX_DELAY_SECONDS,
    TRANSPORT_RECOVERY_MAX_RETRIES,
    VLLM_AGENT_SETTINGS_PATH,
    VLLM_API_KEY_ENV,
    VLLM_AUTO_COMPACT_WINDOW,
    VLLM_BASE_URL_ENV,
    VLLM_MODEL_PREFIX,
)
from src.models import (
    ExperimentConfig,
    PhaseResult,
    ReconnectEvent,
    TokenSpendLimit,
    ToolCall,
)
from src.openrouter_routing import route_for
from src.prompts import system_prompt
from src.token_pool import TokenPool

log = logging.getLogger("solver")

STOP_BUDGET_EXHAUSTED: str = "budget_exhausted"

_RETRYABLE_TRANSPORT_MARKERS: tuple[str, ...] = (
    "stream ended without receiving any events",
    "stream idle timeout - no chunks received",
    "peer closed connection",
    "incomplete chunked read",
    "remoteprotocolerror",
    "connection reset",
    "connection terminated",
    "disconnect/reset before headers",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "api error: 502",
    "api error: 503",
    "api error: 504",
    "status code: 502",
    "status code: 503",
    "status code: 504",
)


def is_retryable_transport_error(value: object) -> bool:
    """True only for transient stream/proxy failures safe to transcript-resume."""
    text = str(value).lower()
    return any(marker in text for marker in _RETRYABLE_TRANSPORT_MARKERS)


def _assistant_transport_error(message: AssistantMessage) -> str | None:
    """Return a synthetic CLI API-error block, never normal model prose."""
    text = "\n".join(
        block.text for block in message.content if isinstance(block, TextBlock)
    ).strip()
    if message.error == "server_error":
        return text or "Claude server error"
    if text.lower().startswith("api error:") and is_retryable_transport_error(text):
        return text
    return None


def session_recovery_policy() -> dict[str, object]:
    """Non-secret live recovery controls recorded with every result."""
    return {
        "transport_recovery": "same_transcript_continue_v4",
        "empty_success_recovery": "bounded_same_transcript_retry",
        "transport_recovery_max_retries": TRANSPORT_RECOVERY_MAX_RETRIES,
        "transport_recovery_base_delay_seconds": (
            TRANSPORT_RECOVERY_BASE_DELAY_SECONDS
        ),
        "transport_recovery_max_delay_seconds": TRANSPORT_RECOVERY_MAX_DELAY_SECONDS,
    }


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


OptionsFactory = Callable[[str, str | None, str | None, StderrTail], ClaudeAgentOptions]


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
        self._interrupt_requested = False
        # A resumed CLI can replay the transcript tail before it answers the
        # pending query.  run_phase consumes this guard only after observing a
        # genuinely new provider message and a nonzero terminal result.
        self._connection_requires_fresh_output = self._resume_on_enter
        # If a response completed before a late rate-limit/transport envelope,
        # commit it immediately and defer the credential handoff until the next
        # query.  Otherwise a valid answer can sit uncommitted for hours while
        # every replacement credential is cooling.
        self._deferred_recovery: tuple[str, int | None, str] | None = None
        self._recovery_signal_count = 0

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

    @property
    def recovery_signal_count(self) -> int:
        """Number of rate/spend/transport failures observed, handoff or deferred."""
        return self._recovery_signal_count

    @property
    def connection_requires_fresh_output(self) -> bool:
        """Whether this CLI was opened by transcript resume, not a fresh session."""
        return self._connection_requires_fresh_output

    def confirm_fresh_response(self) -> None:
        """Consume the replay guard after run_phase proves a response completed."""
        self._connection_requires_fresh_output = False

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
                        "%s was unusable at session startup; trying another credential",
                        self._pool.credential_label(token),
                    )
                    continue
                await self._pool.release(token)
                raise
            self._token = token
            self._stderr_tail = stderr_tail
            self._client = client
            self._connection_id = str(uuid.uuid4())
            self._connection_requires_fresh_output = resume
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
        elif reason == "transport":
            # The credential/sidecar itself is not known to be exhausted.  A
            # reconnect may therefore reuse it (or round-robin to another
            # configured sidecar) without changing pool eligibility.
            pass
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

    async def _defer_completed_recovery(
        self, reason: str, resets_at: int | None
    ) -> None:
        """Retire a failed credential now; reopen only when another query exists."""
        token = self._token
        if token is None:
            raise RuntimeError("Cannot defer recovery with no leased credential")
        old_label = self._pool.credential_label(token)
        if reason == "rate_limit":
            await self._pool.mark_rate_limited(token, resets_at or 0)
        elif reason == "spend_limit":
            await self._pool.mark_dead(token)
        elif reason != "transport":
            raise ValueError(f"Cannot defer recovery reason: {reason}")
        self._deferred_recovery = (reason, resets_at, old_label)

    async def _finish_deferred_recovery(self) -> None:
        """Rotate a retired completed leg immediately before the next query."""
        deferred = self._deferred_recovery
        if deferred is None:
            return
        reason, resets_at, old_label = deferred
        self._deferred_recovery = None
        await self._close(suppress_errors=True)
        await self._open(resume=True)
        if self._token is None:
            raise RuntimeError("Deferred recovery opened no replacement credential")
        new_label = self._pool.credential_label(self._token)
        self._reconnects.append(
            ReconnectEvent(
                reason=reason,
                resets_at=resets_at,
                from_credential=old_label,
                to_credential=new_label,
            )
        )
        log.warning(
            "Session %s resumed after deferred %s (%s -> %s); preserving "
            "accumulated output-token accounting",
            self._session_id,
            reason,
            old_label,
            new_label,
        )

    async def query(self, prompt: str) -> None:
        """Send a normal experiment prompt to the active conversation."""
        await self._finish_deferred_recovery()
        if self._client is None:
            raise RuntimeError("Session is not open")
        self._pending_prompt = prompt
        self._interrupt_requested = False
        await self._client.query(prompt)

    async def receive_response(self) -> AsyncIterator[object]:
        """Stream one response with bounded, same-transcript live recovery."""
        transport_retries = 0
        while True:
            if self._client is None or self._stderr_tail is None:
                raise RuntimeError("Session is not open")
            rate_limit_reset: int | None = None
            spend_limited = False
            transport_failure: str | None = None
            completed_response = False
            leg_saw_provider_output = False
            leg_saw_fatal_result = False
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
                    if isinstance(message, AssistantMessage):
                        if message.error == "rate_limit":
                            # Claude Code emits this zero-token quota prose as
                            # an AssistantMessage; it is infrastructure, never
                            # experimental model output.
                            if rate_limit_reset is None:
                                rate_limit_reset = 0
                            # Let the phase collector retract any stream deltas
                            # that preceded this synthetic quota envelope. It
                            # never treats error Assistants as model output.
                            yield message
                            continue
                        assistant_error = _assistant_transport_error(message)
                        if assistant_error is not None:
                            transport_failure = assistant_error
                            # Expose the envelope only as a discard signal for
                            # already-streamed infrastructure prose.
                            yield message
                            continue
                        if message.content or message.usage:
                            leg_saw_provider_output = True
                    elif isinstance(message, StreamEvent):
                        leg_saw_provider_output = True
                    if isinstance(message, ResultMessage) and (
                        message.is_error
                        and is_retryable_transport_error(
                            "\n".join(
                                str(part)
                                for part in (message.result, message.errors)
                                if part is not None
                            )
                        )
                    ):
                        transport_failure = str(message.result or message.errors)
                        # Never expose an infrastructure error as the terminal
                        # ResultMessage of an experimental phase.
                        continue
                    if isinstance(message, ResultMessage):
                        if message.is_error:
                            if rate_limit_reset is not None:
                                # The rejected leg's error terminator is not an
                                # experimental completion.
                                continue
                            leg_saw_fatal_result = True
                        else:
                            usage = message.usage or {}
                            output_tokens = usage.get("output_tokens")
                            completed_response = completed_response or (
                                message.num_turns > 0
                                and (
                                    leg_saw_provider_output
                                    or (
                                        output_tokens is not None
                                        and int(str(output_tokens)) > 0
                                    )
                                )
                            )
                    # A rejected/error event can precede valid buffered
                    # Assistant/User/Stream envelopes from the same CLI. The
                    # specific infrastructure envelopes were skipped above;
                    # retain every remaining envelope so a credential handoff
                    # cannot silently erase paid model output or tool calls.
                    yield message
            except Exception as exc:
                if rate_limit_reset is None:
                    try:
                        self._stderr_tail.raise_if_spend_limit()
                    except TokenSpendLimit:
                        spend_limited = True
                    diagnostic = "\n".join([str(exc), *self._stderr_tail.lines[-8:]])
                    if not spend_limited and is_retryable_transport_error(diagnostic):
                        transport_failure = diagnostic
                    elif not spend_limited:
                        raise

            # Claude CLI can terminate a resumed query with a synthetic
            # zero-turn success (usually "No response requested.") after an
            # upstream empty-stream failure.  It is neither a valid model
            # completion nor a fatal API result.  Treat it as the same bounded
            # transport failure so the pending prompt is retried on the stable
            # transcript instead of making the whole benchmark task stop.
            if (
                rate_limit_reset is None
                and not spend_limited
                and transport_failure is None
                and not completed_response
                and not leg_saw_fatal_result
                and not self._interrupt_requested
            ):
                transport_failure = (
                    "provider returned no completed response with fresh output"
                )

            if rate_limit_reset is not None:
                self._recovery_signal_count += 1
                if completed_response:
                    await self._defer_completed_recovery("rate_limit", rate_limit_reset)
                    return
                await self._recover("rate_limit", rate_limit_reset)
            elif spend_limited:
                self._recovery_signal_count += 1
                if completed_response:
                    await self._defer_completed_recovery("spend_limit", None)
                    return
                await self._recover("spend_limit", None)
            elif transport_failure is not None:
                self._recovery_signal_count += 1
                if completed_response:
                    await self._defer_completed_recovery("transport", None)
                    return
                if transport_retries >= TRANSPORT_RECOVERY_MAX_RETRIES:
                    raise RuntimeError(
                        "Transient provider transport failure persisted after "
                        f"{TRANSPORT_RECOVERY_MAX_RETRIES} same-transcript "
                        f"retries: {transport_failure}"
                    )
                transport_retries += 1
                delay = min(
                    TRANSPORT_RECOVERY_MAX_DELAY_SECONDS,
                    TRANSPORT_RECOVERY_BASE_DELAY_SECONDS
                    * (2 ** (transport_retries - 1)),
                )
                # Stable per-session jitter prevents eight simultaneous
                # benchmark attempts from retrying the proxy in lockstep.
                jitter = int(self._session_id.replace("-", "")[:4], 16) / 65_535
                delay += jitter
                log.warning(
                    "Session %s hit transient transport failure; retrying same "
                    "transcript in %.1f s (%d/%d): %s",
                    self._session_id,
                    delay,
                    transport_retries,
                    TRANSPORT_RECOVERY_MAX_RETRIES,
                    transport_failure,
                )
                await anyio.sleep(delay)
                await self._recover("transport", None)
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
        self._interrupt_requested = True
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
        # Zero-token/message-start-only ids must also survive phase boundaries;
        # otherwise a later transcript replay could masquerade as fresh work.
        self._seen_message_ids: set[str] = set()
        # A successful terminal Result seals the stable message ids generated
        # by that query. Its aggregate usage has already trued up the phase, so
        # a later transcript replay must not charge those ids a second time.
        self._sealed_message_ids: set[str] = set()
        self._current_phase_streamed_tokens = 0
        self._prev_session_cost = 0.0
        self._prev_connection_id: str | None = None

    def add(self, message_id: str | None, usage: dict[str, object] | None) -> int:
        """Accumulate the current phase's streamed usage (max snapshot per id).

        Usage snapshots for one API message (message_start, then the real
        count in message_delta) share one id and grow — track the max, never
        the first (undercounts) nor the sum (double counts). Return the newly
        charged delta so callers can distinguish fresh output from a replay.
        """
        if usage is None:
            return 0
        tokens = int(str(usage.get("output_tokens", 0)))
        if message_id is None:
            self._current_phase_streamed_tokens += tokens
            self.spent += tokens
            return tokens
        if message_id in self._sealed_message_ids:
            return 0
        previous = self._message_tokens.get(message_id, 0)
        if tokens > previous:
            delta = tokens - previous
            self._current_phase_streamed_tokens += delta
            self.spent += delta
            self._message_tokens[message_id] = tokens
            return delta
        return 0

    @property
    def known_message_ids(self) -> frozenset[str]:
        """Stable API message ids already charged in this conversation."""
        return (
            frozenset(self._message_tokens)
            | frozenset(self._seen_message_ids)
            | frozenset(self._sealed_message_ids)
        )

    def observe_message_id(self, message_id: str) -> None:
        """Persist an id even when no usage snapshot accompanied its envelope."""
        self._seen_message_ids.add(message_id)

    def seal_message_ids(self, message_ids: set[str]) -> None:
        """Bind a completed query's Result true-up to its stable message ids."""
        self._sealed_message_ids.update(message_ids)

    @property
    def current_phase_streamed_tokens(self) -> int:
        """Unique streamed tokens charged in the active phase, including restarts."""
        return self._current_phase_streamed_tokens

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

    def phase_cost_delta(self, session_cost_usd: float, connection_id: str) -> float:
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
            "seen_message_ids": sorted(self._seen_message_ids),
            "sealed_message_ids": sorted(self._sealed_message_ids),
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
        if (
            int(str(snapshot["budget_tokens"])) != budget_tokens
            or int(str(snapshot["soft_limit_tokens"])) != expected_soft
        ):
            raise ValueError("Checkpoint token budget does not match this invocation")
        tracker.spent = int(str(snapshot["spent"]))
        tracker._completed_phases_tokens = int(str(snapshot["completed_phases_tokens"]))
        raw_messages = snapshot.get("message_tokens", {})
        if not isinstance(raw_messages, dict):
            raise TypeError("Checkpoint message_tokens must be an object")
        tracker._message_tokens = {
            str(message_id): int(tokens) for message_id, tokens in raw_messages.items()
        }
        raw_seen = snapshot["seen_message_ids"]
        if not isinstance(raw_seen, list):
            raise TypeError("Checkpoint seen_message_ids must be a list")
        tracker._seen_message_ids = {str(message_id) for message_id in raw_seen}
        raw_sealed = snapshot["sealed_message_ids"]
        if not isinstance(raw_sealed, list):
            raise TypeError("Checkpoint sealed_message_ids must be a list")
        tracker._sealed_message_ids = {str(message_id) for message_id in raw_sealed}
        tracker._current_phase_streamed_tokens = int(
            str(snapshot["current_phase_streamed_tokens"])
        )
        tracker._prev_session_cost = float(str(snapshot["prev_session_cost"]))
        raw_connection_id = snapshot.get("prev_connection_id")
        tracker._prev_connection_id = (
            str(raw_connection_id) if raw_connection_id else None
        )
        if tracker.spent != (
            tracker._completed_phases_tokens + tracker._current_phase_streamed_tokens
        ):
            raise ValueError("Checkpoint BudgetTracker counters are inconsistent")
        return tracker


def uses_openrouter(model: str) -> bool:
    """True for 'vendor/model' ids, which route through OpenRouter."""
    return "/" in model and not uses_litellm(model) and not uses_vllm(model)


def uses_litellm(model: str) -> bool:
    """True for models routed to the local Codex-subscription sidecar pool."""
    return model.startswith(LITELLM_MODEL_PREFIX)


def uses_vllm(model: str) -> bool:
    """True for models served by a tunneled native Anthropic vLLM endpoint."""
    return model.startswith(VLLM_MODEL_PREFIX)


def agent_settings_path(model: str) -> Path:
    """Return the frozen Claude CLI settings profile for this model route."""
    return VLLM_AGENT_SETTINGS_PATH if uses_vllm(model) else AGENT_SETTINGS_PATH


def auto_compact_window(model: str) -> str:
    """Return the transcript threshold appropriate to the model context."""
    if uses_vllm(model):
        return VLLM_AUTO_COMPACT_WINDOW
    if provider_model_name(model).rsplit("/", 1)[-1] == "gpt-5.4-mini":
        return GPT_5_4_MINI_AUTO_COMPACT_WINDOW
    return AUTO_COMPACT_WINDOW


@cache
def _named_settings_denies(path: Path) -> tuple[str, ...]:
    """Read named tool denies; Bash patterns remain enforced by --settings."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    permissions = raw.get("permissions")
    if not isinstance(permissions, dict):
        raise ValueError(f"{path}: missing permissions object")
    denies = permissions.get("deny")
    if not isinstance(denies, list) or not all(
        isinstance(item, str) for item in denies
    ):
        raise ValueError(f"{path}: permissions.deny must be a string list")
    return tuple(item for item in denies if "(" not in item)


def disallowed_tools_for_model(model: str) -> list[str]:
    """Strip the smaller VLLM profile's named tools from its advertised set."""
    denied = set(DISALLOWED_TOOLS)
    if uses_vllm(model):
        denied.update(_named_settings_denies(agent_settings_path(model)))
    return sorted(denied)


def agent_runtime_policy(model: str) -> dict[str, object]:
    """Non-secret agent controls included in VLLM checkpoint identity."""
    return {
        "settings_profile": agent_settings_path(model).name,
        "autocompact": auto_compact_window(model),
        "disallowed_tools": disallowed_tools_for_model(model),
    }


def provider_model_name(model: str) -> str:
    """Translate a harness model id into the model alias sent to the provider."""
    if uses_litellm(model):
        provider_model = model.removeprefix(LITELLM_MODEL_PREFIX)
        if not provider_model:
            raise ValueError("LiteLLM model id must include a model after 'litellm/'")
        return provider_model
    if uses_vllm(model):
        provider_model = model.removeprefix(VLLM_MODEL_PREFIX)
        if not provider_model:
            raise ValueError("vLLM model id must include a model after 'vllm/'")
        return provider_model
    return model


def token_env_name(model: str) -> str:
    """Name of the env var family holding this model's provider API keys."""
    if uses_litellm(model):
        # Pool entries are isolated sidecar URLs rather than bearer secrets.
        return LITELLM_BASE_URL_ENV
    if uses_vllm(model):
        # Pool entries are native Anthropic endpoint URLs. One vLLM server may
        # internally data-parallelize across several GPUs.
        return VLLM_BASE_URL_ENV
    return OPENROUTER_KEY_ENV if uses_openrouter(model) else OAUTH_TOKEN_ENV


def provider_transport_policy(model: str) -> dict[str, object]:
    """Non-secret transport controls that define a reproducible attempt."""
    openrouter_route = route_for(model)
    if openrouter_route is not None:
        return {
            "policy": "openrouter_frozen_route_v1",
            "route": openrouter_route,
        }
    if uses_vllm(model):
        return {
            "policy": "vllm_native_anthropic_stream_v1",
            "api_timeout_ms": CLAUDE_API_TIMEOUT_MS,
            "stream_watchdog_enabled": True,
            "stream_idle_timeout_ms": CLAUDE_STREAM_IDLE_TIMEOUT_MS,
            "nonstreaming_fallback_enabled": False,
            "automatic_api_retries": CLAUDE_MAX_API_RETRIES,
        }
    if not uses_litellm(model):
        return {"policy": "provider_default_v1"}
    return {
        "policy": "litellm_chatgpt_stream_v2",
        "api_timeout_ms": CLAUDE_API_TIMEOUT_MS,
        "stream_watchdog_enabled": True,
        "stream_idle_timeout_ms": CLAUDE_STREAM_IDLE_TIMEOUT_MS,
        "nonstreaming_fallback_enabled": False,
        "automatic_api_retries": CLAUDE_MAX_API_RETRIES,
        "litellm_router_retries": 0,
        "litellm_timeout_seconds": 3_600,
        "litellm_stream_timeout_seconds": 3_600,
        "litellm_upstream_http_transport": "httpx",
    }


def provider_env(
    model: str,
    api_key: str,
    max_output_tokens_per_response: int = MAX_OUTPUT_TOKENS_PER_RESPONSE,
) -> dict[str, str]:
    """Per-session provider auth and an explicit per-response output cap."""
    cap = {MAX_OUTPUT_TOKENS_ENV: str(max_output_tokens_per_response)}
    if uses_litellm(model):
        proxy_key = os.environ.get(LITELLM_API_KEY_ENV, "").strip()
        if not proxy_key:
            raise ValueError(f"{LITELLM_API_KEY_ENV} is required for {model}")
        return {
            ANTHROPIC_BASE_URL_ENV: api_key.rstrip("/"),
            ANTHROPIC_AUTH_TOKEN_ENV: proxy_key,
            ANTHROPIC_API_KEY_ENV: "",
            CLAUDE_API_TIMEOUT_ENV: str(CLAUDE_API_TIMEOUT_MS),
            CLAUDE_ENABLE_STREAM_WATCHDOG_ENV: "1",
            CLAUDE_STREAM_IDLE_TIMEOUT_ENV: str(CLAUDE_STREAM_IDLE_TIMEOUT_MS),
            CLAUDE_DISABLE_NONSTREAMING_FALLBACK_ENV: "1",
            CLAUDE_MAX_API_RETRIES_ENV: str(CLAUDE_MAX_API_RETRIES),
            **cap,
        }
    if uses_vllm(model):
        vllm_key = os.environ.get(VLLM_API_KEY_ENV, "").strip()
        if not vllm_key:
            raise ValueError(f"{VLLM_API_KEY_ENV} is required for {model}")
        return {
            ANTHROPIC_BASE_URL_ENV: api_key.rstrip("/"),
            ANTHROPIC_AUTH_TOKEN_ENV: vllm_key,
            ANTHROPIC_API_KEY_ENV: "",
            CLAUDE_API_TIMEOUT_ENV: str(CLAUDE_API_TIMEOUT_MS),
            CLAUDE_ENABLE_STREAM_WATCHDOG_ENV: "1",
            CLAUDE_STREAM_IDLE_TIMEOUT_ENV: str(CLAUDE_STREAM_IDLE_TIMEOUT_MS),
            CLAUDE_DISABLE_NONSTREAMING_FALLBACK_ENV: "1",
            CLAUDE_MAX_API_RETRIES_ENV: str(CLAUDE_MAX_API_RETRIES),
            **cap,
        }
    if uses_openrouter(model):
        return {
            ANTHROPIC_BASE_URL_ENV: os.environ.get(
                OPENROUTER_PROXY_URL_ENV, OPENROUTER_BASE_URL
            ).rstrip("/"),
            ANTHROPIC_AUTH_TOKEN_ENV: api_key,
            ANTHROPIC_API_KEY_ENV: "",
            **cap,
        }
    return {OAUTH_TOKEN_ENV: api_key, **cap}


def isolated_session_env(
    model: str,
    api_key: str,
    scratch_dir: str,
    max_output_tokens_per_response: int = MAX_OUTPUT_TOKENS_PER_RESPONSE,
) -> dict[str, str]:
    """Provider auth plus a transcript store private to this attempt."""
    return {
        **provider_env(model, api_key, max_output_tokens_per_response),
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
    max_output_tokens_per_response: int = MAX_OUTPUT_TOKENS_PER_RESPONSE,
    max_turns: int | None = None,
    tools_enabled: bool = True,
) -> ClaudeAgentOptions:
    """Construct agent options for one attempt.

    Tool policy comes from the route-specific settings profile plus
    disallowed_tools; network isolation comes from the container firewall.
    oauth_token is the pool-assigned key for this attempt's session.

    NOTE: user/project settings are excluded via extra_args — the SDK drops
    a falsy setting_sources=[] instead of sending it, so the flag must be
    passed explicitly with an empty value.
    """
    return ClaudeAgentOptions(
        model=provider_model_name(config.model),
        cli_path=os.environ.get(CLI_PATH_ENV),
        env=isolated_session_env(
            config.model,
            oauth_token,
            scratch_dir,
            max_output_tokens_per_response,
        ),
        effort=config.effort,  # type: ignore[arg-type]
        stderr=stderr_tail,
        system_prompt=system_prompt(),
        allowed_tools=list(ALLOWED_TOOLS) if tools_enabled else [],
        disallowed_tools=(
            disallowed_tools_for_model(config.model)
            if tools_enabled
            else sorted(set(DISALLOWED_TOOLS) | set(ALLOWED_TOOLS))
        ),
        settings=str(agent_settings_path(config.model)),
        extra_args={
            "setting-sources": "",
            "autocompact": auto_compact_window(config.model),
        },
        permission_mode=PERMISSION_MODE,
        max_turns=max_turns if max_turns is not None else config.max_turns_per_phase,
        # The provider rejects task budgets below 20k.  This is only its
        # pacing envelope: run_phase still interrupts at the exact local
        # stop_at_tokens cutoff, and wrap-up prompts state the true remainder.
        task_budget={"total": max(PROVIDER_MIN_TASK_BUDGET_TOKENS, budget_tokens)},
        include_partial_messages=True,
        cwd=scratch_dir,
        session_id=session_id,
        resume=resume_session_id,
    )


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
    discarded_text_block_keys: list[str] | None = None,
    discarded_message_ids: list[str] | None = None,
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
    if reconnect_start is None:
        reconnect_start = client.reconnect_count
    recovery_signal_start = int(getattr(client, "recovery_signal_count", 0))
    await client.query(query_prompt if query_prompt is not None else prompt)
    resumed_connection_at_query = bool(
        getattr(client, "connection_requires_fresh_output", False)
    )

    assistant_text_parts: list[str] = []
    seen_message_ids: set[str] = set(discarded_message_ids or [])
    for discarded_message_id in seen_message_ids:
        tracker.observe_message_id(discarded_message_id)
    phase_initial_message_ids = set(tracker.known_message_ids) | seen_message_ids
    globally_seen_message_ids = set(phase_initial_message_ids)
    seen_text_block_keys: set[str] = set(discarded_text_block_keys or [])
    tool_uses: dict[str, ToolUseBlock] = {}
    tool_results: dict[str, ToolResultBlock] = {}
    result_records: list[tuple[str, ResultMessage]] = []
    connection_known_before: dict[str, set[str]] = {}
    connection_resumed: dict[str, bool] = {}
    connection_fresh_message_ids: dict[str, set[str]] = {}
    connection_fresh_output_ids: dict[str, set[str]] = {}
    connection_observed_tokens: dict[str, dict[str, int]] = {}
    result_messages_by_connection: dict[str, ResultMessage] = {}
    result_connection_order: list[str] = []
    interrupted = False
    interrupted_connections: set[str] = set()

    current_stream_id: str | None = None
    streamed_text: dict[str, list[str]] = {}

    def register_connection(connection_id: str) -> None:
        """Snapshot transcript history before this CLI emits any envelopes."""
        if connection_id in connection_known_before:
            return
        connection_known_before[connection_id] = set(globally_seen_message_ids)
        connection_resumed[connection_id] = bool(
            getattr(client, "connection_requires_fresh_output", False)
        )

    def register_message(connection_id: str, message_id: str | None) -> bool:
        """Return true only for a stable id created by this connection/query."""
        register_connection(connection_id)
        if message_id is None:
            return False
        is_fresh = message_id not in connection_known_before[connection_id]
        globally_seen_message_ids.add(message_id)
        seen_message_ids.add(message_id)
        tracker.observe_message_id(message_id)
        if is_fresh:
            connection_fresh_message_ids.setdefault(connection_id, set()).add(
                message_id
            )
        return is_fresh

    def message_is_fresh(connection_id: str, message_id: str | None) -> bool:
        """Classify a stable id without confusing a larger replay with new work."""
        register_message(connection_id, message_id)
        return bool(
            message_id is not None
            and message_id in connection_fresh_message_ids.get(connection_id, set())
        )

    def add_streamed_usage(
        connection_id: str,
        accounting_id: str | None,
        stable_message_id: str | None,
        usage: dict[str, object] | None,
    ) -> int:
        """Charge global novelty while retaining raw local maxima for Result use."""
        fresh_message = register_message(connection_id, stable_message_id)
        if usage is not None:
            tokens = int(str(usage.get("output_tokens", 0)))
            if fresh_message and stable_message_id is not None and tokens > 0:
                connection_fresh_output_ids.setdefault(connection_id, set()).add(
                    stable_message_id
                )
            if accounting_id is not None:
                local = connection_observed_tokens.setdefault(connection_id, {})
                local[accounting_id] = max(local.get(accounting_id, 0), tokens)
        return tracker.add(accounting_id, usage)

    def text_block_key(message_id: str, text: str) -> str:
        """Deduplicate fragments only within the same stable provider message."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"v2:{message_id}:{digest}"

    def anonymous_text_was_discarded(text: str) -> bool:
        """Deduplicate discarded output that had no stable provider message id."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"sha256:{digest}" in seen_text_block_keys

    def progress_record() -> dict[str, object]:
        """Serializable prefix retained if the local process is killed."""
        return {
            "text_parts": [
                *assistant_text_parts,
                *("".join(parts) for parts in streamed_text.values() if parts),
            ],
            "seen_message_ids": sorted(seen_message_ids),
            "seen_text_block_keys": sorted(seen_text_block_keys),
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
        message_connection_id = client.connection_id
        register_connection(message_connection_id)
        if isinstance(message, AssistantMessage):
            # API message ids are stable if a resumed CLI replays the tail;
            # SDK envelope UUIDs need not be. One API message id may arrive in
            # several envelopes (thinking, tools, then final text), so merge
            # blocks instead of dropping the whole repeated message.
            stable_assistant_id = message.message_id or current_stream_id
            accounting_id = stable_assistant_id or message.uuid
            assistant_error = (
                message.error == "rate_limit"
                or _assistant_transport_error(message) is not None
            )
            if assistant_error:
                # Stream deltas can precede the SDK's synthetic quota/server
                # error Assistant. Remove their prose and disqualify their id
                # as fresh experimental output. Any token charge already seen
                # is retained as a conservative upper bound.
                error_message_ids = {
                    message_id
                    for message_id in (stable_assistant_id, current_stream_id)
                    if message_id is not None
                }
                for error_message_id in error_message_ids:
                    streamed_text.pop(error_message_id, None)
                    connection_fresh_output_ids.setdefault(
                        message_connection_id, set()
                    ).discard(error_message_id)
                register_message(message_connection_id, stable_assistant_id)
                save_progress()
                continue
            anonymous_allowed = (
                stable_assistant_id is None
                and not connection_resumed[message_connection_id]
            )
            # A live reconnect may reveal a TextBlock/tool envelope that was
            # generated on the first leg but not delivered before transport
            # failure. Preserve that within-phase evidence. By contrast, ids
            # known before this phase are transcript history (or deliberately
            # discarded crash prefixes) and must never enter the new artifact.
            content_allowed = bool(
                stable_assistant_id is not None
                and stable_assistant_id not in phase_initial_message_ids
            )
            if content_allowed or anonymous_allowed:
                block_identity = stable_assistant_id or accounting_id
                for block in message.content:
                    if isinstance(block, TextBlock):
                        if block_identity is None:
                            if not anonymous_text_was_discarded(block.text):
                                assistant_text_parts.append(block.text)
                        else:
                            key = text_block_key(block_identity, block.text)
                            if key not in seen_text_block_keys:
                                assistant_text_parts.append(block.text)
                                seen_text_block_keys.add(key)
                    elif isinstance(block, ToolUseBlock):
                        tool_uses[block.id] = block
                if stable_assistant_id is not None:
                    streamed_text.pop(stable_assistant_id, None)
                if current_stream_id is not None:
                    streamed_text.pop(current_stream_id, None)
            # SDK assistant envelopes should carry the API message id. If one
            # does not, reuse the most recent stream id so its initial usage
            # snapshot cannot be double-counted against message_delta.
            add_streamed_usage(
                message_connection_id,
                accounting_id,
                stable_assistant_id,
                message.usage,
            )
            save_progress()
            await interrupt_if_over_budget()
        elif isinstance(message, StreamEvent):
            # The real per-message output count arrives in message_delta;
            # AssistantMessage.usage only carries the initial tiny snapshot.
            event_type = message.event.get("type")
            if event_type == "message_start":
                raw_stream_id = message.event.get("message", {}).get("id")
                current_stream_id = (
                    str(raw_stream_id) if raw_stream_id is not None else None
                )
                if current_stream_id is not None and message_is_fresh(
                    message_connection_id, current_stream_id
                ):
                    streamed_text.setdefault(current_stream_id, [])
                save_progress()
                # If a quota handoff happened after the previous CLI was
                # interrupted, stop the replacement before it can continue
                # materially beyond the same attempt cutoff.
                await interrupt_if_over_budget()
            elif event_type == "message_delta":
                add_streamed_usage(
                    message_connection_id,
                    current_stream_id or message.uuid,
                    current_stream_id,
                    message.event.get("usage"),
                )
                save_progress()
                await interrupt_if_over_budget()
            elif event_type == "content_block_delta":
                delta = message.event.get("delta", {})
                fresh_stream = message_is_fresh(
                    message_connection_id, current_stream_id
                )
                if (
                    current_stream_id is not None
                    and fresh_stream
                    and isinstance(delta, dict)
                    and delta.get("type") == "text_delta"
                ):
                    streamed_text.setdefault(current_stream_id, []).append(
                        str(delta.get("text", ""))
                    )
                    save_progress()
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if (
                    isinstance(block, ToolResultBlock)
                    and block.tool_use_id in tool_uses
                ):
                    tool_results[block.tool_use_id] = block
            save_progress()
        elif isinstance(message, ResultMessage):
            result_records.append((message_connection_id, message))
            if message_connection_id not in result_messages_by_connection:
                result_connection_order.append(message_connection_id)
            result_messages_by_connection[message_connection_id] = message

    if not result_records:
        raise RuntimeError(f"Agent produced no ResultMessage for phase '{label}'")

    reconnects = client.reconnect_events[reconnect_start:]
    recovered_phase = bool(
        process_resume_count > 0
        or resumed_connection_at_query
        or reconnects
        or int(getattr(client, "recovery_signal_count", 0)) > recovery_signal_start
    )

    if interrupted:
        result_connection_id, result_message = result_records[-1]
    else:
        completed_results = [
            (connection_id, message)
            for connection_id, message in result_records
            if message.num_turns > 0
            and bool(connection_fresh_output_ids.get(connection_id))
        ]
        if not completed_results:
            last_connection_id, last_message = result_records[-1]
            if last_message.is_error:
                result_connection_id, result_message = (
                    last_connection_id,
                    last_message,
                )
            else:
                raise RuntimeError(
                    f"Phase '{label}' returned no fresh completed provider "
                    "response; leaving its checkpoint incomplete"
                )
        else:
            result_connection_id, result_message = completed_results[-1]

    # An API-error result (auth failure, server error, max-turns error, ...)
    # must never be written as a completed attempt: fail the task so the
    # resumable rerun retries it. A deliberate interrupt is not an error.
    if result_message.is_error and not interrupted:
        raise RuntimeError(
            f"Phase '{label}' failed: subtype={result_message.subtype} "
            f"errors={result_message.errors} result={result_message.result!r}"
        )

    text = "\n".join(part for part in assistant_text_parts if part)
    # Some Claude CLI/SDK versions persist the final assistant text to the raw
    # transcript and emit it as stream deltas, but return result="" and no
    # TextBlock envelope.  Never let that empty aggregate erase real output.
    # Residual streamed text is safe to append: completed AssistantMessage ids
    # are removed from streamed_text above.
    residual_streamed_text = [
        "".join(parts) for parts in streamed_text.values() if "".join(parts).strip()
    ]
    if residual_streamed_text:
        text = "\n".join(part for part in [text, *residual_streamed_text] if part)
    result_text = result_message.result
    if isinstance(result_text, str) and result_text.strip() and not interrupted:
        if not recovered_phase:
            text = result_text
        else:
            # A resumed CLI can return the completed final answer only in the
            # terminal ResultMessage, without replaying its AssistantMessage.
            # Preserve any already-streamed prefix and append the aggregate
            # only when it is not already represented there.
            if result_text.strip() not in text:
                text = "\n".join(part for part in (text, result_text) if part)
    # The tracker charges each stable API message once across all reconnects.
    # Result usage is per query in the Claude SDK. Reconcile it only against
    # messages created by the terminal recovery query; replayed transcript
    # history was generated by an earlier query and is already represented by
    # the tracker's global message-id maxima. If a proxy instead reports a
    # transcript aggregate, this deliberately overcounts rather than risking
    # an over-budget experimental artifact.
    reconciled_phase_tokens = tracker.current_phase_streamed_tokens
    usage = result_message.usage or {}
    raw_tokens = usage.get("output_tokens")
    result_can_supply_unstreamed_tokens = bool(
        connection_fresh_output_ids.get(result_connection_id)
    )
    unstreamed_result_tokens = 0
    result_usage_covers_observed = False
    if raw_tokens is not None and result_can_supply_unstreamed_tokens:
        result_output_tokens = int(str(raw_tokens))
        observed = connection_observed_tokens.get(result_connection_id, {})
        fresh_observed_ids = connection_fresh_message_ids.get(
            result_connection_id, set()
        )
        observed_message_ids = (
            fresh_observed_ids
            if fresh_observed_ids or recovered_phase
            else set(observed)
        )
        observed_on_connection = sum(
            tokens
            for message_id, tokens in observed.items()
            if message_id in observed_message_ids
        )
        unstreamed_result_tokens = max(0, result_output_tokens - observed_on_connection)
        reconciled_phase_tokens += unstreamed_result_tokens
        result_usage_covers_observed = result_output_tokens >= observed_on_connection
    if result_usage_covers_observed and not interrupted:
        tracker.seal_message_ids(
            connection_fresh_message_ids.get(result_connection_id, set())
        )
    phase_tokens = tracker.finish_phase(reconciled_phase_tokens)

    phase_cost = tracker.phase_cost_delta(
        result_message.total_cost_usd
        if result_message.total_cost_usd is not None
        else 0.0,
        result_connection_id,
    )
    provider_usage: dict[str, object] = dict(usage)
    if recovered_phase:
        # Keep every provider-specific field losslessly while exposing the
        # normalized cross-connection count used by the experiment.
        provider_usage["_terminal_result_usage"] = dict(usage)
        provider_usage["_result_usage_by_connection"] = [
            {
                "connection_id": connection_id,
                "usage": dict(message.usage or {}),
            }
            for connection_id in result_connection_order
            for message in [result_messages_by_connection[connection_id]]
        ]
        provider_usage["_recovery_output_accounting"] = (
            "stable_message_ids_plus_per_query_result_conservative"
        )
        provider_usage["output_tokens"] = phase_tokens

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
        num_turns=result_message.num_turns,
        duration_ms=result_message.duration_ms,
        total_cost_usd=phase_cost,
        is_error=result_message.is_error,
        stop_reason=stop_reason,
        budget_exhausted=interrupted,
        tool_calls=_collect_tool_calls(tool_uses, tool_results),
        reconnects=reconnects,
        provider_usage=provider_usage,
        process_resume_count=process_resume_count,
        discarded_output_text=discarded_output_text,
        discarded_tool_calls=list(discarded_tool_calls or []),
    )
    confirm_response = getattr(client, "confirm_fresh_response", None)
    if callable(confirm_response) and bool(
        connection_fresh_output_ids.get(result_connection_id)
    ):
        confirm_response()
    if on_complete is not None:
        on_complete(phase)
    return phase
