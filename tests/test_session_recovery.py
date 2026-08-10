"""Deterministic recovery tests; no provider calls or credentials required."""

import os
import time
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    RateLimitEvent,
    RateLimitInfo,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from src.concurrency import run_all
from src.config import load_config
from src.constants import (
    CONFIG_PATH,
    ANTHROPIC_API_KEY_ENV,
    ANTHROPIC_AUTH_TOKEN_ENV,
    ANTHROPIC_BASE_URL_ENV,
    CLAUDE_API_TIMEOUT_ENV,
    CLAUDE_DISABLE_NONSTREAMING_FALLBACK_ENV,
    CLAUDE_ENABLE_STREAM_WATCHDOG_ENV,
    CLAUDE_MAX_API_RETRIES_ENV,
    CLAUDE_STREAM_IDLE_TIMEOUT_ENV,
    LITELLM_API_KEY_ENV,
    LITELLM_BASE_URL_ENV,
    MAX_OUTPUT_TOKENS_ENV,
    OAUTH_TOKEN_ENV,
    SESSION_RECOVERY_PROMPT,
    VLLM_API_KEY_ENV,
    VLLM_BASE_URL_ENV,
)
from src.solver import (
    BudgetTracker,
    ResumableClaudeSession,
    StderrTail,
    build_options,
    provider_env,
    provider_model_name,
    provider_transport_policy,
    run_phase,
    session_recovery_policy,
    token_env_name,
)
from src.models import Problem
from src.run import run_checkpoint_identity
from src.token_pool import TokenPool


class FakeClaudeSDKClient:
    """Two-credential SDK script: partial output, rejection, then resume."""

    instances: list["FakeClaudeSDKClient"] = []
    queries: list[tuple[str, str]] = []

    def __init__(self, options: ClaudeAgentOptions) -> None:
        self.options = options
        self.token = options.env[OAUTH_TOKEN_ENV]
        self.session_id = options.session_id or options.resume
        if self.session_id is None:
            raise AssertionError("test sessions must have an explicit id")
        self.interrupted = False
        self.response_count = 0
        self.instances.append(self)

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def query(self, prompt: str) -> None:
        self.queries.append((self.token, prompt))

    async def interrupt(self) -> None:
        self.interrupted = True

    async def receive_response(self):  # type: ignore[no-untyped-def]
        self.response_count += 1
        if self.token == "token-a":
            yield StreamEvent(
                uuid="stream-a-start",
                session_id=self.session_id,
                event={"type": "message_start", "message": {"id": "message-a"}},
            )
            yield StreamEvent(
                uuid="stream-a-delta",
                session_id=self.session_id,
                event={"type": "message_delta", "usage": {"output_tokens": 10}},
            )
            yield AssistantMessage(
                content=[TextBlock("partial proof")],
                model="test-model",
                usage={"output_tokens": 2},
                message_id="message-a",
                session_id=self.session_id,
                uuid="assistant-a",
            )
            yield RateLimitEvent(
                rate_limit_info=RateLimitInfo(
                    status="rejected", resets_at=int(time.time()) + 3600
                ),
                uuid="limit-a",
                session_id=self.session_id,
            )
            return

        if self.token != "token-b":
            raise AssertionError(f"unexpected credential: {self.token}")
        if self.response_count == 2:
            yield StreamEvent(
                uuid="stream-c-start",
                session_id=self.session_id,
                event={"type": "message_start", "message": {"id": "message-c"}},
            )
            yield StreamEvent(
                uuid="stream-c-delta",
                session_id=self.session_id,
                event={"type": "message_delta", "usage": {"output_tokens": 4}},
            )
            yield AssistantMessage(
                content=[TextBlock("next phase")],
                model="test-model",
                usage={"output_tokens": 1},
                message_id="message-c",
                session_id=self.session_id,
                uuid="assistant-c",
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=50,
                duration_api_ms=40,
                is_error=False,
                num_turns=2,
                session_id=self.session_id,
                stop_reason="end_turn",
                total_cost_usd=0.02,
                usage={"output_tokens": 4},
                result="next phase",
            )
            return
        # A resume implementation may replay the last persisted message. The
        # harness must deduplicate by message id/uuid rather than double-charge
        # or duplicate its text.
        yield StreamEvent(
            uuid="stream-a-start",
            session_id=self.session_id,
            event={"type": "message_start", "message": {"id": "message-a"}},
        )
        yield StreamEvent(
            uuid="stream-a-delta",
            session_id=self.session_id,
            event={"type": "message_delta", "usage": {"output_tokens": 10}},
        )
        yield AssistantMessage(
            content=[TextBlock("partial proof")],
            model="test-model",
            usage={"output_tokens": 2},
            message_id="message-a",
            session_id=self.session_id,
            uuid="assistant-a",
        )
        yield StreamEvent(
            uuid="stream-b-start",
            session_id=self.session_id,
            event={"type": "message_start", "message": {"id": "message-b"}},
        )
        yield StreamEvent(
            uuid="stream-b-delta",
            session_id=self.session_id,
            event={"type": "message_delta", "usage": {"output_tokens": 7}},
        )
        yield AssistantMessage(
            content=[TextBlock("completed proof")],
            model="test-model",
            usage={"output_tokens": 1},
            message_id="message-b",
            session_id=self.session_id,
            uuid="assistant-b",
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=90,
            is_error=False,
            num_turns=1,
            session_id=self.session_id,
            stop_reason="end_turn",
            total_cost_usd=0.01,
            usage={"output_tokens": 7},
            result="completed proof",
        )


class FakeTransportRecoverySDKClient:
    """First CLI loses its stream; resumed CLI replays and completes it."""

    instances: list["FakeTransportRecoverySDKClient"] = []
    queries: list[str] = []

    def __init__(self, options: ClaudeAgentOptions) -> None:
        self.options = options
        self.session_id = options.session_id or options.resume
        if self.session_id is None:
            raise AssertionError("test sessions must have an explicit id")
        self.ordinal = len(self.instances) + 1
        self.instances.append(self)

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def query(self, prompt: str) -> None:
        self.queries.append(prompt)

    async def interrupt(self) -> None:
        raise AssertionError("unexpected interrupt")

    async def receive_response(self):  # type: ignore[no-untyped-def]
        yield StreamEvent(
            uuid=f"start-{self.ordinal}",
            session_id=self.session_id,
            event={"type": "message_start", "message": {"id": "message-a"}},
        )
        yield StreamEvent(
            uuid=f"delta-{self.ordinal}",
            session_id=self.session_id,
            event={"type": "message_delta", "usage": {"output_tokens": 6}},
        )
        if self.ordinal == 1:
            yield AssistantMessage(
                content=[
                    TextBlock("API Error: Stream ended without receiving any events")
                ],
                model="test-model",
                usage={"output_tokens": 0},
                message_id="synthetic-api-error",
                session_id=self.session_id,
                uuid="synthetic-api-error-envelope",
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=10,
                duration_api_ms=5,
                is_error=True,
                num_turns=0,
                session_id=self.session_id,
                stop_reason=None,
                total_cost_usd=0.0,
                usage=None,
                result="API Error: Stream ended without receiving any events",
            )
            return
        yield AssistantMessage(
            content=[TextBlock("partial proof")],
            model="test-model",
            usage={"output_tokens": 1},
            message_id="message-a",
            session_id=self.session_id,
            uuid="assistant-a",
        )
        yield StreamEvent(
            uuid="start-b",
            session_id=self.session_id,
            event={"type": "message_start", "message": {"id": "message-b"}},
        )
        yield StreamEvent(
            uuid="delta-b",
            session_id=self.session_id,
            event={"type": "message_delta", "usage": {"output_tokens": 3}},
        )
        yield AssistantMessage(
            content=[TextBlock("completed proof")],
            model="test-model",
            usage={"output_tokens": 1},
            message_id="message-b",
            session_id=self.session_id,
            uuid="assistant-b",
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=20,
            duration_api_ms=15,
            is_error=False,
            num_turns=1,
            session_id=self.session_id,
            stop_reason="end_turn",
            total_cost_usd=0.01,
            usage={"output_tokens": 3},
            result="completed proof",
        )


class SessionRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        FakeClaudeSDKClient.instances.clear()
        FakeClaudeSDKClient.queries.clear()
        FakeTransportRecoverySDKClient.instances.clear()
        FakeTransportRecoverySDKClient.queries.clear()

    async def test_transient_empty_stream_resumes_without_double_counting(
        self,
    ) -> None:
        pool = TokenPool(["token-a"], "TEST_TOKEN")
        tracker = BudgetTracker(100, 0)

        def options_factory(
            token: str,
            session_id: str | None,
            resume_id: str | None,
            stderr: StderrTail,
        ) -> ClaudeAgentOptions:
            del stderr
            return ClaudeAgentOptions(
                env={OAUTH_TOKEN_ENV: token},
                session_id=session_id,
                resume=resume_id,
                task_budget={"total": max(1, tracker.remaining)},
                include_partial_messages=True,
            )

        with (
            patch("src.solver.ClaudeSDKClient", FakeTransportRecoverySDKClient),
            patch("src.solver.anyio.sleep", new=AsyncMock()) as sleep,
        ):
            async with ResumableClaudeSession(pool, options_factory) as session:
                result = await run_phase(session, "solve", "solve", tracker, 100)

        self.assertEqual(result.output_tokens, 9)
        self.assertEqual(result.text, "partial proof\ncompleted proof")
        self.assertEqual(len(result.reconnects), 1)
        self.assertEqual(result.reconnects[0].reason, "transport")
        self.assertEqual(result.reconnects[0].from_credential, "credential_1")
        self.assertEqual(result.reconnects[0].to_credential, "credential_1")
        self.assertEqual(len(FakeTransportRecoverySDKClient.instances), 2)
        self.assertIsNotNone(
            FakeTransportRecoverySDKClient.instances[0].options.session_id
        )
        self.assertEqual(
            FakeTransportRecoverySDKClient.instances[1].options.resume,
            FakeTransportRecoverySDKClient.instances[0].options.session_id,
        )
        self.assertEqual(
            FakeTransportRecoverySDKClient.queries,
            [
                "solve",
                f"{SESSION_RECOVERY_PROMPT}\n\nPending request:\nsolve",
            ],
        )
        sleep.assert_awaited_once()

    async def test_zero_turn_success_retries_same_transcript(self) -> None:
        class ZeroThenProofSDKClient:
            instances: list["ZeroThenProofSDKClient"] = []
            queries: list[str] = []

            def __init__(self, options: ClaudeAgentOptions) -> None:
                self.options = options
                self.session_id = options.session_id or options.resume
                if self.session_id is None:
                    raise AssertionError("test sessions must have an explicit id")
                self.ordinal = len(self.instances) + 1
                self.instances.append(self)

            async def connect(self) -> None:
                return None

            async def disconnect(self) -> None:
                return None

            async def query(self, prompt: str) -> None:
                self.queries.append(prompt)

            async def interrupt(self) -> None:
                raise AssertionError("unexpected interrupt")

            async def receive_response(self):  # type: ignore[no-untyped-def]
                if self.ordinal == 1:
                    yield ResultMessage(
                        subtype="success",
                        duration_ms=1,
                        duration_api_ms=1,
                        is_error=False,
                        num_turns=0,
                        session_id=self.session_id,
                        stop_reason="end_turn",
                        total_cost_usd=0.0,
                        usage={"output_tokens": 0},
                        result="No response requested.",
                    )
                    return
                yield StreamEvent(
                    uuid="recovered-start",
                    session_id=self.session_id,
                    event={
                        "type": "message_start",
                        "message": {"id": "recovered-message"},
                    },
                )
                yield StreamEvent(
                    uuid="recovered-delta",
                    session_id=self.session_id,
                    event={
                        "type": "message_delta",
                        "usage": {"output_tokens": 6},
                    },
                )
                yield AssistantMessage(
                    content=[TextBlock("## Final Solution\nRecovered proof.")],
                    model="test-model",
                    usage={"output_tokens": 1},
                    message_id="recovered-message",
                    session_id=self.session_id,
                    uuid="recovered-assistant",
                )
                yield ResultMessage(
                    subtype="success",
                    duration_ms=2,
                    duration_api_ms=2,
                    is_error=False,
                    num_turns=1,
                    session_id=self.session_id,
                    stop_reason="end_turn",
                    total_cost_usd=0.01,
                    usage={"output_tokens": 6},
                    result="## Final Solution\nRecovered proof.",
                )

        pool = TokenPool(["token-a"], "TEST_TOKEN")
        tracker = BudgetTracker(100, 0)

        def options_factory(
            token: str,
            session_id: str | None,
            resume_id: str | None,
            stderr: StderrTail,
        ) -> ClaudeAgentOptions:
            del stderr
            return ClaudeAgentOptions(
                env={OAUTH_TOKEN_ENV: token},
                session_id=session_id,
                resume=resume_id,
                include_partial_messages=True,
            )

        with (
            patch("src.solver.ClaudeSDKClient", ZeroThenProofSDKClient),
            patch("src.solver.anyio.sleep", new=AsyncMock()) as sleep,
        ):
            async with ResumableClaudeSession(pool, options_factory) as session:
                result = await run_phase(session, "solve", "solve", tracker, 100)

        self.assertEqual(result.text, "## Final Solution\nRecovered proof.")
        self.assertEqual(result.output_tokens, 6)
        self.assertEqual(len(result.reconnects), 1)
        self.assertEqual(result.reconnects[0].reason, "transport")
        self.assertEqual(len(ZeroThenProofSDKClient.instances), 2)
        self.assertEqual(
            ZeroThenProofSDKClient.queries,
            [
                "solve",
                f"{SESSION_RECOVERY_PROMPT}\n\nPending request:\nsolve",
            ],
        )
        sleep.assert_awaited_once()

    async def test_nontransient_api_error_is_not_retried(self) -> None:
        class FatalSDKClient(FakeTransportRecoverySDKClient):
            async def receive_response(self):  # type: ignore[no-untyped-def]
                yield ResultMessage(
                    subtype="error",
                    duration_ms=1,
                    duration_api_ms=1,
                    is_error=True,
                    num_turns=0,
                    session_id=self.session_id,
                    stop_reason=None,
                    total_cost_usd=0.0,
                    usage=None,
                    result="API Error: 401 Unauthorized",
                )

        pool = TokenPool(["token-a"], "TEST_TOKEN")
        tracker = BudgetTracker(100, 0)

        def options_factory(
            token: str,
            session_id: str | None,
            resume_id: str | None,
            stderr: StderrTail,
        ) -> ClaudeAgentOptions:
            del stderr
            return ClaudeAgentOptions(
                env={OAUTH_TOKEN_ENV: token},
                session_id=session_id,
                resume=resume_id,
            )

        with patch("src.solver.ClaudeSDKClient", FatalSDKClient):
            async with ResumableClaudeSession(pool, options_factory) as session:
                with self.assertRaisesRegex(RuntimeError, "401 Unauthorized"):
                    await run_phase(session, "solve", "solve", tracker, 100)
        self.assertEqual(len(FatalSDKClient.instances), 1)

    async def test_transient_recovery_is_bounded(self) -> None:
        class PersistentTransientSDKClient(FakeTransportRecoverySDKClient):
            async def receive_response(self):  # type: ignore[no-untyped-def]
                yield ResultMessage(
                    subtype="success",
                    duration_ms=1,
                    duration_api_ms=1,
                    is_error=True,
                    num_turns=0,
                    session_id=self.session_id,
                    stop_reason=None,
                    total_cost_usd=0.0,
                    usage=None,
                    result="API Error: Stream ended without receiving any events",
                )

        pool = TokenPool(["token-a"], "TEST_TOKEN")
        tracker = BudgetTracker(100, 0)

        def options_factory(
            token: str,
            session_id: str | None,
            resume_id: str | None,
            stderr: StderrTail,
        ) -> ClaudeAgentOptions:
            del stderr
            return ClaudeAgentOptions(
                env={OAUTH_TOKEN_ENV: token},
                session_id=session_id,
                resume=resume_id,
            )

        with (
            patch("src.solver.ClaudeSDKClient", PersistentTransientSDKClient),
            patch("src.solver.anyio.sleep", new=AsyncMock()) as sleep,
        ):
            async with ResumableClaudeSession(pool, options_factory) as session:
                with self.assertRaisesRegex(
                    RuntimeError, "persisted after 6 same-transcript retries"
                ):
                    await run_phase(session, "solve", "solve", tracker, 100)
        self.assertEqual(len(PersistentTransientSDKClient.instances), 7)
        self.assertEqual(sleep.await_count, 6)

    async def test_mid_phase_rate_limit_resumes_and_preserves_budget(self) -> None:
        pool = TokenPool(["token-a", "token-b"], "TEST_TOKEN")
        tracker = BudgetTracker(100, 0)

        def options_factory(
            token: str,
            session_id: str | None,
            resume_id: str | None,
            stderr: StderrTail,
        ) -> ClaudeAgentOptions:
            del stderr
            return ClaudeAgentOptions(
                env={OAUTH_TOKEN_ENV: token},
                session_id=session_id,
                resume=resume_id,
                task_budget={"total": max(1, tracker.remaining)},
                include_partial_messages=True,
            )

        with patch("src.solver.ClaudeSDKClient", FakeClaudeSDKClient):
            async with ResumableClaudeSession(pool, options_factory) as session:
                result = await run_phase(
                    session,
                    "solve",
                    "solve",
                    tracker,
                    100,
                    process_resume_count=1,
                )
                next_result = await run_phase(
                    session, "critique", "critique", tracker, 100
                )

        self.assertEqual(result.output_tokens, 17)
        self.assertEqual(result.cumulative_output_tokens, 17)
        self.assertEqual(result.text, "partial proof\ncompleted proof")
        self.assertEqual(result.process_resume_count, 1)
        self.assertEqual(len(result.reconnects), 1)
        self.assertEqual(result.reconnects[0].from_credential, "credential_1")
        self.assertEqual(result.reconnects[0].to_credential, "credential_2")
        self.assertEqual(next_result.output_tokens, 4)
        self.assertEqual(next_result.cumulative_output_tokens, 21)
        self.assertEqual(next_result.reconnects, [])

        first, second = FakeClaudeSDKClient.instances
        self.assertIsNotNone(first.options.session_id)
        self.assertIsNone(first.options.resume)
        self.assertEqual(second.options.resume, first.options.session_id)
        self.assertIsNone(second.options.session_id)
        self.assertEqual(second.options.task_budget, {"total": 90})
        self.assertEqual(
            FakeClaudeSDKClient.queries,
            [
                ("token-a", "solve"),
                (
                    "token-b",
                    f"{SESSION_RECOVERY_PROMPT}\n\nPending request:\nsolve",
                ),
                ("token-b", "critique"),
            ],
        )

    async def test_output_after_rate_limit_event_is_not_discarded(self) -> None:
        class BufferedAfterRejectionSDKClient:
            def __init__(self, options: ClaudeAgentOptions) -> None:
                self.options = options
                self.token = options.env[OAUTH_TOKEN_ENV]
                self.session_id = options.session_id or options.resume
                if self.session_id is None:
                    raise AssertionError("test sessions must have an explicit id")

            async def connect(self) -> None:
                return None

            async def disconnect(self) -> None:
                return None

            async def query(self, prompt: str) -> None:
                self.prompt = prompt

            async def interrupt(self) -> None:
                raise AssertionError("unexpected interrupt")

            async def receive_response(self):  # type: ignore[no-untyped-def]
                if self.token == "token-a":
                    yield RateLimitEvent(
                        rate_limit_info=RateLimitInfo(
                            status="rejected", resets_at=int(time.time()) + 3600
                        ),
                        uuid="limit-first",
                        session_id=self.session_id,
                    )
                    yield StreamEvent(
                        uuid="buffered-start",
                        session_id=self.session_id,
                        event={
                            "type": "message_start",
                            "message": {"id": "buffered-message"},
                        },
                    )
                    yield StreamEvent(
                        uuid="buffered-delta",
                        session_id=self.session_id,
                        event={
                            "type": "message_delta",
                            "usage": {"output_tokens": 6},
                        },
                    )
                    yield AssistantMessage(
                        content=[
                            TextBlock("buffered work"),
                            ToolUseBlock(
                                id="tool-1",
                                name="Bash",
                                input={"command": "true"},
                            ),
                        ],
                        model="test-model",
                        usage={"output_tokens": 1},
                        message_id="buffered-message",
                        session_id=self.session_id,
                        uuid="buffered-assistant",
                    )
                    yield UserMessage(
                        content=[
                            ToolResultBlock(
                                tool_use_id="tool-1", content="command completed"
                            )
                        ],
                        uuid="buffered-tool-result",
                    )
                    return

                yield StreamEvent(
                    uuid="resumed-start",
                    session_id=self.session_id,
                    event={
                        "type": "message_start",
                        "message": {"id": "resumed-message"},
                    },
                )
                yield StreamEvent(
                    uuid="resumed-delta",
                    session_id=self.session_id,
                    event={
                        "type": "message_delta",
                        "usage": {"output_tokens": 4},
                    },
                )
                yield ResultMessage(
                    subtype="success",
                    duration_ms=20,
                    duration_api_ms=15,
                    is_error=False,
                    num_turns=1,
                    session_id=self.session_id,
                    stop_reason="end_turn",
                    total_cost_usd=0.01,
                    usage={"output_tokens": 4},
                    result="## Final Solution\nRecovered proof.",
                )

        pool = TokenPool(["token-a", "token-b"], "TEST_TOKEN")
        tracker = BudgetTracker(100, 0)

        def options_factory(
            token: str,
            session_id: str | None,
            resume_id: str | None,
            stderr: StderrTail,
        ) -> ClaudeAgentOptions:
            del stderr
            return ClaudeAgentOptions(
                env={OAUTH_TOKEN_ENV: token},
                session_id=session_id,
                resume=resume_id,
                include_partial_messages=True,
            )

        with patch("src.solver.ClaudeSDKClient", BufferedAfterRejectionSDKClient):
            async with ResumableClaudeSession(pool, options_factory) as session:
                result = await run_phase(session, "solve", "solve", tracker, 100)

        self.assertEqual(
            result.text,
            "buffered work\n## Final Solution\nRecovered proof.",
        )
        self.assertEqual(result.output_tokens, 10)
        self.assertEqual(len(result.reconnects), 1)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "Bash")
        self.assertEqual(result.tool_calls[0].result, "command completed")

    async def test_fresh_result_before_handoff_survives_zero_turn_replay(self) -> None:
        class FinalThenReplaySDKClient:
            queries: list[tuple[str, str]] = []

            def __init__(self, options: ClaudeAgentOptions) -> None:
                self.options = options
                self.token = options.env[OAUTH_TOKEN_ENV]
                self.session_id = options.session_id or options.resume
                if self.session_id is None:
                    raise AssertionError("test sessions must have an explicit id")

            async def connect(self) -> None:
                return None

            async def disconnect(self) -> None:
                return None

            async def query(self, prompt: str) -> None:
                self.queries.append((self.token, prompt))

            async def interrupt(self) -> None:
                raise AssertionError("unexpected interrupt")

            async def receive_response(self):  # type: ignore[no-untyped-def]
                if self.token == "token-a":
                    yield StreamEvent(
                        uuid="final-start",
                        session_id=self.session_id,
                        event={
                            "type": "message_start",
                            "message": {"id": "final-message"},
                        },
                    )
                    yield StreamEvent(
                        uuid="final-delta",
                        session_id=self.session_id,
                        event={
                            "type": "message_delta",
                            "usage": {"output_tokens": 8},
                        },
                    )
                    yield RateLimitEvent(
                        rate_limit_info=RateLimitInfo(
                            status="rejected", resets_at=int(time.time()) + 3600
                        ),
                        uuid="limit-first",
                        session_id=self.session_id,
                    )
                    yield ResultMessage(
                        subtype="success",
                        duration_ms=20,
                        duration_api_ms=15,
                        is_error=False,
                        num_turns=1,
                        session_id=self.session_id,
                        stop_reason="end_turn",
                        total_cost_usd=0.01,
                        usage={"output_tokens": 8},
                        result="## Final Solution\nProof before handoff.",
                    )
                    return

                yield ResultMessage(
                    subtype="success",
                    duration_ms=1,
                    duration_api_ms=1,
                    is_error=False,
                    num_turns=0,
                    session_id=self.session_id,
                    stop_reason="end_turn",
                    total_cost_usd=0.0,
                    usage={"output_tokens": 0},
                    result="No response requested.",
                )

        pool = TokenPool(["token-a", "token-b"], "TEST_TOKEN")
        tracker = BudgetTracker(100, 0)

        def options_factory(
            token: str,
            session_id: str | None,
            resume_id: str | None,
            stderr: StderrTail,
        ) -> ClaudeAgentOptions:
            del stderr
            return ClaudeAgentOptions(
                env={OAUTH_TOKEN_ENV: token},
                session_id=session_id,
                resume=resume_id,
            )

        with patch("src.solver.ClaudeSDKClient", FinalThenReplaySDKClient):
            async with ResumableClaudeSession(pool, options_factory) as session:
                result = await run_phase(session, "solve", "solve", tracker, 100)

        self.assertEqual(result.text, "## Final Solution\nProof before handoff.")
        self.assertEqual(result.output_tokens, 8)
        # The response completed before handoff, so it is committed now and a
        # credential rotation is deferred until a later phase needs a query.
        self.assertEqual(len(result.reconnects), 0)
        self.assertEqual(FinalThenReplaySDKClient.queries, [("token-a", "solve")])

    async def test_replayed_message_cannot_certify_recovered_completion(self) -> None:
        class ReplayOnlySession:
            reconnect_count = 0
            reconnect_events: list[object] = []
            connection_id = "replacement"

            async def query(self, prompt: str) -> None:
                self.prompt = prompt

            async def interrupt(self) -> None:
                raise AssertionError("unexpected interrupt")

            async def receive_response(self):  # type: ignore[no-untyped-def]
                yield StreamEvent(
                    uuid="replayed-start",
                    session_id="session",
                    event={
                        "type": "message_start",
                        "message": {"id": "old-message"},
                    },
                )
                yield StreamEvent(
                    uuid="replayed-delta",
                    session_id="session",
                    event={
                        "type": "message_delta",
                        "usage": {"output_tokens": 10},
                    },
                )
                yield AssistantMessage(
                    content=[TextBlock("old partial proof")],
                    model="test-model",
                    usage={"output_tokens": 10},
                    message_id="old-message",
                    session_id="session",
                    uuid="replayed-assistant",
                )
                yield ResultMessage(
                    subtype="success",
                    duration_ms=1,
                    duration_api_ms=1,
                    is_error=False,
                    num_turns=1,
                    session_id="session",
                    stop_reason="end_turn",
                    total_cost_usd=0.0,
                    usage={"output_tokens": 10},
                    result="old partial proof",
                )

        tracker = BudgetTracker(100, 0)
        tracker.add("old-message", {"output_tokens": 10})
        with self.assertRaisesRegex(
            RuntimeError, "no fresh completed provider response"
        ):
            await run_phase(  # type: ignore[arg-type]
                ReplayOnlySession(),
                "solve",
                "solve",
                tracker,
                100,
                process_resume_count=1,
            )
        self.assertEqual(tracker.spent, 10)

    async def test_rate_limit_assistant_prose_is_never_model_output(self) -> None:
        class QuotaErrorSDKClient:
            def __init__(self, options: ClaudeAgentOptions) -> None:
                self.options = options
                self.token = options.env[OAUTH_TOKEN_ENV]
                self.session_id = options.session_id or options.resume
                if self.session_id is None:
                    raise AssertionError("test sessions must have an explicit id")

            async def connect(self) -> None:
                return None

            async def disconnect(self) -> None:
                return None

            async def query(self, prompt: str) -> None:
                self.prompt = prompt

            async def interrupt(self) -> None:
                raise AssertionError("unexpected interrupt")

            async def receive_response(self):  # type: ignore[no-untyped-def]
                if self.token == "token-a":
                    yield RateLimitEvent(
                        rate_limit_info=RateLimitInfo(
                            status="rejected", resets_at=int(time.time()) + 3600
                        ),
                        uuid="quota-event",
                        session_id=self.session_id,
                    )
                    yield AssistantMessage(
                        content=[TextBlock("You've hit your weekly usage limit")],
                        model="test-model",
                        error="rate_limit",
                        usage={"output_tokens": 0},
                        message_id="quota-error",
                        session_id=self.session_id,
                        uuid="quota-assistant",
                    )
                    yield ResultMessage(
                        subtype="error",
                        duration_ms=1,
                        duration_api_ms=1,
                        is_error=True,
                        num_turns=0,
                        session_id=self.session_id,
                        stop_reason=None,
                        total_cost_usd=0.0,
                        usage={"output_tokens": 0},
                        result="rate_limit",
                    )
                    return

                yield ResultMessage(
                    subtype="success",
                    duration_ms=1,
                    duration_api_ms=1,
                    is_error=False,
                    num_turns=0,
                    session_id=self.session_id,
                    stop_reason="end_turn",
                    total_cost_usd=0.0,
                    usage={"output_tokens": 0},
                    result="No response requested.",
                )

        pool = TokenPool(["token-a", "token-b"], "TEST_TOKEN")
        tracker = BudgetTracker(100, 0)

        def options_factory(
            token: str,
            session_id: str | None,
            resume_id: str | None,
            stderr: StderrTail,
        ) -> ClaudeAgentOptions:
            del stderr
            return ClaudeAgentOptions(
                env={OAUTH_TOKEN_ENV: token},
                session_id=session_id,
                resume=resume_id,
            )

        with (
            patch("src.solver.ClaudeSDKClient", QuotaErrorSDKClient),
            patch("src.solver.anyio.sleep", new=AsyncMock()),
        ):
            async with ResumableClaudeSession(pool, options_factory) as session:
                with self.assertRaisesRegex(
                    RuntimeError, "transport failure persisted"
                ):
                    await run_phase(session, "solve", "solve", tracker, 100)

        self.assertEqual(tracker.spent, 0)

    async def test_same_message_id_fragments_preserve_final_text(self) -> None:
        class FragmentedSession:
            reconnect_count = 0
            reconnect_events: list[object] = []
            connection_id = "fragmented"

            async def query(self, prompt: str) -> None:
                self.prompt = prompt

            async def interrupt(self) -> None:
                raise AssertionError("unexpected interrupt")

            async def receive_response(self):  # type: ignore[no-untyped-def]
                yield StreamEvent(
                    uuid="fragment-start",
                    session_id="session",
                    event={
                        "type": "message_start",
                        "message": {"id": "shared-message"},
                    },
                )
                yield StreamEvent(
                    uuid="fragment-delta",
                    session_id="session",
                    event={
                        "type": "message_delta",
                        "usage": {"output_tokens": 8},
                    },
                )
                yield AssistantMessage(
                    content=[ThinkingBlock("private reasoning", "signature")],
                    model="test-model",
                    usage={"output_tokens": 8},
                    message_id="shared-message",
                    session_id="session",
                    uuid="thinking-envelope",
                )
                yield AssistantMessage(
                    content=[TextBlock("## Final Solution\nComplete proof.")],
                    model="test-model",
                    usage={"output_tokens": 8},
                    message_id="shared-message",
                    session_id="session",
                    uuid="text-envelope",
                )
                yield ResultMessage(
                    subtype="success",
                    duration_ms=1,
                    duration_api_ms=1,
                    is_error=False,
                    num_turns=1,
                    session_id="session",
                    stop_reason="end_turn",
                    total_cost_usd=0.0,
                    usage={"output_tokens": 8},
                    result="",
                )

        tracker = BudgetTracker(100, 0)
        phase = await run_phase(  # type: ignore[arg-type]
            FragmentedSession(), "solve", "solve", tracker, 100
        )
        self.assertEqual(phase.text, "## Final Solution\nComplete proof.")
        self.assertEqual(phase.output_tokens, 8)

    async def test_round_robin_keys_are_not_concurrency_limits(self) -> None:
        pool = TokenPool(["token-a", "token-b"], "TEST_TOKEN")
        assigned = [await pool.acquire() for _ in range(6)]
        self.assertEqual(
            assigned,
            ["token-a", "token-b", "token-a", "token-b", "token-a", "token-b"],
        )

    async def test_cutoff_is_reapplied_after_credential_handoff(self) -> None:
        pool = TokenPool(["token-a", "token-b"], "TEST_TOKEN")
        tracker = BudgetTracker(100, 0)

        def options_factory(
            token: str,
            session_id: str | None,
            resume_id: str | None,
            stderr: StderrTail,
        ) -> ClaudeAgentOptions:
            del stderr
            return ClaudeAgentOptions(
                env={OAUTH_TOKEN_ENV: token},
                session_id=session_id,
                resume=resume_id,
                task_budget={"total": max(1, tracker.remaining)},
                include_partial_messages=True,
            )

        with patch("src.solver.ClaudeSDKClient", FakeClaudeSDKClient):
            async with ResumableClaudeSession(pool, options_factory) as session:
                result = await run_phase(session, "solve", "solve", tracker, 10)

        first, second = FakeClaudeSDKClient.instances
        self.assertTrue(first.interrupted)
        self.assertTrue(second.interrupted)
        self.assertTrue(result.budget_exhausted)

    async def test_one_credential_reaches_eight_way_concurrency(self) -> None:
        pool = TokenPool(["only-token"], "TEST_TOKEN")
        active = 0
        peak = 0
        lock = anyio.Lock()

        async def task() -> None:
            nonlocal active, peak
            token = await pool.acquire()
            async with lock:
                active += 1
                peak = max(peak, active)
            await anyio.sleep(0.02)
            async with lock:
                active -= 1
            await pool.release(token)

        await run_all([lambda: task() for _ in range(8)], limit=8)
        self.assertEqual(peak, 8)

    def test_expensive_search_controls_use_one_eight_run_bank(self) -> None:
        config = load_config(CONFIG_PATH)
        baseline = config.arms["baseline"].seeds
        banks = config.arms["baseline-parallel"].seeds
        self.assertEqual(config.max_concurrency, 8)
        self.assertEqual(baseline, [1, 2, 3])
        self.assertEqual(banks, [1])

    def test_uniform_strategy_bank_is_exactly_budget_matched(self) -> None:
        config = load_config(CONFIG_PATH)
        arm = config.arms["baseline-uniform-strategy"]
        total = config.budget_tokens(arm)
        executor = (
            total - config.uniform_strategy_plan_tokens
        ) // config.uniform_strategy_branches
        self.assertEqual(arm.seeds, [1])
        self.assertEqual(config.uniform_strategy_branches, 8)
        self.assertEqual(config.uniform_strategy_plan_tokens, 80_000)
        self.assertEqual(executor, 190_000)
        self.assertEqual(
            config.uniform_strategy_plan_tokens
            + config.uniform_strategy_branches * executor,
            total,
        )

    def test_provider_task_budget_respects_twenty_thousand_minimum(self) -> None:
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as scratch:
            options = build_options(
                config,
                scratch,
                6_748,
                "test-token",
                StderrTail(),
                session_id="00000000-0000-0000-0000-000000000000",
            )
        self.assertEqual(options.task_budget, {"total": 20_000})

    def test_litellm_model_uses_sidecar_url_pool_and_bearer_key(self) -> None:
        model = "litellm/gpt-5.4"
        sidecar = "http://olympiad-codex-litellm-2:4000/"
        with patch.dict(os.environ, {LITELLM_API_KEY_ENV: "sk-local"}):
            env = provider_env(model, sidecar)
        self.assertEqual(provider_model_name(model), "gpt-5.4")
        self.assertEqual(token_env_name(model), LITELLM_BASE_URL_ENV)
        self.assertEqual(
            provider_transport_policy(model),
            {
                "policy": "litellm_chatgpt_stream_v2",
                "api_timeout_ms": 3_600_000,
                "stream_watchdog_enabled": True,
                "stream_idle_timeout_ms": 3_600_000,
                "nonstreaming_fallback_enabled": False,
                "automatic_api_retries": 0,
                "litellm_router_retries": 0,
                "litellm_timeout_seconds": 3_600,
                "litellm_stream_timeout_seconds": 3_600,
                "litellm_upstream_http_transport": "httpx",
            },
        )
        self.assertEqual(
            session_recovery_policy(),
            {
                "transport_recovery": "same_transcript_continue_v4",
                "empty_success_recovery": "bounded_same_transcript_retry",
                "transport_recovery_max_retries": 6,
                "transport_recovery_base_delay_seconds": 2.0,
                "transport_recovery_max_delay_seconds": 30.0,
            },
        )
        self.assertEqual(
            env,
            {
                ANTHROPIC_BASE_URL_ENV: sidecar.rstrip("/"),
                ANTHROPIC_AUTH_TOKEN_ENV: "sk-local",
                ANTHROPIC_API_KEY_ENV: "",
                CLAUDE_API_TIMEOUT_ENV: "3600000",
                CLAUDE_ENABLE_STREAM_WATCHDOG_ENV: "1",
                CLAUDE_STREAM_IDLE_TIMEOUT_ENV: "3600000",
                CLAUDE_DISABLE_NONSTREAMING_FALLBACK_ENV: "1",
                CLAUDE_MAX_API_RETRIES_ENV: "0",
                MAX_OUTPUT_TOKENS_ENV: "64000",
            },
        )

    def test_vllm_model_uses_native_anthropic_tunnel(self) -> None:
        model = "vllm/qed-nano"
        endpoint = "http://host.docker.internal:8000/"
        with patch.dict(os.environ, {VLLM_API_KEY_ENV: "qed-local-key"}):
            env = provider_env(model, endpoint)
        self.assertEqual(provider_model_name(model), "qed-nano")
        self.assertEqual(token_env_name(model), VLLM_BASE_URL_ENV)
        self.assertEqual(
            provider_transport_policy(model),
            {
                "policy": "vllm_native_anthropic_stream_v1",
                "api_timeout_ms": 3_600_000,
                "stream_watchdog_enabled": True,
                "stream_idle_timeout_ms": 3_600_000,
                "nonstreaming_fallback_enabled": False,
                "automatic_api_retries": 0,
            },
        )
        self.assertEqual(
            env,
            {
                ANTHROPIC_BASE_URL_ENV: endpoint.rstrip("/"),
                ANTHROPIC_AUTH_TOKEN_ENV: "qed-local-key",
                ANTHROPIC_API_KEY_ENV: "",
                CLAUDE_API_TIMEOUT_ENV: "3600000",
                CLAUDE_ENABLE_STREAM_WATCHDOG_ENV: "1",
                CLAUDE_STREAM_IDLE_TIMEOUT_ENV: "3600000",
                CLAUDE_DISABLE_NONSTREAMING_FALLBACK_ENV: "1",
                CLAUDE_MAX_API_RETRIES_ENV: "0",
                MAX_OUTPUT_TOKENS_ENV: "64000",
            },
        )

    def test_transport_policy_versions_only_local_checkpoint_identity(
        self,
    ) -> None:
        config = load_config(CONFIG_PATH)
        arm = config.arms["baseline"]
        problem = Problem("test", "statement", "combinatorics", None, None, None)
        anthropic_identity = run_checkpoint_identity(config, arm, problem, 1)
        litellm_identity = run_checkpoint_identity(
            replace(config, model="litellm/gpt-5.4"), arm, problem, 1
        )
        vllm_identity = run_checkpoint_identity(
            replace(config, model="vllm/qed-nano"), arm, problem, 1
        )
        self.assertNotIn("provider_transport_policy", anthropic_identity)
        self.assertEqual(
            litellm_identity["provider_transport_policy"],
            provider_transport_policy("litellm/gpt-5.4"),
        )
        self.assertEqual(
            vllm_identity["provider_transport_policy"],
            provider_transport_policy("vllm/qed-nano"),
        )

    def test_wrap_options_are_one_turn_tool_free_and_capped_at_twenty_k(
        self,
    ) -> None:
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as scratch:
            options = build_options(
                config,
                scratch,
                20_000,
                "test-token",
                StderrTail(),
                session_id="00000000-0000-0000-0000-000000000000",
                max_output_tokens_per_response=20_000,
                max_turns=1,
                tools_enabled=False,
            )
        self.assertEqual(options.task_budget, {"total": 20_000})
        self.assertEqual(options.env[MAX_OUTPUT_TOKENS_ENV], "20000")
        self.assertEqual(options.max_turns, 1)
        self.assertEqual(options.allowed_tools, [])

    def test_budget_tracker_handles_missing_ids_and_cross_phase_replay(self) -> None:
        anonymous = BudgetTracker(100, 0)
        anonymous.add(None, {"output_tokens": 5})
        self.assertEqual(anonymous.finish_phase(None), 5)
        self.assertEqual(anonymous.spent, 5)

        tracker = BudgetTracker(100, 0)
        tracker.add("old-message", {"output_tokens": 10})
        self.assertEqual(tracker.finish_phase(7), 10)
        # A resumed CLI replaying a completed-phase message must add zero.
        tracker.add("old-message", {"output_tokens": 10})
        tracker.add("new-message", {"output_tokens": 4})
        self.assertEqual(tracker.finish_phase(4), 4)
        self.assertEqual(tracker.spent, 14)

    def test_cost_counter_resets_only_when_cli_connection_changes(self) -> None:
        tracker = BudgetTracker(100, 0)
        self.assertAlmostEqual(tracker.phase_cost_delta(0.050, "cli-a"), 0.050)
        self.assertAlmostEqual(tracker.phase_cost_delta(0.084, "cli-a"), 0.034)
        # A resumed CLI has a fresh cumulative counter. Its first phase can
        # cost more than the old process's total, so numeric comparisons alone
        # cannot identify this boundary.
        self.assertAlmostEqual(tracker.phase_cost_delta(0.200, "cli-b"), 0.200)
        self.assertAlmostEqual(tracker.phase_cost_delta(0.230, "cli-b"), 0.030)
        restored = BudgetTracker.restore(tracker.snapshot(), 100, 0)
        self.assertAlmostEqual(restored.phase_cost_delta(0.250, "cli-b"), 0.020)
        self.assertAlmostEqual(restored.phase_cost_delta(0.010, "cli-c"), 0.010)

    async def test_empty_sdk_result_preserves_streamed_final_text(self) -> None:
        class StreamOnlySession:
            reconnect_count = 0
            reconnect_events: list[object] = []
            connection_id = "stream-only"

            async def query(self, prompt: str) -> None:
                self.prompt = prompt

            async def interrupt(self) -> None:
                raise AssertionError("unexpected interrupt")

            async def receive_response(self):  # type: ignore[no-untyped-def]
                yield StreamEvent(
                    uuid="start",
                    session_id="session",
                    event={"type": "message_start", "message": {"id": "final"}},
                )
                yield StreamEvent(
                    uuid="text",
                    session_id="session",
                    event={
                        "type": "content_block_delta",
                        "delta": {
                            "type": "text_delta",
                            "text": "## Final Solution\nRecovered proof.",
                        },
                    },
                )
                yield StreamEvent(
                    uuid="usage",
                    session_id="session",
                    event={"type": "message_delta", "usage": {"output_tokens": 8}},
                )
                yield ResultMessage(
                    subtype="success",
                    duration_ms=1,
                    duration_api_ms=1,
                    is_error=False,
                    num_turns=1,
                    session_id="session",
                    stop_reason="end_turn",
                    total_cost_usd=0.0,
                    usage={"output_tokens": 8},
                    result="",
                )

        tracker = BudgetTracker(100, 0)
        phase = await run_phase(  # type: ignore[arg-type]
            StreamOnlySession(), "solve", "solve", tracker, 100
        )
        self.assertEqual(phase.text, "## Final Solution\nRecovered proof.")
        self.assertEqual(phase.output_tokens, 8)

    async def test_empty_zero_turn_process_replay_cannot_complete_attempt(
        self,
    ) -> None:
        class EmptyReplaySession:
            reconnect_count = 0
            reconnect_events: list[object] = []
            connection_id = "empty-replay"

            async def query(self, prompt: str) -> None:
                self.prompt = prompt

            async def interrupt(self) -> None:
                raise AssertionError("unexpected interrupt")

            async def receive_response(self):  # type: ignore[no-untyped-def]
                yield ResultMessage(
                    subtype="success",
                    duration_ms=1,
                    duration_api_ms=1,
                    is_error=False,
                    num_turns=0,
                    session_id="session",
                    stop_reason="end_turn",
                    total_cost_usd=0.0,
                    usage={"output_tokens": 0},
                    result="",
                )

        tracker = BudgetTracker(100, 0)
        tracker.add("pre-crash", {"output_tokens": 12})
        with self.assertRaisesRegex(
            RuntimeError, "no fresh completed provider response"
        ):
            await run_phase(  # type: ignore[arg-type]
                EmptyReplaySession(),
                "solve",
                "solve",
                tracker,
                100,
                process_resume_count=1,
            )
        self.assertEqual(tracker.spent, 12)


if __name__ == "__main__":
    unittest.main()
