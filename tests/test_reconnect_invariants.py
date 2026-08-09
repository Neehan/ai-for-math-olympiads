"""Adversarial invariants for transcript replay and reconnect accounting.

These tests intentionally model awkward event orderings observed from resumed
Claude CLI transcripts.  They are kept separate from the ordinary recovery
tests so a permissive collector change cannot silently weaken the invariants.
"""

from __future__ import annotations

import hashlib
import time
import unittest
from collections.abc import AsyncIterator
from unittest.mock import patch

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    RateLimitEvent,
    RateLimitInfo,
    ResultMessage,
    StreamEvent,
    TextBlock,
)

from src.constants import OAUTH_TOKEN_ENV
from src.solver import BudgetTracker, ResumableClaudeSession, StderrTail, run_phase
from src.token_pool import TokenPool


def _start(message_id: str) -> StreamEvent:
    return StreamEvent(
        uuid=f"start-{message_id}",
        session_id="session",
        event={"type": "message_start", "message": {"id": message_id}},
    )


def _usage(message_id: str, output_tokens: int) -> StreamEvent:
    return StreamEvent(
        uuid=f"usage-{message_id}-{output_tokens}",
        session_id="session",
        event={
            "type": "message_delta",
            "usage": {"output_tokens": output_tokens},
        },
    )


def _text_delta(message_id: str, text: str) -> StreamEvent:
    return StreamEvent(
        uuid=f"text-{message_id}",
        session_id="session",
        event={
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": text},
        },
    )


def _assistant(message_id: str, text: str, output_tokens: int) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text)],
        model="test-model",
        usage={"output_tokens": output_tokens},
        message_id=message_id,
        session_id="session",
        uuid=f"assistant-{message_id}-{hashlib.sha256(text.encode()).hexdigest()[:8]}",
    )


def _result(
    output_tokens: int,
    text: str,
    *,
    num_turns: int = 1,
) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=num_turns,
        session_id="session",
        stop_reason="end_turn",
        total_cost_usd=0.01,
        usage={"output_tokens": output_tokens},
        result=text,
    )


def _result_without_usage(text: str, *, num_turns: int = 1) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=num_turns,
        session_id="session",
        stop_reason="end_turn",
        total_cost_usd=0.01,
        usage=None,
        result=text,
    )


class StaticSession:
    """Small run_phase-compatible session with one deterministic connection."""

    reconnect_count = 0
    reconnect_events: list[object] = []
    connection_id = "replacement"

    def __init__(self, messages: list[object]) -> None:
        self.messages = messages

    async def query(self, prompt: str) -> None:
        self.prompt = prompt

    async def interrupt(self) -> None:
        raise AssertionError("unexpected interrupt")

    async def receive_response(self) -> AsyncIterator[object]:
        for message in self.messages:
            yield message


class ReconnectInvariantTests(unittest.IsolatedAsyncioTestCase):
    async def test_larger_snapshot_of_old_id_cannot_certify_completion(self) -> None:
        tracker = BudgetTracker(100, 0)
        tracker.add("old", {"output_tokens": 10})
        session = StaticSession(
            [
                _start("old"),
                _usage("old", 15),
                _result(15, "stale transcript aggregate"),
            ]
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "no fresh completed provider response",
        ):
            await run_phase(  # type: ignore[arg-type]
                session,
                "solve",
                "solve",
                tracker,
                100,
                process_resume_count=1,
            )

        # The larger provider snapshot is charged once, never as 10 + 15.
        self.assertEqual(tracker.spent, 15)

    async def test_new_zero_token_id_cannot_certify_stale_result(self) -> None:
        tracker = BudgetTracker(100, 0)
        tracker.add("old", {"output_tokens": 10})
        session = StaticSession(
            [
                _start("empty-new-id"),
                _start("old"),
                _usage("old", 10),
                _result(10, "stale transcript aggregate"),
            ]
        )

        with self.assertRaisesRegex(
            RuntimeError, "no fresh completed provider response"
        ):
            await run_phase(  # type: ignore[arg-type]
                session,
                "solve",
                "solve",
                tracker,
                100,
                process_resume_count=1,
            )

        self.assertEqual(tracker.spent, 10)

    async def test_zero_token_assistant_cannot_certify_stale_result(self) -> None:
        tracker = BudgetTracker(100, 0)
        tracker.add("old", {"output_tokens": 10})
        session = StaticSession(
            [
                _start("synthetic"),
                _assistant("synthetic", "No response requested.", 0),
                _start("old"),
                _usage("old", 10),
                _result(10, "stale transcript aggregate"),
            ]
        )

        with self.assertRaisesRegex(
            RuntimeError, "no fresh completed provider response"
        ):
            await run_phase(  # type: ignore[arg-type]
                session,
                "solve",
                "solve",
                tracker,
                100,
                process_resume_count=1,
            )
        self.assertEqual(tracker.spent, 10)

    async def test_unmetered_fresh_proof_is_rejected(self) -> None:
        session = StaticSession(
            [
                _start("fresh"),
                AssistantMessage(
                    content=[TextBlock("PROOF")],
                    model="test-model",
                    usage=None,
                    message_id="fresh",
                    session_id="session",
                    uuid="assistant-fresh",
                ),
                _result_without_usage("PROOF"),
            ]
        )

        with self.assertRaisesRegex(
            RuntimeError, "no fresh completed provider response"
        ):
            await run_phase(  # type: ignore[arg-type]
                session,
                "solve",
                "solve",
                BudgetTracker(100, 0),
                100,
            )

    async def test_error_stream_prose_is_removed_before_recovery_proof(self) -> None:
        for error in ("rate_limit", "server_error"):
            with self.subTest(error=error):
                error_text = (
                    "You've hit your weekly usage limit"
                    if error == "rate_limit"
                    else "API Error: Stream idle timeout - no chunks received"
                )
                session = StaticSession(
                    [
                        _start("infrastructure"),
                        _text_delta("infrastructure", error_text),
                        _usage("infrastructure", 0),
                        AssistantMessage(
                            content=[TextBlock(error_text)],
                            model="test-model",
                            error=error,
                            usage={"output_tokens": 0},
                            message_id="infrastructure",
                            session_id="session",
                            uuid=f"assistant-{error}",
                        ),
                        _start("fresh"),
                        _usage("fresh", 4),
                        _assistant("fresh", "PROOF", 4),
                        _result(4, "PROOF"),
                    ]
                )

                phase = await run_phase(  # type: ignore[arg-type]
                    session,
                    "solve",
                    "solve",
                    BudgetTracker(100, 0),
                    100,
                )

                self.assertEqual(phase.text, "PROOF")
                self.assertNotIn(error_text, phase.text)

    async def test_discarded_zero_usage_id_remains_known_after_phase(self) -> None:
        tracker = BudgetTracker(100, 0)
        first = StaticSession(
            [
                _start("fresh"),
                _usage("fresh", 4),
                _assistant("fresh", "PROOF", 4),
                _result(4, "PROOF"),
            ]
        )
        await run_phase(  # type: ignore[arg-type]
            first,
            "solve",
            "solve",
            tracker,
            100,
            discarded_message_ids=["discarded-zero-usage"],
        )

        restored = BudgetTracker.restore(tracker.snapshot(), 100, 0)
        replay = StaticSession(
            [
                _start("discarded-zero-usage"),
                _usage("discarded-zero-usage", 5),
                _assistant("discarded-zero-usage", "STALE", 5),
                _result(5, "STALE"),
            ]
        )
        with self.assertRaisesRegex(
            RuntimeError, "no fresh completed provider response"
        ):
            await run_phase(  # type: ignore[arg-type]
                replay,
                "revise",
                "revise",
                restored,
                100,
                process_resume_count=1,
            )

    async def test_terminal_true_up_is_sealed_across_process_restart(self) -> None:
        tracker = BudgetTracker(100, 0)
        first = StaticSession(
            [
                _start("old"),
                _assistant("old", "OLD", 1),
                _result(10, "OLD"),
            ]
        )
        phase_one = await run_phase(  # type: ignore[arg-type]
            first, "solve", "solve", tracker, 100
        )
        self.assertEqual(phase_one.output_tokens, 10)

        restored = BudgetTracker.restore(tracker.snapshot(), 100, 0)
        second = StaticSession(
            [
                _start("old"),
                _usage("old", 10),
                _assistant("old", "OLD", 10),
                _start("fresh"),
                _usage("fresh", 4),
                _assistant("fresh", "NEW", 4),
                _result(4, "NEW"),
            ]
        )
        phase_two = await run_phase(  # type: ignore[arg-type]
            second,
            "revise",
            "revise",
            restored,
            100,
            process_resume_count=1,
        )

        self.assertEqual(phase_two.text, "NEW")
        self.assertEqual(phase_two.output_tokens, 4)
        self.assertEqual(phase_two.cumulative_output_tokens, 14)

    async def test_equal_terminal_usage_also_seals_completed_id(self) -> None:
        tracker = BudgetTracker(100, 0)
        first = StaticSession(
            [
                _start("old"),
                _usage("old", 10),
                _assistant("old", "OLD", 10),
                _result(10, "OLD"),
            ]
        )
        await run_phase(  # type: ignore[arg-type]
            first, "solve", "solve", tracker, 100
        )

        restored = BudgetTracker.restore(tracker.snapshot(), 100, 0)
        replay = StaticSession([_start("old"), _usage("old", 12), _result(12, "OLD")])
        with self.assertRaisesRegex(
            RuntimeError, "no fresh completed provider response"
        ):
            await run_phase(  # type: ignore[arg-type]
                replay,
                "revise",
                "revise",
                restored,
                100,
                process_resume_count=1,
            )
        self.assertEqual(restored.spent, 10)

    async def test_cutoff_does_not_true_up_from_stale_recovery_result(self) -> None:
        class InterruptibleSession(StaticSession):
            async def interrupt(self) -> None:
                self.interrupted = True

        tracker = BudgetTracker(100, 0)
        tracker.add("old", {"output_tokens": 10})
        session = InterruptibleSession([_start("old"), _result(10, "stale")])

        phase = await run_phase(  # type: ignore[arg-type]
            session,
            "solve",
            "solve",
            tracker,
            10,
            process_resume_count=1,
        )

        self.assertTrue(phase.budget_exhausted)
        self.assertEqual(phase.output_tokens, 10)
        self.assertEqual(phase.cumulative_output_tokens, 10)

    async def test_replayed_old_and_fresh_id_aggregate_counts_once(self) -> None:
        tracker = BudgetTracker(100, 0)
        tracker.add("old", {"output_tokens": 6})
        session = StaticSession(
            [
                _start("old"),
                _usage("old", 6),
                _start("fresh"),
                _usage("fresh", 4),
                _assistant("fresh", "## Final Solution\nNew proof.", 4),
                # Claude SDK Result usage is per query, so the terminal leg
                # reports only the fresh suffix.
                _result(4, "## Final Solution\nNew proof."),
            ]
        )

        phase = await run_phase(  # type: ignore[arg-type]
            session,
            "solve",
            "solve",
            tracker,
            100,
            process_resume_count=1,
        )

        self.assertEqual(phase.output_tokens, 10)
        self.assertEqual(phase.cumulative_output_tokens, 10)

    async def test_aggregate_result_usage_is_conservatively_overcounted(self) -> None:
        tracker = BudgetTracker(100, 0)
        tracker.add("old", {"output_tokens": 6})
        session = StaticSession(
            [
                _start("old"),
                _usage("old", 6),
                _start("fresh"),
                _usage("fresh", 4),
                _assistant("fresh", "complete", 4),
                # A nonstandard proxy might report old+fresh here. Treating
                # that as per-query usage overcounts but can never admit an
                # artifact beyond its stated compute budget.
                _result(10, "complete"),
            ]
        )

        phase = await run_phase(  # type: ignore[arg-type]
            session,
            "solve",
            "solve",
            tracker,
            100,
            process_resume_count=1,
        )

        self.assertEqual(phase.output_tokens, 16)

    async def test_zero_turn_replay_at_phase_boundary_is_rejected(self) -> None:
        tracker = BudgetTracker(100, 0)
        tracker.add("prior-phase", {"output_tokens": 8})
        tracker.finish_phase(8)
        session = StaticSession([_result(0, "No response requested.", num_turns=0)])
        # This is a clean new phase after the controller restarted: there is no
        # active-phase resume counter and no reconnect inside this phase.
        session.connection_requires_fresh_output = True

        with self.assertRaisesRegex(
            RuntimeError,
            "no fresh completed provider response",
        ):
            await run_phase(  # type: ignore[arg-type]
                session,
                "solve",
                "solve",
                tracker,
                100,
            )
        self.assertEqual(tracker.spent, 8)

    async def test_same_text_new_id_is_preserved_same_id_replay_is_suppressed(
        self,
    ) -> None:
        repeated = "A necessary repeated sentence."
        session = StaticSession(
            [
                _start("message-a"),
                _usage("message-a", 2),
                _assistant("message-a", repeated, 2),
                _assistant("message-a", repeated, 2),
                _start("message-b"),
                _usage("message-b", 2),
                _assistant("message-b", repeated, 2),
                _result(4, ""),
            ]
        )

        phase = await run_phase(  # type: ignore[arg-type]
            session,
            "solve",
            "solve",
            BudgetTracker(100, 0),
            100,
        )

        self.assertEqual(phase.text, f"{repeated}\n{repeated}")
        self.assertEqual(phase.output_tokens, 4)

    async def test_streamed_normal_response_accepts_missing_result_usage(self) -> None:
        result = _result(0, "## Final Solution\nProof.")
        result = ResultMessage(
            subtype=result.subtype,
            duration_ms=result.duration_ms,
            duration_api_ms=result.duration_api_ms,
            is_error=False,
            num_turns=1,
            session_id=result.session_id,
            stop_reason=result.stop_reason,
            total_cost_usd=result.total_cost_usd,
            usage=None,
            result=result.result,
        )
        session = StaticSession([_start("fresh"), _usage("fresh", 8), result])

        phase = await run_phase(  # type: ignore[arg-type]
            session,
            "solve",
            "solve",
            BudgetTracker(100, 0),
            100,
        )

        self.assertEqual(phase.output_tokens, 8)
        self.assertEqual(phase.text, "## Final Solution\nProof.")

    async def test_replayed_stream_text_is_not_committed(self) -> None:
        old_text = "## Final Solution\nDiscarded partial proof."
        new_text = "## Final Solution\nFresh complete proof."
        old_key = "sha256:" + hashlib.sha256(old_text.encode()).hexdigest()
        tracker = BudgetTracker(100, 0)
        tracker.add("old", {"output_tokens": 5})
        session = StaticSession(
            [
                _start("old"),
                _text_delta("old", old_text),
                _usage("old", 5),
                _start("fresh"),
                _usage("fresh", 4),
                _assistant("fresh", new_text, 4),
                _result(4, ""),
            ]
        )

        phase = await run_phase(  # type: ignore[arg-type]
            session,
            "solve",
            "solve",
            tracker,
            100,
            process_resume_count=1,
            discarded_text_block_keys=[old_key],
        )

        self.assertEqual(phase.text, new_text)
        self.assertNotIn("Discarded partial proof", phase.text)
        self.assertEqual(phase.output_tokens, 9)

    async def test_result_before_rate_limit_does_not_query_replacement(self) -> None:
        class ResultThenRateLimitSDKClient:
            queries: list[tuple[str, str]] = []

            def __init__(self, options: ClaudeAgentOptions) -> None:
                self.options = options
                self.token = options.env[OAUTH_TOKEN_ENV]
                self.session_id = options.session_id or options.resume
                if self.session_id is None:
                    raise AssertionError("test sessions need an explicit id")

            async def connect(self) -> None:
                return None

            async def disconnect(self) -> None:
                return None

            async def query(self, prompt: str) -> None:
                self.queries.append((self.token, prompt))

            async def interrupt(self) -> None:
                raise AssertionError("unexpected interrupt")

            async def receive_response(self) -> AsyncIterator[object]:
                if self.token == "token-a":
                    yield _start("final")
                    yield _usage("final", 8)
                    yield _result(8, "## Final Solution\nCompleted before quota event.")
                    # A later synthetic success must not erase the already
                    # completed-response latch.
                    yield _result(0, "No response requested.", num_turns=0)
                    yield RateLimitEvent(
                        rate_limit_info=RateLimitInfo(
                            status="rejected",
                            resets_at=int(time.time()) + 3600,
                        ),
                        uuid="late-rate-limit",
                        session_id=self.session_id,
                    )
                    return
                yield _result(0, "No response requested.", num_turns=0)

        pool = TokenPool(["token-a", "token-b"], "TEST_TOKEN")

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

        with patch("src.solver.ClaudeSDKClient", ResultThenRateLimitSDKClient):
            async with ResumableClaudeSession(pool, options_factory) as session:
                phase = await run_phase(
                    session,
                    "solve",
                    "solve",
                    BudgetTracker(100, 0),
                    100,
                )

        self.assertEqual(phase.output_tokens, 8)
        self.assertEqual(
            ResultThenRateLimitSDKClient.queries,
            [("token-a", "solve")],
        )

    async def test_late_spend_limit_defers_handoff_until_next_query(self) -> None:
        stderr_by_token: dict[str, StderrTail] = {}

        class ResultThenSpendLimitSDKClient:
            queries: list[tuple[str, str]] = []

            def __init__(self, options: ClaudeAgentOptions) -> None:
                self.token = options.env[OAUTH_TOKEN_ENV]
                self.session_id = options.session_id or options.resume
                if self.session_id is None:
                    raise AssertionError("test sessions need an explicit id")

            async def connect(self) -> None:
                return None

            async def disconnect(self) -> None:
                return None

            async def query(self, prompt: str) -> None:
                self.queries.append((self.token, prompt))

            async def interrupt(self) -> None:
                raise AssertionError("unexpected interrupt")

            async def receive_response(self) -> AsyncIterator[object]:
                message_id = "first" if self.token == "token-a" else "second"
                text = "FIRST" if self.token == "token-a" else "SECOND"
                yield _start(message_id)
                yield _usage(message_id, 4)
                yield _assistant(message_id, text, 4)
                yield _result(4, text)
                if self.token == "token-a":
                    stderr_by_token[self.token]("usage limit reached")
                    raise RuntimeError("Claude CLI exited")

        pool = TokenPool(["token-a", "token-b"], "TEST_TOKEN")

        def options_factory(
            token: str,
            session_id: str | None,
            resume_id: str | None,
            stderr: StderrTail,
        ) -> ClaudeAgentOptions:
            stderr_by_token[token] = stderr
            return ClaudeAgentOptions(
                env={OAUTH_TOKEN_ENV: token},
                session_id=session_id,
                resume=resume_id,
                include_partial_messages=True,
            )

        with patch("src.solver.ClaudeSDKClient", ResultThenSpendLimitSDKClient):
            async with ResumableClaudeSession(pool, options_factory) as session:
                tracker = BudgetTracker(100, 0)
                first = await run_phase(session, "solve", "solve", tracker, 100)
                self.assertEqual(first.text, "FIRST")
                self.assertEqual(first.reconnects, [])
                self.assertEqual(
                    ResultThenSpendLimitSDKClient.queries,
                    [("token-a", "solve")],
                )

                second = await run_phase(session, "critique", "critique", tracker, 100)

        self.assertEqual(second.text, "SECOND")
        self.assertEqual(len(second.reconnects), 1)
        self.assertEqual(second.reconnects[0].reason, "spend_limit")
        self.assertEqual(
            ResultThenSpendLimitSDKClient.queries,
            [("token-a", "solve"), ("token-b", "critique")],
        )

    async def test_partial_then_zero_turn_rate_result_requires_recovery(self) -> None:
        class PartialZeroThenRecoverySDKClient:
            queries: list[tuple[str, str]] = []

            def __init__(self, options: ClaudeAgentOptions) -> None:
                self.token = options.env[OAUTH_TOKEN_ENV]
                self.session_id = options.session_id or options.resume
                if self.session_id is None:
                    raise AssertionError("test sessions need an explicit id")

            async def connect(self) -> None:
                return None

            async def disconnect(self) -> None:
                return None

            async def query(self, prompt: str) -> None:
                self.queries.append((self.token, prompt))

            async def interrupt(self) -> None:
                raise AssertionError("unexpected interrupt")

            async def receive_response(self) -> AsyncIterator[object]:
                if self.token == "token-a":
                    yield _start("partial")
                    yield _usage("partial", 4)
                    yield _assistant("partial", "unfinished", 4)
                    yield RateLimitEvent(
                        rate_limit_info=RateLimitInfo(
                            status="rejected",
                            resets_at=int(time.time()) + 3600,
                        ),
                        uuid="rate-limit",
                        session_id=self.session_id,
                    )
                    yield _result(0, "No response requested.", num_turns=0)
                    return
                yield _start("replacement")
                yield _usage("replacement", 3)
                yield _assistant("replacement", "complete", 3)
                yield _result(3, "complete")

        pool = TokenPool(["token-a", "token-b"], "TEST_TOKEN")

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

        with patch("src.solver.ClaudeSDKClient", PartialZeroThenRecoverySDKClient):
            async with ResumableClaudeSession(pool, options_factory) as session:
                phase = await run_phase(
                    session,
                    "solve",
                    "solve",
                    BudgetTracker(100, 0),
                    100,
                )

        self.assertEqual(phase.output_tokens, 7)
        self.assertEqual(phase.text, "unfinished\ncomplete")
        self.assertEqual(len(PartialZeroThenRecoverySDKClient.queries), 2)


if __name__ == "__main__":
    unittest.main()
