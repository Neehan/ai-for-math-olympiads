"""Deterministic recovery tests; no provider calls or credentials required."""

import time
import unittest
from unittest.mock import patch

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    RateLimitEvent,
    RateLimitInfo,
    ResultMessage,
    StreamEvent,
    TextBlock,
)

from src.concurrency import run_all
from src.config import load_config
from src.constants import CONFIG_PATH, OAUTH_TOKEN_ENV, SESSION_RECOVERY_PROMPT
from src.solver import (
    BudgetTracker,
    ResumableClaudeSession,
    StderrTail,
    run_phase,
)
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


class SessionRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        FakeClaudeSDKClient.instances.clear()
        FakeClaudeSDKClient.queries.clear()

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
                result = await run_phase(session, "solve", "solve", tracker, 100)
                next_result = await run_phase(
                    session, "critique", "critique", tracker, 100
                )

        self.assertEqual(result.output_tokens, 17)
        self.assertEqual(result.cumulative_output_tokens, 17)
        self.assertEqual(result.text, "partial proof\ncompleted proof")
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
                ("token-b", SESSION_RECOVERY_PROMPT),
                ("token-b", "critique"),
            ],
        )

    async def test_round_robin_keys_are_not_concurrency_limits(self) -> None:
        pool = TokenPool(["token-a", "token-b"], "TEST_TOKEN")
        assigned = [await pool.acquire() for _ in range(6)]
        self.assertEqual(
            assigned,
            ["token-a", "token-b", "token-a", "token-b", "token-a", "token-b"],
        )

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

    def test_parallel_bank_is_exactly_eight_disjoint_seeds(self) -> None:
        config = load_config(CONFIG_PATH)
        baseline = config.arms["baseline"].seeds
        extension = config.arms["baseline-parallel"].seeds
        self.assertEqual(config.max_concurrency, 8)
        self.assertEqual(set(baseline) & set(extension), set())
        self.assertEqual(sorted(baseline + extension), list(range(1, 9)))

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


if __name__ == "__main__":
    unittest.main()
