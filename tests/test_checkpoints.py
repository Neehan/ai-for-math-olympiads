"""Durable checkpoint invariants; no provider calls or credentials required."""

import hashlib
import os
import shutil
import tempfile
import unittest
import fcntl
from pathlib import Path
from unittest.mock import patch

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    TextBlock,
)

from src.checkpoint import AttemptCheckpoint, protocol_fingerprint
from src.constants import (
    CHECKPOINT_ROOT_ENV,
    DEFER_CHECKPOINT_CLEANUP_ENV,
    LATE_CONTINUATION_PROMPT_FILE,
    SELECTION_PROMPT_FILE,
    SELECTION_WRAP_PROMPT_FILE,
    STATE_AUDIT_PROMPT_FILE,
    UNIFORM_COMPRESS_PROMPT_FILE,
)
from src.models import PhaseResult, ReconnectEvent
from src.run import _checkpointed_phase
from src.solver import process_recovery_prompt
from src.storage import archive_audit_scratches


def _phase(tokens: int) -> PhaseResult:
    return PhaseResult(
        label="solve",
        prompt="problem",
        text="complete proof",
        output_tokens=tokens,
        cumulative_output_tokens=tokens,
        num_turns=1,
        duration_ms=10,
        total_cost_usd=0.1,
        is_error=False,
        stop_reason="end_turn",
        budget_exhausted=False,
        tool_calls=[],
        reconnects=[],
        provider_usage={
            "input_tokens": 21,
            "cache_read_input_tokens": 8,
            "output_tokens": tokens,
        },
    )


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(
            os.environ, {CHECKPOINT_ROOT_ENV: self.temp.name}, clear=False
        )
        self.env.start()
        self.identity = {
            "stage": "run",
            "model": "test-model",
            "arm": "baseline-sequential",
            "problem": "p1",
            "seed": 1,
        }

    def tearDown(self) -> None:
        self.env.stop()
        self.temp.cleanup()

    def test_posthoc_prompts_do_not_change_legacy_protocol_fingerprint(self) -> None:
        root = Path(self.temp.name)
        prompts = root / "prompts"
        prompts.mkdir()
        settings = root / "settings.json"
        settings.write_text("settings", encoding="utf-8")
        (prompts / "solve.md").write_text("solve-v1", encoding="utf-8")
        protocol_fingerprint.cache_clear()
        with patch("src.checkpoint.PROMPTS_DIR", prompts):
            baseline = protocol_fingerprint(settings)
            (prompts / STATE_AUDIT_PROMPT_FILE).write_text(
                "state-v1", encoding="utf-8"
            )
            (prompts / UNIFORM_COMPRESS_PROMPT_FILE).write_text(
                "compress-v1", encoding="utf-8"
            )
            (prompts / SELECTION_PROMPT_FILE).write_text(
                "selection-v1", encoding="utf-8"
            )
            (prompts / SELECTION_WRAP_PROMPT_FILE).write_text(
                "selection-wrap-v1", encoding="utf-8"
            )
            (prompts / LATE_CONTINUATION_PROMPT_FILE).write_text(
                "late-continuation-v1", encoding="utf-8"
            )
            protocol_fingerprint.cache_clear()
            self.assertEqual(protocol_fingerprint(settings), baseline)
            protocol_fingerprint.cache_clear()
            self.assertNotEqual(
                protocol_fingerprint(settings, (UNIFORM_COMPRESS_PROMPT_FILE,)),
                baseline,
            )
            protocol_fingerprint.cache_clear()
            self.assertNotEqual(
                protocol_fingerprint(settings, (LATE_CONTINUATION_PROMPT_FILE,)),
                baseline,
            )
        protocol_fingerprint.cache_clear()

    def test_mid_phase_tracker_and_prefix_survive_process_restart(self) -> None:
        first = AttemptCheckpoint(self.identity)
        scratch = first.scratch_dir("main")
        (scratch / "work.txt").write_text("lemma", encoding="utf-8")
        tracker = first.tracker("main", 100, 10)
        first.save_session("main", "session-uuid", [])
        first.begin_phase("main", "solve", "problem", 90, tracker)
        tracker.add("message-1", {"output_tokens": 12})
        first.save_progress(
            "main",
            tracker,
            "session-uuid",
            [],
            {
                "text_parts": ["partial proof", "stream suffix"],
                "seen_message_ids": ["message-1"],
                "seen_text_block_keys": ["v2:message-1:already-recorded"],
                "current_stream_id": "message-2",
                "tool_uses": {},
                "tool_results": {},
            },
        )
        first.close()

        second = AttemptCheckpoint(self.identity)
        self.assertEqual(second.session_id("main"), "session-uuid")
        self.assertEqual(
            (second.scratch_dir("main") / "work.txt").read_text(encoding="utf-8"),
            "lemma",
        )
        restored = second.tracker("main", 100, 10)
        self.assertEqual(restored.spent, 12)
        # The per-message maximum is restored: transcript replay costs zero.
        restored.add("message-1", {"output_tokens": 12})
        self.assertEqual(restored.spent, 12)
        active = second.prepare_process_resume("main")
        self.assertEqual(active["process_resume_count"], 1)
        self.assertEqual(
            active["discarded_output_text"], "partial proof\nstream suffix"
        )
        self.assertEqual(
            set(active["discarded_text_block_keys"]),
            {
                "v2:message-1:already-recorded",
                "sha256:" + hashlib.sha256(b"partial proof").hexdigest(),
                "sha256:" + hashlib.sha256(b"stream suffix").hexdigest(),
            },
        )
        self.assertEqual(active["discarded_message_ids"], ["message-1", "message-2"])
        second.clear()

    def test_retained_prefix_restores_exact_opaque_workspace_name(self) -> None:
        snapshot = Path(self.temp.name) / "snapshot"
        snapshot.mkdir()
        (snapshot / "proof.txt").write_text("work", encoding="utf-8")
        checkpoint = AttemptCheckpoint(self.identity)

        restored = checkpoint.restore_scratch_dir("main", "1234abcd", snapshot)

        self.assertEqual(restored.name, "1234abcd")
        self.assertEqual(
            (restored / "proof.txt").read_text(encoding="utf-8"), "work"
        )
        self.assertEqual(checkpoint.scratch_dir("main"), restored)
        checkpoint.clear()

    def test_phase_first_commit_reconciles_controller_crash(self) -> None:
        first = AttemptCheckpoint(self.identity)
        tracker = first.tracker("main", 100, 0)
        first.save_session("main", "session-uuid", [])
        first.begin_phase("main", "solve", "problem", 100, tracker)
        tracker.add("message-1", {"output_tokens": 9})
        tracker.finish_phase(9)

        real_save = first._save
        with patch.object(first, "_save", side_effect=RuntimeError("killed")):
            with self.assertRaisesRegex(RuntimeError, "killed"):
                first.finish_phase("main", _phase(9), tracker, "session-uuid", [])
        first._save = real_save
        first.close()

        second = AttemptCheckpoint(self.identity)
        phases = second.phases("main")
        restored = second.tracker("main", 100, 0)
        self.assertEqual([phase.text for phase in phases], ["complete proof"])
        self.assertEqual(
            phases[0].provider_usage,
            {
                "input_tokens": 21,
                "cache_read_input_tokens": 8,
                "output_tokens": 9,
            },
        )
        self.assertEqual(restored.spent, 9)
        self.assertIsNone(second.active("main"))
        second.clear()

    def test_audit_calls_and_uuid_are_resumable(self) -> None:
        checkpoint = AttemptCheckpoint({**self.identity, "stage": "audit"})
        checkpoint.begin_call("full", "grade this")
        reconnect = ReconnectEvent(
            reason="rate_limit",
            resets_at=123,
            from_credential="credential_1",
            to_credential="credential_2",
        )
        checkpoint.finish_call(
            "full",
            {
                "verdict": {"score": 7, "note": "valid"},
                "reconnects": [],
                "process_resume_count": 0,
            },
            "judge-uuid",
            [reconnect],
        )
        checkpoint.close()

        restored = AttemptCheckpoint({**self.identity, "stage": "audit"})
        self.assertEqual(restored.session_id("full"), "judge-uuid")
        self.assertEqual(restored.call_result("full")["verdict"]["score"], 7)
        self.assertEqual(restored.reconnects("full"), [reconnect])
        restored.clear()

    def test_exact_attempt_lock_prevents_concurrent_controller(self) -> None:
        first = AttemptCheckpoint(self.identity)
        with self.assertRaisesRegex(RuntimeError, "already running"):
            AttemptCheckpoint(self.identity)
        first.close()
        reopened = AttemptCheckpoint(self.identity)
        reopened.clear()

    def test_incompatible_state_releases_file_lock(self) -> None:
        checkpoint = AttemptCheckpoint(self.identity)
        state_path = checkpoint.state_path
        lock_path = checkpoint.path / ".lock"
        state = checkpoint.state
        state["schema_version"] = -1
        checkpoint._save()
        checkpoint.close()

        with self.assertRaisesRegex(ValueError, "Incompatible checkpoint"):
            AttemptCheckpoint(self.identity)
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        shutil.rmtree(state_path.parent)

    def test_completion_marker_rejects_path_traversal(self) -> None:
        checkpoint = AttemptCheckpoint(self.identity)
        with self.assertRaisesRegex(ValueError, "Invalid relative result marker"):
            checkpoint.prepare_completion("../../results/meta.json")
        checkpoint.clear()

    def test_run_completion_can_defer_cleanup_until_host_merge(self) -> None:
        checkpoint = AttemptCheckpoint(self.identity)
        path = checkpoint.path
        workspace = checkpoint.scratch_dir("main")
        with patch.dict(os.environ, {DEFER_CHECKPOINT_CLEANUP_ENV: "1"}, clear=False):
            checkpoint.prepare_completion("model/arm/p1/seed_1/meta.json")
            checkpoint.complete()
        checkpoint.close()
        self.assertTrue(path.exists())
        self.assertTrue(workspace.exists())
        restored = AttemptCheckpoint(self.identity)
        self.assertIs(restored.state["completed"], True)
        restored.clear()

    def test_missing_paid_workspace_fails_loud(self) -> None:
        checkpoint = AttemptCheckpoint(self.identity)
        workspace = checkpoint.scratch_dir("main")
        checkpoint.save_session("main", "paid-session", [])
        checkpoint.close()
        shutil.rmtree(workspace)

        restored = AttemptCheckpoint(self.identity)
        with self.assertRaisesRegex(FileNotFoundError, "transcript reset"):
            restored.scratch_dir("main")
        restored.clear()

    def test_audit_archive_is_retryable_and_excludes_runtime(self) -> None:
        checkpoint = AttemptCheckpoint({**self.identity, "stage": "audit"})
        scratch = checkpoint.scratch_dir("full")
        (scratch / "calculation.txt").write_text("first", encoding="utf-8")
        runtime = scratch / ".claude-runtime"
        runtime.mkdir()
        (runtime / "transcript.jsonl").write_text("private", encoding="utf-8")
        output = checkpoint.path / "output"
        output.mkdir()

        archive_audit_scratches(output, {"full": scratch})
        self.assertTrue((scratch / "calculation.txt").exists())
        self.assertTrue((runtime / "transcript.jsonl").exists())
        self.assertEqual(
            (output / "audit_scratch" / "full" / "calculation.txt").read_text(
                encoding="utf-8"
            ),
            "first",
        )
        self.assertFalse(
            (output / "audit_scratch" / "full" / ".claude-runtime").exists()
        )

        (scratch / "calculation.txt").write_text("second", encoding="utf-8")
        archive_audit_scratches(output, {"full": scratch})
        self.assertEqual(
            (output / "audit_scratch" / "full" / "calculation.txt").read_text(
                encoding="utf-8"
            ),
            "second",
        )
        checkpoint.clear()


class _KilledPhaseClient:
    session_id = "solver-uuid"
    reconnect_events: list[ReconnectEvent] = []

    def __init__(self, *, killed: bool) -> None:
        self.killed = killed
        self.connection_id = "killed-cli" if killed else "replacement-cli"
        self.queries: list[str] = []

    async def query(self, prompt: str) -> None:
        self.queries.append(prompt)

    async def interrupt(self) -> None:
        return None

    async def receive_response(self):  # type: ignore[no-untyped-def]
        message_id = "partial-message" if self.killed else "replacement-message"
        tokens = 12 if self.killed else 7
        yield StreamEvent(
            uuid=f"{message_id}-start",
            session_id=self.session_id,
            event={"type": "message_start", "message": {"id": message_id}},
        )
        yield StreamEvent(
            uuid=f"{message_id}-delta",
            session_id=self.session_id,
            event={"type": "message_delta", "usage": {"output_tokens": tokens}},
        )
        yield AssistantMessage(
            content=[TextBlock("partial proof" if self.killed else "complete proof")],
            model="test-model",
            usage={"output_tokens": 1},
            message_id=message_id,
            session_id=self.session_id,
            uuid=f"{message_id}-assistant",
        )
        if self.killed:
            raise RuntimeError("process killed")
        yield ResultMessage(
            subtype="success",
            duration_ms=10,
            duration_api_ms=9,
            is_error=False,
            num_turns=1,
            session_id=self.session_id,
            stop_reason="end_turn",
            total_cost_usd=0.01,
            usage={"output_tokens": tokens},
            result="complete proof",
        )


class _MidTextKilledClient(_KilledPhaseClient):
    def __init__(self) -> None:
        super().__init__(killed=True)

    async def receive_response(self):  # type: ignore[no-untyped-def]
        yield StreamEvent(
            uuid="mid-start",
            session_id=self.session_id,
            event={"type": "message_start", "message": {"id": "mid-message"}},
        )
        yield StreamEvent(
            uuid="mid-text",
            session_id=self.session_id,
            event={
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "half sentence"},
            },
        )
        raise RuntimeError("process killed mid-text")


class CrossProcessPhaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_mid_message_text_is_preserved_as_discarded_evidence(self) -> None:
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(os.environ, {CHECKPOINT_ROOT_ENV: root}, clear=False),
        ):
            identity = {"stage": "run", "attempt": "mid-text-kill"}
            first = AttemptCheckpoint(identity)
            tracker = first.tracker("main", 100, 0)
            first.save_session("main", "solver-uuid", [])
            killed = _MidTextKilledClient()
            with self.assertRaisesRegex(RuntimeError, "mid-text"):
                await _checkpointed_phase(
                    first,  # type: ignore[arg-type]
                    "main",
                    killed,  # type: ignore[arg-type]
                    tracker,
                    "solve this",
                    "solve",
                    100,
                )
            first.close()

            second = AttemptCheckpoint(identity)
            restored = second.tracker("main", 100, 0)
            replacement = _KilledPhaseClient(killed=False)
            phase = await _checkpointed_phase(
                second,  # type: ignore[arg-type]
                "main",
                replacement,  # type: ignore[arg-type]
                restored,
                "solve this",
                "solve",
                100,
            )
            self.assertEqual(phase.discarded_output_text, "half sentence")
            self.assertEqual(phase.output_tokens, 7)
            second.clear()

    async def test_pre_query_kill_recovery_repeats_pending_request(self) -> None:
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(os.environ, {CHECKPOINT_ROOT_ENV: root}, clear=False),
        ):
            identity = {"stage": "run", "attempt": "pre-query-kill"}
            first = AttemptCheckpoint(identity)
            tracker = first.tracker("main", 100, 0)
            first.save_session("main", "solver-uuid", [])
            first.begin_phase("main", "solve", "solve this", 100, tracker)
            first.close()

            second = AttemptCheckpoint(identity)
            restored = second.tracker("main", 100, 0)
            replacement = _KilledPhaseClient(killed=False)
            phase = await _checkpointed_phase(
                second,  # type: ignore[arg-type]
                "main",
                replacement,  # type: ignore[arg-type]
                restored,
                "solve this",
                "solve",
                100,
            )
            self.assertEqual(
                replacement.queries, [process_recovery_prompt("solve this")]
            )
            self.assertEqual(phase.output_tokens, 7)
            self.assertEqual(phase.process_resume_count, 1)
            second.clear()

    async def test_killed_phase_resumes_without_losing_reported_tokens(self) -> None:
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(os.environ, {CHECKPOINT_ROOT_ENV: root}, clear=False),
        ):
            identity = {"stage": "run", "attempt": "cross-process"}
            first = AttemptCheckpoint(identity)
            tracker = first.tracker("main", 100, 0)
            first.save_session("main", "solver-uuid", [])
            killed = _KilledPhaseClient(killed=True)
            with self.assertRaisesRegex(RuntimeError, "process killed"):
                await _checkpointed_phase(
                    first,  # type: ignore[arg-type]
                    "main",
                    killed,  # type: ignore[arg-type]
                    tracker,
                    "solve this",
                    "solve",
                    100,
                )
            self.assertEqual(tracker.spent, 12)
            first.close()

            second = AttemptCheckpoint(identity)
            self.assertEqual(second.phases("main"), [])
            restored = second.tracker("main", 100, 0)
            replacement = _KilledPhaseClient(killed=False)
            phase = await _checkpointed_phase(
                second,  # type: ignore[arg-type]
                "main",
                replacement,  # type: ignore[arg-type]
                restored,
                "solve this",
                "solve",
                100,
            )
            self.assertEqual(
                replacement.queries, [process_recovery_prompt("solve this")]
            )
            self.assertEqual(phase.text, "complete proof")
            self.assertEqual(phase.discarded_output_text, "partial proof")
            self.assertEqual(phase.output_tokens, 19)
            self.assertEqual(phase.cumulative_output_tokens, 19)
            self.assertEqual(phase.process_resume_count, 1)
            self.assertEqual(second.phases("main"), [phase])
            second.clear()

    async def test_killed_phase_at_cutoff_finishes_without_another_query(self) -> None:
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(os.environ, {CHECKPOINT_ROOT_ENV: root}, clear=False),
        ):
            identity = {"stage": "run", "attempt": "cutoff-recovery"}
            first = AttemptCheckpoint(identity)
            tracker = first.tracker("main", 100, 10)
            first.save_session("main", "solver-uuid", [])
            first.begin_phase("main", "solve", "solve this", 90, tracker)
            tracker.add("cutoff-message", {"output_tokens": 95})
            first.save_progress(
                "main",
                tracker,
                "solver-uuid",
                [],
                {"text_parts": ["partial at cutoff"]},
            )
            first.close()

            second = AttemptCheckpoint(identity)
            restored = second.tracker("main", 100, 10)
            replacement = _KilledPhaseClient(killed=False)
            phase = await _checkpointed_phase(
                second,  # type: ignore[arg-type]
                "main",
                replacement,  # type: ignore[arg-type]
                restored,
                "solve this",
                "solve",
                90,
            )
            self.assertEqual(replacement.queries, [])
            self.assertEqual(phase.text, "partial at cutoff")
            self.assertEqual(phase.output_tokens, 95)
            self.assertEqual(phase.cumulative_output_tokens, 95)
            self.assertTrue(phase.budget_exhausted)
            self.assertEqual(phase.process_resume_count, 1)
            self.assertEqual(second.phases("main"), [phase])
            second.clear()


if __name__ == "__main__":
    unittest.main()
