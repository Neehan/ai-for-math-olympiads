"""Retained native-prefix tests."""

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.config import load_config
from src.constants import CHECKPOINT_ROOT_ENV, CONFIG_PATH
from src.late_intervention import (
    fork_native_session,
    load_prefix_source,
    save_prefix_source,
)
from src.models import PhaseResult, Problem


def _phase(text: str = "Attempted proof.") -> PhaseResult:
    return PhaseResult(
        label="wrap_up",
        prompt="Write the proof.",
        text=text,
        output_tokens=100,
        cumulative_output_tokens=100,
        num_turns=1,
        duration_ms=1,
        total_cost_usd=0.0,
        is_error=False,
        stop_reason="end_turn",
        budget_exhausted=False,
        tool_calls=[],
        reconnects=[],
    )


class LateInterventionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoints = self.root / "checkpoints"
        self.env = patch.dict(
            os.environ, {CHECKPOINT_ROOT_ENV: str(self.checkpoints)}, clear=False
        )
        self.env.start()
        self.config = replace(load_config(CONFIG_PATH), model="source/model")
        self.problem = Problem(
            "p1", "Prove P.", "algebra", None, "Use the key lemma.", None
        )

    def tearDown(self) -> None:
        self.env.stop()
        self.temporary.cleanup()

    def test_every_retained_prefix_is_loadable_without_outcome_gating(self) -> None:
        scratch = self.root / "1234abcd"
        (scratch / ".claude-runtime").mkdir(parents=True)
        (scratch / "work.txt").write_text("lemma", encoding="utf-8")
        source = save_prefix_source(
            self.config,
            self.problem,
            1,
            scratch,
            "11111111-1111-4111-8111-111111111111",
            [_phase()],
            590_000,
        )

        loaded, reason = load_prefix_source(self.config, self.problem, 1)

        self.assertEqual(reason, "eligible")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.session_id, source.session_id)
        self.assertEqual(loaded.scratch_name, "1234abcd")
        self.assertEqual(
            (loaded.workspace / "work.txt").read_text(encoding="utf-8"),
            "lemma",
        )

    def test_native_fork_uses_the_snapshot_config_and_restores_environment(self) -> None:
        scratch = self.root / "1234abcd"
        (scratch / ".claude-runtime").mkdir(parents=True)
        old = os.environ.get("CLAUDE_CONFIG_DIR")
        os.environ["CLAUDE_CONFIG_DIR"] = "original"
        try:
            with patch(
                "src.late_intervention.fork_session",
                return_value=SimpleNamespace(
                    session_id="22222222-2222-4222-8222-222222222222"
                ),
            ) as mocked:
                forked = fork_native_session(
                    scratch, "11111111-1111-4111-8111-111111111111"
                )
            self.assertEqual(forked, "22222222-2222-4222-8222-222222222222")
            mocked.assert_called_once_with(
                "11111111-1111-4111-8111-111111111111",
                directory=str(scratch),
            )
            self.assertEqual(os.environ.get("CLAUDE_CONFIG_DIR"), "original")
        finally:
            if old is None:
                os.environ.pop("CLAUDE_CONFIG_DIR", None)
            else:
                os.environ["CLAUDE_CONFIG_DIR"] = old


if __name__ == "__main__":
    unittest.main()
