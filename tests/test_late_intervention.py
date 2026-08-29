"""Problem eligibility and retained native-prefix tests."""

import json
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
    LateProblemEligibility,
    fork_native_session,
    load_prefix_source,
    problem_eligibility,
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
        self.results = self.root / "results"
        self.checkpoints = self.root / "checkpoints"
        self.results_patch = patch(
            "src.late_intervention.RESULTS_ROOT", self.results
        )
        self.results_patch.start()
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
        self.results_patch.stop()
        self.temporary.cleanup()

    def _write_original_audits(self, scores: list[int]) -> None:
        for seed, score in enumerate(scores, start=1):
            path = (
                self.results
                / self.config.model_dirname
                / "baseline-sequential"
                / "p1"
                / f"seed_{seed}"
                / "audit.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "problem_id": "p1",
                        "arm": "baseline-sequential",
                        "seed": seed,
                        "solver_model": "source/model",
                        "budget_cuts": {
                            "3x": {
                                "audit_score": score,
                                "audit_model": "judge",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

    def test_problem_eligibility_uses_fewer_than_two_of_three_passes(self) -> None:
        self._write_original_audits([0, 5, 0])

        eligibility, reason = problem_eligibility(self.config, self.problem)

        self.assertEqual(reason, "eligible")
        self.assertIsNotNone(eligibility)
        assert eligibility is not None
        self.assertEqual(eligibility.provenance["eligibility_pass_count"], 1)
        self.assertEqual(
            eligibility.provenance["eligibility_scores"],
            {"1": 0, "2": 5, "3": 0},
        )

    def test_problem_is_excluded_after_two_original_3x_passes(self) -> None:
        self._write_original_audits([5, 7, 0])

        eligibility, reason = problem_eligibility(self.config, self.problem)

        self.assertIsNone(eligibility)
        self.assertIn("passed 2/3", reason)

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
            LateProblemEligibility(
                provenance={
                    "eligibility_rule": "fewer_than_2_of_3_pass_at_3x",
                    "eligibility_pass_count": 0,
                }
            ),
        )

        loaded, reason = load_prefix_source(self.config, self.problem, 1)

        self.assertEqual(reason, "eligible")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.session_id, source.session_id)
        self.assertEqual(loaded.scratch_name, "1234abcd")
        self.assertEqual(loaded.provenance["eligibility_pass_count"], 0)
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
