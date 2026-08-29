"""Late-replay eligibility, reconstruction, and isolation-boundary tests."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import zstandard

from src.config import load_config
from src.constants import CONFIG_PATH
from src.late_replay import (
    load_late_replay_source,
    remove_staged_sources,
    source_output_dir,
)
from src.models import Problem


class LateReplaySourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.results_patch = patch("src.late_replay.RESULTS_ROOT", self.root)
        self.results_patch.start()
        self.config = replace(load_config(CONFIG_PATH), model="source/model")
        self.problem = Problem(
            "p1", "Prove P.", "algebra", None, "Use the key lemma.", None
        )
        self.output = source_output_dir(self.config, "p1", 1)
        self.output.mkdir(parents=True)

    def tearDown(self) -> None:
        self.results_patch.stop()
        self.temporary.cleanup()

    def _write_source(self, score: int = 0) -> None:
        meta = {
            "problem_id": "p1",
            "arm": "baseline-sequential",
            "mode": "sequential",
            "hint": "none",
            "model": "source/model",
            "seed": 1,
            "budget_output_tokens": 1_600_000,
            "termination_reason": "token_limit",
        }
        (self.output / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        records = [
            {
                "label": "solve",
                "prompt": "Solve P.",
                "text": "First attempted proof.",
                "cumulative_output_tokens": 100_000,
                "budget_exhausted": False,
                "tool_calls": [
                    {
                        "name": "Bash",
                        "input": {"command": "python check.py"},
                        "result": "checked",
                        "is_error": False,
                    }
                ],
            },
            {
                "label": "critique",
                "prompt": "Critique it.",
                "text": "The lemma is missing.",
                "cumulative_output_tokens": 150_000,
                "budget_exhausted": False,
                "tool_calls": [],
            },
            {
                "label": "revise",
                "prompt": "Revise it.",
                "text": "Second attempted proof.",
                "cumulative_output_tokens": 250_000,
                "budget_exhausted": False,
                "tool_calls": [],
            },
            {
                "label": "critique",
                "prompt": "Late prompt that crossed the cutoff.",
                "text": "Post-cut information.",
                "cumulative_output_tokens": 650_000,
                "budget_exhausted": False,
                "tool_calls": [],
            },
        ]
        log_text = "\n".join(json.dumps(record) for record in records) + "\n"
        (self.output / "logs.jsonl.zst").write_bytes(
            zstandard.ZstdCompressor().compress(log_text.encode("utf-8"))
        )
        proof_artifact = "Second attempted proof.\n"
        digest = hashlib.sha256(proof_artifact.encode("utf-8")).hexdigest()
        audit = {
            "problem_id": "p1",
            "arm": "baseline-sequential",
            "seed": 1,
            "solver_model": "source/model",
            "audit_model": "judge",
            "budget_cuts": {
                "3x": {
                    "audit_score": score,
                    "solution_sha256": digest,
                    "audit_model": "judge",
                }
            },
        }
        (self.output / "audit.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )

    def test_replays_only_complete_phases_through_3x(self) -> None:
        self._write_source()

        source, reason = load_late_replay_source(self.config, self.problem, 1)

        self.assertEqual(reason, "eligible")
        self.assertIsNotNone(source)
        assert source is not None
        self.assertIn("First attempted proof.", source.history)
        self.assertIn("checked", source.history)
        self.assertIn("Second attempted proof.", source.history)
        self.assertNotIn("Post-cut information.", source.history)
        self.assertEqual(source.provenance["source_completed_phase_count"], 3)
        self.assertEqual(source.provenance["source_last_completed_phase_tokens"], 250_000)
        self.assertEqual(source.provenance["source_3x_audit_score"], 0)

    def test_excludes_source_already_solved_by_3x(self) -> None:
        self._write_source(score=7)

        source, reason = load_late_replay_source(self.config, self.problem, 1)

        self.assertIsNone(source)
        self.assertIn("already solved by 3x", reason)

    def test_rejects_audit_not_bound_to_replayed_proof(self) -> None:
        self._write_source()
        audit_path = self.output / "audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["budget_cuts"]["3x"]["solution_sha256"] = "0" * 64
        audit_path.write_text(json.dumps(audit), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "not bound"):
            load_late_replay_source(self.config, self.problem, 1)

    def test_staged_source_tree_is_deleted_before_solver(self) -> None:
        self._write_source()

        remove_staged_sources(self.config)

        self.assertFalse(self.output.parents[1].exists())


if __name__ == "__main__":
    unittest.main()
