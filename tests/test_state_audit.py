"""Compact route-state annotation and mechanical-label tests."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio

from src.models import ArmConfig, ExperimentConfig, Problem
from src.state_audit import derive_state, state_audit_seed
from src.storage import _outline_reference
from src.token_pool import TokenPool


class _FakeCheckpoint:
    def __init__(self, _: object, root: Path) -> None:
        self.root = root

    def scratch_dir(self, role: str) -> Path:
        path = self.root / role
        path.mkdir(parents=True, exist_ok=True)
        return path

    def prepare_completion(self, _: object) -> None:
        pass

    def complete(self) -> None:
        pass

    def close(self) -> None:
        pass


def _config(arm: ArmConfig) -> ExperimentConfig:
    return ExperimentConfig(
        model="solver",
        audit_model="judge",
        effort="high",
        unit_output_tokens=200_000,
        wrap_up_reserve_tokens=20_000,
        uniform_strategy_plan_tokens=20_000,
        uniform_strategy_plan_wrap_up_reserve_tokens=2_000,
        uniform_strategy_branches=8,
        max_turns_per_phase=64,
        audit_max_turns=64,
        max_concurrency=8,
        arms={arm.name: arm},
    )


class StateDerivationTests(unittest.TestCase):
    def test_only_three_present_steps_are_productive(self) -> None:
        self.assertEqual(
            derive_state(
                {
                    "steps": [
                        {"present": True, "reason": "Explicit."},
                        {"present": True, "reason": "Explicit."},
                        {"present": True, "reason": "Explicit."},
                    ],
                }
            ),
            "P",
        )
        self.assertEqual(
            derive_state(
                {
                    "steps": [
                        {"present": True, "reason": "Explicit."},
                        {"present": False, "reason": "Missing."},
                        {"present": True, "reason": "Explicit."},
                    ],
                }
            ),
            "U",
        )

    def test_outline_reference_requires_one_explicit_marker(self) -> None:
        record = {
            "problem_id": "p",
            "reference_solutions": [
                {"route_id": "hard_hint", "solution": "Aligned route."},
                {"solution": "Different route."},
            ],
        }
        self.assertEqual(_outline_reference(record), "Aligned route.")
        record["reference_solutions"].reverse()
        with self.assertRaisesRegex(ValueError, r"reference_solutions\[0\]"):
            _outline_reference(record)
        record["reference_solutions"][0]["route_id"] = "hard_hint"
        with self.assertRaisesRegex(ValueError, "exactly one reference"):
            _outline_reference(record)


class StateAuditSeedTests(unittest.TestCase):
    def test_missing_solved_and_duplicate_unsolved_artifacts(self) -> None:
        arm = ArmConfig("baseline-sequential", "none", "sequential", 8, [1])
        config = _config(arm)
        problem = Problem(
            "p",
            "Prove it.",
            "algebra",
            None,
            None,
            "1. First route step.\n2. Second route step.\n3. Third route step.",
        )
        unsolved = "## Final Solution\nA route with one execution gap.\n"
        solved = "## Final Solution\nA complete proof.\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results" / "solver" / arm.name / "p" / "seed_1"
            state_output = (
                root / "state-results" / "solver" / arm.name / "p" / "seed_1"
            )
            output.mkdir(parents=True)
            (output / "solution.md").write_text(solved, encoding="utf-8")
            (output / "solution_2x.md").write_text(unsolved, encoding="utf-8")
            (output / "solution_4x.md").write_text(unsolved, encoding="utf-8")
            (output / "audit.json").write_text(
                json.dumps(
                    {
                        "problem_id": "p",
                        "arm": arm.name,
                        "seed": 1,
                        "solver_model": "solver",
                        "audit_model": "proof-judge",
                        "audit_score": 7,
                        "solution_sha256": hashlib.sha256(
                            solved.encode("utf-8")
                        ).hexdigest(),
                        "budget_cuts": {
                            "1x": {"audit_score": 0},
                            "2x": {
                                "audit_score": 0,
                                "solution_sha256": hashlib.sha256(
                                    unsolved.encode("utf-8")
                                ).hexdigest(),
                            },
                            "4x": {
                                "audit_score": 0,
                                "solution_sha256": hashlib.sha256(
                                    unsolved.encode("utf-8")
                                ).hexdigest(),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            judge = AsyncMock(
                return_value=(
                    {
                        "steps": [
                            {"present": True, "reason": "Step 1 is explicit."},
                            {"present": True, "reason": "Step 2 is explicit."},
                            {"present": True, "reason": "Step 3 is explicit."},
                        ],
                    },
                    [],
                )
            )

            def checkpoint_factory(identity: object) -> _FakeCheckpoint:
                return _FakeCheckpoint(identity, root / "checkpoint")

            with (
                patch("src.state_audit.STATE_RESULTS_ROOT", root / "state-results"),
                patch("src.state_audit.seed_output_dir", return_value=output),
                patch(
                    "src.state_audit.state_output_dir", return_value=state_output
                ),
                patch(
                    "src.state_audit.AttemptCheckpoint",
                    side_effect=checkpoint_factory,
                ),
                patch("src.state_audit._judge", judge),
                patch("src.state_audit.protocol_fingerprint", return_value="test"),
            ):
                anyio.run(
                    state_audit_seed,
                    config,
                    arm,
                    problem,
                    1,
                    TokenPool(["unused"], "TEST_TOKEN"),
                    "A full outline-matching reference proof.",
                )

            self.assertEqual(judge.await_count, 1)
            call = judge.await_args
            self.assertIsNotNone(call)
            assert call is not None
            rendered_prompt = call.args[1]
            self.assertIn("A full outline-matching reference proof.", rendered_prompt)
            self.assertIn("1. First route step.", rendered_prompt)
            self.assertIn(unsolved.strip(), rendered_prompt)
            record = json.loads(
                (state_output / "state_audit.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(record["checkpoints"]["1x"]["state"])
            self.assertEqual(record["checkpoints"]["1x"]["steps"], [])
            self.assertEqual(record["checkpoints"]["2x"]["state"], "P")
            self.assertEqual(record["checkpoints"]["4x"]["state"], "P")
            self.assertEqual(
                record["checkpoints"]["4x"]["state_audit_reused_from"], "2x"
            )
            self.assertEqual(record["checkpoints"]["8x"]["state"], "S")
            self.assertEqual(record["checkpoints"]["8x"]["steps"], [])


if __name__ == "__main__":
    unittest.main()
