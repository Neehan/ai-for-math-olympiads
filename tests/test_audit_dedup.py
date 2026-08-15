"""Identical sequential snapshots must receive one immutable audit verdict."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio

from src.audit import audit_seed
from src.models import ArmConfig, ExperimentConfig, Problem
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

    def session_ids(self) -> dict[str, str]:
        return {}

    def data(self) -> dict[str, object]:
        return {"calls": {}}

    def complete(self) -> None:
        pass

    def close(self) -> None:
        pass


class AuditDedupTests(unittest.TestCase):
    def test_full_and_identical_cuts_are_judged_once(self) -> None:
        arm = ArmConfig("baseline-sequential", "none", "sequential", 8, [1])
        config = ExperimentConfig(
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
        problem = Problem("p", "Prove it.", "algebra", None, None, None)
        proof = "## Final Solution\nA valid proof.\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results" / "solver" / arm.name / "p" / "seed_1"
            output.mkdir(parents=True)
            (output / "solution.md").write_text(proof, encoding="utf-8")
            for multiplier in (1, 2, 4):
                (output / f"solution_{multiplier}x.md").write_text(
                    proof, encoding="utf-8"
                )
            judge = AsyncMock(return_value=({"score": 7, "note": "Valid."}, []))

            def checkpoint_factory(identity: object) -> _FakeCheckpoint:
                return _FakeCheckpoint(identity, root / "checkpoint")

            with (
                patch("src.audit.RESULTS_ROOT", root / "results"),
                patch("src.audit.seed_output_dir", return_value=output),
                patch("src.audit.AttemptCheckpoint", side_effect=checkpoint_factory),
                patch("src.audit._judge", judge),
                patch("src.audit.protocol_fingerprint", return_value="test"),
            ):
                anyio.run(
                    audit_seed,
                    config,
                    arm,
                    problem,
                    1,
                    TokenPool(["unused"], "TEST_TOKEN"),
                )

            self.assertEqual(judge.await_count, 1)
            record = json.loads((output / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(record["audit_score"], 7)
            for multiplier in (1, 2, 4):
                cut = record["budget_cuts"][f"{multiplier}x"]
                self.assertEqual(cut["audit_score"], 7)
                self.assertEqual(cut["audit_reused_from"], "full")
                self.assertEqual(cut["solution_sha256"], record["solution_sha256"])


if __name__ == "__main__":
    unittest.main()
