"""Identical sequential snapshots must receive one immutable audit verdict."""

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio
import zstandard

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
    def test_dense_extension_reuses_frozen_sparse_verdicts(self) -> None:
        arm = ArmConfig("baseline-sequential", "none", "sequential", 8, [1])
        config = ExperimentConfig(
            model="solver",
            audit_model="judge-new",
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
        digest = hashlib.sha256(proof.encode("utf-8")).hexdigest()
        new_proof = "## Final Solution\nA new incomplete proof.\n"
        new_digest = hashlib.sha256(new_proof.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results" / "solver" / arm.name / "p" / "seed_1"
            output.mkdir(parents=True)
            (output / "solution.md").write_text(proof, encoding="utf-8")
            phases = [
                {
                    "label": "solve",
                    "text": proof,
                    "cumulative_output_tokens": 100_000,
                    "budget_exhausted": False,
                },
                {
                    "label": "revise",
                    "text": new_proof,
                    "cumulative_output_tokens": 500_000,
                    "budget_exhausted": False,
                },
                {
                    "label": "revise",
                    "text": proof,
                    "cumulative_output_tokens": 700_000,
                    "budget_exhausted": False,
                },
            ]
            (output / "logs.jsonl.zst").write_bytes(
                zstandard.ZstdCompressor().compress(
                    ("\n".join(json.dumps(phase) for phase in phases) + "\n").encode(
                        "utf-8"
                    )
                )
            )
            for multiplier in (1, 2, 4):
                (output / f"solution_{multiplier}x.md").write_text(
                    proof, encoding="utf-8"
                )
            frozen_cut = {
                "audit_score": 7,
                "note": "Valid.",
                "solution_sha256": digest,
                "session_reconnect_count": 0,
                "session_reconnects": [],
            }
            (output / "audit.json").write_text(
                json.dumps(
                    {
                        "problem_id": "p",
                        "arm": arm.name,
                        "seed": 1,
                        "solver_model": "solver",
                        "audit_model": "judge-old",
                        "audit_score": 7,
                        "note": "Valid.",
                        "solution_sha256": digest,
                        "session_reconnect_count": 0,
                        "session_reconnects": [],
                        "provider_session_ids": {},
                        "process_resume_count": 0,
                        "budget_cuts": {
                            f"{multiplier}x": dict(frozen_cut)
                            for multiplier in (1, 2, 4)
                        },
                    }
                ),
                encoding="utf-8",
            )
            judge = AsyncMock(
                return_value=({"score": 0, "note": "Incomplete."}, [])
            )

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
                    lambda: audit_seed(
                        config,
                        arm,
                        problem,
                        1,
                        TokenPool(["unused"], "TEST_TOKEN"),
                        "A verified reference proof.",
                        all_checkpoints=True,
                    )
                )

            self.assertEqual(judge.await_count, 1)
            record = json.loads((output / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(set(record["budget_cuts"]), {f"{n}x" for n in range(1, 8)})
            self.assertEqual(record["audit_model"], "judge-old")
            for multiplier in (1, 2, 4, 5, 6, 7):
                self.assertEqual(
                    record["budget_cuts"][f"{multiplier}x"]["audit_model"],
                    "judge-old",
                )
            self.assertEqual(record["budget_cuts"]["3x"]["audit_score"], 0)
            self.assertEqual(record["budget_cuts"]["3x"]["solution_sha256"], new_digest)
            self.assertEqual(
                record["budget_cuts"]["3x"]["audit_model"], "judge-new"
            )

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
                    "A verified reference proof.",
                )

            self.assertEqual(judge.await_count, 1)
            rendered_prompt = judge.await_args.args[1]
            self.assertIn("A verified reference proof.", rendered_prompt)
            self.assertIn(proof.strip(), rendered_prompt)
            record = json.loads((output / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(record["audit_score"], 7)
            for multiplier in (1, 2, 4):
                cut = record["budget_cuts"][f"{multiplier}x"]
                self.assertEqual(cut["audit_score"], 7)
                self.assertEqual(cut["audit_reused_from"], "full")
                self.assertEqual(cut["solution_sha256"], record["solution_sha256"])


if __name__ == "__main__":
    unittest.main()
