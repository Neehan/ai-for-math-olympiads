"""Compact route-state annotation and mechanical-label tests."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio
import zstandard

from src.models import ArmConfig, ExperimentConfig, Problem
from src.state_audit import (
    _bank_targets,
    derive_state,
    recognized_step_count,
    state_audit_seed,
    state_audit_strategy_artifact,
)
from src.storage import (
    _outline_reference,
    bank_run_output_dir,
    compile_arm_state_audit,
)
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
    def test_complete_step_count_is_acquired(self) -> None:
        self.assertEqual(
            derive_state(
                {
                    "steps": [
                        {"present": True, "reason": "Explicit."},
                        {"present": True, "reason": "Explicit."},
                        {"present": True, "reason": "Explicit."},
                    ],
                },
                2,
            ),
            ("S", 3),
        )
        self.assertEqual(
            derive_state(
                {
                    "steps": [
                        {"present": True, "reason": "Explicit."},
                        {"present": False, "reason": "Missing."},
                        {"present": True, "reason": "Explicit."},
                    ],
                },
                2,
            ),
            ("U", 2),
        )

    def test_flat_and_decreasing_step_counts_are_unproductive(self) -> None:
        verdict = {
            "steps": [
                {"present": True, "reason": "Explicit."},
                {"present": False, "reason": "Missing."},
                {"present": False, "reason": "Missing."},
            ]
        }
        self.assertEqual(recognized_step_count(verdict), 1)
        self.assertEqual(derive_state(verdict, 1), ("U", 1))
        self.assertEqual(derive_state(verdict, 2), ("U", 1))

    def test_complete_route_is_acquired_even_when_count_is_flat(self) -> None:
        verdict = {
            "steps": [
                {"present": True, "reason": "Explicit."},
                {"present": True, "reason": "Explicit."},
                {"present": True, "reason": "Explicit."},
            ]
        }
        self.assertEqual(derive_state(verdict, 3), ("S", 3))

    def test_first_artifact_is_compared_with_zero(self) -> None:
        verdict = {
            "steps": [
                {"present": True, "reason": "Explicit."},
                {"present": False, "reason": "Missing."},
                {"present": False, "reason": "Missing."},
            ]
        }
        self.assertEqual(derive_state(verdict, 0), ("P", 1))

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
    def test_empty_planner_bank_is_observed_as_unacquired_without_judge_call(self) -> None:
        arm = ArmConfig(
            "baseline-uniform-strategy-only",
            "none",
            "uniform_strategy_only",
            1,
            [1],
        )
        config = _config(arm)
        problem = Problem(
            "p",
            "Prove it.",
            "algebra",
            None,
            "Frozen oracle strategy.",
            "1. First route step.\n2. Second route step.\n3. Third route step.",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results" / "solver" / arm.name / "p" / "seed_1"
            output.mkdir(parents=True)
            (output / "strategies.json").write_text(
                json.dumps({"strategies": [], "run_strategy_indices": []}),
                encoding="utf-8",
            )
            judge = AsyncMock()

            def checkpoint_factory(identity: object) -> _FakeCheckpoint:
                return _FakeCheckpoint(identity, root / "checkpoint")

            with (
                patch("src.state_audit.RESULTS_ROOT", root / "results"),
                patch("src.state_audit.seed_output_dir", return_value=output),
                patch("src.state_audit.AttemptCheckpoint", side_effect=checkpoint_factory),
                patch("src.state_audit._judge", judge),
            ):
                anyio.run(
                    state_audit_strategy_artifact,
                    config,
                    arm,
                    problem,
                    1,
                    TokenPool(["unused"], "TEST_TOKEN"),
                    "A reference proof.",
                )

            record = json.loads(
                (output / "state_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["state"], "U")
            self.assertEqual(record["strategies"], [])
            self.assertEqual(judge.await_count, 0)

    def test_planner_bank_audits_each_strategy_without_proof_audit(self) -> None:
        arm = ArmConfig(
            "baseline-uniform-strategy-only",
            "none",
            "uniform_strategy_only",
            1,
            [1],
        )
        config = _config(arm)
        problem = Problem(
            "p",
            "Prove it.",
            "algebra",
            None,
            "Frozen oracle strategy.",
            "1. First route step.\n2. Second route step.\n3. Third route step.",
        )
        verdicts = [
            {
                "oracle_strategy_match": True,
                "reason": "The proposal follows the frozen oracle route.",
            },
            {
                "oracle_strategy_match": False,
                "reason": "The proposal follows a different route.",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results" / "solver" / arm.name / "p" / "seed_1"
            output.mkdir(parents=True)
            (output / "strategies.json").write_text(
                json.dumps(
                    {
                        "strategies": ["Complete route.", "Partial route."],
                        "run_strategy_indices": [],
                    }
                ),
                encoding="utf-8",
            )
            judge = AsyncMock(side_effect=[(verdict, []) for verdict in verdicts])

            def checkpoint_factory(identity: object) -> _FakeCheckpoint:
                return _FakeCheckpoint(identity, root / "checkpoint")

            with (
                patch("src.state_audit.RESULTS_ROOT", root / "results"),
                patch("src.state_audit.seed_output_dir", return_value=output),
                patch("src.state_audit.AttemptCheckpoint", side_effect=checkpoint_factory),
                patch("src.state_audit._judge", judge),
            ):
                anyio.run(
                    state_audit_strategy_artifact,
                    config,
                    arm,
                    problem,
                    1,
                    TokenPool(["unused"], "TEST_TOKEN"),
                    "A reference proof.",
                )

            record = json.loads(
                (output / "state_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["state"], "S")
            self.assertEqual(
                [item["oracle_strategy_match"] for item in record["strategies"]],
                [True, False],
            )
            self.assertTrue(
                all(
                    "strategy_acquired" not in item
                    and "acquisition_basis" not in item
                    for item in record["strategies"]
                )
            )
            first_prompt = judge.await_args_list[0].args[1]
            self.assertIn("Frozen oracle strategy.", first_prompt)
            self.assertNotIn("Reference solution outline:", first_prompt)
            self.assertFalse((output / "audit.json").exists())

    def test_compressed_candidates_have_no_state_audit(self) -> None:
        arm = ArmConfig(
            "baseline-uniform-compress", "none", "uniform_compress", 1, [1]
        )
        config = _config(arm)
        problem = Problem(
            "p",
            "Prove it.",
            "algebra",
            None,
            None,
            "1. First route step.\n2. Second route step.\n3. Third route step.",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results" / "solver" / arm.name / "p" / "seed_1"
            output.mkdir(parents=True)
            with patch("src.state_audit.seed_output_dir", return_value=output):
                with self.assertRaisesRegex(
                    ValueError, "Unsupported strategy-artifact mode"
                ):
                    anyio.run(
                        state_audit_strategy_artifact,
                        config,
                        arm,
                        problem,
                        1,
                        TokenPool(["unused"], "TEST_TOKEN"),
                        "A reference proof.",
                    )

    def test_dense_state_extension_reuses_sparse_step_annotations(self) -> None:
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
        solution = "## Final Solution\nAn incomplete proof.\n"
        digest = hashlib.sha256(solution.encode("utf-8")).hexdigest()
        new_solution = "## Final Solution\nA different incomplete proof.\n"
        new_digest = hashlib.sha256(new_solution.encode("utf-8")).hexdigest()
        steps = [
            {"present": True, "reason": "Step 1 is explicit."},
            {"present": True, "reason": "Step 2 is explicit."},
            {"present": False, "reason": "Step 3 is missing."},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results" / "solver" / arm.name / "p" / "seed_1"
            output.mkdir(parents=True)
            (output / "solution.md").write_text(solution, encoding="utf-8")
            phases = [
                {
                    "label": "solve",
                    "text": solution,
                    "cumulative_output_tokens": 100_000,
                    "budget_exhausted": False,
                },
                {
                    "label": "revise",
                    "text": new_solution,
                    "cumulative_output_tokens": 500_000,
                    "budget_exhausted": False,
                },
                {
                    "label": "revise",
                    "text": solution,
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
                    solution, encoding="utf-8"
                )
            old_cut_audit = {
                "audit_score": 0,
                "note": "Incomplete.",
                "solution_sha256": digest,
                "audit_model": "judge-old",
            }
            new_cut_audit = {
                "audit_score": 0,
                "note": "Incomplete.",
                "solution_sha256": new_digest,
                "audit_model": "judge",
            }
            (output / "audit.json").write_text(
                json.dumps(
                    {
                        "audit_score": 0,
                        "audit_model": "judge-old",
                        "budget_cuts": {
                            **{
                                f"{multiplier}x": dict(old_cut_audit)
                                for multiplier in (1, 2, 4, 5, 6, 7)
                            },
                            "3x": new_cut_audit,
                        },
                    }
                ),
                encoding="utf-8",
            )
            state_piece = {
                "state": "P",
                "steps": steps,
                "note": "Frozen sparse annotation.",
                "solution_sha256": digest,
            }
            (output / "state_audit.json").write_text(
                json.dumps(
                    {
                        "problem_id": "p",
                        "arm": arm.name,
                        "seed": 1,
                        "solver_model": "solver",
                        "audit_model": "judge-old",
                        **state_piece,
                        "budget_cuts": {
                            f"{multiplier}x": dict(state_piece)
                            for multiplier in (1, 2, 4)
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
                            {"present": False, "reason": "Step 2 is missing."},
                            {"present": False, "reason": "Step 3 is missing."},
                        ]
                    },
                    [],
                )
            )

            def checkpoint_factory(identity: object) -> _FakeCheckpoint:
                return _FakeCheckpoint(identity, root / "checkpoint")

            with (
                patch("src.state_audit.RESULTS_ROOT", root / "results"),
                patch("src.state_audit.seed_output_dir", return_value=output),
                patch(
                    "src.state_audit.AttemptCheckpoint",
                    side_effect=checkpoint_factory,
                ),
                patch("src.state_audit._judge", judge),
            ):
                anyio.run(
                    lambda: state_audit_seed(
                        config,
                        arm,
                        problem,
                        1,
                        TokenPool(["unused"], "TEST_TOKEN"),
                        "A reference proof.",
                        all_checkpoints=True,
                    )
                )

            self.assertEqual(judge.await_count, 1)
            record = json.loads(
                (output / "state_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(record["budget_cuts"]), {f"{n}x" for n in range(1, 8)})
            self.assertEqual(
                {
                    label: cut["state"]
                    for label, cut in record["budget_cuts"].items()
                },
                {
                    "1x": "P",
                    "2x": "U",
                    "3x": "U",
                    "4x": "P",
                    "5x": "U",
                    "6x": "U",
                    "7x": "U",
                },
            )
            self.assertEqual(record["state"], "U")
            self.assertEqual(record["audit_model"], "judge-old")
            self.assertEqual(record["budget_cuts"]["3x"]["state"], "U")
            self.assertEqual(record["budget_cuts"]["3x"]["audit_model"], "judge")
            for multiplier in (1, 2, 4, 5, 6, 7):
                self.assertEqual(
                    record["budget_cuts"][f"{multiplier}x"]["audit_model"],
                    "judge-old",
                )

    def test_single_arm_has_one_final_state_and_no_budget_cuts(self) -> None:
        arm = ArmConfig("baseline", "none", "single", 1, [1])
        config = _config(arm)
        problem = Problem(
            "p",
            "Prove it.",
            "algebra",
            None,
            None,
            "1. First route step.\n2. Second route step.\n3. Third route step.",
        )
        solution = "## Final Solution\nAn incomplete proof.\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results" / "solver" / arm.name / "p" / "seed_1"
            output.mkdir(parents=True)
            (output / "solution.md").write_text(solution, encoding="utf-8")
            (output / "audit.json").write_text(
                json.dumps(
                    {
                        "audit_model": "proof-judge",
                        "audit_score": 0,
                        "budget_cuts": {},
                    }
                ),
                encoding="utf-8",
            )
            judge = AsyncMock(
                return_value=(
                    {
                        "steps": [
                            {"present": True, "reason": "Present."},
                            {"present": False, "reason": "Missing."},
                            {"present": True, "reason": "Present."},
                        ]
                    },
                    [],
                )
            )

            def checkpoint_factory(identity: object) -> _FakeCheckpoint:
                return _FakeCheckpoint(identity, root / "checkpoint")

            with (
                patch("src.state_audit.RESULTS_ROOT", root / "results"),
                patch("src.state_audit.seed_output_dir", return_value=output),
                patch(
                    "src.state_audit.AttemptCheckpoint",
                    side_effect=checkpoint_factory,
                ),
                patch("src.state_audit._judge", judge),
            ):
                anyio.run(
                    state_audit_seed,
                    config,
                    arm,
                    problem,
                    1,
                    TokenPool(["unused"], "TEST_TOKEN"),
                    "A reference proof.",
                )

            record = json.loads(
                (output / "state_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["state"], "P")
            self.assertEqual(
                record["note"],
                "Recognized outline-step count increased from 0/3 to 2/3.",
            )
            self.assertEqual(record["budget_cuts"], {})
            self.assertEqual(judge.await_count, 1)

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
                patch("src.state_audit.RESULTS_ROOT", root / "results"),
                patch("src.state_audit.seed_output_dir", return_value=output),
                patch(
                    "src.state_audit.AttemptCheckpoint",
                    side_effect=checkpoint_factory,
                ),
                patch("src.state_audit._judge", judge),
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
                (output / "state_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(record),
                {
                    "problem_id",
                    "arm",
                    "seed",
                    "solver_model",
                    "audit_model",
                    "state",
                    "steps",
                    "note",
                    "solution_sha256",
                    "budget_cuts",
                },
            )
            self.assertEqual(set(record["budget_cuts"]), {"1x", "2x", "4x"})
            self.assertIsNone(record["budget_cuts"]["1x"]["state"])
            self.assertEqual(record["budget_cuts"]["1x"]["steps"], [])
            self.assertNotIn("solution_sha256", record["budget_cuts"]["1x"])
            self.assertEqual(record["budget_cuts"]["2x"]["state"], "S")
            self.assertEqual(record["budget_cuts"]["4x"]["state"], "S")
            self.assertEqual(
                record["budget_cuts"]["4x"]["note"],
                "A complete strategy was observed earlier; acquired state is carried forward.",
            )
            self.assertEqual(
                record["budget_cuts"]["2x"]["solution_sha256"],
                hashlib.sha256(unsolved.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(record["state"], "S")
            self.assertEqual(record["steps"], [])
            self.assertEqual(
                record["solution_sha256"],
                hashlib.sha256(solved.encode("utf-8")).hexdigest(),
            )

    def test_solved_state_is_carried_forward_after_proof_regression(self) -> None:
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
        solved = "## Final Solution\nA complete proof.\n"
        regressed = "## Final Solution\nA later invalid revision.\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results" / "solver" / arm.name / "p" / "seed_1"
            output.mkdir(parents=True)
            (output / "solution_1x.md").write_text(solved, encoding="utf-8")
            (output / "solution_2x.md").write_text(regressed, encoding="utf-8")
            (output / "solution.md").write_text(regressed, encoding="utf-8")
            (output / "audit.json").write_text(
                json.dumps(
                    {
                        "audit_model": "proof-judge",
                        "audit_score": 0,
                        "budget_cuts": {
                            "1x": {
                                "audit_model": "proof-judge",
                                "audit_score": 7,
                            },
                            "2x": {
                                "audit_model": "proof-judge",
                                "audit_score": 0,
                            },
                            "4x": {
                                "audit_model": "proof-judge",
                                "audit_score": 0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            judge = AsyncMock()

            def checkpoint_factory(identity: object) -> _FakeCheckpoint:
                return _FakeCheckpoint(identity, root / "checkpoint")

            with (
                patch("src.state_audit.RESULTS_ROOT", root / "results"),
                patch("src.state_audit.seed_output_dir", return_value=output),
                patch(
                    "src.state_audit.AttemptCheckpoint",
                    side_effect=checkpoint_factory,
                ),
                patch("src.state_audit._judge", judge),
            ):
                anyio.run(
                    state_audit_seed,
                    config,
                    arm,
                    problem,
                    1,
                    TokenPool(["unused"], "TEST_TOKEN"),
                    "A reference proof.",
                )

            record = json.loads(
                (output / "state_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(judge.await_count, 0)
            self.assertEqual(record["budget_cuts"]["1x"]["state"], "S")
            self.assertEqual(record["budget_cuts"]["2x"]["state"], "S")
            self.assertEqual(record["budget_cuts"]["4x"]["state"], "S")
            self.assertEqual(record["state"], "S")
            self.assertNotIn(
                "solution_sha256", record["budget_cuts"]["4x"]
            )

    def test_parallel_bank_resolves_each_executor_proof(self) -> None:
        arm = ArmConfig("baseline-parallel", "none", "parallel", 8, [1])
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "seed_1"
            bank.mkdir(parents=True)
            (bank / "audit.json").write_text(
                json.dumps(
                    {
                        "audit_score": 7,
                        "runs": [
                            {"run": 1, "audit_score": 7},
                            {"run": 2, "audit_score": 7},
                        ],
                        "budget_cuts": {},
                    }
                ),
                encoding="utf-8",
            )
            for run in (1, 2):
                run_dir = bank_run_output_dir(bank, run)
                run_dir.mkdir(parents=True)
                (run_dir / "audit.json").write_text(
                    json.dumps({"audit_score": 7, "budget_cuts": {}}),
                    encoding="utf-8",
                )

            targets = _bank_targets(arm, bank, 1)
            self.assertEqual(
                [target.name for target, _ in targets], ["run_01", "run_02"]
            )
            self.assertEqual(
                [extra["parallel_run"] for _, extra in targets], [1, 2]
            )

    def test_parallel_executor_states_compile_at_arm_level(self) -> None:
        arm = ArmConfig("baseline-parallel", "none", "parallel", 8, [1])
        config = _config(arm)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            bank = root / "solver" / arm.name / "p" / "seed_1"
            for run in (1, 2):
                run_dir = bank_run_output_dir(bank, run)
                run_dir.mkdir(parents=True)
                (run_dir / "state_audit.json").write_text(
                    json.dumps(
                        {
                            "problem_id": "p",
                            "seed": 1,
                            "parallel_run": run,
                            "state": "S" if run == 1 else "U",
                        }
                    ),
                    encoding="utf-8",
                )
            with patch("src.storage.RESULTS_ROOT", root):
                path, count = compile_arm_state_audit(config, arm)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(count, 2)
            self.assertEqual([record["parallel_run"] for record in records], [1, 2])


if __name__ == "__main__":
    unittest.main()
