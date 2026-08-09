"""Fresh-IID Parallel-8 layout, accounting, migration, and audit tests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio

from src.audit import audit_parallel_bank
from src.config import load_config
from src.constants import (
    CONFIG_PATH,
    META_FILENAME,
    PARALLEL_BANK_PROTOCOL,
    RUN_REFERENCE_FILENAME,
    SEED_AUDIT_FILENAME,
    SOLUTION_FILENAME,
)
from src.models import Problem
from src.run import _solve_parallel_run, solve_parallel_bank
from src.storage import (
    bank_run_output_dir,
    compile_arm_audit,
    parallel_bank_done,
    write_parallel_bank_meta,
)
from src.token_pool import TokenPool


def _meta(tokens: int, run: int) -> dict[str, object]:
    return {
        "problem_id": "p",
        "arm": "baseline-parallel",
        "mode": "parallel",
        "model": "claude-opus-4-8",
        "seed": 1,
        "budget_output_tokens": 200_000,
        "output_tokens_spent": tokens,
        "process_resume_count": 0,
        "session_reconnect_count": 0,
        "session_reconnects": [],
        "provider_usage_totals": {"output_tokens": tokens},
        "gradeable_solution_emitted": True,
        "parallel_bank_seed": 1,
        "parallel_run": run,
        "parallel_run_budget_output_tokens": 200_000,
    }


def _write_member(bank_dir: Path, run: int, *, tokens: int | None = None) -> None:
    run_dir = bank_run_output_dir(bank_dir, run)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / META_FILENAME).write_text(
        json.dumps(_meta(tokens if tokens is not None else run, run)),
        encoding="utf-8",
    )
    (run_dir / SOLUTION_FILENAME).write_text("## Final Solution\nProof.", encoding="utf-8")


class ParallelBankTests(unittest.TestCase):
    def test_run_directories_are_fixed_and_zero_padded(self) -> None:
        root = Path("bank")
        self.assertEqual(bank_run_output_dir(root, 1), root / "run_01")
        self.assertEqual(bank_run_output_dir(root, 8), root / "run_08")
        with self.assertRaises(ValueError):
            bank_run_output_dir(root, 0)
        with self.assertRaises(ValueError):
            bank_run_output_dir(root, 9)

    def test_parallel_worker_accepts_fresh_run_01(self) -> None:
        """Regression: the controller and worker must agree on runs 01..08."""
        config = load_config(CONFIG_PATH)
        arm = config.arms["baseline-parallel"]
        problem = Problem("p", "Prove it.", "algebra", None, None, None)
        with tempfile.TemporaryDirectory() as temp:
            bank_dir = Path(temp)
            run_01 = bank_run_output_dir(bank_dir, 1)
            run_01.mkdir()
            (run_01 / META_FILENAME).write_text("{}", encoding="utf-8")
            with patch("src.run.seed_output_dir", return_value=bank_dir):
                anyio.run(
                    _solve_parallel_run,
                    config,
                    arm,
                    problem,
                    1,
                    1,
                    TokenPool(["unused"], "TEST_TOKEN"),
                )
                with self.assertRaisesRegex(ValueError, "01..08"):
                    anyio.run(
                        _solve_parallel_run,
                        config,
                        arm,
                        problem,
                        1,
                        0,
                        TokenPool(["unused"], "TEST_TOKEN"),
                    )

    def test_old_seven_fresh_bank_resumes_only_new_run_01(self) -> None:
        """The live pilot's paid runs 02..08 survive the protocol migration."""
        config = load_config(CONFIG_PATH)
        arm = config.arms["baseline-parallel"]
        problem = Problem("p", "Prove it.", "algebra", None, None, None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def output_dir(
                _config: object, selected_arm: object, _problem_id: str, seed: int
            ) -> Path:
                return root / getattr(selected_arm, "name") / f"seed_{seed}"

            bank_dir = output_dir(config, arm, "p", 1)
            for run in range(2, 9):
                _write_member(bank_dir, run)
            run_01 = bank_run_output_dir(bank_dir, 1)
            run_01.mkdir(parents=True)
            (run_01 / RUN_REFERENCE_FILENAME).write_text("{}", encoding="utf-8")
            worker = AsyncMock()
            with (
                patch("src.run.seed_output_dir", side_effect=output_dir),
                patch("src.run._solve_parallel_run", worker),
                patch("src.run.write_parallel_bank_meta"),
            ):
                anyio.run(
                    solve_parallel_bank,
                    config,
                    arm,
                    problem,
                    1,
                    TokenPool(["unused"], "TEST_TOKEN"),
                )
            self.assertEqual([call.args[4] for call in worker.await_args_list], [1])
            self.assertFalse((run_01 / RUN_REFERENCE_FILENAME).exists())

    def test_bank_meta_accounts_for_eight_fresh_local_runs(self) -> None:
        config = load_config(CONFIG_PATH)
        arm = config.arms["baseline-parallel"]
        problem = Problem("p", "Prove it.", "algebra", None, None, None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank_dir = root / config.model_dirname / arm.name / "p" / "seed_1"
            for run in range(1, 9):
                _write_member(bank_dir, run)

            with patch("src.storage.RESULTS_ROOT", root):
                write_parallel_bank_meta(config, arm, problem, 1, bank_dir)

            meta = json.loads((bank_dir / META_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(meta["parallel_bank_protocol"], PARALLEL_BANK_PROTOCOL)
            self.assertEqual(meta["parallel_run_count"], 8)
            self.assertEqual(meta["budget_output_tokens"], 1_600_000)
            self.assertEqual(meta["output_tokens_spent"], sum(range(1, 9)))
            self.assertTrue(all(record["local_result_path"] for record in meta["runs"]))
            self.assertNotIn("run_01_source_arm", meta)
            self.assertTrue(parallel_bank_done(bank_dir))
            # Generation staging intentionally preloads metadata only. The
            # last-written protocol marker must still make a finished bank
            # skippable without exposing old solutions to the solver.
            for run in range(1, 9):
                (bank_run_output_dir(bank_dir, run) / SOLUTION_FILENAME).unlink()
            self.assertTrue(parallel_bank_done(bank_dir))

    def test_bank_meta_rejects_a_stale_run_from_another_bank(self) -> None:
        config = load_config(CONFIG_PATH)
        arm = config.arms["baseline-parallel"]
        problem = Problem("p", "Prove it.", "algebra", None, None, None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank_dir = root / config.model_dirname / arm.name / "p" / "seed_1"
            for run in range(1, 9):
                _write_member(bank_dir, run)
            stale_path = bank_run_output_dir(bank_dir, 6) / META_FILENAME
            stale = json.loads(stale_path.read_text(encoding="utf-8"))
            stale["parallel_bank_seed"] = 2
            stale_path.write_text(json.dumps(stale), encoding="utf-8")
            with (
                patch("src.storage.RESULTS_ROOT", root),
                self.assertRaisesRegex(ValueError, "run_06.*identity mismatch"),
            ):
                write_parallel_bank_meta(config, arm, problem, 1, bank_dir)

    def test_old_top_level_marker_cannot_certify_new_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bank_dir = Path(temp)
            for run in range(1, 9):
                _write_member(bank_dir, run)
            (bank_dir / META_FILENAME).write_text(
                json.dumps({"parallel_run_count": 8}), encoding="utf-8"
            )
            self.assertFalse(parallel_bank_done(bank_dir))

    def test_bank_audit_reports_coverage_and_unbiased_pass_at_k(self) -> None:
        config = load_config(CONFIG_PATH)
        arm = config.arms["baseline-parallel"]
        problem = Problem("p", "Prove it.", "algebra", None, None, None)
        scores = [0, 0, 5, 0, 7, 0, 0, 0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def output_dir(
                _config: object, selected_arm: object, _problem_id: str, seed: int
            ) -> Path:
                return root / getattr(selected_arm, "name") / f"seed_{seed}"

            bank_dir = output_dir(config, arm, "p", 1)
            for run, score in enumerate(scores, start=1):
                _write_member(bank_dir, run)
                run_dir = bank_run_output_dir(bank_dir, run)
                (run_dir / SEED_AUDIT_FILENAME).write_text(
                    json.dumps(
                        {
                            "problem_id": "p",
                            "arm": arm.name,
                            "seed": 1,
                            "solver_model": config.model,
                            "audit_model": config.audit_model,
                            "parallel_bank_seed": 1,
                            "parallel_run": run,
                            "audit_score": score,
                            "note": f"run {run}",
                        }
                    ),
                    encoding="utf-8",
                )
            (bank_dir / META_FILENAME).write_text(
                json.dumps(
                    {
                        "parallel_bank_protocol": PARALLEL_BANK_PROTOCOL,
                        "parallel_run_count": 8,
                    }
                ),
                encoding="utf-8",
            )

            with patch("src.audit.seed_output_dir", side_effect=output_dir):
                anyio.run(
                    audit_parallel_bank,
                    config,
                    arm,
                    problem,
                    1,
                    TokenPool(["unused"], "TEST_TOKEN"),
                )

            record = json.loads(
                (bank_dir / SEED_AUDIT_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(record["parallel_bank_protocol"], PARALLEL_BANK_PROTOCOL)
            self.assertEqual(record["candidate_pass_count"], 2)
            self.assertEqual(record["first_success_run"], 3)
            self.assertAlmostEqual(record["pass_at_k"]["1"], 0.25)
            self.assertAlmostEqual(record["pass_at_k"]["2"], 13 / 28)
            self.assertAlmostEqual(record["pass_at_k"]["4"], 55 / 70)
            self.assertEqual(record["pass_at_k"]["8"], 1.0)
            self.assertEqual(record["budget_cuts"], {})

    def test_compiled_audit_excludes_retired_protocol_and_seeds(self) -> None:
        config = load_config(CONFIG_PATH)
        arm = config.arms["baseline-parallel"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            arm_root = root / config.model_dirname / arm.name / "p"
            for seed, protocol in ((1, "retired"), (2, PARALLEL_BANK_PROTOCOL)):
                seed_dir = arm_root / f"seed_{seed}"
                seed_dir.mkdir(parents=True)
                (seed_dir / SEED_AUDIT_FILENAME).write_text(
                    json.dumps(
                        {
                            "problem_id": "p",
                            "seed": seed,
                            "parallel_bank_protocol": protocol,
                        }
                    ),
                    encoding="utf-8",
                )
            with patch("src.storage.RESULTS_ROOT", root):
                path, count = compile_arm_audit(config, arm)
            self.assertEqual(count, 0)

            seed_1 = arm_root / "seed_1" / SEED_AUDIT_FILENAME
            seed_1.write_text(
                json.dumps(
                    {
                        "problem_id": "p",
                        "seed": 1,
                        "parallel_bank_protocol": PARALLEL_BANK_PROTOCOL,
                    }
                ),
                encoding="utf-8",
            )
            with patch("src.storage.RESULTS_ROOT", root):
                path, count = compile_arm_audit(config, arm)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(count, 1)
            self.assertEqual([record["seed"] for record in records], [1])


if __name__ == "__main__":
    unittest.main()
