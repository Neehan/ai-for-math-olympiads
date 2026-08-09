"""Replicated Parallel-bank layout, accounting, and audit tests."""

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
    RUN_REFERENCE_FILENAME,
    SEED_AUDIT_FILENAME,
)
from src.models import Problem
from src.run import parallel_source_arm_name, solve_parallel_bank
from src.storage import (
    bank_run_output_dir,
    compile_arm_audit,
    write_parallel_bank_meta,
)
from src.token_pool import TokenPool


def _meta(
    tokens: int,
    *,
    arm: str = "baseline",
    mode: str = "single",
    run: int | None = None,
    gradeable: bool = True,
) -> dict[str, object]:
    record: dict[str, object] = {
        "problem_id": "p",
        "arm": arm,
        "mode": mode,
        "model": "claude-opus-4-8",
        "seed": 1,
        "budget_output_tokens": 200_000,
        "output_tokens_spent": tokens,
        "process_resume_count": 0,
        "session_reconnect_count": 0,
        "session_reconnects": [],
        "provider_usage_totals": {"output_tokens": tokens},
        "gradeable_solution_emitted": gradeable,
    }
    if run is not None:
        record.update(
            {
                "parallel_bank_seed": 1,
                "parallel_run": run,
                "parallel_run_budget_output_tokens": 200_000,
            }
        )
    return record


class ParallelBankTests(unittest.TestCase):
    def test_run_directories_are_fixed_and_zero_padded(self) -> None:
        root = Path("bank")
        self.assertEqual(bank_run_output_dir(root, 1), root / "run_01")
        self.assertEqual(bank_run_output_dir(root, 8), root / "run_08")
        with self.assertRaises(ValueError):
            bank_run_output_dir(root, 0)
        with self.assertRaises(ValueError):
            bank_run_output_dir(root, 9)

    def test_only_prespecified_parallel_arms_can_select_a_source(self) -> None:
        config = load_config(CONFIG_PATH)
        self.assertEqual(
            parallel_source_arm_name(config.arms["baseline-parallel"]), "baseline"
        )
        self.assertEqual(parallel_source_arm_name(config.arms["hint-parallel"]), "hint")
        with self.assertRaises(ValueError):
            parallel_source_arm_name(config.arms["baseline"])

    def test_bank_resume_launches_only_unfinished_fresh_runs(self) -> None:
        config = load_config(CONFIG_PATH)
        arm = config.arms["baseline-parallel"]
        problem = Problem("p", "Prove it.", "algebra", None, None, None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def output_dir(
                _config: object, selected_arm: object, _problem_id: str, seed: int
            ) -> Path:
                return root / getattr(selected_arm, "name") / f"seed_{seed}"

            source_dir = output_dir(config, config.arms["baseline"], "p", 1)
            source_dir.mkdir(parents=True)
            (source_dir / META_FILENAME).write_text("{}", encoding="utf-8")
            completed = bank_run_output_dir(output_dir(config, arm, "p", 1), 4)
            completed.mkdir(parents=True)
            (completed / META_FILENAME).write_text("{}", encoding="utf-8")
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
            launched = {call.args[4] for call in worker.await_args_list}
            self.assertEqual(launched, {2, 3, 5, 6, 7, 8})

    def test_bank_meta_reuses_run_01_and_accounts_for_eight_runs(self) -> None:
        config = load_config(CONFIG_PATH)
        arm = config.arms["baseline-parallel"]
        source_arm = config.arms["baseline"]
        problem = Problem("p", "Prove it.", "algebra", None, None, None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir = root / "claude-opus-4-8" / "baseline" / "p" / "seed_1"
            bank_dir = (
                root / "claude-opus-4-8" / "baseline-parallel" / "p" / "seed_1"
            )
            source_dir.mkdir(parents=True)
            (source_dir / META_FILENAME).write_text(
                json.dumps(_meta(10)), encoding="utf-8"
            )
            for run in range(2, 9):
                run_dir = bank_run_output_dir(bank_dir, run)
                run_dir.mkdir(parents=True)
                (run_dir / META_FILENAME).write_text(
                    json.dumps(
                        _meta(
                            run,
                            arm="baseline-parallel",
                            mode="parallel",
                            run=run,
                        )
                    ),
                    encoding="utf-8",
                )

            with patch("src.storage.RESULTS_ROOT", root):
                write_parallel_bank_meta(
                    config,
                    arm,
                    problem,
                    1,
                    bank_dir,
                    source_arm,
                    source_dir,
                )

            meta = json.loads((bank_dir / META_FILENAME).read_text(encoding="utf-8"))
            reference = json.loads(
                (bank_run_output_dir(bank_dir, 1) / RUN_REFERENCE_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(meta["parallel_run_count"], 8)
            self.assertEqual(meta["budget_output_tokens"], 1_600_000)
            self.assertEqual(meta["output_tokens_spent"], 10 + sum(range(2, 9)))
            self.assertEqual(meta["runs"][0]["source_result_path"], reference["source_result_path"])
            self.assertFalse((bank_run_output_dir(bank_dir, 1) / META_FILENAME).exists())

    def test_bank_meta_rejects_a_stale_run_from_another_bank(self) -> None:
        config = load_config(CONFIG_PATH)
        arm = config.arms["baseline-parallel"]
        source_arm = config.arms["baseline"]
        problem = Problem("p", "Prove it.", "algebra", None, None, None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir = root / config.model_dirname / "baseline" / "p" / "seed_1"
            bank_dir = root / config.model_dirname / arm.name / "p" / "seed_1"
            source_dir.mkdir(parents=True)
            (source_dir / META_FILENAME).write_text(
                json.dumps(_meta(10)), encoding="utf-8"
            )
            for run in range(2, 9):
                run_dir = bank_run_output_dir(bank_dir, run)
                run_dir.mkdir(parents=True)
                record = _meta(
                    run,
                    arm="baseline-parallel",
                    mode="parallel",
                    run=run,
                )
                if run == 6:
                    record["parallel_bank_seed"] = 2
                (run_dir / META_FILENAME).write_text(
                    json.dumps(record), encoding="utf-8"
                )
            with (
                patch("src.storage.RESULTS_ROOT", root),
                self.assertRaisesRegex(ValueError, "run_06.*identity mismatch"),
            ):
                write_parallel_bank_meta(
                    config,
                    arm,
                    problem,
                    1,
                    bank_dir,
                    source_arm,
                    source_dir,
                )

    def test_bank_audit_uses_ordered_prefixes_and_frozen_source(self) -> None:
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

            source_dir = output_dir(config, config.arms["baseline"], "p", 1)
            bank_dir = output_dir(config, arm, "p", 1)
            source_dir.mkdir(parents=True)
            (source_dir / META_FILENAME).write_text("{}", encoding="utf-8")
            (source_dir / SEED_AUDIT_FILENAME).write_text(
                json.dumps(
                    {
                        "problem_id": "p",
                        "arm": "baseline",
                        "seed": 1,
                        "solver_model": config.model,
                        "audit_model": config.audit_model,
                        "audit_score": scores[0],
                        "note": "source",
                    }
                ),
                encoding="utf-8",
            )
            for run, score in enumerate(scores[1:], start=2):
                run_dir = bank_run_output_dir(bank_dir, run)
                run_dir.mkdir(parents=True)
                (run_dir / META_FILENAME).write_text("{}", encoding="utf-8")
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
            self.assertEqual(record["candidate_pass_count"], 2)
            self.assertEqual(record["first_success_run"], 3)
            self.assertEqual(record["budget_cuts"]["1x"]["audit_score"], 0)
            self.assertEqual(record["budget_cuts"]["2x"]["audit_score"], 0)
            self.assertEqual(record["budget_cuts"]["4x"]["audit_score"], 5)
            self.assertEqual(record["budget_cuts"]["8x"]["audit_score"], 7)
            self.assertEqual(record["runs"][0]["source_arm"], "baseline")

    def test_compiled_audit_excludes_retired_flat_parallel_seeds(self) -> None:
        config = load_config(CONFIG_PATH)
        arm = config.arms["baseline-parallel"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            arm_root = root / config.model_dirname / arm.name / "p"
            for seed in (1, 4, 8):
                seed_dir = arm_root / f"seed_{seed}"
                seed_dir.mkdir(parents=True)
                (seed_dir / SEED_AUDIT_FILENAME).write_text(
                    json.dumps({"problem_id": "p", "seed": seed}),
                    encoding="utf-8",
                )
            with patch("src.storage.RESULTS_ROOT", root):
                path, count = compile_arm_audit(config, arm)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(count, 1)
            self.assertEqual([record["seed"] for record in records], [1])


if __name__ == "__main__":
    unittest.main()
