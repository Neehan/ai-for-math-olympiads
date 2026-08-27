"""Determinism and validation tests for compression/selection controls."""

import dataclasses
import hashlib
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio
from claude_agent_sdk import ResultMessage
from scripts.build_selection_dataset import build_records
from scripts.reuse_uniform_strategies import reuse
from src.audit import _completed_without_verdict
from src.config import load_config
from src.constants import CONFIG_PATH, META_FILENAME, UNIFORM_STRATEGIES_FILENAME
from src.models import Problem
from src.prompts import (
    selection_no_problem_prompt,
    selection_prompt,
    uniform_compress_prompt,
)
from src.storage import uniform_strategy_only_done
from src.strategy_experiments import (
    _selection_candidates,
    compress_uniform_strategies,
    compression_examples,
    run_selection,
    sampled_strategy_indices,
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


class StrategyExperimentTests(unittest.TestCase):
    def test_selector_limit_is_terminal_but_provider_failure_is_not(self) -> None:
        def result(**overrides: object) -> ResultMessage:
            values: dict[str, object] = {
                "subtype": "success",
                "duration_ms": 1,
                "duration_api_ms": 1,
                "is_error": False,
                "num_turns": 1,
                "session_id": "session",
                "stop_reason": "end_turn",
                "total_cost_usd": 0.0,
                "usage": {"output_tokens": 1},
                "result": "",
                "structured_output": None,
            }
            values.update(overrides)
            return ResultMessage(**values)  # type: ignore[arg-type]

        self.assertTrue(
            _completed_without_verdict(result(stop_reason="max_tokens"))
        )
        self.assertTrue(_completed_without_verdict(result()))
        self.assertFalse(
            _completed_without_verdict(
                result(
                    subtype="error",
                    is_error=True,
                    num_turns=0,
                    errors=["API Error: 503"],
                )
            )
        )

    def test_config_contains_configured_auxiliary_arms_and_selection_caps(self) -> None:
        config = load_config(CONFIG_PATH)
        self.assertEqual(
            config.arms["baseline-uniform-strategy-only"].mode,
            "uniform_strategy_only",
        )
        self.assertEqual(
            config.arms["baseline-uniform-compress"].mode, "uniform_compress"
        )
        self.assertEqual(config.arms["selection"].seeds, [1, 2, 3])
        self.assertEqual(config.arms["selection-10k"].seeds, [1, 2, 3])
        self.assertEqual(config.arms["selection-40k"].seeds, [1, 2, 3])
        self.assertEqual(config.arms["selection-no-problem"].seeds, [1, 2, 3])
        self.assertEqual(
            {
                name: config.selection_budget_tokens(config.arms[name])
                for name in (
                    "selection-10k",
                    "selection",
                    "selection-40k",
                    "selection-no-problem",
                )
            },
            {
                "selection-10k": 10_000,
                "selection": 20_000,
                "selection-40k": 40_000,
                "selection-no-problem": 20_000,
            },
        )

    def test_strategy_sampling_is_stable_and_uses_three_distinct_raw_entries(self) -> None:
        first = sampled_strategy_indices("model", "problem", 8)
        second = sampled_strategy_indices("model", "problem", 8)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(set(first)), 3)
        with self.assertRaisesRegex(ValueError, "at least three"):
            sampled_strategy_indices("model", "problem", 2)

    def test_compression_examples_include_problem_and_avoid_target(self) -> None:
        ids = (
            "china-tst-2026-3",
            "rmm-2026-03",
            "apmo-2026-05",
            "china-tst-2026-12",
            "china-tst-2026-6",
            "imo-2026-06",
        )
        problems = [
            Problem(pid, f"Statement {pid}", "algebra", None, f"Hint {pid}", None)
            for pid in ids
        ]
        examples = compression_examples(problems, ids[0])
        self.assertEqual(len(examples), 5)
        self.assertNotIn((f"Statement {ids[0]}", f"Hint {ids[0]}"), examples)
        prompt = uniform_compress_prompt(
            Problem("target", "Target statement", "algebra", None, None, None),
            "Raw route",
            examples,
        )
        self.assertIn("Problem:\nStatement", prompt)
        self.assertIn("Sketch:\nHint", prompt)
        self.assertIn("at most 25", prompt)
        self.assertIn("Do not repair", prompt)

    def test_selection_order_is_reproducible_and_shared_across_controls(self) -> None:
        problem = Problem("p", "Statement", "algebra", None, "Oracle", None)
        record = {
            "oracle_strategy": "Oracle",
            "generated_strategies": [
                {
                    "candidate_id": f"g{i}",
                    "strategy": f"G{i}",
                    "strategy_acquired": i == 1,
                    "acquisition_basis": (
                        "reference_steps" if i == 1 else "none"
                    ),
                }
                for i in range(1, 4)
            ],
        }
        first, oracle_position = _selection_candidates(problem, record, 2, "model")
        second, second_position = _selection_candidates(problem, record, 2, "model")
        self.assertEqual(first, second)
        self.assertEqual(oracle_position, second_position)
        texts = [str(item["strategy"]) for item in first]
        with_problem = selection_prompt(problem, texts)
        without_problem = selection_no_problem_prompt(texts)
        for index, text in enumerate(texts, start=1):
            candidate = f"Strategy {index}: {text}"
            self.assertIn(candidate, with_problem)
            self.assertIn(candidate, without_problem)
        self.assertIn("complete correct proof", with_problem)
        self.assertIn("human-written frozen reference", without_problem)

    def test_selection_accepts_documented_human_alternative(self) -> None:
        problem = Problem("p", "Statement", "algebra", None, "Oracle", None)
        record = {
            "oracle_strategy": "Oracle",
            "generated_strategies": [
                {
                    "candidate_id": "g1",
                    "strategy": "A different valid route.",
                    "strategy_acquired": True,
                    "acquisition_basis": "human_alternative",
                    "adjudication_note": "Expert verified a complete alternative route.",
                },
                {
                    "candidate_id": "g2",
                    "strategy": "Incomplete route two.",
                    "strategy_acquired": False,
                    "acquisition_basis": "none",
                },
                {
                    "candidate_id": "g3",
                    "strategy": "Incomplete route three.",
                    "strategy_acquired": False,
                    "acquisition_basis": "none",
                },
            ],
        }
        candidates, _ = _selection_candidates(problem, record, 1, "model")
        alternative = next(
            candidate for candidate in candidates if candidate["candidate_id"] == "g1"
        )
        self.assertTrue(alternative["strategy_acquired"])
        self.assertEqual(alternative["acquisition_basis"], "human_alternative")
        self.assertIn("Expert verified", alternative["adjudication_note"])

        del record["generated_strategies"][0]["adjudication_note"]
        with self.assertRaisesRegex(ValueError, "adjudication_note"):
            _selection_candidates(problem, record, 1, "model")

    def test_compression_and_selection_emit_complete_auxiliary_artifacts(self) -> None:
        config = dataclasses.replace(load_config(CONFIG_PATH), model="source/model")
        compress_arm = config.arms["baseline-uniform-compress"]
        selection_arm = config.arms["selection"]
        target = Problem("target", "Target statement", "algebra", None, "Oracle", None)
        ids = (
            "china-tst-2026-3",
            "rmm-2026-03",
            "apmo-2026-05",
            "china-tst-2026-12",
            "china-tst-2026-6",
            "imo-2026-06",
        )
        examples = [
            Problem(pid, f"Statement {pid}", "algebra", None, f"Hint {pid}", None)
            for pid in ids
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "results"
            compression_output = (
                root
                / "source-model"
                / compress_arm.name
                / target.problem_id
                / "seed_1"
            )
            selection_output = (
                root
                / "source-model"
                / selection_arm.name
                / target.problem_id
                / "seed_1"
            )
            verdicts = [
                ({"strategy": "Compressed route one"}, []),
                ({"strategy": "Compressed route two"}, []),
                ({"strategy": "Compressed route three"}, []),
            ]

            def checkpoint_factory(_: object) -> _FakeCheckpoint:
                return _FakeCheckpoint(_, Path(temp) / "checkpoints")

            with (
                patch("src.strategy_experiments.RESULTS_ROOT", root),
                patch(
                    "src.strategy_experiments.seed_output_dir",
                    return_value=compression_output,
                ),
                patch(
                    "src.strategy_experiments._load_raw_strategies",
                    return_value=[f"Raw strategy {index}" for index in range(8)],
                ),
                patch(
                    "src.strategy_experiments.AttemptCheckpoint",
                    side_effect=checkpoint_factory,
                ),
                patch("src.audit._judge", AsyncMock(side_effect=verdicts)),
            ):
                anyio.run(
                    compress_uniform_strategies,
                    config,
                    compress_arm,
                    target,
                    1,
                    TokenPool(["unused"], "TEST_TOKEN"),
                    [*examples, target],
                    "worker/model",
                )
            artifact = json.loads(
                (compression_output / "compressed_strategies.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(artifact["generated_strategies"]), 3)
            self.assertTrue((compression_output / "meta.json").is_file())

            frozen = {
                "oracle_strategy": "Oracle",
                "generated_strategies": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "strategy": candidate["strategy"],
                        "strategy_acquired": False,
                        "acquisition_basis": "none",
                    }
                    for candidate in artifact["generated_strategies"]
                ],
            }
            selector_judge = AsyncMock(
                return_value=(
                    {"ranking": [1, 2, 3, 4], "reason": "Ranked."},
                    [],
                )
            )
            with (
                patch("src.strategy_experiments.RESULTS_ROOT", root),
                patch(
                    "src.strategy_experiments.seed_output_dir",
                    return_value=selection_output,
                ),
                patch(
                    "src.strategy_experiments.AttemptCheckpoint",
                    side_effect=checkpoint_factory,
                ),
                patch(
                    "src.audit._judge",
                    selector_judge,
                ),
            ):
                anyio.run(
                    partial(
                        run_selection,
                        include_problem=True,
                    ),
                    config,
                    selection_arm,
                    target,
                    1,
                    TokenPool(["unused"], "TEST_TOKEN"),
                    frozen,
                    "source/model",
                )
            audit = json.loads(
                (selection_output / "audit.json").read_text(encoding="utf-8")
            )
            self.assertIn(audit["oracle_rank"], {1, 2, 3, 4})
            self.assertIsInstance(audit["strategy_acquired_top1"], bool)
            self.assertEqual(audit["strategy_acquired_candidate_count"], 1)
            self.assertEqual(
                audit["random_strategy_acquired_top1_probability"], 0.25
            )
            self.assertTrue((selection_output / "selection.json").is_file())
            self.assertEqual(
                selector_judge.await_args.kwargs["max_output_tokens_per_response"],
                20_000,
            )
            self.assertEqual(selector_judge.await_args.kwargs["max_turns"], 1)
            self.assertTrue(
                selector_judge.await_args.kwargs["terminal_no_verdict"]
            )

    def test_selector_no_decision_is_terminal_and_scored_false(self) -> None:
        config = dataclasses.replace(load_config(CONFIG_PATH), model="source/model")
        arm = config.arms["selection"]
        problem = Problem("p", "Statement", "algebra", None, "Oracle", None)
        frozen = {
            "oracle_strategy": "Oracle",
            "generated_strategies": [
                {
                    "candidate_id": f"g{index}",
                    "strategy": f"G{index}",
                    "strategy_acquired": False,
                    "acquisition_basis": "none",
                }
                for index in range(1, 4)
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "results"
            output = root / "source-model" / arm.name / "p" / "seed_1"

            def checkpoint_factory(identity: object) -> _FakeCheckpoint:
                return _FakeCheckpoint(identity, Path(temp) / "checkpoints")

            with (
                patch("src.strategy_experiments.RESULTS_ROOT", root),
                patch("src.strategy_experiments.seed_output_dir", return_value=output),
                patch(
                    "src.strategy_experiments.AttemptCheckpoint",
                    side_effect=checkpoint_factory,
                ),
                patch(
                    "src.audit._judge",
                    AsyncMock(
                        return_value=(
                            {
                                "decision_status": "no_decision",
                                "stop_reason": "max_tokens",
                                "usage": {"output_tokens": 20_000},
                            },
                            [],
                        )
                    ),
                ),
            ):
                anyio.run(
                    partial(run_selection, include_problem=True),
                    config,
                    arm,
                    problem,
                    1,
                    TokenPool(["unused"], "TEST_TOKEN"),
                    frozen,
                    "source/model",
                )
            audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
            meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["decision_status"], "no_decision")
            self.assertFalse(audit["oracle_top1"])
            self.assertFalse(audit["strategy_acquired_top1"])
            self.assertIsNone(audit["oracle_rank"])
            self.assertEqual(meta["output_tokens_spent"], 20_000)

    def test_reuse_copies_only_planner_artifacts_and_marks_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model" / "baseline-uniform-strategy" / "p" / "seed_1"
            source.mkdir(parents=True)
            (source / UNIFORM_STRATEGIES_FILENAME).write_text(
                json.dumps(
                    {
                        "strategies": ["One", "Two", "Three"],
                        "run_strategy_indices": [1, 2, 3, 1, 2, 3, 1, 2],
                    }
                ),
                encoding="utf-8",
            )
            (source / META_FILENAME).write_text(
                json.dumps(
                    {
                        "problem_id": "p",
                        "uniform_strategy_plan_budget_output_tokens": 80000,
                        "uniform_strategy_plan_output_tokens_spent": 123,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(reuse(root, "model", {"other"}), 0)
            self.assertEqual(reuse(root, "model", {"p"}), 1)
            target = (
                root / "model" / "baseline-uniform-strategy-only" / "p" / "seed_1"
            )
            self.assertTrue(uniform_strategy_only_done(target))
            copied = json.loads(
                (target / UNIFORM_STRATEGIES_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(copied["strategies"], ["One", "Two", "Three"])
            self.assertEqual(copied["run_strategy_indices"], [])
            self.assertFalse((target / "run_01").exists())

    def test_reuse_preserves_a_completed_zero_proposal_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model" / "baseline-uniform-strategy" / "p" / "seed_1"
            source.mkdir(parents=True)
            (source / UNIFORM_STRATEGIES_FILENAME).write_text(
                json.dumps({"strategies": [], "run_strategy_indices": []}),
                encoding="utf-8",
            )
            (source / META_FILENAME).write_text(
                json.dumps(
                    {
                        "problem_id": "p",
                        "planner_failure": "No eligible strategy was emitted.",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(reuse(root, "model", {"p"}), 1)
            target = (
                root / "model" / "baseline-uniform-strategy-only" / "p" / "seed_1"
            )
            self.assertTrue(uniform_strategy_only_done(target))
            target_meta = json.loads(
                (target / META_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(target_meta["strategy_count"], 0)
            self.assertIn("planner_failure", target_meta)

    def test_selection_builder_aggregates_audited_strategy_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hints = root / "hints.jsonl"
            hints.write_text(
                json.dumps({"problem_id": "p", "hint": "Frozen oracle route"})
                + "\n",
                encoding="utf-8",
            )
            for source_model in ("model/a", "model/b"):
                model_dir = source_model.replace("/", "-")
                output = (
                    root
                    / "results"
                    / model_dir
                    / "baseline-uniform-compress"
                    / "p"
                    / "seed_1"
                )
                output.mkdir(parents=True)
                generated = [
                    {
                        "candidate_id": f"generated_{index}",
                        "raw_strategy_index": index,
                        "strategy": (
                            "First generated route"
                            if index == 1
                            else f"Generated route {index}"
                        ),
                    }
                    for index in range(1, 4)
                ]
                (output / "compressed_strategies.json").write_text(
                    json.dumps(
                        {
                            "problem_id": "p",
                            "source_model": source_model,
                            "oracle_strategy": "Frozen oracle route",
                            "generated_strategies": generated,
                        }
                    ),
                    encoding="utf-8",
                )
                (output / "meta.json").write_text(
                    json.dumps(
                        {
                            "problem_id": "p",
                            "model": source_model,
                            "mode": "uniform_compress",
                            "sample_seed": 20260827,
                            "sampled_raw_strategy_indices": [1, 2, 3],
                        }
                    ),
                    encoding="utf-8",
                )
                (output / "state_audit.json").write_text(
                    json.dumps(
                        {
                            "problem_id": "p",
                            "arm": "baseline-uniform-compress",
                            "solver_model": source_model,
                            "strategies": [
                                {
                                    "strategy_index": index,
                                    "candidate_id": candidate["candidate_id"],
                                    "raw_strategy_index": candidate[
                                        "raw_strategy_index"
                                    ],
                                    "strategy_sha256": hashlib.sha256(
                                        candidate["strategy"].encode("utf-8")
                                    ).hexdigest(),
                                    "state": "S" if index == 1 else "U",
                                    "strategy_acquired": index == 1,
                                    "acquisition_basis": (
                                        "reference_steps" if index == 1 else "none"
                                    ),
                                }
                                for index, candidate in enumerate(generated, 1)
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            records = build_records(root / "results", hints)
            self.assertEqual(
                [(record["source_model"], record["problem_id"]) for record in records],
                [("model/a", "p"), ("model/b", "p")],
            )
            first_a = records[0]["generated_strategies"][0]
            first_b = records[1]["generated_strategies"][0]
            self.assertIs(first_a["strategy_acquired"], True)
            self.assertIs(first_b["strategy_acquired"], True)
            self.assertEqual(first_a["acquisition_basis"], "reference_steps")
            self.assertEqual(first_b["acquisition_basis"], "reference_steps")


if __name__ == "__main__":
    unittest.main()
