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
from scripts.stage_selection_markers import stage_markers
from src.audit import _completed_without_verdict
from src.config import load_config
from src.constants import CONFIG_PATH, META_FILENAME, UNIFORM_STRATEGIES_FILENAME
from src.models import Problem
from src.models import PhaseResult
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
    sampled_strategy_indices,
)
from src.run import _parse_selection_decision, run_selection
from src.solver import BudgetTracker
from src.token_pool import TokenPool


class _FakeCheckpoint:
    def __init__(self, _: object, root: Path) -> None:
        self.root = root

    def scratch_dir(self, role: str) -> Path:
        path = self.root / role
        path.mkdir(parents=True, exist_ok=True)
        return path

    def phases(self, _: str) -> list[PhaseResult]:
        return []

    def tracker(
        self, _: str, budget_tokens: int, reserve_tokens: int
    ) -> BudgetTracker:
        return BudgetTracker(budget_tokens, reserve_tokens)

    def active(self, _: str) -> None:
        return None

    def session_id(self, _: str) -> None:
        return None

    def reconnects(self, _: str) -> list[object]:
        return []

    def save_session(self, *_: object) -> None:
        pass

    def session_ids(self) -> dict[str, str]:
        return {}

    def prepare_completion(self, _: object) -> None:
        pass

    def complete(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeSession:
    def __init__(self, *_: object, **__: object) -> None:
        self.session_id = "selection-session"
        self.reconnect_events: list[object] = []

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *_: object) -> None:
        pass


def _selection_phase(
    *,
    decision: dict[str, object] | None,
    label: str = "selection_work",
    cumulative_tokens: int = 100,
    budget_exhausted: bool = False,
) -> PhaseResult:
    text = ""
    if decision is not None:
        ranking = decision["ranking"]
        reason = decision["reason"]
        if not isinstance(ranking, list):
            raise TypeError("ranking must be a list")
        text = (
            f"<ranking>{','.join(str(value) for value in ranking)}</ranking>\n"
            f"<reason>{reason}</reason>"
        )
    return PhaseResult(
        label=label,
        prompt="selection",
        text=text,
        output_tokens=cumulative_tokens,
        cumulative_output_tokens=cumulative_tokens,
        num_turns=1,
        duration_ms=1,
        total_cost_usd=0.0,
        is_error=False,
        stop_reason="budget_exhausted" if budget_exhausted else "end_turn",
        budget_exhausted=budget_exhausted,
        tool_calls=[],
        reconnects=[],
    )


class StrategyExperimentTests(unittest.TestCase):
    def test_selection_staging_exposes_only_opaque_completion_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "results"
            destination = root / "staging"
            completed = source / "source-model" / "selection" / "p" / "seed_1"
            completed.mkdir(parents=True)
            (completed / "meta.json").write_text(
                json.dumps({"oracle_position": 3}), encoding="utf-8"
            )
            (completed / "selection.json").write_text(
                json.dumps({"candidates": [{"provenance": "oracle"}]}),
                encoding="utf-8",
            )
            self.assertEqual(
                stage_markers(source, destination, "source-model", "selection"),
                1,
            )
            staged = destination / completed.relative_to(source)
            self.assertEqual(
                json.loads((staged / "meta.json").read_text(encoding="utf-8")),
                {},
            )
            self.assertFalse((staged / "selection.json").exists())

    def test_selection_decision_parser_is_strict(self) -> None:
        self.assertEqual(
            _parse_selection_decision(
                "<ranking>4, 2, 1, 3</ranking>\n<reason>Fourth is strongest.</reason>"
            ),
            {"ranking": [4, 2, 1, 3], "reason": "Fourth is strongest."},
        )
        self.assertIsNone(
            _parse_selection_decision(
                "<ranking>1,1,2,3</ranking><reason>Duplicate.</reason>"
            )
        )
        self.assertIsNone(_parse_selection_decision("Ranking: 1,2,3,4"))

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

    def test_config_gives_selection_arms_the_standard_one_x_budget(self) -> None:
        config = load_config(CONFIG_PATH)
        self.assertEqual(
            config.arms["baseline-uniform-strategy-only"].mode,
            "uniform_strategy_only",
        )
        self.assertEqual(
            config.arms["baseline-uniform-compress"].mode, "uniform_compress"
        )
        self.assertEqual(
            config.arms["baseline-uniform-strategy-only"].seeds, [1]
        )
        self.assertEqual(
            config.arms["baseline-uniform-compress"].seeds, [1]
        )
        self.assertEqual(config.arms["selection"].seeds, [1, 2, 3])
        self.assertEqual(config.arms["selection-no-problem"].seeds, [1, 2, 3])
        self.assertEqual(config.budget_tokens(config.arms["selection"]), 200_000)
        self.assertEqual(
            config.budget_tokens(config.arms["selection-no-problem"]), 200_000
        )

    def test_strategy_sampling_is_stable_and_uses_three_distinct_raw_entries(self) -> None:
        first = sampled_strategy_indices("model", "problem", 1, 8)
        second = sampled_strategy_indices("model", "problem", 1, 8)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(set(first)), 3)
        with self.assertRaisesRegex(ValueError, "at least three"):
            sampled_strategy_indices("model", "problem", 1, 2)

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
        self.assertIn("18--25", prompt)
        self.assertIn("load-bearing route", prompt)
        self.assertIn("do not repair", prompt.lower())

    def test_selection_order_is_reproducible_and_shared_across_controls(self) -> None:
        problem = Problem("p", "Statement", "algebra", None, "Oracle", None)
        record = {
            "proposal_seed": 1,
            "oracle_strategy": "Oracle",
            "generated_strategies": [
                {
                    "candidate_id": f"g{i}",
                    "strategy": f"G{i}",
                    "oracle_strategy_match": i == 1,
                }
                for i in range(1, 4)
            ],
        }
        first, oracle_position = _selection_candidates(problem, record, 2, "model")
        second, second_position = _selection_candidates(problem, record, 2, "model")
        self.assertEqual(first, second)
        self.assertEqual(oracle_position, second_position)
        texts = [str(item["strategy"]) for item in first]
        with_problem = selection_prompt(problem, texts, 200_000, 20_000)
        without_problem = selection_no_problem_prompt(texts, 200_000, 20_000)
        for index, text in enumerate(texts, start=1):
            candidate = f"Strategy {index}: {text}"
            self.assertIn(candidate, with_problem)
            self.assertIn(candidate, without_problem)
        self.assertIn("complete correct proof", with_problem)
        self.assertIn("complete correct proof", without_problem)
        self.assertIn("Exactly one is derived", with_problem)
        self.assertIn("Exactly one is derived", without_problem)
        self.assertIn("200000 output tokens", with_problem)
        self.assertIn("180000 output tokens for this exploration", with_problem)
        self.assertIn("reserves up to 20000 output tokens", with_problem)
        self.assertIn("provided offline scratch tools", with_problem)
        self.assertIn("Problem statement:\n\nStatement", with_problem)
        self.assertIn(
            "Problem statement:\n\n[WITHHELD FOR THIS CONTROL]",
            without_problem,
        )
        self.assertEqual(
            with_problem.replace("Statement", "[WITHHELD FOR THIS CONTROL]"),
            without_problem,
        )

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
                (
                    {
                        "strategy": "Define the first proposed construction; apply its stated lemma, preserve its unsupported transition, and derive the claimed terminal bound."
                    },
                    [],
                ),
                (
                    {
                        "strategy": "Encode the second proposed route; invoke its named reduction, retain the unresolved bridge, and conclude through the proposed induction."
                    },
                    [],
                ),
                (
                    {
                        "strategy": "Construct the third proposed object; use its asserted invariant, preserve the missing justification, and close with the suggested counting step."
                    },
                    [],
                ),
            ]

            checkpoint_identities: list[object] = []

            def checkpoint_factory(identity: object) -> _FakeCheckpoint:
                checkpoint_identities.append(identity)
                return _FakeCheckpoint(identity, Path(temp) / "checkpoints")

            compressor_judge = AsyncMock(side_effect=verdicts)
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
                patch("src.audit._judge", compressor_judge),
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
            self.assertEqual(compressor_judge.await_count, 3)
            self.assertTrue(
                all(
                    "Oracle" not in call.args[1]
                    for call in compressor_judge.await_args_list
                )
            )
            self.assertEqual(len(artifact["generated_strategies"]), 3)
            self.assertEqual(artifact["oracle_strategy"], "Oracle")
            self.assertTrue((compression_output / "meta.json").is_file())

            frozen = {
                "proposal_seed": 1,
                "oracle_strategy": artifact["oracle_strategy"],
                "generated_strategies": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "strategy": candidate["strategy"],
                        "oracle_strategy_match": False,
                    }
                    for candidate in artifact["generated_strategies"]
                ],
            }
            selection_phase_runner = AsyncMock(
                return_value=_selection_phase(
                    decision={
                        "ranking": [1, 2, 3, 4],
                        "reason": "Ranked.",
                    }
                )
            )
            with (
                patch("src.run.RESULTS_ROOT", root),
                patch(
                    "src.run.seed_output_dir",
                    return_value=selection_output,
                ),
                patch(
                    "src.run.AttemptCheckpoint",
                    side_effect=checkpoint_factory,
                ),
                patch("src.run.ResumableClaudeSession", _FakeSession),
                patch("src.run._checkpointed_phase", selection_phase_runner),
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
                )
            audit = json.loads(
                (selection_output / "audit.json").read_text(encoding="utf-8")
            )
            self.assertIn(audit["oracle_rank"], {1, 2, 3, 4})
            self.assertIsInstance(audit["oracle_strategy_match_top1"], bool)
            self.assertNotIn("oracle_strategy_match_candidate_count", audit)
            self.assertNotIn("random_oracle_strategy_match_top1_probability", audit)
            self.assertNotIn("style_leakage_oracle_top1", audit)
            self.assertNotIn("worker_model", audit)
            self.assertNotIn(
                "worker_model",
                json.loads(
                    (selection_output / "selection.json").read_text(encoding="utf-8")
                ),
            )
            self.assertNotIn(
                "worker_model",
                json.loads((selection_output / "meta.json").read_text(encoding="utf-8")),
            )
            self.assertTrue((selection_output / "selection.json").is_file())
            self.assertEqual(selection_phase_runner.await_count, 1)
            self.assertEqual(
                selection_phase_runner.await_args.args[5], "selection_work"
            )
            self.assertEqual(selection_phase_runner.await_args.args[6], 180_000)
            meta = json.loads(
                (selection_output / "meta.json").read_text(encoding="utf-8")
            )
            self.assertTrue(meta["tools_enabled_during_work"])
            self.assertFalse(meta["tools_enabled_during_wrap"])
            self.assertEqual(meta["working_output_tokens"], 180_000)
            self.assertEqual(meta["wrap_up_reserve_tokens"], 20_000)
            selection_identity = checkpoint_identities[-1]
            self.assertIsInstance(selection_identity, dict)
            if not isinstance(selection_identity, dict):
                self.fail("selection checkpoint identity must be a dictionary")
            self.assertNotIn("candidate_order", selection_identity)
            self.assertNotIn("oracle_position", selection_identity)
            self.assertEqual(
                len(selection_identity["candidate_strategy_sha256s"]), 4
            )

    def test_selector_no_decision_is_terminal_and_scored_false(self) -> None:
        config = dataclasses.replace(load_config(CONFIG_PATH), model="source/model")
        arm = config.arms["selection"]
        problem = Problem("p", "Statement", "algebra", None, "Oracle", None)
        frozen = {
            "proposal_seed": 1,
            "oracle_strategy": "Oracle",
            "generated_strategies": [
                {
                    "candidate_id": f"g{index}",
                    "strategy": f"G{index}",
                    "oracle_strategy_match": False,
                }
                for index in range(1, 4)
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "results"
            output = root / "source-model" / arm.name / "p" / "seed_1"

            def checkpoint_factory(identity: object) -> _FakeCheckpoint:
                return _FakeCheckpoint(identity, Path(temp) / "checkpoints")

            async def exhaust_work_then_wrap(*args: object) -> PhaseResult:
                tracker = args[3]
                label = str(args[5])
                if not isinstance(tracker, BudgetTracker):
                    raise TypeError("Expected BudgetTracker")
                message_id = f"message-{label}"
                phase_tokens = 180_000 if label == "selection_work" else 20_000
                tracker.add(message_id, {"output_tokens": phase_tokens})
                tracker.finish_phase(None)
                return _selection_phase(
                    decision=None,
                    label=label,
                    cumulative_tokens=tracker.spent,
                    budget_exhausted=True,
                )

            phase_runner = AsyncMock(side_effect=exhaust_work_then_wrap)
            with (
                patch("src.run.RESULTS_ROOT", root),
                patch("src.run.seed_output_dir", return_value=output),
                patch(
                    "src.run.AttemptCheckpoint",
                    side_effect=checkpoint_factory,
                ),
                patch("src.run.ResumableClaudeSession", _FakeSession),
                patch(
                    "src.run._checkpointed_phase",
                    phase_runner,
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
                )
            audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
            meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["decision_status"], "no_decision")
            self.assertFalse(audit["oracle_top1"])
            self.assertFalse(audit["oracle_strategy_match_top1"])
            self.assertIsNone(audit["oracle_rank"])
            self.assertEqual(meta["output_tokens_spent"], 200_000)
            self.assertEqual(phase_runner.await_count, 2)
            self.assertEqual(
                [call.args[5] for call in phase_runner.await_args_list],
                ["selection_work", "selection_wrap"],
            )
            self.assertEqual(
                [call.args[6] for call in phase_runner.await_args_list],
                [180_000, 200_000],
            )

    def test_selector_rejects_an_over_budget_ranking(self) -> None:
        config = dataclasses.replace(load_config(CONFIG_PATH), model="source/model")
        arm = config.arms["selection"]
        problem = Problem("p", "Statement", "algebra", None, "Oracle", None)
        frozen = {
            "proposal_seed": 1,
            "oracle_strategy": "Oracle",
            "generated_strategies": [
                {
                    "candidate_id": f"g{index}",
                    "strategy": f"G{index}",
                    "oracle_strategy_match": False,
                }
                for index in range(1, 4)
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "results"
            output = root / "source-model" / arm.name / "p" / "seed_1"

            def checkpoint_factory(identity: object) -> _FakeCheckpoint:
                return _FakeCheckpoint(identity, Path(temp) / "checkpoints")

            async def exceed_budget(*args: object) -> PhaseResult:
                tracker = args[3]
                if not isinstance(tracker, BudgetTracker):
                    raise TypeError("Expected BudgetTracker")
                tracker.add("over-budget", {"output_tokens": 200_001})
                tracker.finish_phase(None)
                return _selection_phase(
                    decision={
                        "ranking": [1, 2, 3, 4],
                        "reason": "Over-budget ranking.",
                    },
                    cumulative_tokens=tracker.spent,
                    budget_exhausted=True,
                )

            with (
                patch("src.run.RESULTS_ROOT", root),
                patch("src.run.seed_output_dir", return_value=output),
                patch(
                    "src.run.AttemptCheckpoint",
                    side_effect=checkpoint_factory,
                ),
                patch("src.run.ResumableClaudeSession", _FakeSession),
                patch(
                    "src.run._checkpointed_phase",
                    AsyncMock(side_effect=exceed_budget),
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
                )
            audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
            artifact = json.loads(
                (output / "selection.json").read_text(encoding="utf-8")
            )
            meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["decision_status"], "no_decision")
            self.assertIsNone(audit["oracle_rank"])
            self.assertEqual(artifact["ranking_positions"], [])
            self.assertEqual(artifact["stop_reason"], "budget_exhausted")
            self.assertFalse(meta["budget_eligible"])

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
                for seed in (1,):
                    raw_output = (
                        root
                        / "results"
                        / model_dir
                        / "baseline-uniform-strategy-only"
                        / "p"
                        / f"seed_{seed}"
                    )
                    output = (
                        root
                        / "results"
                        / model_dir
                        / "baseline-uniform-compress"
                        / "p"
                        / f"seed_{seed}"
                    )
                    raw_output.mkdir(parents=True)
                    output.mkdir(parents=True)
                    raw_strategies = [
                        f"Uncompressed raw strategy {index} for seed {seed}."
                        for index in range(1, 4)
                    ]
                    generated = [
                        {
                            "candidate_id": f"generated_{index}",
                            "raw_strategy_index": index,
                            "raw_strategy": raw_strategies[index - 1],
                            "strategy": (
                                f"Define generated route {index} for seed {seed}; apply its "
                                "stated construction, preserve the unresolved lemma, and "
                                "finish through its proposed counting argument."
                            ),
                        }
                        for index in range(1, 4)
                    ]
                    (raw_output / "strategies.json").write_text(
                        json.dumps(
                            {
                                "strategies": raw_strategies,
                                "run_strategy_indices": [],
                            }
                        ),
                        encoding="utf-8",
                    )
                    (raw_output / "meta.json").write_text(
                        json.dumps(
                            {
                                "problem_id": "p",
                                "model": source_model,
                                "mode": "uniform_strategy_only",
                                "seed": seed,
                            }
                        ),
                        encoding="utf-8",
                    )
                    (raw_output / "state_audit.json").write_text(
                        json.dumps(
                            {
                                "problem_id": "p",
                                "arm": "baseline-uniform-strategy-only",
                                "solver_model": source_model,
                                "seed": seed,
                                "strategies": [
                                    {
                                        "strategy_index": index,
                                        "candidate_id": f"strategy_{index}",
                                        "strategy_sha256": hashlib.sha256(
                                            strategy.encode("utf-8")
                                        ).hexdigest(),
                                        "oracle_strategy_match": index == 1,
                                    }
                                    for index, strategy in enumerate(raw_strategies, 1)
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
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
                                "seed": seed,
                                "sample_seed": 20260827,
                                "sampled_raw_strategy_indices": [1, 2, 3],
                                "oracle_strategy_sha256": hashlib.sha256(
                                    b"Frozen oracle route"
                                ).hexdigest(),
                                "oracle_word_count": 3,
                            }
                        ),
                        encoding="utf-8",
                    )
            records = build_records(root / "results", hints)
            self.assertEqual(
                [
                    (
                        record["source_model"],
                        record["problem_id"],
                        record["proposal_seed"],
                    )
                    for record in records
                ],
                [
                    ("model/a", "p", 1),
                    ("model/b", "p", 1),
                ],
            )
            first_a = records[0]["generated_strategies"][0]
            first_b = records[1]["generated_strategies"][0]
            self.assertIs(first_a["oracle_strategy_match"], True)
            self.assertIs(first_b["oracle_strategy_match"], True)
            self.assertNotIn("strategy_acquired", first_a)
            self.assertNotIn("acquisition_basis", first_a)


if __name__ == "__main__":
    unittest.main()
