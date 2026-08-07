"""Uniform Strategy parsing and exact bank-accounting tests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anyio

from src.audit import audit_uniform_strategy_bank
from src.config import load_config
from src.constants import (
    CONFIG_PATH,
    META_FILENAME,
    SEED_AUDIT_FILENAME,
    SOLUTION_FILENAME,
    UNIFORM_STRATEGIES_FILENAME,
)
from src.models import PhaseResult, Problem
from src.run import _proposed_strategies
from src.storage import uniform_branch_output_dir, write_uniform_strategy_bank_meta
from src.token_pool import TokenPool


def _phase(text: str, tokens: int = 100) -> PhaseResult:
    return PhaseResult(
        label="plan",
        prompt="prompt",
        text=text,
        output_tokens=tokens,
        cumulative_output_tokens=tokens,
        num_turns=1,
        duration_ms=1,
        total_cost_usd=0.0,
        is_error=False,
        stop_reason="complete",
        budget_exhausted=False,
        tool_calls=[],
        reconnects=[],
    )


class UniformStrategyTests(unittest.TestCase):
    def test_parser_deduplicates_and_caps_tagged_strategies(self) -> None:
        phases = [
            _phase(
                "## Strategy Set\n"
                "<strategy>Use parity.</strategy>\n"
                "<strategy>  Use   parity. </strategy>\n"
                "<strategy>Use an extremal counterexample.</strategy>"
            )
        ]
        self.assertEqual(
            _proposed_strategies(phases, 8, 200),
            ["Use parity.", "Use an extremal counterexample."],
        )
        self.assertEqual(_proposed_strategies(phases, 1, 200), ["Use parity."])

    def test_parser_preserves_substantive_malformed_output_as_one_strategy(
        self,
    ) -> None:
        self.assertEqual(
            _proposed_strategies([_phase("Try an invariant and induction.")], 8, 200),
            ["Try an invariant and induction."],
        )

    def test_parser_rejects_strategy_text_ending_beyond_planner_budget(self) -> None:
        phase = _phase("<strategy>Use parity.</strategy>", 201)
        phase.cumulative_output_tokens = 201
        with self.assertRaises(RuntimeError):
            _proposed_strategies([phase], 8, 200)

    def test_bank_meta_accounts_for_exact_eight_x_cap(self) -> None:
        config = load_config(CONFIG_PATH)
        arm = config.arms["baseline-uniform-strategy"]
        problem = Problem("p", "Prove it.", "algebra", None, None, None)
        executor_budget = 190_000
        with tempfile.TemporaryDirectory() as temp:
            bank_dir = Path(temp)
            for branch in range(1, 9):
                branch_dir = uniform_branch_output_dir(bank_dir, branch)
                branch_dir.mkdir()
                (branch_dir / META_FILENAME).write_text(
                    json.dumps(
                        {
                            "output_tokens_spent": branch,
                            "process_resume_count": 0,
                            "provider_session_ids": {"main": f"uuid-{branch}"},
                            "gradeable_solution_emitted": True,
                        }
                    ),
                    encoding="utf-8",
                )
            write_uniform_strategy_bank_meta(
                config,
                arm,
                problem,
                1,
                bank_dir,
                [_phase("<strategy>Use parity.</strategy>", 20)],
                1,
                [1] * 8,
                executor_budget,
                {"plan": "plan-uuid"},
            )
            meta = json.loads((bank_dir / META_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(meta["budget_output_tokens"], 1_600_000)
            self.assertEqual(
                meta["uniform_strategy_executor_budget_output_tokens_each"],
                executor_budget,
            )
            self.assertEqual(meta["output_tokens_spent"], 20 + sum(range(1, 9)))
            self.assertTrue((bank_dir / SOLUTION_FILENAME).is_file())

    def test_bank_audit_aggregates_independent_branch_verdicts(self) -> None:
        config = load_config(CONFIG_PATH)
        arm = config.arms["baseline-uniform-strategy"]
        problem = Problem("p", "Prove it.", "algebra", None, None, None)
        with tempfile.TemporaryDirectory() as temp:
            bank_dir = Path(temp)
            (bank_dir / UNIFORM_STRATEGIES_FILENAME).write_text(
                json.dumps(
                    {
                        "strategies": ["Use parity.", "Use extremality."],
                        "branch_strategy_indices": [1, 2, 1, 2, 1, 2, 1, 2],
                    }
                ),
                encoding="utf-8",
            )
            scores = [0, 5, 6, 7, 0, 0, 5, 0]
            for branch, score in enumerate(scores, start=1):
                branch_dir = uniform_branch_output_dir(bank_dir, branch)
                branch_dir.mkdir()
                (branch_dir / SEED_AUDIT_FILENAME).write_text(
                    json.dumps({"audit_score": score, "note": f"score {score}"}),
                    encoding="utf-8",
                )
            with patch("src.audit.seed_output_dir", return_value=bank_dir):
                anyio.run(
                    audit_uniform_strategy_bank,
                    config,
                    arm,
                    problem,
                    1,
                    TokenPool(["unused"], "TEST_TOKEN"),
                )
            record = json.loads(
                (bank_dir / SEED_AUDIT_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(record["audit_score"], 7)
            self.assertEqual(record["candidate_pass_count"], 4)
            self.assertEqual(len(record["branches"]), 8)


if __name__ == "__main__":
    unittest.main()
