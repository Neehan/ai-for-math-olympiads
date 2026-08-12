"""Prompt-rendering tests; no provider calls or credentials required."""

import unittest

from src.models import Problem
from src.prompts import (
    task_prompt,
    uniform_strategy_execute_prompt,
    uniform_strategy_plan_prompt,
)


class UniformStrategyPromptTests(unittest.TestCase):
    def test_planner_budget_split_comes_from_arguments(self) -> None:
        problem = Problem("test", "Prove the claim.", "algebra", None, None, None)

        prompt = uniform_strategy_plan_prompt(problem, "/tmp/scratch", 30_000, 3_500, 8)

        self.assertIn("30,000 output tokens total", prompt)
        self.assertIn("approximately 26,500 to explore", prompt)
        self.assertIn("reserving approximately 3,500", prompt)
        self.assertIn("between 1 and 8 strategies", prompt)
        self.assertNotIn("20,000", prompt)
        self.assertNotIn("18,000", prompt)
        self.assertNotIn("2,000", prompt)

    def test_executor_prompt_is_identical_to_oracle_hint_prompt(self) -> None:
        problem = Problem("test", "Prove the claim.", "algebra", None, None, None)
        strategy = "Use an extremal counterexample."
        prompt = uniform_strategy_execute_prompt(
            problem,
            strategy,
            "/tmp/scratch",
            190_000,
        )

        self.assertEqual(
            prompt,
            task_prompt(problem, strategy, "/tmp/scratch", 190_000),
        )
        self.assertIn("The following hint is correct", prompt)


if __name__ == "__main__":
    unittest.main()
