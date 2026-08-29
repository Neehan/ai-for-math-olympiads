"""Prompt-rendering tests; no provider calls or credentials required."""

import unittest

from src.models import Problem
from src.prompts import (
    late_replay_prompt,
    task_prompt,
    uniform_strategy_execute_prompt,
    uniform_strategy_plan_prompt,
    uniform_strategy_plan_wrap_up_prompt,
)


class UniformStrategyPromptTests(unittest.TestCase):
    def test_late_replay_prompts_differ_only_by_hint_block(self) -> None:
        prior_work = "Old attempted proof."
        control = late_replay_prompt(prior_work, None, "/tmp/scratch", 200_000)
        prompt = late_replay_prompt(
            prior_work,
            "Use the key lemma.",
            "/tmp/scratch",
            200_000,
        )

        self.assertIn("## Prior attempted solution", prompt)
        self.assertIn("200,000 output tokens", prompt)
        self.assertIn("Use the key lemma.", prompt)
        self.assertIn("Old attempted proof.", prompt)
        self.assertLess(
            prompt.index("Old attempted proof."), prompt.index("Use the key lemma.")
        )
        self.assertEqual(control.count("Continue solving the problem."), 1)
        self.assertEqual(prompt.count("Continue solving the problem."), 1)
        self.assertEqual(
            prompt.replace(
                "Use the following proposed strategy as the basis of your solution. "
                "Check it carefully and repair any issues you find; you remain "
                "responsible for proving every step.\n\nProposed strategy:\n"
                "Use the key lemma.\n",
                "",
            ),
            control,
        )

    def test_planner_budget_split_comes_from_arguments(self) -> None:
        problem = Problem("test", "Prove the claim.", "algebra", None, None, None)

        prompt = uniform_strategy_plan_prompt(problem, "/tmp/scratch", 30_000, 3_500, 8)

        self.assertIn("30,000 output tokens total", prompt)
        self.assertIn("approximately 26,500 to explore", prompt)
        self.assertIn("reserving approximately 3,500", prompt)
        self.assertIn("between 1 and 8 strategies", prompt)
        self.assertIn("standalone plan for the entire problem", prompt)
        self.assertIn("load-bearing proof mechanisms differ", prompt)
        self.assertIn("Return fewer than 8", prompt)
        self.assertIn("split is strict", prompt)
        self.assertIn("at most 500 words", prompt)
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
        self.assertIn("following proposed strategy", prompt)
        self.assertIn("repair any issues you find", prompt)
        self.assertNotIn("following hint is correct", prompt)

    def test_planner_wrap_up_repeats_whole_proof_constraints(self) -> None:
        prompt = uniform_strategy_plan_wrap_up_prompt(4_000, 8)

        self.assertIn("semantically distinct whole-proof strategies", prompt)
        self.assertIn("combine complementary components", prompt)
        self.assertIn("return fewer entries rather than fragments", prompt)
        self.assertIn("entire response must fit", prompt)
        self.assertIn("at most 500 words", prompt)


if __name__ == "__main__":
    unittest.main()
