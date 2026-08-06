"""Prompt-rendering tests; no provider calls or credentials required."""

import unittest

from src.models import Problem
from src.prompts import ideasearch_plan_prompt


class IdeaSearchPromptTests(unittest.TestCase):
    def test_planner_budget_split_comes_from_arguments(self) -> None:
        problem = Problem("test", "Prove the claim.", "algebra", None, None, None)

        prompt = ideasearch_plan_prompt(problem, "/tmp/scratch", 30_000, 3_500)

        self.assertIn("30,000 output tokens total", prompt)
        self.assertIn("approximately 26,500 to explore", prompt)
        self.assertIn("reserving approximately 3,500", prompt)
        self.assertNotIn("20,000", prompt)
        self.assertNotIn("18,000", prompt)
        self.assertNotIn("2,000", prompt)


if __name__ == "__main__":
    unittest.main()
