"""Dataset routing and reference-selection tests."""

import os
import subprocess
import sys
import unittest

from src import storage


class DatasetSelectionTests(unittest.TestCase):
    def test_placebo_is_next_hint_in_sorted_domain_cycle(self) -> None:
        problems = [
            {"problem_id": "c", "domain": "algebra"},
            {"problem_id": "a", "domain": "algebra"},
            {"problem_id": "b", "domain": "algebra"},
            {"problem_id": "y", "domain": "combinatorics"},
            {"problem_id": "x", "domain": "combinatorics"},
        ]
        hints = {
            problem_id: {"hint": f"hint-{problem_id}"}
            for problem_id in ("a", "b", "c", "x", "y")
        }

        self.assertEqual(
            storage._domain_shifted_placebos(problems, hints),
            {
                "a": "hint-b",
                "b": "hint-c",
                "c": "hint-a",
                "x": "hint-y",
                "y": "hint-x",
            },
        )

    def test_placebo_shift_rejects_single_problem_domain(self) -> None:
        with self.assertRaises(ValueError):
            storage._domain_shifted_placebos(
                [{"problem_id": "only", "domain": "algebra"}],
                {"only": {"hint": "own hint"}},
            )

    def test_imobench_selects_its_four_frozen_files(self) -> None:
        script = (
            "from src.constants import HINTS_URL, OUTLINES_URL, PROBLEMS_URL, "
            "SOLUTIONS_URL; "
            "print(PROBLEMS_URL); print(HINTS_URL); print(OUTLINES_URL); "
            "print(SOLUTIONS_URL)"
        )
        env = dict(os.environ)
        env["HARNESS_DATASET"] = "imobench"
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        lines = result.stdout.splitlines()
        self.assertTrue(lines[0].endswith("/imobench_problems.jsonl"))
        self.assertTrue(lines[1].endswith("/imobench_hints.jsonl"))
        self.assertTrue(lines[2].endswith("/imobench_outlines.jsonl"))
        self.assertTrue(lines[3].endswith("/imobench_solutions.jsonl"))

    def test_imobench_reference_uses_marked_index_zero(self) -> None:
        record = {
            "problem_id": "PB-Advanced-001",
            "reference_solutions": [
                {"route_id": "hard_hint", "solution": "Official proof."}
            ],
        }
        self.assertEqual(storage._outline_reference(record), "Official proof.")

    def test_math_contests_reference_keeps_route_marker_check(self) -> None:
        record = {
            "problem_id": "problem-1",
            "reference_solutions": [{"solution": "Proof without marker."}],
        }
        with self.assertRaises(ValueError):
            storage._outline_reference(record)


if __name__ == "__main__":
    unittest.main()
