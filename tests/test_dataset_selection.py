"""Dataset routing and reference-selection tests."""

import json
import os
import subprocess
import sys
import unittest

from src import storage


def _run_with_dataset(
    dataset: str, script: str, *args: str, check: bool = True
) -> "subprocess.CompletedProcess[str] | str":
    """Run one script in a subprocess under HARNESS_DATASET.

    Dataset routing is resolved when src.constants is imported, so a test can
    only exercise another dataset from a fresh interpreter. Returns stdout for
    the success path and the whole result when the caller expects a failure.
    """
    env = dict(os.environ)
    env["HARNESS_DATASET"] = dataset
    result = subprocess.run(
        [sys.executable, "-c", script, *args],
        check=check,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout if check else result


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

    def test_aime26_selects_answer_graded_sources_only(self) -> None:
        script = (
            "from src.constants import DATASET_ANSWER_GRADED, "
            "DATASET_HAS_STRATEGY_ARTIFACTS, HINTS_URL, OUTLINES_URL, "
            "PROBLEMS_URL, SELECTION_URL, SOLUTIONS_URL; "
            "print(PROBLEMS_URL); print(SOLUTIONS_URL); print(HINTS_URL); "
            "print(OUTLINES_URL); print(SELECTION_URL); "
            "print(DATASET_ANSWER_GRADED); print(DATASET_HAS_STRATEGY_ARTIFACTS)"
        )
        lines = _run_with_dataset("aime26", script).splitlines()
        self.assertTrue(lines[0].endswith("/aime26_problems.jsonl"))
        self.assertTrue(lines[1].endswith("/aime26_solutions.jsonl"))
        self.assertEqual(lines[2:5], ["None", "None", "None"])
        self.assertEqual(lines[5:7], ["True", "False"])

    def test_aime26_problems_carry_no_hint_tiers(self) -> None:
        records = [
            {
                "problem_id": "aime-2026-01",
                "statement": "Find n.",
                "domain": "aime",
                "task": "answer_only",
            }
        ]
        script = (
            "import json, sys; from src import storage; "
            "storage._fetch_jsonl = lambda env, url: json.loads(sys.argv[1]); "
            "problem = storage.load_problems()[0]; "
            "print(problem.problem_id); print(problem.domain); "
            "print(problem.hint_h1); print(problem.hint_h2); print(problem.hint_h3)"
        )
        lines = _run_with_dataset(
            "aime26", script, json.dumps(records)
        ).splitlines()
        self.assertEqual(lines, ["aime-2026-01", "aime", "None", "None", "None"])

    def test_aime26_reference_is_the_published_answer(self) -> None:
        records = [
            {
                "problem_id": "aime-2026-01",
                "statement": "Find n.",
                "answer": "204",
            }
        ]
        script = (
            "import json, sys; from src import storage; "
            "storage._fetch_jsonl = lambda env, url: json.loads(sys.argv[1]); "
            "print(storage.load_audit_references()['aime-2026-01'][1])"
        )
        self.assertEqual(
            _run_with_dataset("aime26", script, json.dumps(records)).strip(), "204"
        )

    def test_aime26_grades_with_the_answer_equivalence_prompt(self) -> None:
        script = (
            "from src.models import Problem; from src.prompts import audit_prompt; "
            "problem = Problem(problem_id='aime-2026-01', statement='Find n.', "
            "domain='aime', hint_h1=None, hint_h2=None, hint_h3=None); "
            "rendered = audit_prompt(problem, '204', '## Final Solution\\n204'); "
            "print('ground-truth answer' in rendered); "
            "print('reference solution' in rendered)"
        )
        self.assertEqual(
            _run_with_dataset("aime26", script).splitlines(), ["True", "False"]
        )

    def test_aime26_refuses_arms_needing_strategy_artifacts(self) -> None:
        script = (
            "from src.config import load_config, require_supported_arm; "
            "from src.constants import CONFIG_PATH; "
            "config = load_config(CONFIG_PATH); "
            "require_supported_arm(config.arms['baseline']); "
            "require_supported_arm(config.arms['baseline-parallel']); "
            "print('no-hint arms allowed'); "
            "require_supported_arm(config.arms['hint'])"
        )
        result = _run_with_dataset("aime26", script, check=False)
        self.assertIn("no-hint arms allowed", result.stdout)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot run on dataset 'aime26'", result.stderr)

    def test_proof_datasets_still_grade_against_a_reference_solution(self) -> None:
        script = (
            "from src.models import Problem; from src.prompts import audit_prompt; "
            "problem = Problem(problem_id='problem-1', statement='Prove it.', "
            "domain='algebra', hint_h1=None, hint_h2=None, hint_h3=None); "
            "rendered = audit_prompt(problem, 'Official proof.', "
            "'## Final Solution\\nQED'); "
            "print('reference solution' in rendered); "
            "print('ground-truth answer' in rendered)"
        )
        self.assertEqual(
            _run_with_dataset("imobench", script).splitlines(), ["True", "False"]
        )

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
