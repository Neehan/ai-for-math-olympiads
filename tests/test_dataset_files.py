"""Which dataset files each stage may stage into a container.

`run.sh --dataset-dir` copies exactly the files this helper lists, so an entry
added here is an entry handed to a live container. A generation container must
never receive reference solutions.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts" / "dataset_files.py"


def _listing(dataset: str, stage: str, *flags: str) -> dict[str, str]:
    env = dict(os.environ)
    env["HARNESS_DATASET"] = dataset
    result = subprocess.run(
        [sys.executable, str(HELPER), stage, *flags],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )
    return dict(line.split(" ", 1) for line in result.stdout.splitlines())


class DatasetFileListingTests(unittest.TestCase):
    def test_generation_never_receives_reference_solutions(self) -> None:
        for dataset in ("math-contests-2026", "imobench", "aime26"):
            with self.subTest(dataset=dataset):
                self.assertNotIn("solutions.jsonl", _listing(dataset, "run"))

    def test_audit_stages_receive_reference_solutions(self) -> None:
        for stage in ("audit", "state-audit"):
            with self.subTest(stage=stage):
                listing = _listing("math-contests-2026", stage)
                self.assertEqual(listing["solutions.jsonl"], "hard_solutions.jsonl")

    def test_selection_candidates_only_for_selection_arms(self) -> None:
        self.assertNotIn("selection.jsonl", _listing("math-contests-2026", "run"))
        self.assertEqual(
            _listing("math-contests-2026", "run", "--selection-arm")["selection.jsonl"],
            "hard_hint_selection.jsonl",
        )

    def test_answer_graded_dataset_lists_only_what_it_publishes(self) -> None:
        self.assertEqual(
            _listing("aime26", "run"), {"problems.jsonl": "aime26_problems.jsonl"}
        )
        self.assertEqual(
            _listing("aime26", "audit"),
            {
                "problems.jsonl": "aime26_problems.jsonl",
                "solutions.jsonl": "aime26_solutions.jsonl",
            },
        )
        # An arm needing them is refused before staging, but the listing must
        # not invent hint or outline sources the dataset never published.
        self.assertNotIn("hints.jsonl", _listing("aime26", "audit"))
        self.assertNotIn("outlines.jsonl", _listing("aime26", "audit"))

    def test_proof_datasets_stage_their_full_hint_ladder(self) -> None:
        listing = _listing("imobench", "run")
        self.assertEqual(listing["problems.jsonl"], "imobench_problems.jsonl")
        self.assertEqual(listing["hints.jsonl"], "imobench_hints.jsonl")
        self.assertEqual(listing["outlines.jsonl"], "imobench_outlines.jsonl")


if __name__ == "__main__":
    unittest.main()
