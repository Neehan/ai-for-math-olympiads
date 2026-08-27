"""Generation-staging pruning tests for interrupted search banks."""

import tempfile
import unittest
from pathlib import Path

from scripts.prune_staging import prune_staging


class StagingMergeTests(unittest.TestCase):
    def test_incomplete_bank_keeps_only_completed_child_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seed = root / "model" / "arm" / "problem" / "seed_1"
            complete = seed / "run_02"
            marker_only = seed / "run_03"
            partial = seed / "run_04"
            for path in (complete, marker_only, partial):
                path.mkdir(parents=True)
            (seed / "meta.json").write_text("preseed", encoding="utf-8")
            (complete / "meta.json").write_text("done", encoding="utf-8")
            (complete / "solution.md").write_text("proof", encoding="utf-8")
            (marker_only / "meta.json").write_text("preseed", encoding="utf-8")
            (partial / "solution.md").write_text("partial", encoding="utf-8")

            prune_staging(root)

            self.assertTrue(complete.is_dir())
            self.assertFalse(marker_only.exists())
            self.assertFalse(partial.exists())
            self.assertFalse((seed / "meta.json").exists())

    def test_completed_top_level_attempt_is_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seed = root / "model" / "arm" / "problem" / "seed_1"
            seed.mkdir(parents=True)
            (seed / "meta.json").write_text("done", encoding="utf-8")
            (seed / "solution.md").write_text("proof", encoding="utf-8")
            (seed / "scratch.txt").write_text("evidence", encoding="utf-8")

            prune_staging(root)

            self.assertTrue((seed / "scratch.txt").is_file())

    def test_completed_selection_attempt_retains_deterministic_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seed = root / "model" / "selection" / "problem" / "seed_1"
            seed.mkdir(parents=True)
            for name in ("meta.json", "solution.md", "selection.json", "audit.json"):
                (seed / name).write_text("frozen", encoding="utf-8")

            prune_staging(root)

            self.assertTrue((seed / "selection.json").is_file())
            self.assertTrue((seed / "audit.json").is_file())

    def test_marker_only_normal_attempt_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seed = root / "model" / "arm" / "problem" / "seed_1"
            seed.mkdir(parents=True)
            (seed / "meta.json").write_text("preseed", encoding="utf-8")

            prune_staging(root)

            self.assertFalse(seed.exists())


if __name__ == "__main__":
    unittest.main()
