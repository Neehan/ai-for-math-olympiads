"""Host-side completed-checkpoint cleanup tests."""

import fcntl
import json
import tempfile
import unittest
from pathlib import Path

from src.cleanup_checkpoints import cleanup_completed


class CheckpointCleanupTests(unittest.TestCase):
    def _checkpoint(
        self,
        root: Path,
        results: Path,
        name: str,
        scratch: str,
        *,
        completed: bool,
        marker_exists: bool = True,
    ) -> tuple[Path, Path]:
        attempt = root / "attempts" / name
        workspace = root / "w" / scratch
        attempt.mkdir(parents=True)
        workspace.mkdir(parents=True)
        marker = Path("model") / "arm" / name / "meta.json"
        if marker_exists:
            canonical = results / marker
            canonical.parent.mkdir(parents=True)
            canonical.write_text("{}", encoding="utf-8")
        (attempt / "state.json").write_text(
            json.dumps(
                {
                    "completed": completed,
                    "completion_marker": marker.as_posix(),
                    "roles": {"main": {"scratch_name": scratch}},
                }
            ),
            encoding="utf-8",
        )
        return attempt, workspace

    def test_only_completed_attempts_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            done = self._checkpoint(
                root, results, "a" * 24, "b" * 8, completed=False
            )
            live = self._checkpoint(
                root,
                results,
                "c" * 24,
                "d" * 8,
                completed=True,
                marker_exists=False,
            )
            self.assertEqual(cleanup_completed(root, results), 1)
            self.assertFalse(done[0].exists())
            self.assertFalse(done[1].exists())
            self.assertTrue(live[0].exists())
            self.assertTrue(live[1].exists())

    def test_malformed_workspace_aborts_before_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            safe = self._checkpoint(
                root, results, "a" * 24, "b" * 8, completed=True
            )
            bad_attempt = root / "attempts" / ("c" * 24)
            bad_attempt.mkdir(parents=True)
            bad_marker = results / "model/arm/bad/meta.json"
            bad_marker.parent.mkdir(parents=True, exist_ok=True)
            bad_marker.write_text("{}", encoding="utf-8")
            (bad_attempt / "state.json").write_text(
                json.dumps(
                    {
                        "completed": True,
                        "completion_marker": "model/arm/bad/meta.json",
                        "roles": {"main": {"scratch_name": "../../escape"}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "malformed workspace"):
                cleanup_completed(root, results)
            self.assertTrue(safe[0].exists())
            self.assertTrue(safe[1].exists())

    def test_live_attempt_lock_prevents_host_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            attempt, workspace = self._checkpoint(
                root, results, "a" * 24, "b" * 8, completed=True
            )
            with (attempt / ".lock").open("a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(cleanup_completed(root, results), 0)
                self.assertTrue(attempt.exists())
                self.assertTrue(workspace.exists())
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

            self.assertEqual(cleanup_completed(root, results), 1)
            self.assertFalse(attempt.exists())
            self.assertFalse(workspace.exists())


if __name__ == "__main__":
    unittest.main()
