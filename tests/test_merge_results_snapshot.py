import tempfile
import unittest
from pathlib import Path

from scripts.merge_results_snapshot import merge_snapshot


class MergeResultsSnapshotTests(unittest.TestCase):
    def test_python_caches_neither_conflict_nor_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "remote"
            local = root / "local"
            relative_seed = Path("results/model/arm/problem/seed_1")
            remote_seed = remote / relative_seed
            local_seed = local / relative_seed
            for seed in (remote_seed, local_seed):
                seed.mkdir(parents=True)
                (seed / "solution.md").write_text("same proof", encoding="utf-8")
                (seed / "meta.json").write_text("{}", encoding="utf-8")

            remote_cache = remote_seed / "scratch/__pycache__/test.cpython-39.pyc"
            local_cache = local_seed / "scratch/__pycache__/test.cpython-312.pyc"
            remote_cache.parent.mkdir(parents=True)
            local_cache.parent.mkdir(parents=True)
            remote_cache.write_bytes(b"remote cache")
            local_cache.write_bytes(b"local cache")

            self.assertEqual(merge_snapshot(remote, local, dry_run=False), 0)
            self.assertTrue(local_cache.exists())
            self.assertFalse((local_cache.parent / remote_cache.name).exists())


if __name__ == "__main__":
    unittest.main()
