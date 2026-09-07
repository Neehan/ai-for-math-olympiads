"""Checkpoint namespaces ignore operational replication-count changes."""

import unittest

from scripts.checkpoint_namespace import _canonicalize_seed_roster
from src.models import ArmConfig, arm_checkpoint_identity


class CheckpointNamespaceTests(unittest.TestCase):
    def test_sequential_seed_expansion_preserves_namespace_input(self) -> None:
        old = b"""{
  "arms": {
    "baseline-sequential": { "hint": "none", "mode": "sequential", "budget_units": 8, "seeds": [1, 2, 3] },
    "hint-sequential": { "hint": "h2", "mode": "sequential", "budget_units": 8, "seeds": [1, 2, 3] }
  }
}
"""
        expanded = old.replace(b"[1, 2, 3]", b"[1, 2, 3, 4, 5]")
        self.assertEqual(
            _canonicalize_seed_roster(expanded),
            _canonicalize_seed_roster(old),
        )

    def test_sequential_seed_expansion_preserves_attempt_identity(self) -> None:
        for name, hint in (
            ("baseline-sequential", "none"),
            ("hint-sequential", "h2"),
        ):
            with self.subTest(arm=name):
                old = ArmConfig(name, hint, "sequential", 8, [1, 2, 3])
                expanded = ArmConfig(
                    name, hint, "sequential", 8, [1, 2, 3, 4, 5]
                )
                self.assertEqual(
                    arm_checkpoint_identity(expanded), arm_checkpoint_identity(old)
                )


if __name__ == "__main__":
    unittest.main()
