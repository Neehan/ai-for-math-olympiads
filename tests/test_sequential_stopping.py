"""Regression tests for the sequential self-convergence controller."""

import unittest

from src.constants import (
    SEQUENTIAL_MIN_ROUNDS_BEFORE_CONVERGENCE,
    SEQUENTIAL_NO_GAP_STREAK_TO_STOP,
)
from src.run import _sequential_self_converged


class SequentialStoppingTests(unittest.TestCase):
    def test_no_gap_streak_cannot_stop_before_round_floor(self) -> None:
        self.assertFalse(
            _sequential_self_converged(
                SEQUENTIAL_MIN_ROUNDS_BEFORE_CONVERGENCE - 1,
                SEQUENTIAL_NO_GAP_STREAK_TO_STOP + 20,
            )
        )

    def test_round_floor_alone_does_not_stop(self) -> None:
        self.assertFalse(
            _sequential_self_converged(
                SEQUENTIAL_MIN_ROUNDS_BEFORE_CONVERGENCE,
                SEQUENTIAL_NO_GAP_STREAK_TO_STOP - 1,
            )
        )

    def test_both_conditions_stop_at_round_floor(self) -> None:
        self.assertTrue(
            _sequential_self_converged(
                SEQUENTIAL_MIN_ROUNDS_BEFORE_CONVERGENCE,
                SEQUENTIAL_NO_GAP_STREAK_TO_STOP,
            )
        )


if __name__ == "__main__":
    unittest.main()
