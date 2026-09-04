import math
import unittest

from scripts.report_allocation_model import (
    brute_force_allocation_probability,
    exact_allocation_probability,
)


class AllocationModelEstimatorTests(unittest.TestCase):
    def test_compressed_estimator_matches_literal_equation_7(self) -> None:
        proposals = [True, False, True, False]
        executions = [{1: False, 2: True}, {1: True, 2: True}]
        expected = brute_force_allocation_probability(
            proposals,
            executions,
            n_arms=2,
            blocks_per_arm=2,
        )
        actual = exact_allocation_probability(
            proposals,
            executions,
            n_arms=2,
            blocks_per_arm=2,
        )
        self.assertTrue(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12))

    def test_two_arm_estimator_removes_plugin_diagonal(self) -> None:
        proposals = [True, False, False]
        executions = [{1: True}, {1: True}]
        actual = exact_allocation_probability(
            proposals,
            executions,
            n_arms=2,
            blocks_per_arm=1,
        )
        self.assertAlmostEqual(actual, 2 / 3)
        plugin = 1 - (1 - 1 / 3) ** 2
        self.assertAlmostEqual(plugin, 5 / 9)
        self.assertGreater(actual, plugin)

    def test_no_acquisition_means_zero_success(self) -> None:
        actual = exact_allocation_probability(
            [False, False, False, False],
            [{1: True, 2: True}, {1: True, 2: True}],
            n_arms=2,
            blocks_per_arm=2,
        )
        self.assertEqual(actual, 0.0)

    def test_requires_distinct_observations_for_all_arms(self) -> None:
        with self.assertRaisesRegex(ValueError, "Need 4 Parallel observations"):
            exact_allocation_probability(
                [True, False, False],
                [{1: True, 2: True}, {1: True, 2: True}],
                n_arms=2,
                blocks_per_arm=2,
            )
        with self.assertRaisesRegex(ValueError, "Need 2 oracle observations"):
            exact_allocation_probability(
                [True, False],
                [{1: True}],
                n_arms=2,
                blocks_per_arm=1,
            )


if __name__ == "__main__":
    unittest.main()
