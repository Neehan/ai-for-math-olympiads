import math
import unittest

from scripts.report_allocation_model import (
    RootData,
    _matched_observations,
    _oracle_aligned_proof_curve,
    _oracle_plan_acquired,
    brute_force_allocation_probability,
    exact_allocation_probability,
)


class AllocationModelEstimatorTests(unittest.TestCase):
    def test_oracle_acquisition_requires_three_explicit_step_labels(self) -> None:
        complete = {
            "solution_sha256": "digest",
            "steps": [{"present": True}] * 3,
        }
        partial = {
            "solution_sha256": "digest",
            "steps": [{"present": True}, {"present": False}, {"present": True}],
        }
        self.assertTrue(_oracle_plan_acquired(complete))
        self.assertFalse(_oracle_plan_acquired(partial))
        self.assertFalse(_oracle_plan_acquired({"steps": []}))
        with self.assertRaisesRegex(ValueError, "lacks a complete three-step"):
            _oracle_plan_acquired({"solution_sha256": "digest", "steps": []})

    def test_observed_success_requires_correctness_and_oracle_alignment(self) -> None:
        proof = {
            "arm": "baseline-sequential",
            "problem_id": "p",
            "seed": 1,
            "audit_score": 7,
            "budget_cuts": {"1x": {"audit_score": 7}},
        }
        state = {
            "arm": "baseline-sequential",
            "problem_id": "p",
            "seed": 1,
            "solution_sha256": "final",
            "steps": [{"present": True}] * 3,
            "budget_cuts": {
                "1x": {
                    "solution_sha256": "cut",
                    "steps": [
                        {"present": True},
                        {"present": False},
                        {"present": True},
                    ],
                }
            },
        }
        self.assertEqual(
            _oracle_aligned_proof_curve(
                proof, state, final_block=2, threshold=5
            ),
            {1: False, 2: True},
        )

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

    def test_matched_observations_excludes_partial_seed_sets(self) -> None:
        data = RootData(
            proposals={},
            executions={},
            observed={
                "first": {
                    ("complete", 1): {1: True},
                    ("complete", 2): {1: False},
                    ("complete", 3): {1: True},
                    ("partial", 1): {1: True},
                    ("partial", 2): {1: True},
                },
                "second": {
                    ("complete", 1): {1: False},
                    ("complete", 2): {1: True},
                    ("complete", 3): {1: False},
                    ("partial", 1): {1: True},
                    ("partial", 2): {1: True},
                },
            },
        )
        weights, observed = _matched_observations(
            data,
            ("first", "second"),
            max_blocks=1,
            required_seeds=(1, 2, 3),
        )
        self.assertEqual(weights, {"complete": 3})
        self.assertEqual(observed, [3])


if __name__ == "__main__":
    unittest.main()
