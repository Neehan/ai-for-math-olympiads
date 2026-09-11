import math
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.report_allocation_model import (
    PAPER_PROFILES,
    RootData,
    _matched_observations,
    _parallel_acquired,
    _proof_curve,
    brute_force_allocation_probability,
    exact_allocation_probability,
)


class AllocationModelEstimatorTests(unittest.TestCase):
    def test_gpt55_uses_both_complete_parallel_seeds(self) -> None:
        profiles = [p for p in PAPER_PROFILES if p.model == "litellm-gpt-5.5"]
        self.assertEqual({p.key for p in profiles}, {"gpt55-n1", "gpt55-n2"})
        for profile in profiles:
            self.assertEqual(profile.proposal_seeds, (1, 2))
            self.assertEqual(profile.results_roots, ("results", "results-imobench"))

    def test_proof_curve_is_cumulative(self) -> None:
        record = {
            "arm": "baseline-sequential",
            "problem_id": "example",
            "seed": 1,
            "audit_score": 0,
            "budget_cuts": {
                "1x": {"audit_score": 0},
                "2x": {"audit_score": 5},
                "3x": {"audit_score": 0},
            },
        }
        self.assertEqual(
            _proof_curve(record, final_block=4, threshold=5),
            {1: False, 2: True, 3: True, 4: True},
        )

    def test_acquisition_is_union_not_intersection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            branch = Path(tmp)
            (branch / "solution.md").write_text("partial proof")
            state = {
                "solution_sha256": hashlib.sha256(b"partial proof").hexdigest(),
                "steps": [{"present": True} for _ in range(3)],
            }
            (branch / "state_audit.json").write_text(json.dumps(state))
            self.assertTrue(_parallel_acquired({"audit_score": 0}, branch, threshold=5))
            state["steps"][0]["present"] = False
            (branch / "state_audit.json").write_text(json.dumps(state))
            self.assertFalse(_parallel_acquired({"audit_score": 0}, branch, threshold=5))
            self.assertTrue(_parallel_acquired({"audit_score": 5}, branch, threshold=5))

    def test_missing_stale_and_empty_acquisition_audits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            branch = Path(tmp)
            self.assertFalse(_parallel_acquired({"audit_score": 0}, branch, threshold=5))
            (branch / "solution.md").write_text("partial proof")
            with self.assertRaisesRegex(ValueError, "Missing Parallel state audit"):
                _parallel_acquired({"audit_score": 0}, branch, threshold=5)
            state = {"solution_sha256": "stale", "steps": []}
            (branch / "state_audit.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "Stale Parallel state audit"):
                _parallel_acquired({"audit_score": 0}, branch, threshold=5)
            state["solution_sha256"] = hashlib.sha256(b"partial proof").hexdigest()
            (branch / "state_audit.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "Incomplete Parallel state steps"):
                _parallel_acquired({"audit_score": 0}, branch, threshold=5)

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
