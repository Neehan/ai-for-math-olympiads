import math
import unittest

import numpy as np

from scripts.report_geometric_gof import (
    BANK, CHECKPOINTS, HORIZON, LOOKUP, conditional_support, holm,
    null_mle, residual, rmse, simulate, tail_summary,
)


class GeometricGoodnessOfFitTests(unittest.TestCase):
    def test_first_completion_and_censoring(self):
        times = np.array([[1, 4, 9], [2, 8, 9]])
        x = np.array([0, 24])
        # All-pass second fresh bank predicts three at every checkpoint.
        np.testing.assert_array_equal(residual(x, times), [-2, -1, -1, 0, 0, 0, 0, 1])

    def test_null_mle_counts_only_exposure_before_first_completion(self):
        q = null_mle(np.array([4, 0, 24]), np.array([[1, 4, 9], [9, 9, 9], [1, 1, 1]]))
        np.testing.assert_allclose(q, [6/37, 0, 1])
        grid = np.linspace(.001, .999, 10000)
        ll = 6*np.log(grid) + 31*np.log1p(-grid)
        self.assertLess(abs(grid[np.argmax(ll)] - q[0]), .0001)

    def test_finite_bank_prediction_unbiased(self):
        for q in [.01, .2, .7, .99]:
            weights = np.array([math.comb(BANK, x)*q**x*(1-q)**(BANK-x) for x in range(BANK+1)])
            np.testing.assert_allclose(weights @ LOOKUP, 1-(1-q)**CHECKPOINTS, atol=1e-13)

    def test_simulation_boundaries_and_monotone_curves(self):
        x, t = simulate(np.array([0., .2, 1.]), 1000, np.random.default_rng(10))
        self.assertTrue(np.all(x[:, 0] == 0) and np.all(t[:, 0] == 9))
        self.assertTrue(np.all(x[:, 2] == 24) and np.all(t[:, 2] == 1))
        self.assertTrue(np.all(np.diff((t[..., None] <= CHECKPOINTS).astype(int), axis=-1) >= 0))
        self.assertEqual(residual(x, t).shape, (1000, 8))

    def test_simulation_mse_matches_analytic_null_variance(self):
        q = np.array([.05, .2, .6])
        variance = np.zeros(HORIZON)
        for qi in q:
            p = 1-(1-qi)**CHECKPOINTS
            weights = np.array([math.comb(BANK, x)*qi**x*(1-qi)**(BANK-x) for x in range(BANK+1)])
            # Three independent target trajectories; one shared estimated bank.
            variance += 3*p*(1-p) + 9*(weights @ (LOOKUP**2) - p**2)
        x, t = simulate(q, 80000, np.random.default_rng(71))
        empirical = rmse(residual(x, t))**2
        self.assertLess(abs(empirical.mean()-variance.mean()), 5*empirical.std()/np.sqrt(len(empirical)))

    def test_conditional_probabilities_cancel_nuisance(self):
        x, t = 5, np.array([2, 5, 9])
        xs, ts, weights = conditional_support(x, t)
        successes = xs + (ts <= HORIZON).sum(axis=1)
        exposures = BANK + np.minimum(ts, HORIZON).sum(axis=1)
        self.assertTrue(np.all(successes == 7))
        self.assertTrue(np.all(exposures == 39))
        for q in [.05, .4, .9]:
            likelihood = np.array([math.comb(BANK, int(v)) for v in xs]) * q**successes * (1-q)**(exposures-successes)
            np.testing.assert_allclose(likelihood/likelihood.sum(), weights)
        self.assertTrue(any(xi == x and np.array_equal(ti, t) for xi, ti in zip(xs, ts)))

    def test_conditional_boundary_is_single_state(self):
        for x, t in [(0, [9, 9, 9]), (24, [1, 1, 1])]:
            xs, ts, weights = conditional_support(x, np.array(t))
            np.testing.assert_array_equal(xs, [x])
            np.testing.assert_array_equal(ts, [t])
            np.testing.assert_array_equal(weights, [1])

    def test_holm_and_finite_monte_carlo_tail(self):
        np.testing.assert_allclose(holm([.4, .016, .02, .8]), [.8, .064, .064, .8])
        self.assertEqual(tail_summary(np.arange(9), 9)['p_value'], .1)


if __name__ == '__main__':
    unittest.main()
