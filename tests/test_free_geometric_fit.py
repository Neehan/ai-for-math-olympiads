import unittest

import numpy as np

from scripts.report_free_geometric_fit import geometric_counts, two_rate_counts, summarize


class FreeGeometricFitTests(unittest.TestCase):
    def test_single_rate_and_boundaries(self):
        np.testing.assert_array_equal(geometric_counts([0, 1]), np.full(8, 3))
        np.testing.assert_allclose(geometric_counts([.2]*5), 15*(1-.8**np.arange(1, 9)))

    def test_two_rate_mixture_matches_problem_rates(self):
        np.testing.assert_allclose(two_rate_counts([.6, .8, .03], 15),
                                   geometric_counts([.8]*3+[.03]*2))

    def test_error_uses_eight_aggregate_counts(self):
        result = summarize(np.arange(8), np.zeros(8))
        self.assertAlmostEqual(result['rmse'], np.sqrt(17.5))
        self.assertAlmostEqual(result['mae'], 3.5)


if __name__ == '__main__':
    unittest.main()
