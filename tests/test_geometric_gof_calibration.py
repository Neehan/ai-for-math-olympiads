import unittest
from unittest.mock import patch

import numpy as np

from scripts.check_geometric_gof_calibration import bootstrap_p, calibration, rate_summary, wilson
from scripts.report_geometric_gof import null_mle


class GeometricCalibrationTests(unittest.TestCase):
    def test_wilson_reference(self):
        np.testing.assert_allclose(wilson(100, 2000), [.04128, .06045], atol=.0001)
        self.assertAlmostEqual(wilson(0, 100)[0], 0)
        self.assertAlmostEqual(wilson(100, 100)[1], 1)

    def test_rejection_includes_threshold(self):
        r = rate_summary(np.array([.01, .05, .051, .9]), .05)
        self.assertEqual(r['rejections'], 2)
        self.assertEqual(r['rate'], .5)

    def test_inner_simulation_uses_refitted_q(self):
        x = np.array([3, 10])
        t = np.array([[1, 4, 9], [2, 3, 5]])
        with patch('scripts.check_geometric_gof_calibration.simulate',
                   return_value=(np.broadcast_to(x, (10, 2)), np.broadcast_to(t, (10, 2, 3)))) as sim:
            p = bootstrap_p(x, t, 10, np.random.default_rng(1))
            np.testing.assert_allclose(sim.call_args.args[0], null_mle(x, t))
            self.assertEqual(p, 1)

    def test_degenerate_null_never_rejects(self):
        result = calibration(np.array([0., 1.]), 10, 19, 123, 'unit', known_reference=100)
        for key in ('known_q_control', 'fitted_q_procedure'):
            self.assertTrue(all(r['rejections'] == 0 for r in result[key]))

    def test_deterministic_reproduction(self):
        first = calibration(np.array([.1, .5]), 10, 19, 213, 'unit', known_reference=100)
        second = calibration(np.array([.1, .5]), 10, 19, 213, 'unit', known_reference=100)
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
