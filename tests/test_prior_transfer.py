import unittest
from unittest.mock import patch

import numpy as np

from scripts.allocation_estimators import Joint
from scripts.report_prior_transfer import frozen_curves


def row(epsilon):
    return dict(solved=2, parallel_n=8, oracle_n=3, epsilon=epsilon)


class PriorTransferTests(unittest.TestCase):
    def test_strict_transfer_rejects_unseen_category(self):
        train = Joint([row([1/3]*8)])
        result, missing = frozen_curves(train, np.log([2, 3, 4, 5]),
                                        [row([1/3]+[2/3]*7)])
        self.assertIsNone(result)
        self.assertEqual(missing, [2])

    def test_positive_support_allows_new_category_without_fitting(self):
        train = Joint([row([1/3]*8)])
        train.active = np.arange(9)
        theta = np.log([2, 3] + [1]*9)
        before = theta.copy()
        with patch.object(Joint, 'fit', side_effect=AssertionError('Target fit forbidden')):
            result, missing = frozen_curves(train, theta, [row([1/3]+[2/3]*7)])
        self.assertEqual(missing, [])
        np.testing.assert_array_equal(theta, before)
        for n, horizon in [(1, 8), (2, 4), (4, 2)]:
            self.assertEqual(result[n].shape, (1, horizon))
            self.assertTrue(np.all(np.diff(result[n], axis=1) >= -1e-10))
            self.assertTrue(np.all((result[n] >= 0) & (result[n] <= 1)))

    def test_target_subset_preserves_source_parameter_mapping(self):
        train = Joint([row([1/3]*3+[2/3]*5)])
        theta = np.log([2, 3, 4, 5, 6])
        target = [row([1/3]*8)]
        result, missing = frozen_curves(train, theta, target)
        expected = Joint(target)
        expected.active = train.active.copy()
        np.testing.assert_allclose(result[1], expected.predict(theta)[0])
        self.assertEqual(missing, [])


if __name__ == '__main__':
    unittest.main()
