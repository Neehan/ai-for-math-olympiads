import unittest

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from scripts.allocation_estimators import Joint
from scripts.report_measurement_checks import constraint_check, fit_sample, mse_delta, objective_gradient


def row(solved=4, e=1/3, problem='p'):
    return dict(dataset='results', problem=problem, solved=solved,
                parallel_n=24, oracle_n=3, epsilon=[e]*8)


class MeasurementCheckTests(unittest.TestCase):
    def test_empirical_violation_is_not_every_alpha_one(self):
        result=constraint_check([row(9),row(8),row(0,0),row(24,1)])
        self.assertEqual(result['violations'],1)
        self.assertEqual(result['alpha_one'],3)
        self.assertTrue(result['details'][2]['unidentified'])
        self.assertAlmostEqual(result['max_excess_pp'],100/24)

    def test_gradient(self):
        rows=[row(4),row(12,2/3),row(0,0)]
        model=Joint(rows);theta=np.log([.4,.7,.3,.2]);weights=np.array([2.,1.,3.])
        _,gradient=objective_gradient(model,theta,weights)
        numeric=approx_derivative(lambda x:objective_gradient(model,x,weights)[0],theta).ravel()
        np.testing.assert_allclose(gradient,numeric,atol=1e-6,rtol=1e-5)

    def test_weighted_likelihood_matches_repeated_problems(self):
        rows=[row(4),row(12,2/3)];weights=np.array([2.,3.]);theta=np.log([.4,.7,.3,.2])
        compressed=Joint(rows);expanded=Joint([rows[0]]*2+[rows[1]]*3)
        value,_=objective_gradient(compressed,theta,weights)
        self.assertAlmostEqual(value,expanded.objective(theta),places=9)

    def test_error_is_of_aggregate_curve(self):
        observed=np.zeros((2,8));geo=np.ones((2,8));rde=np.zeros((2,8))
        self.assertEqual(mse_delta(observed,geo,rde,np.ones(2)),36.)
        # Opposite signed problem errors cancel in aggregate, by definition.
        geo[1]=-1
        self.assertEqual(mse_delta(observed,geo,rde,np.ones(2)),0.)

    def test_saturated_execution_limit(self):
        curves,diagnostic=fit_sample([row(4,1),row(12,1),row(20,1)],np.ones(3))
        self.assertTrue(diagnostic['saturated_execution'])
        self.assertEqual(curves.shape,(3,8))
        self.assertTrue(np.all(np.diff(curves,axis=1)>=0))
        self.assertTrue(np.all((curves>=0)&(curves<=1)))


if __name__=='__main__': unittest.main()
