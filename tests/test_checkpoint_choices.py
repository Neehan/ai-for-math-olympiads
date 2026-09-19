import unittest

import numpy as np

from scripts.allocation_estimators import Joint
from scripts.report_checkpoint_choices import action, de_choices, evaluate, rde_choices, render_tables


class CheckpointChoicesTests(unittest.TestCase):
    def test_paper_tables_preserve_policy_order_and_ties(self):
        row = dict(always_continue=3., always_restart=0., RDE=0.)
        report = dict(restart_estimator='separate-three', models={
            model: dict(equal_horizon_average=row, summaries={str(h): row for h in (1, 2, 3, 4)})
            for model in ('muse', 'gpt54', 'gpt55', 'opus')})
        tables = render_tables(report)
        self.assertEqual(set(tables), {'checkpoint_regret_table.tex'})
        self.assertIn(r'Average & 3.00 & \textbf{0.00} & \textbf{0.00}', tables['checkpoint_regret_table.tex'])
        extra = render_tables(report, include_supplementary=True)
        self.assertEqual(extra['checkpoint_regret_budgets_table.tex'].count(r'$4\times$'), 4)
        report['restart_estimator'] = 'leave-one-out-five'
        self.assertEqual(render_tables(report), {})
        self.assertEqual(set(render_tables(report, include_supplementary=True)), {'checkpoint_regret_loo_table.tex'})

    def test_paper_tables_require_all_models(self):
        with self.assertRaises(ValueError):
            render_tables(dict(restart_estimator='separate-three', models={}))

    def test_geometric_is_indifferent(self):
        q = np.array([.1, .4, .8])
        curves = 1 - (1 - q[:, None]) ** np.arange(1, 9)
        c, r = de_choices(curves)
        np.testing.assert_allclose(c, r, atol=1e-10)

    def test_zero_survival_is_not_silently_clipped(self):
        c, r = de_choices([[1.] * 8])
        self.assertTrue(np.isnan(c).all())
        self.assertTrue(np.isnan(r).all())
        self.assertEqual(action(np.nan), 'undefined')

    def test_at_risk_and_average_of_separate_restart_seeds(self):
        curves = [[[0, 1, 1, 1, 1, 1, 1, 1], [1] * 8, [0] * 8]]
        c = np.full((1, 7), .3)
        r = np.full((1, 7), .2)
        out = evaluate(curves, [[0, 1, 1]], {'model': (c, r)})
        self.assertEqual(out[0]['at_risk'], 2)
        self.assertEqual(out[0]['continue_solved'], 1)
        self.assertAlmostEqual(out[0]['restart_solved'], 4/3)
        self.assertEqual(out[1]['at_risk'], 1)
        self.assertEqual(out[1]['continue_solved'], 0)
        self.assertAlmostEqual(out[1]['restart_solved'], 2/3)
        self.assertAlmostEqual(out[1]['frameworks']['model']['empirical_gap'], 2/3)

    def test_horizon_uses_correct_endpoint_and_checkpoint_count(self):
        curves = [[[0, 0, 0, 1, 1, 1, 1, 1]]*3]
        for h in (1, 2, 3, 4):
            predicted = {'model': (np.full((1, 8-h), .5), np.full((1, 8-h), .2))}
            out = evaluate(curves, [[0, 1, 0]], predicted, h)
            self.assertEqual(len(out), 8-h)
            self.assertEqual(out[0]['continue_solved'], 3 if h >= 3 else 0)

    def test_leave_one_out_excludes_current_trajectory_at_restart_horizon(self):
        curves = [[[0, 1, 1, 1, 1, 1, 1, 1], [1]*8, [0]*8]]
        h = 2
        predictions = {'model': (np.full((1, 6), .5), np.full((1, 6), .2))}
        out = evaluate(curves, [[1, 0, 0]], predictions, h, 'leave-one-out-five')
        # At k=1, seeds 1 and 3 remain. Their other-five success rates
        # at h=2 are respectively 2/5 and 3/5; seed 1's future success
        # must not enter its own restart estimate.
        self.assertEqual(out[0]['at_risk'], 2)
        self.assertAlmostEqual(out[0]['restart_solved'], 1.)
        self.assertAlmostEqual(out[1]['restart_solved'], 3/5)

    def test_conditional_posterior_against_monte_carlo(self):
        # Independent simulation of complete posterior draws and convolution.
        row = dict(solved=2, parallel_n=24, oracle_n=3,
                   epsilon=[1/3, 1/3, 2/3, 2/3, 2/3, 2/3, 2/3, 2/3])
        joint = Joint([row])
        theta = np.log([1.2, 4., 2., 1., 3.])
        _, w, ap, bp, ep, rp, d = joint.components(theta)
        rng = np.random.default_rng(20260918)
        count = 300000
        mixture = rng.choice(w.shape[1], count, p=w[0])
        alpha = rng.beta(ap[0, mixture], bp[0, mixture])
        e1 = rng.beta(np.broadcast_to(ep, w.shape)[0, mixture], rp[0, mixture])
        indices = joint.active[joint.active > 0]
        tail = rng.dirichlet(d[indices] + joint.c[0, indices], count)
        masses = np.zeros((count, 9))
        masses[:, 0] = e1
        masses[:, indices] = (1 - e1[:, None]) * tail
        epsilon = masses[:, :8].cumsum(axis=1)
        curves = np.zeros((count, 8))
        for k in range(1, 9):
            for j in range(1, k + 1):
                curves[:, k-1] += alpha*(1-alpha)**(j-1)*epsilon[:, k-j]
        for h in (1, 2, 3, 4):
            c, r = rde_choices(joint, theta, h)
            survival = (1-curves[:, :8-h]).mean(axis=0)
            sim_c = (curves[:, h:]-curves[:, :8-h]).mean(axis=0) / survival
            sim_r = ((1-curves[:, :8-h])*curves[:, h-1:h]).mean(axis=0) / survival
            np.testing.assert_allclose(c[0], sim_c, atol=.001)
            np.testing.assert_allclose(r[0], sim_r, atol=.001)
            self.assertTrue(np.all(r[0] < curves[:, h-1].mean()))


if __name__ == '__main__':
    unittest.main()
