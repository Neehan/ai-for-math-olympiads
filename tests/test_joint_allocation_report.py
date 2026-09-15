import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.allocation_estimators import (
    Joint, check_posterior_n1, check_posterior_n2, fit_de, posterior_n2, posterior_n4,
)
from scripts.report_joint_allocation import (
    MODELS, collect_interventions, collect_targets, fit_interventions, geometric,
    indexed_audits, passing, proof_curve, summarize,
)


def row(x=2, m=5, epsilon=None):
    return dict(solved=x, parallel_n=m, oracle_n=3,
                epsilon=[1/3]*3+[2/3]*5 if epsilon is None else epsilon)


class JointEstimatorTests(unittest.TestCase):
    def test_de_interior(self):
        fit = fit_de(row(8, 24, [2/3]*8))
        self.assertAlmostEqual(fit['alpha'], .5)
        self.assertEqual(fit['epsilon1'], 2/3)
        self.assertAlmostEqual(fit['prediction_unidentified_alpha0'][0], 8/24)

    def test_de_clip_and_joint_boundary_adjustment(self):
        fit = fit_de(row(12, 24, [1/3]*3+[2/3]*5))
        self.assertEqual(fit['alpha'], 1)
        self.assertAlmostEqual(fit['epsilon1'], 13/27)
        self.assertAlmostEqual(fit['epsilon'][3], 13/27 + (1-13/27)/2)
        np.testing.assert_allclose(fit['prediction_unidentified_alpha0'], fit['epsilon'])

    def test_de_zero_oracle_with_fresh_success(self):
        fit = fit_de(row(2, 24, [0]*3+[1/3]*5))
        self.assertEqual(fit['alpha'], 1)
        self.assertAlmostEqual(fit['epsilon1'], 2/27)
        self.assertFalse(fit['unidentified'])

    def test_de_unidentified_keeps_sensitivity(self):
        fit = fit_de(row(0, 24, [0]*3+[1/3]*5))
        self.assertTrue(fit['unidentified'])
        self.assertEqual(fit['prediction_unidentified_alpha0'], [0]*8)
        self.assertEqual(fit['prediction_unidentified_alpha1'], [0]*3+[1/3]*5)

    def test_de_geometric_special_case(self):
        fit = fit_de(row(6, 24, [1]*8))
        np.testing.assert_allclose(fit['prediction_unidentified_alpha0'], [1-.75**k for k in range(1, 9)])

    def test_posterior_against_independent_quadrature(self):
        check_posterior_n1()
        check_posterior_n2()

    def test_n2_not_plugin(self):
        model = Joint([row()])
        theta = np.log([2, 3, 2, 3, 4])
        single, _ = model.predict(theta)
        double = posterior_n2(model, theta)
        self.assertTrue(np.all(double >= single[:, :4]))
        self.assertTrue(np.all(double < 1-(1-single[:, :4])**2))

    def test_n4_independent_integration(self):
        from numpy.polynomial.legendre import leggauss
        from scipy.special import betaln
        z, w = leggauss(32)
        z, w = (z+1)/2, w/2
        a, e, r = z[:, None, None], z[None, :, None], z[None, None, :]
        def pdf(x, aa, bb):
            return np.exp((aa-1)*np.log(x)+(bb-1)*np.log1p(-x)-betaln(aa, bb))
        # Dirichlet prior (2,3,4); oracle counts (1,1,1).
        model = Joint([row(epsilon=[1/3]+[2/3]*7)])
        theta = np.log([2, 3, 2, 3, 4])
        mass = w[:, None, None]*w[None, :, None]*w[None, None, :]
        mass = mass*pdf(a, 2, 3)*pdf(e, 3, 9)*pdf(r, 4, 5)*(a*e)**2*(1-a*e)**3
        mass /= mass.sum()
        expected = [np.sum(mass*(1-(1-s)**4)) for s in [a*e, a*(2-a)*e+a*(1-e)*r]]
        np.testing.assert_allclose(posterior_n4(model, theta)[0], expected, atol=1e-11)
        # No block-2 completions: r is a point mass at zero.
        sparse = Joint([row()])
        actual = posterior_n4(sparse, theta)
        self.assertEqual(actual.shape, (1, 2))
        self.assertTrue(np.all(actual >= posterior_n2(sparse, theta)[:, :2]))

    def test_finite_bank_against_enumeration(self):
        for m in range(1, 7):
            for x in range(m+1):
                bank = [True]*x+[False]*(m-x)
                for k in range(1, m+1):
                    expected = sum(any(v) for v in itertools.combinations(bank, k))/math.comb(m, k)
                    self.assertAlmostEqual(geometric(x, m, k), expected)

    def test_input_validation(self):
        for bad in [row(6, 5), row(epsilon=[.2]*8), row(epsilon=[1, 0]*4), row(epsilon=[float('nan')]*8)]:
            with self.assertRaises(ValueError):
                fit_de(bad)
        with self.assertRaisesRegex(ValueError, 'requires some oracle completions'):
            Joint([row(epsilon=[0]*8)])

    def test_targets_cannot_change_fit(self):
        rows = [
            dict(row(x, 8), problem=f'p{i}',
                 dataset='results' if i < 2 else 'results-imobench')
            for i, x in enumerate((0, 2, 5))
        ]
        changed = [dict(r, observed=[1]*8, weight=100, acquired=999) for r in rows]
        for prior_dataset in (None, 'results'):
            with self.subTest(prior_dataset=prior_dataset):
                before, diagnostics = fit_interventions(rows, prior_dataset=prior_dataset)
                after, changed_diagnostics = fit_interventions(changed, prior_dataset=prior_dataset)
                self.assertEqual(before, after)
                self.assertEqual(diagnostics, changed_diagnostics)
                expected_training = ['p0', 'p1', 'p2'] if prior_dataset is None else ['p0', 'p1']
                self.assertEqual(diagnostics['training_problems'], expected_training)


class AuditAndAggregationTests(unittest.TestCase):
    def test_comparison_tables_use_count_rmse(self):
        from scripts.render_allocation_report import comparison_table
        from scripts.report_joint_allocation import METHODS
        reports = {}
        for n, models in [(1, ['muse', 'gpt54', 'gpt55', 'opus']), (2, ['muse', 'gpt54', 'gpt55']), (4, ['muse', 'gpt55'])]:
            for model in models:
                rows = []
                for dataset, size, count_error in [('results', 35, 10.), ('results-imobench', 22, 20.)]:
                    rows.extend(
                        dict(dataset=dataset, problem=f'{dataset}-{j}', weight=3,
                             observed=[0.]*(8//n),
                             predictions={method: [(count_error+i)/(3*size)]*(8//n)
                                          for i, method in enumerate(METHODS)})
                        for j in range(size)
                    )
                reports[f'{model}-n{n}'] = summarize(rows)
        for n in (1, 2):
            table = comparison_table(dict(reports=reports), n)
            self.assertIn('RMSE in solved-trial counts', table)
            self.assertIn(r'\textbf{10.00}', table)
            self.assertIn(r'\textbf{20.00}', table)
            self.assertIn('AOBench', table)
            self.assertIn('IMO-ProofBench', table)
            self.assertNotIn('5.85', table)
        self.assertIn('$N=4$', comparison_table(dict(reports=reports), 2))

    def test_cumulative_and_missing_scores(self):
        r = dict(problem_id='p', audit_score=0, budget_cuts={'1x': {'audit_score': 5}, '2x': {'audit_score': 0}})
        self.assertEqual(proof_curve(r, 3, 5), [1, 1, 1])
        r['budget_cuts']['2x']['audit_score'] = None
        with self.assertRaises(ValueError):
            proof_curve(r, 3, 5)
        for value in (True, None, '5', -1, 8):
            with self.assertRaises(ValueError):
                passing(value, 5)

    def test_duplicate_audits_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'audit.jsonl'
            text = json.dumps(dict(problem_id='p', seed=1))+'\n'
            path.write_text(text*2)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                indexed_audits(path, (1,), {})

    def test_gpt55_seed_selection_and_missing_banks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root/'results'/MODELS['gpt55']
            fresh_path = base/'baseline-parallel/audit.jsonl'
            oracle_path = base/'hint-sequential/audit.jsonl'
            fresh_path.parent.mkdir(parents=True)
            oracle_path.parent.mkdir(parents=True)
            fresh = [dict(problem_id='p', seed=s, runs=[dict(run=k, audit_score=7 if s == 1 else 0) for k in range(1, 9)]) for s in (1, 2, 3)]
            oracle = [dict(problem_id='p', seed=s, audit_score=0, budget_cuts={f'{k}x': {'audit_score': 5 if k == 2 else 0} for k in range(1, 8)}) for s in (1, 2, 3, 4, 5)]
            fresh_path.write_text(''.join(json.dumps(r)+'\n' for r in fresh))
            oracle_path.write_text(''.join(json.dumps(r)+'\n' for r in oracle))
            rows = collect_interventions(root, 'gpt55', 5, {}, {'results': 1})
            self.assertEqual(rows[0]['parallel_n'], 24)
            self.assertEqual(rows[0]['solved'], 8)
            self.assertEqual(rows[0]['oracle_n'], 3)
            self.assertEqual(rows[0]['epsilon'], [0]+[1]*7)
            fresh[1]['runs'].pop()
            fresh_path.write_text(''.join(json.dumps(r)+'\n' for r in fresh))
            with self.assertRaisesRegex(ValueError, 'Incomplete Parallel'):
                collect_interventions(root, 'gpt55', 5, {}, {'results': 1})

    def test_errors_are_aggregate_not_mean_problem_errors(self):
        from scripts.report_joint_allocation import METHODS
        rows = [dict(weight=3, observed=[0, 0], predictions={m: [.25, .25] for m in METHODS}),
                dict(weight=3, observed=[1, 1], predictions={m: [.75, .75] for m in METHODS})]
        report = summarize(rows)
        self.assertEqual(report['trials'], 6)
        for metric in report['metrics'].values():
            self.assertEqual(metric['rmse'], 0)
            self.assertEqual(metric['predicted'], [3, 3])

    def test_n4_fixed_groups_and_missing_audits(self):
        short = {('p', s): dict(problem_id='p', seed=s, audit_score=5 if s == 8 else 0,
                               budget_cuts={'1x': {'audit_score': 5 if s == 1 else 0}})
                 for s in range(1, 13)}
        r = dict(row(m=8), dataset='results', problem='p')
        with patch('scripts.report_joint_allocation.indexed_audits', return_value=short):
            result = collect_targets(Path('/unused'), 'gpt55', 4, [r], [{4: {}}], 5, {})
            self.assertEqual(result[0]['weight'], 3)
            self.assertEqual(result[0]['observed'], [1/3, 2/3])
            del short['p', 12]
            with self.assertRaisesRegex(ValueError, 'Missing N=4 target'):
                collect_targets(Path('/unused'), 'gpt55', 4, [r], [{4: {}}], 5, {})

    def test_n2_pairs_same_seed_and_keeps_cumulative_success(self):
        first = {( 'p', s): dict(problem_id='p', seed=s, audit_score=0,
                  budget_cuts={f'{k}x': {'audit_score': 5 if s == 1 and k == 1 else 0} for k in range(1, 8)}) for s in (1, 2, 3)}
        second = {('p', s): dict(problem_id='p', seed=s, audit_score=0,
                   budget_cuts={f'{k}x': {'audit_score': 5 if s == 2 and k == 2 else 0} for k in range(1, 4)}) for s in (1, 2, 3)}
        def read(path, *args, **kwargs):
            return second if path.parent.name == 'late-baseline-sequential' else first
        r = dict(row(m=8), dataset='results', problem='p')
        with patch('scripts.report_joint_allocation.indexed_audits', side_effect=read):
            result = collect_targets(Path('/unused'), 'gpt54', 2, [r], [{2: {}}], 5, {})
            self.assertEqual(result[0]['weight'], 3)
            self.assertEqual(result[0]['observed'], [1/3, 2/3, 2/3, 2/3])
            del second['p', 3]
            with self.assertRaisesRegex(ValueError, 'Missing N=2 target'):
                collect_targets(Path('/unused'), 'gpt54', 2, [r], [{2: {}}], 5, {})


if __name__ == '__main__':
    unittest.main()
