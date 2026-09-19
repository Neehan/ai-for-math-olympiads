import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from scripts.render_allocation_report import display_series, explanatory_plot, all_allocation_table, n4_allocation_table, render
from scripts.report_joint_allocation import METHODS, summarize


def fixture():
    reports = {}
    for n, models in [(1, ('muse', 'gpt54', 'gpt55', 'opus')),
                      (2, ('muse', 'gpt54', 'gpt55', 'opus')),
                      (4, ('muse', 'gpt55'))]:
        rows = [dict(weight=3, oracle_n=3, epsilon=[.5]*8,
                     observed=(np.arange(1, 8//n+1)/8).tolist(),
                     predictions={m: (np.arange(1, 8//n+1)/9).tolist() for m in METHODS})
                for _ in range(57)]
        for model in models:
            reports[f'{model}-n{n}'] = summarize(rows)
    return dict(reports=reports)


class ExplanatoryFigureTests(unittest.TestCase):
    def test_render_writes_only_current_paper_assets(self):
        data = fixture()
        for report in data['reports'].values():
            for row in report['rows']:
                row['dataset'] = 'results'
        with TemporaryDirectory() as directory, \
                patch('scripts.report_execution_regimes.build', return_value={}), \
                patch('scripts.report_execution_regimes.render', return_value='regimes'), \
                patch('scripts.report_execution_regimes.render_early_gains', return_value='gains'):
            render(data, directory)
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {
                'allocation_report.json', 'execution_scaling_bars.tex',
                'allocation_n2_bars.tex', 'allocation_comparison_table.tex',
                'execution_regimes_table.tex', 'execution_regimes_full_table.tex', 'early_execution_gains_table.tex',
                'allocation_model_fit.tex', 'allocation_n4_table.tex',
            })

    def test_display_checkpoints_are_total_budget(self):
        data = fixture()
        for n in (1, 2, 4):
            x, series = display_series(data['reports'][f'muse-n{n}']['rows'], n)
            np.testing.assert_array_equal(x, np.arange(n, 9, n))
            np.testing.assert_allclose(series['observed'], 171*x/(8*n))

    def test_errors_are_observed_minus_prediction_at_all_checkpoints(self):
        data = fixture()
        for n in (1, 2, 4):
            tex = explanatory_plot(data, n=n)
            panels = 2 if n == 4 else 4
            self.assertEqual(tex.count(r'\nextgroupplot['), panels)
            self.assertNotIn('errorzero', tex)
            self.assertEqual(tex.count('black!65,solid,line width=1.2pt,no marks'), panels)
            self.assertNotIn('ybar', tex)
            for k in range(1, 8//n+1):
                expected = 171*(k/8-k/9)
                self.assertEqual(tex.count(f'({n*k},{expected:.6f})'), 4*panels)

    def test_overprediction_is_negative_and_scale_is_shared(self):
        data = fixture()
        for row in data['reports']['opus-n1']['rows']:
            row['predictions']['solved_geometric'] = [1.0]*8
        tex = explanatory_plot(data)
        self.assertIn('(1,-149.625000)', tex)
        self.assertEqual(tex.count('ymin='), 1)
        self.assertEqual(tex.count('ymax='), 1)
        self.assertIn('ymin=-155', tex)

    def test_error_colors_and_styles(self):
        tex = explanatory_plot(fixture())
        self.assertIn(r'\definecolor{errorrde}{HTML}{D94B40}', tex)
        self.assertIn(r'\definecolor{errorogt}{HTML}{249447}', tex)
        for color in ('D94B40', '2077B4', '8246C5', '249447'):
            self.assertIn('{HTML}{'+color+'}', tex)
        self.assertEqual(tex.count(r'\nextgroupplot['), 4)
        for marker, size in [('square*', '1.3'), ('*', '1.5'), ('triangle*', '2.1'), ('diamond*', '1.9')]:
            self.assertEqual(tex.count(f'solid,line width=1pt,mark={marker},mark size={size}pt'), 4)
            self.assertEqual(tex.count(f'mark={marker},mark size={size}pt'), 5)
        self.assertIn('{Solved $-$ Predicted}', tex)
        self.assertNotIn(r'\small', tex)
        self.assertNotIn(r'\sffamily', tex)
        self.assertIn(r'font=\normalfont\normalsize,text=black', tex)
        self.assertIn('{OGT}', tex)

    def test_combined_table_uses_full_curve_metrics(self):
        data = fixture()
        data['reports']['opus-n1']['metrics']['both_regularized']['rmse'] = 1.2345
        tex = all_allocation_table(data)
        self.assertIn(r'\textbf{1.23}', tex)
        for n in (1, 2):
            self.assertIn(f'$N={n}$', tex)
        self.assertNotIn('$N=4$', tex)
        self.assertNotIn('---', tex)
        self.assertIn('Prediction RMSE (solved-trial counts)', tex)

    def test_n4_table_has_only_two_model_rows_and_preserves_rmse(self):
        data = fixture()
        data['reports']['gpt55-n4']['metrics']['both_regularized']['rmse'] = 0.123
        tex = n4_allocation_table(data)
        self.assertIn(r'\textbf{0.12}', tex)
        self.assertIn('Muse Spark~1.2 & ', tex)
        self.assertIn('GPT-5.5 & ', tex)
        self.assertNotIn('GPT-5.4', tex)
        self.assertNotIn('Opus', tex)
        self.assertNotIn('---', tex)
        for model in ('muse', 'gpt55'):
            for method in METHODS:
                self.assertIn(f"{data['reports'][f'{model}-n4']['metrics'][method]['rmse']:.2f}", tex)
        self.assertIn(r'\label{tab:n4-predictor-comparison}', tex)


if __name__ == '__main__':
    unittest.main()
