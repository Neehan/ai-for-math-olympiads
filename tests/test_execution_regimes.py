import copy
import unittest

import numpy as np

from scripts.report_execution_regimes import METHODS, PANELS, build, render, summarize_subset


def fixture():
    reports = {}
    for n, models in PANELS.items():
        for model in models:
            rows = []
            for i in range(57):
                prediction = (1-0.75**(n*np.arange(1, 8//n+1))).tolist()
                rows.append(dict(dataset='results' if i < 35 else 'results-imobench',
                                 problem=f'p{i}', weight=3, oracle_n=3,
                                 epsilon=[1]*8 if i % 2 == 0 else [1/3]+[1]*7,
                                 solved=6, parallel_n=24, observed=[0]*(8//n),
                                 predictions={m: prediction[:] for m in METHODS}))
            reports[f'{model}-n{n}'] = dict(rows=rows)
    return dict(reports=reports)


class ExecutionRegimeTests(unittest.TestCase):
    def test_groups_partition_and_selection_ignores_targets(self):
        data = fixture()
        before = build(data)
        changed = copy.deepcopy(data)
        for report in changed['reports'].values():
            for row in report['rows']:
                row['observed'] = [1]*len(row['observed'])
        after = build(changed)
        for key, groups in before['panels'].items():
            self.assertEqual(sum(g['trials'] for g in groups.values()), 171)
            self.assertEqual(sum(g['problems'] for g in groups.values()), 57)
            self.assertTrue(set(m['problem'] for m in groups['complete']['members']).isdisjoint(
                m['problem'] for m in groups['incomplete']['members']))
            for name in groups:
                self.assertEqual(groups[name]['members'], after['panels'][key][name]['members'])

    def test_aggregate_error_not_individual_error(self):
        rows = [dict(dataset='results', problem=str(i), weight=3, observed=[i, i],
                     predictions={m: [0.5, 0.5] for m in METHODS}) for i in (0, 1)]
        summary = summarize_subset(rows)
        self.assertEqual(summary['observed'], [3, 3])
        self.assertEqual(summary['metrics']['solved_geometric']['rmse'], 0)

    def test_complete_de_is_geometric(self):
        data = fixture()
        data['reports']['gpt54-n1']['rows'][0]['predictions']['neither_regularized'][0] = 0
        with self.assertRaisesRegex(ValueError, 'does not reduce'):
            build(data)

    def test_table_contains_both_groups_and_allocations(self):
        tex = render(build(fixture()))
        self.assertEqual(tex.count(' & Complete & '), 6)
        self.assertEqual(tex.count(' & Incomplete & '), 6)
        self.assertIn('$N=1$', tex)
        self.assertIn('$N=2$', tex)
        self.assertNotIn('percentage', tex)
        rows = [line.split(' & ')[:2] for line in tex.splitlines()
                if ' & Complete & ' in line or ' & Incomplete & ' in line]
        self.assertEqual(rows, [
            ['GPT-5.4', 'Complete'], ['GPT-5.5', 'Complete'], ['Opus', 'Complete'],
            ['GPT-5.4', 'Incomplete'], ['GPT-5.5', 'Incomplete'], ['Opus', 'Incomplete'],
            ['GPT-5.4', 'Complete'], ['GPT-5.5', 'Complete'], ['Opus', 'Complete'],
            ['GPT-5.4', 'Incomplete'], ['GPT-5.5', 'Incomplete'], ['Opus', 'Incomplete'],
        ])


if __name__ == '__main__':
    unittest.main()
