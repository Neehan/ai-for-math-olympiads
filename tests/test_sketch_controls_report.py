import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.report_external_decay import evaluate
from scripts.report_joint_allocation import fit_interventions
from scripts.report_sketch_controls import load_arm, table


class SketchControlTests(unittest.TestCase):
    def test_main_fit_defaults_to_all_problems(self):
        self.assertIsNone(inspect.signature(fit_interventions).parameters['prior_dataset'].default)

    def test_standalone_hash_and_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for seed in (1, 2, 3):
                d = base / 'hint' / 'p' / f'seed_{seed}'
                d.mkdir(parents=True)
                (d / 'solution.md').write_text('proof')
                (d / 'meta.json').write_text(json.dumps(dict(mode='single', budget_output_tokens=200000)))
                (d / 'audit.json').write_text(json.dumps(dict(problem_id='p', seed=seed, arm='hint',
                    audit_score=7, solution_sha256=hashlib.sha256(b'proof').hexdigest())))
            self.assertEqual(len(load_arm(base, 'hint', {'p'}, {})), 3)
            (base / 'hint/p/seed_2/solution.md').write_text('changed proof')
            with self.assertRaisesRegex(ValueError, 'Stale proof audit'):
                load_arm(base, 'hint', {'p'}, {})
            with self.assertRaises(FileNotFoundError):
                load_arm(base, 'hint', {'missing'}, {})

    def test_count_table_has_no_percentage_or_majority_vote(self):
        tex = table(['Model', 'Trajectories', 'Original', 'Alternate'],
                    [['Test', 72, 38, 38]], 'Solved trajectories.', 'tab:test')
        self.assertIn('Test & 72 & 38 & 38', tex)
        self.assertNotIn('%', tex)
        self.assertNotIn('38/72', tex)

    def test_aobench_decay_ignores_external_unaided_outcomes(self):
        rows = [dict(dataset=dataset, solved=1, parallel_n=3, oracle_n=3,
                     epsilon=[1/3]*8, weight=3, observed=[0, 1/3, 1/3, 1/3, 2/3, 2/3, 2/3, 2/3])
                for dataset in ('results', 'results-imobench')]
        theta = np.zeros(4)  # Beta(a,b), oracle categories 1 and >8.
        before = evaluate(rows, theta, 7, draws=256, dataset='results')
        rows[1]['observed'] = [1]*8
        after = evaluate(rows, theta, 7, draws=256, dataset='results')
        self.assertEqual(before, after)
        self.assertEqual(before['trajectories'], 3)
        combined = evaluate(rows, theta, 7, draws=256)
        self.assertEqual(combined['trajectories'], 6)
        self.assertIsNone(combined['dataset'])


if __name__ == '__main__':
    unittest.main()
