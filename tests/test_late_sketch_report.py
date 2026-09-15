import unittest

from scripts.report_late_sketch import passed, selected_at_3x, render


class LateSketchReportTests(unittest.TestCase):
    def test_selection_uses_3x_not_cumulative_or_final_score(self):
        record = dict(audit_score=7, budget_cuts={
            '1x': {'audit_score': 7}, '2x': {'audit_score': 7},
            '3x': {'audit_score': 4}})
        self.assertTrue(selected_at_3x(record))
        record['budget_cuts']['3x']['audit_score'] = 5
        self.assertFalse(selected_at_3x(record))
        del record['budget_cuts']['3x']
        with self.assertRaises(KeyError):
            selected_at_3x(record)

    def test_paper_threshold_and_missing_scores(self):
        self.assertFalse(passed(4))
        self.assertTrue(passed(5))
        for score in (None, True, 8):
            with self.assertRaises(ValueError):
                passed(score)

    def test_counts_without_percentages(self):
        tex = render({'cohorts': [dict(dataset='AOBench', model='Test', count=16,
                                      early=15, control=7, late=15)]})
        self.assertIn(r'Test & 16 & 7 & 15 & 15', tex)
        self.assertNotIn(r'\%', tex)
        self.assertIn('not cumulative failure', tex)
        self.assertIn('No sketch & Start sketch & Late sketch', tex)
        self.assertIn(r'Test & 16 & 7 & 15 & 15 & $0$', tex)
        self.assertNotIn('multicolumn', tex)

    def test_delta_is_late_minus_start(self):
        tex = render({'cohorts': [dict(dataset='AOBench', model='Muse', count=71,
                                      early=34, control=1, late=17)]})
        self.assertIn(r'Muse & 71 & 1 & 34 & 17 & $-17$', tex)


if __name__ == '__main__':
    unittest.main()
