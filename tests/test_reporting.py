import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from evaluation.demo_report import example_runs
from evaluation.error_analysis import analyze_errors, export_errors
from evaluation.visualizer import load_runs, prepare, build_report


class ReportingTests(unittest.TestCase):
    def test_known_metrics_and_degradation(self):
        rows, warnings = prepare(example_runs())
        by_id = {r['run_id']: r for r in rows}
        self.assertEqual(warnings, [])
        row = by_id['INT8-motion_blur-3']
        self.assertEqual((row['tp'], row['fn']), (1, 1))
        self.assertEqual(row['em_percent'], 0)
        self.assertEqual(row['degradation_percent'], 100)
        self.assertAlmostEqual(row['cer'], 10 / 17)

    def test_failure_matching_duplicates_and_class(self):
        run = example_runs()[0]
        duplicate = deepcopy(run['predictions'][0])
        duplicate['score'] = .8
        run['predictions'].append(duplicate)
        run['predictions'][1]['class_id'] = 99
        kinds = [c['kind'] for c in analyze_errors(run)]
        self.assertEqual(kinds.count('false_positive'), 2)
        self.assertEqual(kinds.count('miss_detection'), 1)

    def test_baseline_requires_same_cohort(self):
        runs = example_runs()[:2]
        runs[1]['ground_truth'][0]['plate_id'] = 'different'
        rows, warnings = prepare(runs)
        self.assertIsNone(rows[1]['degradation_percent'])
        self.assertEqual(len(warnings), 1)

    def test_schema_rejects_nonfinite_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'
            runs = example_runs()[:1]
            runs[0]['latency_ms'] = [float('nan')]
            path.write_text(json.dumps(dict(schema_version=1, runs=runs)), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_runs(path)

    def test_image_path_cannot_escape_root_and_html_is_escaped(self):
        run = example_runs()[1]
        run['run_id'] = '<script>alert(1)</script>'
        run['images'] = {'samples/test2.jpg': '../../outside.jpg'}
        with tempfile.TemporaryDirectory() as directory:
            cases = export_errors([run], directory, directory)
            self.assertIn('outside image_root', cases[0]['image_warning'])
            page = (Path(directory) / 'index.html').read_text(encoding='utf-8')
            self.assertNotIn('<script>', page)

    def test_report_outputs_and_radar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'input.json'
            path.write_text(json.dumps(dict(schema_version=1, runs=example_runs())), encoding='utf-8')
            summary = build_report(path, root / 'report', root)
            self.assertEqual(len(summary['runs']), 20)
            for name in ('report.html', 'report.md', 'summary.csv', 'summary.json',
                         'accuracy_latency.svg', 'failures/index.html'):
                self.assertTrue((root / 'report' / name).is_file(), name)
            self.assertTrue(list((root / 'report').glob('radar_*.png')))


if __name__ == '__main__':
    unittest.main()
