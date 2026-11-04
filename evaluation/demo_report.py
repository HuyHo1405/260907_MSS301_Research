"""Synthetic integration example, NOT experimental model results."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from evaluation.visualizer import build_report


def example_runs():
    runs = []
    targets = [dict(image_id='samples/test1.jpg', plate_id='demo-1',
                    bbox=[30, 30, 200, 100], text='30A12345', layout='one_line'),
               dict(image_id='samples/test2.jpg', plate_id='demo-2',
                    bbox=[30, 30, 160, 140], text='59B112345', layout='two_line')]
    for model in ('FP32', 'INT8'):
        for corruption, severity in [('clean', 0)] + [
                (c, s) for c in ('low_light', 'motion_blur', 'rain') for s in (1, 2, 3)]:
            predictions = [dict(image_id=g['image_id'], bbox=g['bbox'],
                                score=.95, text=g['text']) for g in targets]
            if severity >= 1:
                predictions[1]['text'] = '59B112346'
            if severity == 3:
                predictions = predictions[:1]
            if severity >= 2 and model == 'INT8':
                predictions[0]['text'] = '30A12346'
            runs.append(dict(run_id=f'{model}-{corruption}-{severity}', model=model,
                             source='SYNTHETIC_DEMO', resolution=480,
                             profile='illustrative_only_not_measured', corruption=corruption,
                             severity=severity, ground_truth=deepcopy(targets),
                             predictions=deepcopy(predictions),
                             latency_ms=[100 if model == 'INT8' else 150]))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='output/demo_report')
    parser.add_argument('--image-root', default='.')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / 'synthetic_input.json'
    fixture.write_text(json.dumps({'schema_version': 1, 'runs': example_runs()}, indent=2),
                       encoding='utf-8')
    build_report(fixture, output, args.image_root)
    print(f'SYNTHETIC demonstration only: {output.resolve() / "report.html"}')


if __name__ == '__main__':
    main()
