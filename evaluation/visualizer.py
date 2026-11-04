"""Build reproducible charts and Markdown/HTML reports from benchmark JSON.

Run from repository root: python -m evaluation.visualizer --help
"""

import argparse
import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from evaluation.metrics import (
    evaluate_detection, evaluate_ocr, pair_ocr_records, robustness_degradation,
)


def load_runs(path):
    document = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if document.get('schema_version') != 1 or not document.get('runs'):
        raise ValueError('Expected schema_version=1 and a nonempty runs list')
    ids = set()
    for run in document['runs']:
        for field in ('run_id', 'model', 'source', 'resolution', 'profile',
                      'corruption', 'severity', 'ground_truth', 'predictions'):
            if field not in run:
                raise ValueError(f'Missing field {field}')
        if run['run_id'] in ids:
            raise ValueError('Duplicate run_id')
        ids.add(run['run_id'])
        if run['severity'] not in (0, 1, 2, 3):
            raise ValueError('severity must be 0, 1, 2 or 3')
        if (run['severity'] == 0) != (run['corruption'] == 'clean'):
            raise ValueError('clean must have severity=0; corruptions must have severity 1-3')
        plate_ids = [gt['plate_id'] for gt in run['ground_truth']]
        if len(set(plate_ids)) != len(plate_ids):
            raise ValueError('GT plate_id must be unique within a run')
        for value in run.get('latency_ms', []):
            if not math.isfinite(value) or value < 0:
                raise ValueError('latency_ms must contain finite nonnegative measurements')
        ram = run.get('peak_ram_mb')
        if ram is not None and (not math.isfinite(ram) or ram < 0):
            raise ValueError('peak_ram_mb must be finite and nonnegative')
        # Validate detection and OCR before any output is written.
        summarize(run)
    return document['runs']


def cohort(run):
    """Stable plate identities and labels; transformed boxes may differ."""
    return sorted((str(g['plate_id']), g['text'], g['layout'], g.get('class_id', 1))
                  for g in run['ground_truth'])


def context(run, include_model=True):
    fields = ('source', 'resolution', 'profile')
    key = tuple(str(run[f]) for f in fields)
    key += (run.get('confidence_threshold', 0.25),)
    return ((run['model'],) + key) if include_model else key


def summarize(run):
    threshold = run.get('confidence_threshold', 0.25)
    detection = evaluate_detection(run['ground_truth'], run['predictions'], threshold)
    ocr = evaluate_ocr(pair_ocr_records(run['ground_truth'], run['predictions'],
                                      confidence_threshold=threshold))
    return dict(run_id=run['run_id'], model=run['model'], source=run['source'],
                resolution=run['resolution'], profile=run['profile'],
                corruption=run['corruption'], severity=run['severity'],
                confidence_threshold=threshold,
                map50=detection['map50'], precision=detection['precision'],
                recall=detection['recall'], tp=detection['tp'],
                fp=detection['fp'], fn=detection['fn'],
                em_percent=ocr['overall']['em_percent'], cer=ocr['overall']['cer'],
                count=ocr['overall']['count'], ocr_slices=ocr,
                latency_ms=mean(run['latency_ms']) if run.get('latency_ms') else None,
                peak_ram_mb=run.get('peak_ram_mb'), degradation_percent=None)


def prepare(runs):
    rows = [summarize(run) for run in runs]
    warnings = []
    baselines = defaultdict(list)
    seen = set()
    for run, row in zip(runs, rows):
        cell = context(run) + (run['corruption'], run['severity'])
        if cell in seen:
            raise ValueError('Repeated experiment cell; aggregate repetitions in runner first')
        seen.add(cell)
        if run['severity'] == 0:
            baselines[context(run)].append((run, row))
    for run, row in zip(runs, rows):
        if run['severity'] == 0:
            continue
        baseline = baselines.get(context(run), [])
        if len(baseline) != 1 or cohort(baseline[0][0]) != cohort(run):
            warnings.append(f"{run['run_id']}: no clean baseline with identical plate IDs/labels and configuration")
            continue
        clean_em = baseline[0][1]['em_percent']
        if clean_em is not None and row['em_percent'] is not None:
            row['degradation_percent'] = robustness_degradation(clean_em, row['em_percent'])
        if clean_em == 0:
            warnings.append(f"{run['run_id']}: degradation undefined because clean EM is zero")
    return rows, warnings


def charts(runs, rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    paths = []

    def save(fig, name):
        caption = ('SYNTHETIC DEMO - not measured model performance' if
                   all(r['source'] == 'SYNTHETIC_DEMO' for r in runs) else
                   'Source: benchmark JSON; configuration and run IDs in report summary')
        fig.text(.5, -.025, caption, ha='center', fontsize=9)
        fig.savefig(output / (name + '.png'), dpi=160, bbox_inches='tight')
        fig.savefig(output / (name + '.svg'), bbox_inches='tight')
        plt.close(fig)
        paths.append(name + '.png')

    groups = defaultdict(list)
    for run, row in zip(runs, rows):
        if run['severity'] and row['degradation_percent'] is not None:
            groups[(context(run), run['corruption'])].append(row)
    # Separate contexts prevent mixing sources, resolutions or hardware.
    panels = defaultdict(list)
    for (key, corruption), values in groups.items():
        panels[key[1:]].append((key[0], corruption, values))
    for index, (key, series) in enumerate(panels.items()):
        fig, ax = plt.subplots(figsize=(9, 5))
        for model, corruption, values in series:
            values = sorted(values, key=lambda r: r['severity'])
            ax.plot([r['severity'] for r in values],
                    [r['degradation_percent'] for r in values], marker='o',
                    label=f'{model} / {corruption}')
        ax.set(xlabel='Corruption severity (level)', ylabel='Relative EM degradation (%)',
               title=f'Robustness degradation | {key}', xticks=[1, 2, 3])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        save(fig, f'degradation_{index}')

    comparison = defaultdict(list)
    for run, row in zip(runs, rows):
        # cohort is part of the key: FP32/INT8 must be evaluated on the same plates.
        key = context(run, False) + (run['severity'], json.dumps(cohort(run)))
        if run['severity'] and row['em_percent'] is not None:
            comparison[key].append((run, row))
    radar_groups = defaultdict(dict)
    for key, values in comparison.items():
        for run, row in values:
            radar_groups[key][run['model'], run['corruption']] = row['em_percent']
    for index, (key, values) in enumerate(radar_groups.items()):
        models = sorted({model for model, _ in values})
        if not {'FP32', 'INT8'} <= set(models):
            continue
        axes = sorted({c for m, c in values if m == 'FP32'} &
                      {c for m, c in values if m == 'INT8'})
        if len(axes) < 3:
            continue
        angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})
        for model in ('FP32', 'INT8'):
            scores = [values[model, c] for c in axes]
            ax.plot(angles + angles[:1], scores + scores[:1], label=model)
        ax.set_xticks(angles, axes)
        ax.set_ylim(0, 100)
        ax.set_title(f'End-to-end EM (%) by corruption\n{key[:5]}', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        save(fig, f'radar_{index}')

    points = [(run, row) for run, row in zip(runs, rows)
              if row['latency_ms'] is not None and row['em_percent'] is not None]
    # A scatter shows the tradeoff; no fitted correlation across incompatible contexts.
    if points:
        fig, ax = plt.subplots(figsize=(9, 5))
        for model in sorted({run['model'] for run, _ in points}):
            selected = [(run, row) for run, row in points if run['model'] == model]
            ax.scatter([row['latency_ms'] for _, row in selected],
                       [row['em_percent'] for _, row in selected], label=model)
            coincident = defaultdict(list)
            for run, row in selected:
                coincident[row['latency_ms'], row['em_percent']].append(run['run_id'])
            for point, identifiers in coincident.items():
                label = identifiers[0] if len(identifiers) == 1 else f'{len(identifiers)} runs'
                ax.annotate(label, point, fontsize=7, xytext=(0, 7),
                            ha='center', textcoords='offset points')
        ax.set(xlabel='Mean measured end-to-end latency (ms/image)',
               ylabel='End-to-end exact match (%)', title='Accuracy vs latency', ylim=(-3, 103))
        ax.legend()
        fig.tight_layout()
        save(fig, 'accuracy_latency')
    return paths


def build_report(input_path, output, image_root='.'):
    from evaluation.error_analysis import export_errors

    runs = load_runs(input_path)
    rows, warnings = prepare(runs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figures = charts(runs, rows, output)
    cases = export_errors(runs, output / 'failures', image_root)
    warnings.extend(f"{case['run_id']} / {case['image_id']}: {case['image_warning']}"
                    for case in cases if 'image_warning' in case)
    summary = {'input_sha256': hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
               'schema_version': 1, 'runs': rows, 'warnings': warnings}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                   allow_nan=False), encoding='utf-8')
    columns = [key for key in rows[0] if key != 'ocr_slices']
    with (output / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    note = ('End-to-end OCR includes missed detections. EM: percent; CER: ratio; '
            'AP50: all-points precision envelope. Latency must be measured by runner. '
            'Missing values are N/A. Source: ' + str(input_path))
    header = '| ' + ' | '.join(columns) + ' |'
    table = [header, '| ' + ' | '.join('---' for _ in columns) + ' |']
    for row in rows:
        table.append('| ' + ' | '.join(str(row[k]).replace('|', '\\|').replace('\n', ' ')
                                      if row[k] is not None else 'N/A' for k in columns) + ' |')
    markdown = '# ALPR benchmark report\n\n' + note + '\n\n' + '\n'.join(table)
    for figure in figures:
        markdown += f'\n\n![{figure}]({figure})'
    markdown += '\n\n[Failure gallery](failures/index.html)\n\n## Validation notes\n\n'
    markdown += '\n'.join('- ' + warning for warning in warnings) or 'No validation warnings.'
    (output / 'report.md').write_text(markdown, encoding='utf-8')
    cells = lambda row: ''.join('<td>' + html.escape(str(row[k]) if row[k] is not None else 'N/A')
                                + '</td>' for k in columns)
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>ALPR report</title>'
            '<style>body{font:15px system-ui;margin:36px;color:#17242d}table{border-collapse:collapse}'
            'td,th{padding:8px;border:1px solid #ddd}img{max-width:100%}'
            '.scroll{overflow:auto}</style><h1>ALPR benchmark report</h1><p>' + html.escape(note)
            + '</p><div class="scroll"><table><tr>'
            + ''.join('<th>' + html.escape(k) + '</th>' for k in columns) + '</tr>'
            + ''.join('<tr>' + cells(row) + '</tr>' for row in rows) + '</table></div>'
            + ''.join(f'<figure><img src="{f}" alt="{f}"></figure>' for f in figures)
            + '<p><a href="failures/index.html">Failure gallery</a></p><h2>Validation notes</h2><ul>'
            + ''.join('<li>' + html.escape(w) + '</li>' for w in warnings) + '</ul></html>')
    (output / 'report.html').write_text(page, encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='output/report')
    parser.add_argument('--image-root', default='.')
    args = parser.parse_args()
    build_report(args.input, args.output, args.image_root)
    print(f'Report: {Path(args.output).resolve() / "report.html"}')


if __name__ == '__main__':
    main()
