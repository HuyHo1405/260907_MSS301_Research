"""Failure cases using exactly the same matching rules as metrics.py."""

import argparse
import html
import json
from pathlib import Path

from evaluation.metrics import (
    _detections, _match, normalize_plate, evaluate_detection,
)


def analyze_errors(run):
    """Return misses, unmatched detections and OCR mismatches per run."""
    gt, predictions = run['ground_truth'], run['predictions']
    threshold = run.get('confidence_threshold', 0.25)
    evaluate_detection(gt, predictions, threshold)  # validate even filtered boxes
    targets = _detections(gt)
    selected = [p for p in _detections(predictions, predictions=True)
                if p['score'] >= threshold]
    matched = set()
    failures = []
    for prediction, index in _match(selected, targets, 0.5):
        if index is None:
            failures.append({'kind': 'false_positive', 'image_id': prediction['image_id'],
                             'ground_truth': None, 'prediction': prediction})
        else:
            matched.add(index)
            target = targets[index]
            if normalize_plate(target['text']) != normalize_plate(prediction['text']):
                failures.append({'kind': 'ocr_mismatch', 'image_id': target['image_id'],
                                 'ground_truth': target, 'prediction': prediction})
    for index, target in enumerate(targets):
        if index not in matched:
            failures.append({'kind': 'miss_detection', 'image_id': target['image_id'],
                             'ground_truth': target, 'prediction': None})
    return failures


def export_errors(runs, output, image_root):
    """Export all cases as JSON and an HTML gallery with annotated images.

    Paths must resolve inside image_root. Missing images remain visible as warnings.
    Categories describe symptoms; they do not establish environmental causation.
    """
    from PIL import Image, ImageDraw

    output, image_root = Path(output), Path(image_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases, cards = [], []
    for run in runs:
        for failure in analyze_errors(run):
            case = dict(failure, run_id=run['run_id'])
            image_id = str(case['image_id'])
            source = (image_root / run.get('images', {}).get(image_id, image_id)).resolve()
            index = len(cases)
            try:
                if not source.is_relative_to(image_root):
                    raise ValueError('Image path is outside image_root')
                with Image.open(source) as original:
                    picture = original.convert('RGB')
                draw = ImageDraw.Draw(picture)
                for key, color in [('ground_truth', '#16a34a'), ('prediction', '#dc2626')]:
                    record = case[key]
                    if record:
                        draw.rectangle(tuple(record['bbox']), outline=color, width=3)
                name = f'case_{index:05d}.jpg'
                picture.save(output / name, quality=90)
                case['preview'] = name
                preview = f'<img src="{name}" alt="Annotated failure">'
            except (OSError, ValueError) as exc:
                case['image_warning'] = str(exc)
                preview = '<p>Image unavailable: ' + html.escape(str(exc)) + '</p>'
            cases.append(case)
            text = json.dumps(case, ensure_ascii=False, indent=2)
            cards.append(f'<article>{preview}<pre>{html.escape(text)}</pre></article>')
    (output / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Failure cases</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto}'
        'article{border-top:1px solid #bbb;padding:24px 0}img{max-width:100%;max-height:500px}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
        '<h1>Failure cases</h1><p>Green: GT; red: prediction. '
        'Symptoms only; inspect images and metadata before claiming a cause.</p>'
        + (''.join(cards) if cards else '<p>No failures at the selected threshold.</p>')
        + '</html>', encoding='utf-8')
    return cases


def main():
    from evaluation.visualizer import load_runs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='output/failures')
    parser.add_argument('--image-root', default='.')
    args = parser.parse_args()
    export_errors(load_runs(args.input), args.output, args.image_root)


if __name__ == '__main__':
    main()
