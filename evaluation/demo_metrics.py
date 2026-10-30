"""Run from project root: python -m evaluation.demo_metrics (no model needed)."""

import json

from evaluation.metrics import (
    evaluate_detection, evaluate_ocr, iou_drop_rate, pair_ocr_records,
    predictions_from_pipeline, robustness_degradation,
)


def main():
    targets = [
        {"image_id": "a", "bbox": [0, 0, 100, 40],
         "text": "30A-123.45", "layout": "one_line"},
        {"image_id": "b", "bbox": [10, 10, 60, 60],
         "text": "59B1-123.45", "layout": "two_line"},
    ]
    predictions = predictions_from_pipeline("a", [
        {"box": [0, 0, 100, 40], "detector_conf": 0.95,
         "formatted_plate": "30A12345"},
    ])
    ocr = evaluate_ocr(pair_ocr_records(targets, predictions))
    report = {
        "description": "Synthetic example: one correct plate and one missed plate",
        "detection": evaluate_detection(targets, predictions),
        "ocr_end_to_end": ocr,
        "example_iou_drop": iou_drop_rate([0, 0, 10, 10], [5, 0, 15, 10]),
        "degradation_percent": robustness_degradation(100, ocr["overall"]["em_percent"]),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
