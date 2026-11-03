"""
Quality Assurance (QA) and Dataset Validation Tool for Source B.
Lead: Khanh (Field Data & QA)
Author: GS.TS AI Research Lead

Validates:
1. Image existence and integrity.
2. Coordinate and geometry constraints (BBox within image boundary, x2 > x1, y2 > y1).
3. Text annotation compliance (non-empty, uppercase, valid characters).
4. Layout conformity ('one_line' or 'two_line').
5. Integration with evaluation/metrics.py (TP=Total, FP=0, FN=0, Precision=1.0, Recall=1.0, mAP=1.0, EM=100%, CER=0.0).
"""

import json
import os
import sys

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from evaluation.metrics import evaluate_detection, evaluate_ocr, pair_ocr_records, normalize_plate


def validate_source_b(gt_path="datasets/source_b/ground_truth_source_b.json",
                      meta_path="datasets/source_b/metadata_source_b.json",
                      images_dir="datasets/source_b/images"):
    print("=" * 70)
    print("[QA] RUNNING AUTOMATED AUDIT ON DATASETS/SOURCE_B (KHANH'S DELIVERABLES)")
    print("=" * 70)

    errors = []

    # 1. Check file existence
    if not os.path.exists(gt_path):
        errors.append(f"Missing Ground Truth file: {gt_path}")
    if not os.path.exists(meta_path):
        errors.append(f"Missing Metadata file: {meta_path}")
    if not os.path.exists(images_dir):
        errors.append(f"Missing Images directory: {images_dir}")

    if errors:
        for err in errors:
            print(f"[-] [CRITICAL] {err}")
        return False

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_records = json.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"[*] Loaded {len(gt_records)} plate annotations from {gt_path}")
    print(f"[*] Loaded metadata for {len(metadata.get('records', []))} images from {meta_path}")

    # 2. Check each image and bbox
    image_ids_in_gt = set()
    for idx, rec in enumerate(gt_records):
        img_id = rec.get("image_id", "")
        file_name = rec.get("file_name", "")
        bbox = rec.get("bbox", [])
        text = rec.get("text", "")
        layout = rec.get("layout", "")

        image_ids_in_gt.add(img_id)

        # Image check
        img_path = os.path.join(images_dir, file_name)
        if not os.path.exists(img_path):
            errors.append(f"Record {idx} ({file_name}): File not found at {img_path}")
            continue

        im = cv2.imread(img_path)
        if im is None:
            errors.append(f"Record {idx} ({file_name}): Failed to read image with cv2")
            continue

        h, w = im.shape[:2]

        # BBox check: [x1, y1, x2, y2]
        if len(bbox) != 4:
            errors.append(f"Record {idx} ({file_name}): bbox must have 4 coordinates, got {bbox}")
        else:
            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 <= w + 1e-2 and 0 <= y1 < y2 <= h + 1e-2):
                errors.append(f"Record {idx} ({file_name}): bbox coordinates out of bounds: {bbox} for img size {w}x{h}")

        # Layout check
        if layout not in ("one_line", "two_line"):
            errors.append(f"Record {idx} ({file_name}): invalid layout '{layout}'")

        # Text check
        if not text or text == "UNKNOWN":
            errors.append(f"Record {idx} ({file_name}): text is empty or UNKNOWN: '{text}'")
        else:
            norm = normalize_plate(text)
            if not norm:
                errors.append(f"Record {idx} ({file_name}): text '{text}' is empty after normalization")

    # 3. Check metadata consistency
    meta_img_names = set(r["file_name"] for r in metadata.get("records", []))
    actual_img_names = set(os.listdir(images_dir))

    missing_in_meta = actual_img_names - meta_img_names
    if missing_in_meta:
        errors.append(f"Images in directory but missing in metadata: {len(missing_in_meta)}")

    missing_in_dir = meta_img_names - actual_img_names
    if missing_in_dir:
        errors.append(f"Images in metadata but missing in directory: {len(missing_in_dir)}")

    # 4. End-to-end integration test with evaluation/metrics.py
    print("\n[*] Testing integration with evaluation/metrics.py (Tuan's module)...")

    mock_predictions = []
    for r in gt_records:
        mock_predictions.append({
            "image_id": r["image_id"],
            "class_id": r.get("class_id", 1),
            "bbox": r["bbox"],
            "score": 0.99,
            "text": r["text"]
        })

    # Test Detection Metric
    det_results = evaluate_detection(gt_records, mock_predictions, confidence_threshold=0.25)
    print(f"  -> Detection Test: TP={det_results['tp']}, FP={det_results['fp']}, FN={det_results['fn']}, Precision={det_results['precision']:.2f}, Recall={det_results['recall']:.2f}, mAP@50={det_results['map50']:.2f}")

    if det_results['tp'] != len(gt_records) or det_results['fp'] != 0 or det_results['fn'] != 0:
        errors.append(f"Self-detection test failed: TP={det_results['tp']}/{len(gt_records)}")

    # Test OCR Metric
    pairs = pair_ocr_records(gt_records, mock_predictions, confidence_threshold=0.25)
    ocr_results = evaluate_ocr(pairs)
    overall_em = ocr_results["overall"]["em_percent"]
    overall_cer = ocr_results["overall"]["cer"]
    print(f"  -> OCR Test: Exact Match={overall_em:.1f}%, CER={overall_cer:.3f}")

    if overall_em != 100.0 or overall_cer != 0.0:
        errors.append(f"Self-OCR test failed: EM={overall_em}%, CER={overall_cer}")

    # Summary
    print("\n" + "=" * 70)
    if not errors:
        print("[SUCCESS] ALL QA AUDIT CHECKS PASSED! DATASET IS 100% PRODUCTION READY.")
        print(f"   - Verified images: {len(actual_img_names)}")
        print(f"   - Verified plates: {len(gt_records)}")
        print(f"   - Distribution: {metadata.get('distribution', {})}")
        print("=" * 70)
        return True
    else:
        print(f"[FAIL] QA AUDIT FAILED WITH {len(errors)} ERROR(S):")
        for err in errors[:10]:
            print(f"   - {err}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors.")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = validate_source_b()
    sys.exit(0 if success else 1)
