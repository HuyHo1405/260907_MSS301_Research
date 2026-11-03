"""
Benchmark ALPR Pipeline on Source B (real-world challenging dataset).
Runs detection + recognition, calculates mAP@50, Exact Match %, and CER.
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

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.alpr_pipeline import ALPRPipeline
from evaluation.metrics import evaluate_detection, evaluate_ocr, pair_ocr_records, predictions_from_pipeline


def main():
    weights_path = 'weights/best_int8.onnx'
    gt_path = 'datasets/source_b/ground_truth_source_b.json'
    img_dir = 'datasets/source_b/images'

    print(f"[*] Dang khoi tao ALPRPipeline voi weights: {weights_path}...")
    pipeline = ALPRPipeline(detector_weights=weights_path)

    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)

    print(f"[*] Dang chay benchmark tren {len(gt)} anh Source B...")
    preds = []
    for idx, item in enumerate(gt, 1):
        img_path = os.path.join(img_dir, item['file_name'])
        _, results, latency = pipeline.predict(img_path)
        preds.extend(predictions_from_pipeline(item['image_id'], results))
        if idx % 15 == 0 or idx == len(gt):
            print(f"  -> Da xu ly {idx}/{len(gt)} anh (FPS gan nhat: {latency['fps']:.1f})")

    det = evaluate_detection(gt, preds)
    pairs = pair_ocr_records(gt, preds)
    ocr = evaluate_ocr(pairs)

    print('\n' + '=' * 60)
    print('🏆 KET QUA BENCHMARK THUC TE TREN SOURCE B:')
    print('=' * 60)
    print(f"  • Detection mAP@50:            {det['map50'] * 100:.2f}% (Precision: {det['precision']*100:.2f}%, Recall: {det['recall']*100:.2f}%)")
    print(f"  • Exact Match (EM) Toan Bo:    {ocr['overall']['em_percent']:.2f}%")
    print(f"  • Character Error Rate (CER):  {ocr['overall']['cer']:.4f}")
    if ocr['one_line']['count'] > 0:
        print(f"    - Bien 1 dong (dai):         EM = {ocr['one_line']['em_percent']:.2f}%, CER = {ocr['one_line']['cer']:.4f} ({ocr['one_line']['count']} bien)")
    if ocr['two_line']['count'] > 0:
        print(f"    - Bien 2 dong (vuong):        EM = {ocr['two_line']['em_percent']:.2f}%, CER = {ocr['two_line']['cer']:.4f} ({ocr['two_line']['count']} bien)")
    print('=' * 60)


if __name__ == '__main__':
    main()