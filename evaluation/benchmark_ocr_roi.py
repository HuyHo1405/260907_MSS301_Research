"""
==============================================================================
🔬 BENCHMARK SPRINT: PREPROCESSING & OCR ROI OPTIMIZATION (SPRINT 9)
==============================================================================
Đo đạc định lượng 4 tầng tối ưu hóa OCR trên các thiết bị nhúng:
  1. Baseline (RapidOCR mặc định: limit=736 min, use_cls=True, use_dilation=True)
  2. Step 1: Loại bỏ Text Classifier (use_cls=False)
  3. Step 2: Cắt giảm DBNet Over-processing (limit=320 max, no-dilation)
  4. Step 3: Chuẩn hóa & Downscale ROI (target_height=96)
  5. End-to-End: YOLOv8 INT8 (480x480) + Optimized RapidOCR
==============================================================================
"""

import os
import sys
import time
import argparse
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.detector_onnx import ONNXPlateDetector
from pipeline.recognizer_rapid import RapidOCRRecognizer
from rapidocr_onnxruntime import RapidOCR
from pipeline.postprocess import format_vietnamese_plate


def run_sprint_benchmark(profile_name="Raspberry Pi 4", num_threads=4):
    print("\n" + "=" * 78)
    print(f"🚀 BẮT ĐẦU BENCHMARK OCR ROI & PREPROCESSING: [{profile_name.upper()}]")
    print(f"⚙️ Cấu hình: Threads = {num_threads}")
    print("=" * 78)

    test_images = ['samples/test1.jpg', 'samples/test2.jpg', 'samples/test3.jpg']
    detector = ONNXPlateDetector('weights/best_int8.onnx', imgsz=480, num_threads=num_threads)

    # Thu thập tất cả ROI biển số
    extracted_rois = []
    for p in test_images:
        img = cv2.imread(p)
        boxes, _, rois, _ = detector.detect(img)
        if rois:
            extracted_rois.append((p, rois[0]))

    print(f"📷 Đã phát hiện và trích xuất {len(extracted_rois)} ROI biển số để kiểm thử.")

    # 1. Khởi tạo 2 engine OCR: Gốc vs Tối ưu
    engine_baseline = RapidOCR()
    engine_optimized = RapidOCRRecognizer(target_height=96)

    # Các bước đo đạc đối chứng trên từng ROI
    results_detail = []

    for img_path, roi in extracted_rois:
        print(f"\n🔍 Đang đo đạc ROI từ: '{img_path}' (Kích thước gốc: {roi.shape[1]}x{roi.shape[0]}):")

        # A. Baseline: RapidOCR Default (limit 736 min, use_cls=True, dilation=True)
        engine_baseline(roi, use_cls=True) # warmup
        t0 = time.perf_counter()
        for _ in range(5):
            res_base, _ = engine_baseline(roi, use_cls=True)
        lat_base = (time.perf_counter() - t0) / 5 * 1000
        txt_base = format_vietnamese_plate([x[1] for x in res_base]) if res_base else "MISS"

        # B. Step 1: Bỏ cls (use_cls=False)
        t0 = time.perf_counter()
        for _ in range(5):
            res_s1, _ = engine_baseline(roi, use_cls=False)
        lat_s1 = (time.perf_counter() - t0) / 5 * 1000

        # C. Step 2 & 3: Optimized Engine (limit 320 max, no dilation, no cls, ROI downscale H=96)
        engine_optimized.recognize(roi) # warmup
        t0 = time.perf_counter()
        for _ in range(5):
            raw_opt, txt_opt, _ = engine_optimized.recognize(roi)
        lat_opt = (time.perf_counter() - t0) / 5 * 1000

        speedup = (lat_base / lat_opt) if lat_opt > 0 else 1.0
        time_saved = lat_base - lat_opt

        item_result = {
            'image': img_path,
            'roi_size': f"{roi.shape[1]}x{roi.shape[0]}",
            'baseline_ms': round(lat_base, 2),
            'step1_nocls_ms': round(lat_s1, 2),
            'optimized_ms': round(lat_opt, 2),
            'speedup': round(speedup, 2),
            'time_saved_ms': round(time_saved, 2),
            'text_base': txt_base,
            'text_opt': txt_opt,
            'match': (txt_base == txt_opt)
        }
        results_detail.append(item_result)

        print(f"  🔴 Baseline OCR (Mặc định)    : {lat_base:7.2f} ms | Biển số: {txt_base}")
        print(f"  🟡 Step 1 (Tắt use_cls)       : {lat_s1:7.2f} ms | Tiết kiệm: {lat_base - lat_s1:6.2f} ms")
        print(f"  🟢 Full Optimized (H=96+limit): {lat_opt:7.2f} ms | Biển số: {txt_opt} ➔ Tăng tốc {speedup:4.2f}x (Giảm {time_saved:6.2f} ms)!")

    # Thống kê trung bình
    avg_base = np.mean([r['baseline_ms'] for r in results_detail])
    avg_opt = np.mean([r['optimized_ms'] for r in results_detail])
    total_speedup = avg_base / avg_opt

    summary = {
        'profile': profile_name,
        'threads': num_threads,
        'avg_baseline_ms': round(avg_base, 2),
        'avg_optimized_ms': round(avg_opt, 2),
        'speedup': round(total_speedup, 2),
        'saved_time_ms': round(avg_base - avg_opt, 2),
        'details': results_detail
    }

    print("\n" + "=" * 78)
    print(f"🏆 KẾT QUẢ TỔNG HỢP: [{profile_name.upper()}]")
    print(f"  ⏱️ Baseline OCR trung bình : {avg_base:7.2f} ms")
    print(f"  ⚡ Optimized OCR trung bình: {avg_opt:7.2f} ms")
    print(f"  🚀 TỔNG HỆ SỐ TĂNG TỐC OCR : {total_speedup:4.2f}x (Cắt giảm {avg_base - avg_opt:7.2f} ms!)")
    print("=" * 78 + "\n")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark OCR Preprocessing & ROI Optimization")
    parser.add_argument('--profile', type=str, default='Raspberry Pi 4', help="Tên profile phần cứng")
    parser.add_argument('--threads', type=int, default=4, help="Số luồng CPU")
    parser.add_argument('--output_json', type=str, default='', help="Đường dẫn lưu kết quả JSON")
    args = parser.parse_args()

    res = run_sprint_benchmark(profile_name=args.profile, num_threads=args.threads)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"💾 Đã lưu JSON tại: {args.output_json}")
