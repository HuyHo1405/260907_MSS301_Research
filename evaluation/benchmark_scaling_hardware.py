"""
==============================================================================
🔬 BENCHMARK SUITE: RESOLUTION SCALING & HARDWARE/RUNTIME OPTIMIZATION
==============================================================================
Kiểm chứng định lượng cho:
  - Phương pháp 3: Resolution Scaling (640, 512, 480, 384, 320)
  - Phương pháp 4: ARM NEON SIMD / Multi-Threading (4T, 2T, 1T) & OCR Optimization
Phục vụ mô phỏng 3 cấu hình phần cứng:
  - Raspberry Pi 4 (Quad A72, 4 Cores)
  - Raspberry Pi 3 (Quad A53, 2.5 Cores)
  - Raspberry Pi 2 / Zero (Cortex-A7/A53, 1.5 Cores / 1 Core)
==============================================================================
"""

import os
import sys
import time
import argparse
import json
import cv2
import numpy as np

# Đảm bảo import được module pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.detector_onnx import ONNXPlateDetector
from rapidocr_onnxruntime import RapidOCR
from pipeline.postprocess import format_vietnamese_plate


def compute_iou(boxA, boxB):
    """Tính Intersection over Union (IoU) giữa 2 bounding box [x1, y1, x2, y2]."""
    if not boxA or not boxB:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea


def run_benchmark(profile_name="Pi 4", weights_path="weights/best_int8.onnx", test_images=None, num_threads=4):
    print("\n" + "=" * 78)
    print(f"🚀 BẮT ĐẦU BENCHMARK: [{profile_name.upper()}] - Threads={num_threads}")
    print("=" * 78)

    if test_images is None:
        test_images = ['samples/test1.jpg', 'samples/test2.jpg', 'samples/test3.jpg']
    
    # Lọc ảnh tồn tại
    valid_images = [p for p in test_images if os.path.exists(p)]
    if not valid_images:
        print("❌ Không tìm thấy ảnh test nào hợp lệ!")
        return {}

    print(f"📁 Danh sách ảnh thử nghiệm ({len(valid_images)} ảnh): {valid_images}")

    # Khởi tạo RapidOCR engine dùng chung
    ocr_engine = RapidOCR()

    # --------------------------------------------------------------------------
    # 🧪 PHẦN 1: KHẢO SÁT PHƯƠNG PHÁP 3 - RESOLUTION SCALING
    # --------------------------------------------------------------------------
    resolutions = [640, 512, 480, 384, 320]
    res_results = {}

    print("\n" + "-" * 78)
    print("🔍 [PHƯƠNG PHÁP 3] ĐO ĐẠC RESOLUTION SCALING TRÊN DETECTOR (INT8)")
    print("-" * 78)

    baseline_boxes = {}  # lưu box ở 640 để tính IoU cho các res khác

    for res in resolutions:
        detector = ONNXPlateDetector(weights_path=weights_path, imgsz=res, num_threads=num_threads)
        
        latencies = []
        confs = []
        ious = []
        ocr_matches = []
        plates_detected = 0

        for img_path in valid_images:
            img = cv2.imread(img_path)
            
            # Warmup
            detector.detect(img)
            
            # Đo tốc độ 5 lần
            runs = []
            for _ in range(5):
                t0 = time.perf_counter()
                boxes, scores, rois, _ = detector.detect(img)
                runs.append((time.perf_counter() - t0) * 1000)

            avg_time = np.mean(runs)
            latencies.append(avg_time)

            if len(boxes) > 0:
                plates_detected += 1
                best_box = boxes[0]
                best_conf = scores[0]
                best_roi = rois[0]

                confs.append(best_conf)

                # Lưu box baseline nếu là 640
                if res == 640:
                    baseline_boxes[img_path] = best_box
                    ious.append(1.0)
                else:
                    base_box = baseline_boxes.get(img_path, None)
                    iou = compute_iou(best_box, base_box)
                    ious.append(iou)

                # Test OCR với RapidOCR (use_cls=False)
                ocr_res, _ = ocr_engine(best_roi, use_cls=False)
                raw_texts = [x[1] for x in ocr_res] if ocr_res else []
                fmt_plate = format_vietnamese_plate(raw_texts)
                ocr_matches.append(fmt_plate if fmt_plate else "N/A")
            else:
                confs.append(0.0)
                ious.append(0.0)
                ocr_matches.append("MISS")

        mean_lat = float(np.mean(latencies))
        mean_conf = float(np.mean(confs)) * 100 if confs else 0.0
        mean_iou = float(np.mean(ious)) * 100 if ious else 0.0

        base_lat = res_results.get(640, {}).get('latency_ms', mean_lat)
        speedup = (base_lat / mean_lat) if mean_lat > 0 else 1.0

        res_results[res] = {
            'latency_ms': round(mean_lat, 2),
            'speedup': round(speedup, 2),
            'fps': round(1000.0 / mean_lat, 2) if mean_lat > 0 else 0.0,
            'confidence': round(mean_conf, 1),
            'iou_vs_640': round(mean_iou, 1),
            'detected_ratio': f"{plates_detected}/{len(valid_images)}",
            'plates': ocr_matches
        }

        print(f"  👉 Res {res:3d}x{res:3d} | Latency: {mean_lat:6.2f} ms | Speedup: {speedup:4.2f}x | Conf: {mean_conf:4.1f}% | IoU: {mean_iou:5.1f}% | Plates: {ocr_matches}")

    # --------------------------------------------------------------------------
    # 🧪 PHẦN 2: KHẢO SÁT PHƯƠNG PHÁP 4 - MULTI-THREADING SCALING & OCR TUNING
    # --------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("⚡ [PHƯƠNG PHÁP 4] ĐO ĐẠC MULTI-THREADING (SIMD) & TỐI ƯU HÓA RAPIDOCR")
    print("-" * 78)

    thread_results = {}
    test_res = 480  # Sử dụng res tối ưu 480 để đo thread scaling
    sample_img = cv2.imread(valid_images[0])

    for th in [4, 2, 1]:
        det_th = ONNXPlateDetector(weights_path=weights_path, imgsz=test_res, num_threads=th)
        det_th.detect(sample_img)  # Warmup

        th_runs = []
        for _ in range(5):
            t0 = time.perf_counter()
            det_th.detect(sample_img)
            th_runs.append((time.perf_counter() - t0) * 1000)

        mean_th_lat = float(np.mean(th_runs))
        thread_results[th] = {
            'latency_ms': round(mean_th_lat, 2),
            'fps': round(1000.0 / mean_th_lat, 2)
        }
        print(f"  🧵 {th} Threads | Detector ({test_res}x{test_res}): {mean_th_lat:6.2f} ms | {1000.0/mean_th_lat:4.1f} FPS")

    # Đo đạc RapidOCR: Baseline (use_cls=True) vs Optimized (use_cls=False)
    # Lấy 1 ROI biển số thật để test
    det_tmp = ONNXPlateDetector(weights_path=weights_path, imgsz=480, num_threads=num_threads)
    _, _, sample_rois, _ = det_tmp.detect(sample_img)
    sample_roi = sample_rois[0] if sample_rois else sample_img

    # Warmup
    ocr_engine(sample_roi, use_cls=True)

    # 1. Đo Baseline OCR
    runs_ocr_cls = []
    for _ in range(5):
        t0 = time.perf_counter()
        ocr_engine(sample_roi, use_cls=True)
        runs_ocr_cls.append((time.perf_counter() - t0) * 1000)
    lat_ocr_base = float(np.mean(runs_ocr_cls))

    # 2. Đo Optimized OCR
    runs_ocr_nocls = []
    for _ in range(5):
        t0 = time.perf_counter()
        ocr_engine(sample_roi, use_cls=False)
        runs_ocr_nocls.append((time.perf_counter() - t0) * 1000)
    lat_ocr_opt = float(np.mean(runs_ocr_nocls))

    ocr_speedup = (lat_ocr_base / lat_ocr_opt) if lat_ocr_opt > 0 else 1.0

    ocr_comp = {
        'baseline_ms': round(lat_ocr_base, 2),
        'optimized_ms': round(lat_ocr_opt, 2),
        'speedup': round(ocr_speedup, 2),
        'saved_time_ms': round(lat_ocr_base - lat_ocr_opt, 2)
    }

    print(f"  📝 RapidOCR Baseline (use_cls=True) : {lat_ocr_base:6.2f} ms")
    print(f"  ⚡ RapidOCR Optimized (use_cls=False): {lat_ocr_opt:6.2f} ms | Giảm {lat_ocr_base - lat_ocr_opt:6.2f} ms ({ocr_speedup:.2f}x)")

    # --------------------------------------------------------------------------
    # 📊 TỔNG HỢP PIPELINE END-TO-END TRƯỚC VÀ SAU TỐI ƯU
    # --------------------------------------------------------------------------
    e2e_before = res_results[640]['latency_ms'] + lat_ocr_base
    e2e_after = res_results[480]['latency_ms'] + lat_ocr_opt
    e2e_speedup = (e2e_before / e2e_after) if e2e_after > 0 else 1.0

    summary = {
        'profile': profile_name,
        'threads': num_threads,
        'resolution_scaling': res_results,
        'thread_scaling': thread_results,
        'ocr_optimization': ocr_comp,
        'e2e_comparison': {
            'before_opt_ms': round(e2e_before, 2),
            'after_opt_ms': round(e2e_after, 2),
            'speedup': round(e2e_speedup, 2),
            'fps_before': round(1000.0 / e2e_before, 2),
            'fps_after': round(1000.0 / e2e_after, 2)
        }
    }

    print("\n" + "=" * 78)
    print(f"🏆 KẾT QUẢ END-TO-END PIPELINE: [{profile_name.upper()}]")
    print(f"  🔴 Trước tối ưu (640x640 + OCR cls=True) : {e2e_before:7.2f} ms ({1000.0/e2e_before:.2f} FPS)")
    print(f"  🟢 Sau tối ưu (480x480 + OCR cls=False): {e2e_after:7.2f} ms ({1000.0/e2e_after:.2f} FPS) ➔ Tăng tốc {e2e_speedup:.2f}x!")
    print("=" * 78 + "\n")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark Resolution & Hardware Scaling")
    parser.add_argument('--profile', type=str, default='Raspberry Pi 4', help="Tên cấu hình phần cứng")
    parser.add_argument('--threads', type=int, default=4, help="Số luồng CPU")
    parser.add_argument('--output_json', type=str, default='', help="Đường dẫn lưu file JSON")
    args = parser.parse_args()

    results = run_benchmark(profile_name=args.profile, num_threads=args.threads)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Đã lưu kết quả JSON vào: {args.output_json}")
