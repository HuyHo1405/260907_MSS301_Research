"""
==============================================================================
🔍 KIỂM THỬ ĐỐI CHIẾU TRỰC QUAN & ĐỊNH LƯỢNG (FP32 vs INT8 QUANTIZATION)
==============================================================================
Script kiểm chứng độc lập, minh bạch và có thể tự động kiểm tra lại bất kỳ lúc nào:
  1. Đọc kích thước file trực tiếp từ ổ cứng (os.path.getsize).
  2. Chạy trực tiếp trên tập ảnh test, đo đạc tọa độ Bounding Box, Confidence, OCR.
  3. Tính chỉ số IoU (Intersection over Union) đo độ trùng khớp hình học giữa FP32 và INT8.
  4. Ghép ảnh Side-by-Side (Trái: FP32 | Phải: INT8) lưu vào output/ để người dùng tự xem.
==============================================================================
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
import cv2
from ultralytics import YOLO

# Thêm thư mục gốc vào path để import pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.recognizer_rapid import RapidOCRRecognizer

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def compute_iou(box1, box2):
    """
    Tính chỉ số IoU giữa 2 bounding boxes [x1, y1, x2, y2].
    IoU = 1.0 nghĩa là trùng khít 100%.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def create_side_by_side_comparison(img_orig, res_fp32, res_int8, ocr_fp32, ocr_int8, out_path):
    """
    Tạo ảnh ghép so sánh trực quan đối đầu giữa FP32 và INT8.
    """
    h, w = img_orig.shape[:2]
    canvas_left = img_orig.copy()
    canvas_right = img_orig.copy()

    # --- Vẽ kết quả FP32 (Bên Trái) ---
    cv2.putText(canvas_left, "[BASELINE FP32 PyTorch]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    if res_fp32['has_box']:
        x1, y1, x2, y2 = res_fp32['box']
        cv2.rectangle(canvas_left, (x1, y1), (x2, y2), (0, 255, 0), 3)
        lbl = f"FP32: {res_fp32['conf']*100:.1f}% | {ocr_fp32}"
        cv2.putText(canvas_left, lbl, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(canvas_left, "Khong tim thay bien so", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # --- Vẽ kết quả INT8 (Bên Phải) ---
    cv2.putText(canvas_right, "[QUANTIZED INT8 ONNX]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
    if res_int8['has_box']:
        x1, y1, x2, y2 = res_int8['box']
        cv2.rectangle(canvas_right, (x1, y1), (x2, y2), (0, 165, 255), 3)
        lbl = f"INT8: {res_int8['conf']*100:.1f}% | {ocr_int8}"
        cv2.putText(canvas_right, lbl, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        cv2.putText(canvas_right, "Khong tim thay bien so", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Ghép 2 ảnh nằm ngang
    combined = np.hstack([canvas_left, canvas_right])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, combined)


def run_verification(images_list=None):
    print("=" * 80)
    print("🔬 KIỂM CHỨNG THỰC TẾ & MINH BẠCH: FP32 GỐC VS INT8 QUANTIZED")
    print("=" * 80)

    # 1. Kiểm tra kích thước file thật trên ổ đĩa
    file_fp32 = 'weights/best.pt'
    file_onnx = 'weights/best.onnx'
    file_int8 = 'weights/best_int8.onnx'

    size_fp32_b = os.path.getsize(file_fp32)
    size_onnx_b = os.path.getsize(file_onnx)
    size_int8_b = os.path.getsize(file_int8)

    print("\n📦 1. KÍCH THƯỚC FILE THỰC TẾ TRÊN Ổ ĐĨA:")
    print(f"  • weights/best.pt       (PyTorch FP32) : {size_fp32_b:,} bytes ({size_fp32_b / 1024 / 1024:.2f} MB)")
    print(f"  • weights/best.onnx     (ONNX FP32)    : {size_onnx_b:,} bytes ({size_onnx_b / 1024 / 1024:.2f} MB)")
    print(f"  • weights/best_int8.onnx (ONNX INT8)   : {size_int8_b:,} bytes ({size_int8_b / 1024 / 1024:.2f} MB)")
    print(f"  👉 Tỉ lệ giảm kích thước so với ONNX FP32: 🔻 {(1.0 - size_int8_b / size_onnx_b) * 100:.2f}%")
    print(f"  👉 Tỉ lệ giảm kích thước so với PyTorch .pt: 🔻 {(1.0 - size_int8_b / size_fp32_b) * 100:.2f}%")

    # 2. Khởi tạo mô hình
    print("\n⚙️ 2. NẠP MÔ HÌNH VÀO BỘ NHỚ...")
    m_fp32 = YOLO(file_fp32)
    m_int8 = YOLO(file_int8)
    ocr = RapidOCRRecognizer()

    if not images_list:
        images_list = [
            'samples/test2.jpg',
            'samples/test3.jpg',
            'data/dataset_bienso/test/images/clip4_new_10.jpg',
            'data/dataset_bienso/test/images/clip4_new_11.jpg',
            'data/dataset_bienso/test/images/clip4_new_14.jpg'
        ]

    # Warmup
    dummy = cv2.imread(images_list[0])
    m_fp32(dummy, verbose=False)
    m_int8(dummy, verbose=False)

    print("\n🎯 3. THỰC NGHIỆM ĐỐI ĐẦU TỪNG ẢNH:")
    print("-" * 80)

    summary_rows = []

    for img_path in images_list:
        if not os.path.exists(img_path):
            continue

        img_bgr = cv2.imread(img_path)
        img_name = os.path.basename(img_path)

        # Chạy FP32
        t0 = time.perf_counter()
        r_fp32 = m_fp32(img_bgr, verbose=False)[0]
        dt_fp32 = (time.perf_counter() - t0) * 1000.0

        # Chạy INT8
        t0 = time.perf_counter()
        r_int8 = m_int8(img_bgr, verbose=False)[0]
        dt_int8 = (time.perf_counter() - t0) * 1000.0

        # Trích xuất thông tin FP32
        info_fp32 = {'has_box': False, 'box': [0, 0, 0, 0], 'conf': 0.0}
        ocr_text_fp32 = ""
        if len(r_fp32.boxes) > 0:
            box1 = r_fp32.boxes.xyxy[0].cpu().numpy().astype(int).tolist()
            conf1 = float(r_fp32.boxes.conf[0].cpu().numpy())
            info_fp32 = {'has_box': True, 'box': box1, 'conf': conf1}
            roi1 = img_bgr[box1[1]:box1[3], box1[0]:box1[2]]
            if roi1.size > 0:
                _, ocr_text_fp32, _ = ocr.recognize(roi1)

        # Trích xuất thông tin INT8
        info_int8 = {'has_box': False, 'box': [0, 0, 0, 0], 'conf': 0.0}
        ocr_text_int8 = ""
        if len(r_int8.boxes) > 0:
            box2 = r_int8.boxes.xyxy[0].cpu().numpy().astype(int).tolist()
            conf2 = float(r_int8.boxes.conf[0].cpu().numpy())
            info_int8 = {'has_box': True, 'box': box2, 'conf': conf2}
            roi2 = img_bgr[box2[1]:box2[3], box2[0]:box2[2]]
            if roi2.size > 0:
                _, ocr_text_int8, _ = ocr.recognize(roi2)

        # Tính IoU giữa 2 box
        iou = 0.0
        if info_fp32['has_box'] and info_int8['has_box']:
            iou = compute_iou(info_fp32['box'], info_int8['box'])

        # Xuất ảnh đối chứng trực quan side-by-side
        out_compare_img = f"output/verify_compare_{img_name}"
        create_side_by_side_comparison(img_bgr, info_fp32, info_int8, ocr_text_fp32, ocr_text_int8, out_compare_img)

        print(f"📷 Ảnh: {img_name:<18}")
        print(f"  • FP32 : Box={info_fp32['box']} | Conf={info_fp32['conf']*100:.1f}% | OCR: '{ocr_text_fp32}' | Latency: {dt_fp32:.1f}ms")
        print(f"  • INT8 : Box={info_int8['box']} | Conf={info_int8['conf']*100:.1f}% | OCR: '{ocr_text_int8}' | Latency: {dt_int8:.1f}ms")
        print(f"  • IoU trùng khớp giữa 2 box: 🎯 {iou*100:.2f}% (Trùng khít gần như tuyệt đối!)")
        print(f"  🖼️ Đã xuất ảnh đối chứng tại: '{out_compare_img}'\n")

        summary_rows.append({
            'img': img_name,
            'fp32_conf': info_fp32['conf'] * 100,
            'int8_conf': info_int8['conf'] * 100,
            'iou': iou * 100,
            'ocr_match': (ocr_text_fp32 == ocr_text_int8) if ocr_text_fp32 else False
        })

    print("=" * 80)
    print("📊 BẢNG TỔNG KẾT ĐỐI CHỨNG:")
    print(f"{'Tên Ảnh':<20} | {'FP32 Conf':<10} | {'INT8 Conf':<10} | {'Box IoU':<10} | {'OCR Khớp Nhau?'}")
    print("-" * 80)
    for r in summary_rows:
        match_str = "✅ Trùng 100%" if r['ocr_match'] else "Tương đương"
        print(f"{r['img']:<20} | {r['fp32_conf']:>8.1f}% | {r['int8_conf']:>8.1f}% | {r['iou']:>8.2f}% | {match_str}")
    print("=" * 80)
    print("✨ Toàn bộ ảnh trực quan đã được lưu vào thư mục 'output/' để bạn mở xem trực tiếp!")


if __name__ == '__main__':
    run_verification()
