"""
==============================================================================
🚗 FILE CHẠY THỬ NGHIỆM CHÍNH (ALPR RESEARCH PIPELINE RUNNER)
==============================================================================
Sử dụng:
  - Chạy mặc định:
      python run_demo.py
  - Chạy với ảnh khác:
      python run_demo.py --image samples/test1.jpg
  - Chọn engine so sánh (paddle hoặc lprnet):
      python run_demo.py --engine lprnet
  - Tắt cửa sổ GUI hiển thị (chỉ lưu ảnh kết quả):
      python run_demo.py --no_show
==============================================================================
"""

import os
import sys
import argparse
import cv2

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from pipeline import ALPRPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Nhận diện biển số xe phục vụ nghiên cứu (YOLOv8 + RapidOCR / LPRNet)")
    parser.add_argument('--image', type=str, default='samples/test2.jpg', help='Đường dẫn file ảnh đầu vào')
    parser.add_argument('--engine', type=str, choices=['paddle', 'lprnet'], default='paddle', help='Engine OCR: paddle (PaddleOCR ONNX) hoặc lprnet (LPRNet PyTorch)')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='Đường dẫn weights YOLOv8')
    parser.add_argument('--lpr_weights', type=str, default='weights/lprnet.pth', help='Đường dẫn weights LPRNet')
    parser.add_argument('--output_dir', type=str, default='output', help='Thư mục lưu ảnh kết quả')
    parser.add_argument('--no_show', action='store_true', help='Tắt cửa sổ GUI hiển thị ảnh')
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = args.image

    if not os.path.exists(image_path):
        alt = os.path.join('samples', os.path.basename(image_path))
        if os.path.exists(alt):
            image_path = alt
        else:
            print(f"❌ Error: Không tìm thấy file ảnh '{args.image}'!")
            sys.exit(1)

    print("=================================================================")
    print("🚗 ALPR RESEARCH PIPELINE (TWO-STAGE: DETECTION + RECOGNITION)")
    print("=================================================================")

    # 1. Khởi tạo Pipeline
    pipeline = ALPRPipeline(
        detector_weights=args.weights,
        engine=args.engine,
        lpr_weights=args.lpr_weights
    )

    # 2. Xử lý ảnh
    print(f"\n📷 Đang xử lý ảnh: '{image_path}'...")
    annotated_img, results, latency = pipeline.predict(image_path)

    # 3. In kết quả định lượng
    print("\n📊 --- KẾT QUẢ ĐO ĐẠC THỰC NGHIỆM ---")
    print(f"⏱️ Thời gian Detector (YOLOv8) : {latency['det_time_ms']:.2f} ms")
    print(f"⏱️ Thời gian OCR ({args.engine.upper()})    : {latency['ocr_time_ms']:.2f} ms")
    print(f"⚡ Tổng thời gian suy luận    : {latency['total_time_ms']:.2f} ms (~{latency['fps']:.1f} FPS)")
    print(f"🔍 Số biển số phát hiện       : {len(results)}\n")

    for item in results:
        print(f"  👉 Biển số #{item['plate_id']}:")
        print(f"     - Vị trí Box      : {item['box']}")
        print(f"     - Độ tin cậy YOLO : {item['detector_conf']*100:.1f}%")
        print(f"     - OCR thô đọc được: {item['raw_ocr']}")
        print(f"     - Biển số chuẩn   : ✨ {item['formatted_plate']} ✨")

    # 4. Lưu ảnh kết quả
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"result_{os.path.basename(image_path)}")
    cv2.imwrite(out_file, annotated_img)
    print(f"\n💾 Đã lưu ảnh kết quả tại: '{out_file}'")

    # 5. Hiển thị GUI
    if not args.no_show:
        try:
            cv2.imshow("Ket Qua Nhan Dien Bien So", annotated_img)
            print("Nhấn PHÍM BẤT KỲ trên cửa sổ ảnh để đóng...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass

    print("=================================================================")
    print("✨ Hoàn tất thử nghiệm!")


if __name__ == '__main__':
    main()
