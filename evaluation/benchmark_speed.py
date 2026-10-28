"""
Script Benchmark tốc độ suy luận (Latency & FPS) phục vụ đo đạc số liệu nghiên cứu.
So sánh hiệu năng giữa hai phương pháp OCR: RapidOCR (PaddleOCR ONNX) và LPRNet PyTorch.
"""
import os
import glob
import time
import argparse
import numpy as np

# Thêm thư mục gốc vào sys.path để import pipeline
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import ALPRPipeline


def benchmark(image_dir='samples', runs=5, engine='paddle', weights='weights/best.pt', lpr_weights='weights/lprnet.pth'):
    images = glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png'))
    if not images:
        print(f"❌ Không tìm thấy ảnh nào trong thư mục '{image_dir}'!")
        return

    print("=================================================================")
    print(f"📊 BẮT ĐẦU BENCHMARK [Engine: {engine.upper()} | Số ảnh: {len(images)} | Số lần lặp/ảnh: {runs}]")
    print("=================================================================")

    pipeline = ALPRPipeline(detector_weights=weights, engine=engine, lpr_weights=lpr_weights)

    # Warm-up 1 lần để khởi tạo backend ONNX / PyTorch CUDA
    print("🔥 Warm-up mô hình...")
    pipeline.predict(images[0])

    det_times = []
    ocr_times = []
    total_times = []

    for img_path in images:
        img_name = os.path.basename(img_path)
        for r in range(runs):
            _, results, latency = pipeline.predict(img_path)
            det_times.append(latency['det_time_ms'])
            ocr_times.append(latency['ocr_time_ms'])
            total_times.append(latency['total_time_ms'])

    avg_det = np.mean(det_times)
    std_det = np.std(det_times)
    avg_ocr = np.mean(ocr_times)
    std_ocr = np.std(ocr_times)
    avg_total = np.mean(total_times)
    std_total = np.std(total_times)
    avg_fps = 1000.0 / avg_total if avg_total > 0 else 0.0

    print("\n-----------------------------------------------------------------")
    print("📈 KẾT QUẢ ĐO ĐẠC ĐỊNH LƯỢNG (MEAN ± STD):")
    print(f"  • Detector (YOLOv8) Latency : {avg_det:.2f} ± {std_det:.2f} ms")
    print(f"  • OCR ({engine.upper()}) Latency   : {avg_ocr:.2f} ± {std_ocr:.2f} ms")
    print(f"  • Tổng thời gian suy luận   : {avg_total:.2f} ± {std_total:.2f} ms")
    print(f"  • Tốc độ khung hình (FPS)   : ⚡ {avg_fps:.1f} FPS")
    print("=================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Đo đạc tốc độ suy luận mô hình")
    parser.add_argument('--dir', type=str, default='samples', help='Thư mục chứa ảnh benchmark')
    parser.add_argument('--runs', type=int, default=5, help='Số lần chạy lại trên mỗi ảnh để lấy trung bình')
    parser.add_argument('--engine', type=str, choices=['paddle', 'lprnet'], default='paddle', help='Bộ nhận diện OCR cần benchmark')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='Đường dẫn weights YOLOv8')
    parser.add_argument('--lpr_weights', type=str, default='weights/lprnet.pth', help='Đường dẫn weights LPRNet')
    args = parser.parse_args()

    benchmark(image_dir=args.dir, runs=args.runs, engine=args.engine, weights=args.weights, lpr_weights=args.lpr_weights)
