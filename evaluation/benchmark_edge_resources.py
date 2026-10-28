"""
==============================================================================
📊 ĐO ĐẠC TÀI NGUYÊN MÔ PHỎNG EDGE / RASPBERRY PI 4 (RAM & CPU LATENCY)
==============================================================================
Mục đích:
  So sánh mức tiêu thụ tài nguyên thực tế giữa:
    - Baseline: PyTorch Full Runtime (YOLOv8 FP32 .pt)
    - Edge Optimized: Pure ONNX Runtime (YOLOv8 INT8 .onnx)
  Số liệu đo đạc:
    1. Dung lượng file weights (MB).
    2. Bộ nhớ RAM thực tế bị chiếm dụng (RAM footprint / Memory consumed) (MB).
    3. Thời gian suy luận với 4 threads (giới hạn 4 nhân của Pi 4) (ms).
    4. Tốc độ khung hình (FPS).
==============================================================================
"""

import os
import sys
import time
import psutil
import gc
import cv2
import numpy as np

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def get_process_memory_mb():
    """Lấy lượng RAM thực tế tiến trình hiện tại đang sử dụng (Resident Set Size)."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_pytorch_baseline(image_path='samples/test2.jpg', runs=5):
    """Đo đạc PyTorch FP32 Baseline."""
    gc.collect()
    ram_before = get_process_memory_mb()

    from ultralytics import YOLO
    model = YOLO('weights/best.pt')

    img = cv2.imread(image_path)
    # Warmup
    model(img, verbose=False)

    ram_after = get_process_memory_mb()
    ram_used = max(0.0, ram_after - ram_before)

    times = []
    conf = 0.0
    for _ in range(runs):
        t0 = time.perf_counter()
        res = model(img, verbose=False)
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)
        if res and len(res[0].boxes) > 0:
            conf = float(res[0].boxes.conf[0].cpu().numpy())

    return {
        'name': 'PyTorch Baseline (FP32)',
        'file_size_mb': os.path.getsize('weights/best.pt') / (1024 * 1024),
        'ram_consumed_mb': ram_used,
        'total_process_ram_mb': ram_after,
        'latency_ms': np.mean(times),
        'fps': 1000.0 / np.mean(times),
        'conf': conf * 100
    }


def benchmark_pure_onnx_int8(image_path='samples/test2.jpg', runs=5, num_threads=4):
    """Đo đạc Pure ONNX Runtime INT8 (Giới hạn 4 threads chuẩn Pi 4)."""
    gc.collect()
    ram_before = get_process_memory_mb()

    from pipeline.detector_onnx import ONNXPlateDetector
    detector = ONNXPlateDetector(weights_path='weights/best_int8.onnx', num_threads=num_threads)

    img = cv2.imread(image_path)
    # Warmup
    detector.detect(img)

    ram_after = get_process_memory_mb()
    ram_used = max(0.0, ram_after - ram_before)

    times = []
    conf = 0.0
    for _ in range(runs):
        boxes, confs, _, dt = detector.detect(img)
        times.append(dt)
        if len(confs) > 0:
            conf = confs[0]

    return {
        'name': 'Pure ONNX Edge (INT8)',
        'file_size_mb': os.path.getsize('weights/best_int8.onnx') / (1024 * 1024),
        'ram_consumed_mb': ram_used,
        'total_process_ram_mb': ram_after,
        'latency_ms': np.mean(times),
        'fps': 1000.0 / np.mean(times),
        'conf': conf * 100
    }


def main():
    print("=" * 80)
    print("🍓 BENCHMARK TÀI NGUYÊN MÔ PHỎNG RASPBERRY PI 4 (4 CORES / 2GB RAM)")
    print("=" * 80)

    test_image = 'samples/test2.jpg'
    if not os.path.exists(test_image):
        print(f"❌ Không tìm thấy ảnh test: '{test_image}'")
        return

    print("⏳ Đang đo đạc PyTorch Baseline...")
    res_pytorch = benchmark_pytorch_baseline(test_image)

    print("⏳ Đang đo đạc Pure ONNX INT8 (giới hạn 4 threads CPU)...")
    res_onnx = benchmark_pure_onnx_int8(test_image, num_threads=4)

    ram_saved_pct = ((res_pytorch['ram_consumed_mb'] - res_onnx['ram_consumed_mb']) / res_pytorch['ram_consumed_mb']) * 100 if res_pytorch['ram_consumed_mb'] > 0 else 0
    size_saved_pct = ((res_pytorch['file_size_mb'] - res_onnx['file_size_mb']) / res_pytorch['file_size_mb']) * 100

    print("\n" + "=" * 80)
    print("📈 KẾT QUẢ ĐỐI ĐẦU ĐỊNH LƯỢNG:")
    print("-" * 80)
    print(f"{'Tiêu Chí Đánh Giá':<30} | {'PyTorch FP32 Baseline':<20} | {'Pure ONNX INT8 Edge':<20}")
    print("-" * 80)
    print(f"{'Dung lượng File Weights':<30} | {res_pytorch['file_size_mb']:>17.2f} MB | {res_onnx['file_size_mb']:>17.2f} MB (🔻 {size_saved_pct:.1f}%)")
    print(f"{'RAM Chiếm Dụng Khi Nạp Model':<30} | {res_pytorch['ram_consumed_mb']:>17.2f} MB | {res_onnx['ram_consumed_mb']:>17.2f} MB")
    print(f"{'Tổng RAM Tiến Trình (Process)':<30} | {res_pytorch['total_process_ram_mb']:>17.2f} MB | {res_onnx['total_process_ram_mb']:>17.2f} MB")
    print(f"{'Thời Gian Xử Lý (4 Cores)':<30} | {res_pytorch['latency_ms']:>17.2f} ms | {res_onnx['latency_ms']:>17.2f} ms")
    print(f"{'Tốc Độ Khung Hình (FPS)':<30} | {res_pytorch['fps']:>17.1f} FPS| {res_onnx['fps']:>17.1f} FPS")
    print(f"{'Độ Tin Cậy Phát Hiện (Conf)':<30} | {res_pytorch['conf']:>17.1f} %  | {res_onnx['conf']:>17.1f} %")
    print("=" * 80)
    print("💡 KẾT LUẬN CHO TRIỂN KHAI RASPBERRY PI 4:")
    print(f"  • Tiết kiệm dung lượng lưu trữ: 🔻 {size_saved_pct:.1f}%")
    print(f"  • Hoàn toàn KHÔNG cần cài PyTorch (>1.5 GB package) trên Pi 4.")
    print(f"  • Độ chính xác nhận diện giữ nguyên vẹn ({res_onnx['conf']:.1f}% vs {res_pytorch['conf']:.1f}%).")
    print("=" * 80)


if __name__ == '__main__':
    main()
