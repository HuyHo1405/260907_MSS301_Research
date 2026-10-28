"""
Công cụ đo lường hiệu năng suy luận (Benchmark Inference Latency & Resource Utilization).
Hỗ trợ đo lường mô hình ONNX Runtime trên môi trường Edge Device (Raspberry Pi, Jetson, PC).
"""

import argparse
import os
import time
from pathlib import Path
import numpy as np
import psutil


def run_benchmark(model_path: str, input_size: int = 640, iterations: int = 200, warmup: int = 20):
    path = Path(model_path)
    if not path.exists():
        print(f"[ERROR] Không tìm thấy tệp mô hình: {model_path}")
        return

    print("=" * 65)
    print(" 🚀 BENCHMARK HIỆU NĂNG SUY LUẬN MÔ HÌNH TRÊN EDGE DEVICE")
    print("=" * 65)
    print(f"Mô hình:           {path.name}")
    print(f"Kích thước file:   {path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"Kích thước input:  1x3x{input_size}x{input_size}")
    print(f"Số lượt đo:        {iterations} (Warmup: {warmup})")
    print("-" * 65)

    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] Thư viện onnxruntime chưa được cài đặt: pip install onnxruntime")
        return

    # Khởi tạo ONNX Runtime Session
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=session_options, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

    # Đo tài nguyên trước khi chạy
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 * 1024)

    # Warmup phase
    print(f"[*] Đang thực hiện {warmup} lượt khởi động (Warmup)...")
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})

    # Benchmark loop
    print(f"[*] Đang đo lường {iterations} lượt suy luận...")
    latencies = []
    start_total = time.perf_counter()

    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    total_time = time.perf_counter() - start_total
    ram_after = process.memory_info().rss / (1024 * 1024)

    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    fps = 1000.0 / mean_latency

    print("\n" + "=" * 65)
    print(" 📊 KẾT QUẢ ĐO ĐẠC HIỆU NĂNG (BENCHMARK RESULTS)")
    print("=" * 65)
    print(f"Độ trễ trung bình (Mean Latency):     {mean_latency:.2f} ms")
    print(f"Độ trễ trung vị (Median):             {median_latency:.2f} ms")
    print(f"Phân vị 95 (P95 Latency):             {p95_latency:.2f} ms")
    print(f"Phân vị 99 (P99 Latency):             {p99_latency:.2f} ms")
    print(f"Min / Max Latency:                    {min_latency:.2f} ms / {max_latency:.2f} ms")
    print(f"Thông lượng suy luận (FPS):           {fps:.2f} FPS")
    print(f"Dung lượng RAM tiến trình:            {ram_after:.2f} MB (Tăng {ram_after - ram_before:+.2f} MB)")
    print(f"Tổng thời gian đo:                    {total_time:.2f} s")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Inference Benchmark for SecurityGate")
    parser.add_argument("--model", type=str, required=True, help="Đường dẫn file mô hình .onnx")
    parser.add_argument("--input-size", type=int, default=640, help="Input size ảnh (mặc định 640)")
    parser.add_argument("--iterations", type=int, default=150, help="Số lượt suy luận kiểm thử")
    parser.add_argument("--warmup", type=int, default=20, help="Số lượt warmup bỏ qua")

    args = parser.parse_args()
    run_benchmark(args.model, args.input_size, args.iterations, args.warmup)
