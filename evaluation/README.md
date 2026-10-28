# 📊 Đánh Giá & Benchmark Hiệu Năng (Model Evaluation)

Thư mục chứa các công cụ đo lường và thẩm định mô hình AI khi triển khai trên thiết bị phần cứng thực tế (Edge Hardware: Raspberry Pi 4, Jetson Orin Nano, máy trạm cục bộ).

---

## 🎯 Chỉ Số Đánh Giá Chính (Key Evaluation Metrics)

1. **Hiệu năng thời gian thực (Speed & Latency):**
   - **Inference Latency:** Thời gian suy luận trung bình (Mean), P95, P99 (ms).
   - **Throughput (FPS):** Số khung hình xử lý được trên mỗi giây.
   - **Warmup Phase:** Loại bỏ 20-50 lượt chạy đầu để ổn định cache/GPU context.

2. **Tiêu thụ tài nguyên biên (Resource Consumption):**
   - **RAM Usage:** Dung lượng RAM chiếm dụng (MB) và Peak RAM.
   - **CPU Utilization:** Mức độ tải của các nhân CPU (%).
   - **Nhiệt độ & Throttling (cho RPi4/Jetson):** Giám sát nhiệt độ chip khi hoạt động liên tục.

3. **Chỉ số độ chính xác (Accuracy Metrics):**
   - **Detection:** mAP@0.5, mAP@0.5:0.95 cho biển số và phương tiện.
   - **OCR:** Ký tự đúng hoàn toàn (Exact Match Rate), Levenshtein Distance (Character Error Rate - CER).
   - **Face:** True Accept Rate (TAR) @ False Accept Rate (FAR) = 0.001.

---

## 🚀 Cách Chạy Benchmark

```bash
# Đo đạc mô hình ONNX
python benchmark_edge.py --model ../models/plate_detect.onnx --iterations 200 --warmup 20
```
