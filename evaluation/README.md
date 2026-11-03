# 📊 Đánh Giá & Benchmark Hiệu Năng (Model Evaluation)

## Bộ chỉ số độc lập (Phase 6 — Tuấn)

Xem [METRICS.md](METRICS.md) để biết định dạng dữ liệu, quy ước AP/OCR và cách
tích hợp runner. Chạy từ thư mục gốc dự án, không cần GPU hay model:

```powershell
python -m unittest discover -s tests -p "test_metrics.py" -v
python -m evaluation.demo_metrics
```

---

## 🌪️ Bộ Sinh Nhiễu Nhân Tạo & CLI Batch Generator (Phase 6 — Hy)

Module `evaluation/corruptions.py` hỗ trợ 4 nhóm nhiễu với 10 kịch bản thực tế, chia thành 3 cấp độ (Level 1 $\rightarrow$ 3), tự động cập nhật Bounding Box khi biến đổi hình học:

- **Quang Học & Ánh Sáng**: `low_light`, `overexposure`, `uneven_shadows`
- **Mờ & Chuyển Động**: `motion_blur`, `defocus_blur`
- **Thời Tiết & Truyền Dẫn**: `rain_streaks`, `sensor_noise`, `jpeg_compression`
- **Hình Học & Vật Lý**: `perspective_tilt` (tự động re-project BBox), `partial_occlusion`

### Chạy Unit Test kiểm thử bộ sinh nhiễu:
```powershell
python -m unittest discover -s tests -p "test_corruptions.py" -v
```

### Chạy CLI sinh batch dữ liệu ảnh nhiễu:
```powershell
# Chạy mẫu thử nghiệm (5 ảnh, vẽ kèm BBox trực quan)
python -m evaluation.corruptions --sample-size 5 --visualize

# Chạy toàn bộ Source A cho cả 10 kịch bản x 3 cấp độ
python -m evaluation.corruptions --input-dir datasets/eval_set/images --annotations datasets/eval_set/_annotations.coco.json --output-dir datasets/corrupted --scenarios all --severities 1,2,3

# Chạy riêng kịch bản cụ thể
python -m evaluation.corruptions --scenarios perspective_tilt,overexposure --severities 1,2,3
```

Dữ liệu đầu ra được lưu tại `datasets/corrupted/{scenario}/level_{1,2,3}/` kèm file `_annotations.coco.json` cập nhật và bảng chỉ mục `datasets/corrupted/manifest.json`.

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

## 🚀 Cách Chạy Benchmark Cũ (Phase 1-5)

```bash
# Đo đạc mô hình ONNX
python benchmark_edge.py --model ../models/plate_detect.onnx --iterations 200 --warmup 20
```
