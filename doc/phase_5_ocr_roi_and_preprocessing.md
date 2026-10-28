# 🚀 Phase 5: Đột Phá Tầng OCR — Chuẩn Hóa ROI & Triệt Tiêu Over-Processing

## 1. 🎯 Mục Tiêu Cốt Lõi
- Rà soát các tiêu chí giảm tải tính toán chưa đụng đến ở giai đoạn trước:
  1. **Tiêu chí 1 (nửa sau):** Resize / chuẩn hóa kích thước vùng cắt biển số (ROI Downscale) trước khi nạp vào OCR.
  2. **Tiêu chí 4:** Rà soát và loại bỏ các bước tiền xử lý cồng kềnh thừa thãi (Over-processing) bên trong RapidOCR.
  3. **Tiêu chí 3:** Tối ưu hóa băng thông bộ nhớ Grayscale.
- Giải quyết triệt để nút thắt lớn nhất của hệ thống: Đưa thời gian OCR từ 3s – 8s xuống mức dưới 1 giây mà vẫn đảm bảo đọc chính xác 100% biển số Việt Nam mà không cần fine-tune lại mô hình.

---

## 2. 💡 Phát Hiện Bước Ngoặt (Root Cause & Breakthrough)
Khi "mổ xẻ" mã nguồn bên trong của `rapidocr_onnxruntime`:
* **Nguyên nhân gốc rễ gây chậm:** Mặc định RapidOCR đặt `limit_side_len = 736` và `limit_type = 'min'`.
* **Hậu quả:** Khi YOLO cắt được ảnh biển số nhỏ (ví dụ $120 \times 60\text{px}$), RapidOCR **tự động phóng to (upscale) ảnh lên gấp 5 – 10 lần** để cạnh nhỏ nhất đạt $736\text{px}$. Điều này ép mạng Text Detection (DBNet) phải tính toán trên một ma trận hàng triệu điểm ảnh khổng lồ một cách lãng phí!
* **Giải pháp kỹ thuật được áp dụng:**
  1. Chuẩn hóa chiều cao vùng crop: `target_height = 96px`.
  2. Tinh chỉnh cấu hình RapidOCR: `det_limit_side_len = 320`, `det_limit_type = 'max'` (chống upscale ROI giả tạo).
  3. Loại bỏ phép giãn nở hình thái học: `det_donot_use_dilation = True`.
  4. Loại bỏ phân loại góc xoay: `use_cls = False`.

---

## 3. 💻 Cách Chạy Thực Nghiệm (How to Run)

Chạy script benchmark chuyên biệt cho Sprint 9 trên container Docker mô phỏng 3 thế hệ phần cứng:

```bash
# 1. Chạy trên cấu hình Raspberry Pi 4 (4 Cores, 2GB RAM):
docker run --rm --cpus=4.0 --memory=2048m -v "%cd%:/app" data-alpr-rpi4-simulation:latest python evaluation/benchmark_ocr_roi.py --profile "Raspberry Pi 4" --threads 4 --output_json output/bench_ocr_pi4.json

# 2. Chạy trên cấu hình Raspberry Pi 3 (2.5 Cores, 1GB RAM):
docker run --rm --cpus=2.5 --memory=1024m -v "%cd%:/app" data-alpr-rpi4-simulation:latest python evaluation/benchmark_ocr_roi.py --profile "Raspberry Pi 3" --threads 2 --output_json output/bench_ocr_pi3.json

# 3. Chạy trên cấu hình Raspberry Pi 2 (1.5 Cores, 512MB RAM):
docker run --rm --cpus=1.5 --memory=512m -v "%cd%:/app" data-alpr-rpi4-simulation:latest python evaluation/benchmark_ocr_roi.py --profile "Raspberry Pi 2" --threads 1 --output_json output/bench_ocr_pi2.json
```

---

## 4. 📊 Kết Quả Thu Được Tại Thời Điểm Đó (OCR Optimization Metrics)

### A. Hiệu Quả Tối Ưu Hóa Giai Đoạn OCR (RapidOCR):
| Cấu Hình Thiết Bị | OCR Mặc Định (Baseline) | OCR Bỏ Góc Xoay (Step 1) | OCR Tối Ưu Toàn Diện (H=96 + limit=320 + No Dilation) | Thời Gian Tiết Kiệm | Tỷ Lệ Tăng Tốc (Speedup) | Độ Chính Xác Đọc Biển |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raspberry Pi 4**<br>*(4 Cores / 2048 MB)* | 3,322.86 ms | 2,657.52 ms | **568.42 ms** | 🔻 **Tiết kiệm 2.75 giây** | 🚀 **5.85×** | ✅ Khớp 100% (`68G1-668.86`, `51A-222.22`) |
| **Raspberry Pi 3**<br>*(2.5 Cores / 1024 MB)*| 5,081.26 ms | 4,452.05 ms | **1,040.14 ms** | 🔻 **Tiết kiệm 4.04 giây** | 🚀 **4.89×** | ✅ Khớp 100% (`68G1-668.86`, `51A-222.22`) |
| **Raspberry Pi 2**<br>*(1.5 Cores / 512 MB)* | 8,240.25 ms | 7,701.12 ms | **1,749.40 ms** | 🔻 **Tiết kiệm 6.49 giây** | 🚀 **4.71×** | ✅ Khớp 100% (`68G1-668.86`, `51A-222.22`) |

### B. Bảng Tổng Kết Hiệu Năng Toàn Bộ Pipeline End-to-End (Sau Tất Cả Các Phase):
| Cấu Hình Thiết Bị | Pipeline Ban Đầu (FP32 PyTorch 640 + OCR Full) | Pipeline Tối Ưu Hiện Tại (INT8 480 + RapidOCR Tối Ưu) | Thời Gian Xử Lý Tổng | Tốc Độ Khung Hình | Tổng Tỷ Lệ Tăng Tốc Toàn Hệ Thống |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raspberry Pi 4** | 4,184.24 ms | **~668.74 ms** | 🔻 Nhanh hơn **3.5 giây** | **~1.50 FPS** | 🚀 **6.26×** |
| **Raspberry Pi 3** | 6,597.99 ms | **~1,154.42 ms** | 🔻 Nhanh hơn **5.4 giây** | **~0.87 FPS** | 🚀 **5.71×** |
| **Raspberry Pi 2** | 10,856.81 ms | **~1,900.98 ms** | 🔻 Nhanh hơn **8.9 giây** | **~0.53 FPS** | 🚀 **5.71×** |

---

## 5. 💡 Trạng Thái Mã Nguồn Chính Thức
Toàn bộ cải tiến tối ưu của Phase 5 đã được nạp trực tiếp vào mã nguồn sản xuất tại [pipeline/recognizer_rapid.py](file:///d:/research/data/pipeline/recognizer_rapid.py):
- Tự động downscale vùng ROI về chiều cao $H=96$ khi phát hiện ảnh cắt quá lớn.
- Khởi tạo engine với `det_limit_side_len = 320`, `det_limit_type = 'max'`, `det_donot_use_dilation = True` và `use_cls = False`.
- Khi người dùng gọi `run_demo.py` hoặc chạy container, các tối ưu này tự động có hiệu lực ngay lập tức.
