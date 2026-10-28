# 🐳 Phase 3: Giả Lập Môi Trường Raspberry Pi Trong Docker & Stress-Test Tài Nguyên

## 1. 🎯 Mục Tiêu Cốt Lõi
- Đóng gói toàn bộ Edge Pipeline vào container Docker trên nền tảng **Debian Bookworm** (hệ điều hành chuẩn của Raspberry Pi OS 64-bit).
- Thiết lập giới hạn phần cứng nghiêm ngặt bằng cgroups (`--cpus` và `--memory`) để mô phỏng chính xác năng lực thực tế của các dòng Raspberry Pi.
- Tiến hành **Stress-test** bóp kiệt tài nguyên từ dòng cao cấp (Pi 4) xuống dòng cấp thấp (Pi 3, Pi Zero 2W, và mức cực đoan 256 MB RAM) để tìm ra điểm sụp đổ (Break-point) của hệ thống.

---

## 2. 💻 Cách Chạy Thực Nghiệm (How to Run)

### A. Build Docker Image môi trường Edge siêu nhẹ (< 350 MB):
```bash
docker build -t data-alpr-rpi4-simulation:latest -f Dockerfile.rpi4 .
```

### B. Chạy mô phỏng Raspberry Pi 4 tiêu chuẩn (Docker Compose):
```bash
docker compose -f docker-compose.rpi4.yml run --rm alpr-rpi4-simulation
```

### C. Chạy Stress-test theo từng thế hệ phần cứng:
```bash
# 1. Tier 1: Raspberry Pi 4 (4 Nhân, 2GB RAM)
docker run --rm --cpus=4.0 --memory=2048m -v "%cd%:/app" data-alpr-rpi4-simulation:latest python run_demo.py --weights weights/best_int8.onnx --no_show

# 2. Tier 2: Raspberry Pi 3 B+ (2 Nhân, 1GB RAM)
docker run --rm --cpus=2.0 --memory=1024m -v "%cd%:/app" data-alpr-rpi4-simulation:latest python run_demo.py --weights weights/best_int8.onnx --no_show

# 3. Tier 3: Raspberry Pi Zero 2 W / Pi 2 (1 Nhân, 512MB RAM)
docker run --rm --cpus=1.0 --memory=512m -v "%cd%:/app" data-alpr-rpi4-simulation:latest python run_demo.py --weights weights/best_int8.onnx --no_show

# 4. Tier 4: Cực đoan (1 Nhân CPU, 256MB RAM)
docker run --rm --cpus=1.0 --memory=256m -v "%cd%:/app" data-alpr-rpi4-simulation:latest python run_demo.py --weights weights/best_int8.onnx --no_show
```

---

## 3. 📊 Kết Quả Thu Được Tại Thời Điểm Đó (Stress-Test Metrics)

| Cấu Hình Mô Phỏng | Phần Cứng Tương Đương | Giới Hạn Container | Detector (YOLOv8 INT8) | OCR Engine (RapidOCR) | Tổng Thời Gian Pipeline | Trạng Thái RAM (OOM) | Kết Quả Biển Số Đọc Được |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 1: Pi 4 Profile** | Raspberry Pi 4 (Quad A72) | `4.0 Cores / 2048 MB` | **281.34 ms** | 5,060.00 ms | **5.34 s** | ✅ Mượt mà | `68G1-668.86` (92.2%) |
| **Tier 2: Pi 3 Profile** | Raspberry Pi 3 B+ (Quad A53) | `2.0 Cores / 1024 MB` | **453.68 ms** | 9,486.00 ms | **9.94 s** | ✅ Không lỗi | `68G1-668.86` (92.2%) |
| **Tier 3: Pi Zero 2W** | Pi Zero 2 W / Pi 2 (1-2 Cores)| `1.0 Core  /  512 MB` | **1,095.84 ms** | 16,574.00 ms | **17.67 s** | ✅ Sống sót an toàn | `68G1-668.86` (92.2%) |
| **Tier 4: Extreme Legacy**| Pi 1 / Pi Zero v1 (Single Core)| `1.0 Core  /  256 MB` | **1,000.90 ms** | 33,406.85 ms | **34.40 s** | 🛡️ **Không bị OOM Kill** | `68G1-668.86` (92.2%) |

---

## 4. 💡 Nhận Xét & Phát Hiện Cốt Lõi (Key Findings)
1. **Giải quyết triệt để nút thắt RAM (Memory-bound):** 
   * Bản mô hình Pure ONNX INT8 hoạt động an toàn ngay cả ở mức **256 MB RAM** (mức mà bản PyTorch FP32 gốc chắc chắn 100% bị sập trước khi nạp xong mô hình).
2. **Xác định đúng tâm điểm nút thắt mới (Compute-bound):**
   * YOLOv8 INT8 suy luận rất nhanh: chỉ dao động từ **~280 ms (Pi 4)** đến **~1.0 s (Pi Zero 1 nhân)**.
   * Nhưng **PaddleOCR chiếm tới 85% – 95% thời gian toàn pipeline** (từ 5.3s trên Pi 4 kéo dài tới 34.4s trên Pi 1-core).
3. ➔ **Kết luận chuyển giao:** Vấn đề không còn nằm ở bộ nhớ, mà nằm ở **năng lực tính toán (FLOPs) của CPU ARM**. Cần giải quyết bài toán Compute-bound bằng việc tối ưu hóa độ phân giải đầu vào và cấu hình luồng SIMD (chuyển sang Phase 4).
