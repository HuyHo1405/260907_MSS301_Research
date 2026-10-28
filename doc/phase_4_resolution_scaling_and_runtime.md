# ⚡ Phase 4: Tinh Chỉnh Độ Phân Giải (Resolution Scaling) & Tối Ưu Hóa Runtime SIMD

## 1. 🎯 Mục Tiêu Cốt Lõi
- Giải quyết bài toán **Compute-Bound** trên CPU ARM bằng 2 hướng can thiệp an toàn (không can thiệp cắt tỉa cấu trúc mạng blackbox, không thay thế PaddleOCR để tránh phải fine-tune):
  1. **Phương pháp 3 (Resolution Scaling):** Giảm số phép tính ma trận FLOPs theo hàm bậc hai $O(W \times H)$ bằng cách hạ kích thước ảnh đầu vào của Detector (`640 ➔ 512 ➔ 480 ➔ 384 ➔ 320`).
  2. **Phương pháp 4 (Runtime & Multi-Threading SIMD):** Khảo sát cấu hình luồng ONNX Runtime (`intra_op_num_threads = 1, 2, 4`) và loại bỏ bước phân loại góc xoay ảnh (`use_cls=False`) trong RapidOCR.

---

## 2. 💻 Cách Chạy Thực Nghiệm (How to Run)

Export mô hình YOLOv8 sang dạng **Dynamic Shapes** và thực thi script benchmark đo đạc đối chứng:

```bash
# 1. Chạy trên cấu hình Raspberry Pi 4 (4 Cores, 2GB RAM):
docker run --rm --cpus=4.0 --memory=2048m -v "%cd%:/app" data-alpr-rpi4-simulation:latest python evaluation/benchmark_scaling_hardware.py --profile "Raspberry Pi 4" --threads 4 --output_json output/bench_pi4.json

# 2. Chạy trên cấu hình Raspberry Pi 3 (2.5 Cores, 1GB RAM):
docker run --rm --cpus=2.5 --memory=1024m -v "%cd%:/app" data-alpr-rpi4-simulation:latest python evaluation/benchmark_scaling_hardware.py --profile "Raspberry Pi 3" --threads 2 --output_json output/bench_pi3.json

# 3. Chạy trên cấu hình Raspberry Pi 2 (1.5 Cores, 512MB RAM):
docker run --rm --cpus=1.5 --memory=512m -v "%cd%:/app" data-alpr-rpi4-simulation:latest python evaluation/benchmark_scaling_hardware.py --profile "Raspberry Pi 2" --threads 1 --output_json output/bench_pi2.json
```

---

## 3. 📊 Kết Quả Thu Được Tại Thời Điểm Đó (Resolution & Threading Metrics)

### A. Khảo Sát Tốc Độ & Độ Chính Xác Theo Độ Phân Giải (Phương Pháp 3):
| Cấu Hình Phần Cứng | Độ Phân Giải | Độ Trễ (Latency ms) | Tốc Độ (FPS) | Tỷ Lệ Tăng Tốc | Độ Tin Cậy (Conf) | Kết Quả Đọc Biển Số |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raspberry Pi 4**<br>*(4 Cores / 2GB RAM)* | `640x640` (Gốc) | 187.43 ms | 5.34 FPS | 1.00× (Baseline) | 59.2% | ✅ Đọc đúng 100% |
| | `512x512` | 147.73 ms | 6.77 FPS | **1.27×** | 60.5% | ✅ Đọc đúng 100% |
| | **`480x480` (Sweet Spot)** | **114.17 ms** | **8.76 FPS** | 🚀 **1.64× (+64%)** | **60.3%** | ✅ **Đọc đúng 100%** |
| | **`384x384` (Tốc độ cao)** | **68.64 ms** | **14.57 FPS** | ⚡ **2.73×** | **60.3%** | ✅ **Đọc đúng 100%** |
| | `320x320` | 59.94 ms | 16.68 FPS | 3.13× | ⚠️ 41.7% (Tụt Conf) | ⚠️ Bắt đầu mất nét cạnh |
| **Raspberry Pi 3**<br>*(2.5 Cores / 1GB RAM)* | `640x640` (Gốc) | 236.63 ms | 4.23 FPS | 1.00× (Baseline) | 59.2% | ✅ Đọc đúng 100% |
| | `512x512` | 137.69 ms | 7.26 FPS | **1.72×** | 60.5% | ✅ Đọc đúng 100% |
| | **`480x480` (Sweet Spot)** | **114.33 ms** | **8.75 FPS** | 🚀 **2.07× (+107%)** | **60.3%** | ✅ **Đọc đúng 100%** |
| | **`384x384` (Tốc độ cao)** | **77.39 ms** | **12.92 FPS** | ⚡ **3.06×** | **60.3%** | ✅ **Đọc đúng 100%** |
| | `320x320` | 60.06 ms | 16.65 FPS | 3.94× | ⚠️ 41.7% (Tụt Conf) | ⚠️ Bắt đầu mất nét cạnh |
| **Raspberry Pi 2**<br>*(1.5 Cores / 512MB RAM)* | `640x640` (Gốc) | 296.99 ms | 3.37 FPS | 1.00× (Baseline) | 59.2% | ✅ Đọc đúng 100% |
| | `512x512` | 213.56 ms | 4.68 FPS | **1.39×** | 60.5% | ✅ Đọc đúng 100% |
| | **`480x480` (Sweet Spot)** | **160.54 ms** | **6.23 FPS** | 🚀 **1.85× (+85%)** | **60.3%** | ✅ **Đọc đúng 100%** |
| | **`384x384` (Tốc độ cao)** | **102.22 ms** | **9.78 FPS** | ⚡ **2.91×** | **60.3%** | ✅ **Đọc đúng 100%** |
| | `320x320` | 70.57 ms | 14.17 FPS | 4.21× | ⚠️ 41.7% (Tụt Conf) | ⚠️ Bắt đầu mất nét cạnh |

### B. Hiệu Quả Mở Rộng Đa Luồng (Thread Scaling trên Detector 480x480):
| Cấu Hình | 1 Thread | 2 Threads | 4 Threads | Luồng Tối Ưu Nhất | Nhận Định Kỹ Thuật |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Pi 4 (4 Cores)** | 154.54 ms | 118.12 ms | **100.32 ms** | **4 Threads** | Khai thác tối đa 4 nhân Cortex-A72 thực |
| **Pi 3 (2.5 Cores)** | 155.56 ms | **114.28 ms** | 174.01 ms | **2 Threads** | 4 threads bị chậm do chi phí Context-Switching |
| **Pi 2 (1.5 Cores)** | **151.58 ms** | 161.50 ms | 299.32 ms | **1 Thread** | Ép đa luồng trên chip ít nhân gây nghẽn hàng đợi |

### C. Tối Ưu Hóa Bước Phân Loại Góc Xoay RapidOCR (`use_cls=False`):
- **Pi 4:** Giảm từ 3,996 ms xuống 3,122 ms (**tiết kiệm 874 ms**).
- **Pi 3:** Giảm từ 6,361 ms xuống 5,197 ms (**tiết kiệm 1,163 ms - hơn 1.1s**).
- **Pi 2:** Giảm từ 10,559 ms xuống 8,279 ms (**tiết kiệm 2,280 ms - hơn 2.2s**).

---

## 4. 💡 Nhận Xét & Quyết Định Kỹ Thuật
1. **Tìm ra Điểm cân bằng Pareto (Sweet Spot):** Cấu hình **`480x480`** và **`384x384`** giúp detector tăng tốc từ 1.64x đến 3.06x mà độ tin cậy và OCR vẫn chính xác 100%. Không nên hạ xuống 320x320 vì độ tin cậy bắt đầu tụt.
2. **Chiến lược phân luồng phần cứng:** Bắt buộc cấu hình số luồng ONNX Runtime khớp với số nhân thực tế (Pi 4: 4 luồng; Pi 3: 2 luồng; Pi 2: 1 luồng).
3. ➔ **Kết luận chuyển giao:** Mặc dù Detector đã tối ưu xuống ~100ms, nhưng OCR vẫn đang mất 3s – 5s. Cần rà soát kỹ quy trình tiền xử lý bên trong RapidOCR để tìm nguyên nhân gốc rễ (chuyển sang Phase 5).
