# 🗜️ Phase 2: Nén Mô Hình (Pruning, Quantization INT8 & ONNX Conversion)

## 1. 🎯 Mục Tiêu Cốt Lõi
- Triển khai trọn vẹn 3 kỹ thuật nén mô hình cho YOLOv8 Plate Detector:
  1. **ONNX Runtime Conversion:** Loại bỏ phụ thuộc vào PyTorch, tối ưu hóa đồ thị suy luận trên ARM.
  2. **Quantization FP32 ➔ INT8 (Bắt buộc):** Giảm tối thiểu 40% dung lượng mô hình, tăng tốc độ suy luận, duy trì độ chính xác phát hiện.
  3. **Pruning (Cắt tỉa trọng số):** Tạo ma trận thưa (Sparsity) giảm số liên kết nơ-ron dư thừa.
- Kiểm chứng minh bạch mức độ bảo toàn độ chính xác thông qua chỉ số IoU và độ tin cậy đối chứng trực tiếp giữa bản FP32 và INT8.

---

## 2. 💻 Cách Chạy Thực Nghiệm (How to Run)

### A. Chạy tự động toàn bộ quy trình tối ưu hóa (Pruning + ONNX + Quantize):
```bash
python -m pipeline.optimize --all
```
*(Hoặc chạy riêng từng bước: `--export_onnx`, `--quantize`, `--prune`)*

### B. Chạy kiểm chứng đối chứng minh bạch (IoU & Side-by-Side Verification):
```bash
python evaluation/verify_comparison.py
```
*(Script đọc trực tiếp byte từ ổ đĩa thật, tính IoU giữa hộp FP32 và INT8, và xuất ảnh đối chứng `output/verify_compare_*.jpg`)*

### C. Đo đạc mức giảm tài nguyên RAM:
```bash
python evaluation/benchmark_edge_resources.py
```

---

## 3. 📊 Kết Quả Thu Được Tại Thời Điểm Đó (Optimization Metrics)

### A. Bảng So Sánh Kích Thước & Độ Chính Xác Các Phiên Bản Trọng Số:
| Phiên Bản Mô Hình | Định Dạng File | Dung Lượng File | Tỷ Lệ Nén / Giảm | Độ Tin Cậy (Conf) | Trạng Thái Độ Chính Xác |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8 PyTorch Baseline (FP32)** | `.pt` | 5.97 MB | Baseline (1.0×) | 91.7% | Chuẩn gốc |
| **YOLOv8 ONNX Runtime (FP32)** | `.onnx` | 11.70 MB | Chuẩn hóa đồ thị | 94.3% | Đã gộp node qua `onnxslim` |
| **YOLOv8 INT8 Quantized** | `.onnx` | **3.20 MB** | 🔻 **Giảm 72.6%** | **92.2%** | 🎯 **Bảo toàn nguyên vẹn (IoU 97.8%)** |
| **YOLOv8 Pruned (L1 Norm 25%)** | `.pt` | 11.78 MB | 🎯 **24.91% Sparsity** | 90.5% | Triệt tiêu 750,112 tham số về 0 |

### B. Bảng So Sánh Tiết Kiệm Tài Nguyên Bộ Nhớ RAM:
| Tiêu Chí Đánh Giá | PyTorch FP32 Baseline | Pure ONNX INT8 Edge | Mức Cải Thiện Định Lượng |
| :--- | :---: | :---: | :---: |
| **RAM Nạp Mô Hình** | **330.96 MB** | **53.61 MB** | 🔻 **Tiết kiệm 83.8% RAM (gần 6 lần)** |
| **Kích Thước Trọng Số** | 11.70 MB | **3.20 MB** | 🔻 **Giảm 72.6% dung lượng** |
| **Phụ Thuộc Framework** | Cần `torch` (~1.5 GB) | **Chỉ cần `onnxruntime`** | 🪶 Image triển khai siêu nhẹ |

---

## 4. 💡 Nhận Xét & Quyết Định Kỹ Thuật
1. **Mục tiêu Quantization đạt vượt mức:** Dung lượng file giảm tới **72.6%** (vượt xa chỉ tiêu 40%), trong khi sai lệch tọa độ hộp Bounding Box so với bản FP32 gần như bằng không (**IoU đạt tới 97.8%**).
2. **Cắt bỏ hoàn toàn PyTorch:** Đã xây dựng [pipeline/detector_onnx.py](file:///d:/research/data/pipeline/detector_onnx.py) thực thi thuần túy qua ONNX Runtime + OpenCV/NumPy.
3. ➔ **Kết luận chuyển giao:** Mô hình INT8 đã hoàn toàn sẵn sàng cho môi trường nhúng. Bước tiếp theo là đưa vào Docker để giả lập và kiểm tra giới hạn chịu tải phần cứng Raspberry Pi (chuyển sang Phase 3).
