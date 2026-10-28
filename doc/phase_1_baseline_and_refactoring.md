# 📋 Phase 1: Chuẩn Hóa Codebase Two-Stage & Thiết Lập Baseline

## 1. 🎯 Mục Tiêu Cốt Lõi
- Tái cấu trúc toàn bộ mã nguồn phân tán cũ thành thư viện nghiên cứu dạng module độc lập (`pipeline/`).
- Xây dựng kiến trúc Plug-and-Play cho phép hoán đổi linh hoạt giữa các Detector và OCR Engine.
- Thiết lập kịch bản đo đạc baseline ban đầu (Latency, FPS, RAM) trước khi tiến hành nén mô hình.

---

## 2. 🏗️ Kiến Trúc Được Chuẩn Hóa
Hệ thống được tổ chức thành 2 tầng xử lý nối tiếp (Two-Stage ALPR):
1. **Stage 1 (Detector):** YOLOv8 phát hiện tọa độ Bounding Box của biển số xe máy và ô tô.
2. **Stage 2 (Recognizer):** Cắt vùng ảnh biển số (ROI) và nạp vào OCR Engine để đọc chuỗi ký tự.
3. **Post-processing:** Áp dụng heuristic & biểu thức chính quy (Regex) để chuẩn hóa định dạng biển số xe Việt Nam (biển dài 1 dòng và biển vuông 2 dòng).

---

## 3. 💻 Cách Chạy Thực Nghiệm (How to Run)

### A. Chạy demo nhận diện trên 1 ảnh:
```bash
python run_demo.py --weights weights/best.pt --image samples/test2.jpg --no_show
```

### B. Đo đạc tốc độ suy luận Baseline (Speed Benchmark):
```bash
python evaluation/benchmark_speed.py
```

---

## 4. 📊 Kết Quả Thu Được Tại Thời Điểm Đó (Baseline Metrics)

*Đo đạc trên môi trường host phát triển:*

| Thành Phần Pipeline | Công Nghệ Sử Dụng | Thời Gian Xử Lý (Latency) | Dung Lượng Trọng Số | RAM Chiếm Dụng |
| :--- | :--- | :---: | :---: | :---: |
| **Detector** | YOLOv8n PyTorch (`best.pt`) | ~134.0 ms | 5.97 MB | **330.96 MB** |
| **OCR Baseline 1** | RapidOCR (PaddleOCR ONNX) | ~1,600.0 ms | ~15.0 MB | ~120.0 MB |
| **OCR Baseline 2** | LPRNet PyTorch CTC (`lprnet.pth`)| ~45.0 ms | 1.86 MB | Kèm PyTorch |
| **Toàn bộ Pipeline** | Two-Stage End-to-End | ~1,734.0 ms (~0.58 FPS)| Tổng file weights | **> 450 MB RAM** |

---

## 5. 💡 Nhận Xét & Hạn Chế Phát Hiện Được
1. **Phụ thuộc quá nặng:** Môi trường cần cài đặt toàn bộ gói `torch` và `ultralytics` (>1.5 GB ổ đĩa).
2. **Chiếm dụng RAM lớn:** Bản PyTorch FP32 ngốn tới **330.96 MB RAM** chỉ để nạp mô hình detector. Khi triển khai lên Raspberry Pi cũ (256MB hoặc 512MB RAM), hệ thống sẽ bị Linux OOM Killer tiêu diệt ngay lập tức.
3. ➔ **Kết luận chuyển giao:** Bắt buộc phải chuyển đổi mô hình sang ONNX Runtime siêu nhẹ và thực hiện Quantization INT8 để cắt bỏ hoàn toàn sự phụ thuộc vào PyTorch (chuyển sang Phase 2).
