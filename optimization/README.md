# ⚡ Kỹ Thuật Tối Ưu Hóa Mô Hình (Model Optimization)

Tài liệu và công cụ tối ưu hóa mô hình từ PyTorch sang các định dạng suy luận thời gian thực cho thiết bị Edge.

---

## 🛠️ Các Kỹ Thuật Áp Dụng

1. **Export ONNX & Graph Simplification:**
   - Sử dụng `onnxsim` để gộp các layer liên tiếp (Constant Folding, BatchNorm Folding).
   - Tối ưu biểu diễn đồ thị tính toán.

2. **Lượng tử hóa (Quantization):**
   - **FP16 (Half Precision):** Tăng tốc gấp 1.5x - 2x trên GPU/NPU hỗ trợ FP16, giảm 50% dung lượng bộ nhớ mà gần như không giảm độ chính xác.
   - **INT8 Post-Training Quantization (PTQ):** Lượng tử hóa trọng số và activation về số nguyên 8-bit, sử dụng tập dữ liệu hiệu chuẩn (Calibration dataset) 100-300 ảnh thực tế tại cổng an ninh.

3. **Runtime Engines:**
   - **ONNX Runtime (CPU/OpenVINO/CUDA):** Đa nền tảng, dễ đóng gói container trên x86 và ARM64.
   - **TensorRT (NVIDIA Jetson):** Tối đa hóa FPS và tối thiểu hóa độ trễ trên chip Jetson.
   - **TFLite / NCNN:** Dành riêng cho vi xử lý ARM không có GPU rời (như Raspberry Pi 4).

---

## 🚀 Sử Dụng Script Export

```bash
# Xuất mô hình YOLO sang ONNX tối ưu
python export_onnx.py --weights ../models/best.pt --imgsz 640 --half --simplify
```
