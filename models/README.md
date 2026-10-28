# 🤖 Mô Hình Trí Tuệ Nhân Tạo (AI Models)

Thư mục chứa các cấu hình (configs) và quy chuẩn lưu trữ weights / checkpoints mô hình.

> **Lưu ý:** Không commit trực tiếp các file model weights lớn (`.pt`, `.onnx`, `.engine`, `.tflite` > 50MB) lên Git repository. Hãy tải weights từ Google Drive / S3 / Hugging Face theo link quy định hoặc lưu tại `models/weights/` (được đưa vào `.gitignore`).

---

## 📋 Danh Sách Mô Hình Dự Kiến

| Nhiệm vụ | Kiến trúc cơ sở | Input Size | Định dạng tối ưu | Ghi chú |
| :--- | :--- | :--- | :--- | :--- |
| **Phát hiện xe & biển số** | YOLOv8n / YOLOv11n | 640x640 | ONNX / INT8 | Detect 2 classes: `vehicle`, `plate` |
| **Cắt & phân đoạn biển số** | YOLOv8n-pose / LPD-Net | 320x320 | ONNX FP16 | Canh chỉnh 4 góc biển số nghiêng |
| **Nhận diện ký tự (OCR)** | CRNN / PaddleOCR (SVTR) | 32x100 / 48x320 | ONNX INT8 | Đọc ký tự chữ & số Việt Nam |
| **Nhận diện khuôn mặt** | MobileFaceNet / ArcFace | 112x112 | ONNX FP16 | Embedding 512 chiều đối soát |
