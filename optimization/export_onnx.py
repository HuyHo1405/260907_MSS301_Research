"""
Script xuất mô hình PyTorch / Ultralytics YOLO sang ONNX tối ưu cho Edge Deployment.
Hỗ trợ đơn giản hóa đồ thị tính toán (onnxsim) và xuất dạng bán chính xác (FP16).
"""

import argparse
import sys
from pathlib import Path


def export_model(weights_path: str, imgsz: int = 640, half: bool = False, simplify: bool = True):
    weights = Path(weights_path)
    if not weights.exists():
        print(f"[ERROR] Không tìm thấy file trọng số: {weights_path}")
        sys.exit(1)

    print(f"[INFO] Bắt đầu xuất mô hình: {weights.name}")
    print(f"[INFO] Kích thước đầu vào: {imgsz}x{imgsz} | Half precision (FP16): {half} | Simplify: {simplify}")

    try:
        from ultralytics import YOLO
        model = YOLO(str(weights))
        success = model.export(
            format="onnx",
            imgsz=imgsz,
            half=half,
            simplify=simplify,
            opset=12,
            dynamic=False,
        )
        print(f"[SUCCESS] Xuất mô hình thành công: {success}")
    except ImportError:
        print("[WARNING] Thư viện 'ultralytics' chưa được cài đặt. Vui lòng chạy: pip install ultralytics")
    except Exception as e:
        print(f"[ERROR] Quá trình xuất thất bại: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to optimized ONNX format")
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn file .pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào (mặc định 640)")
    parser.add_argument("--half", action="store_true", help="Sử dụng FP16 Half precision")
    parser.add_argument("--simplify", action="store_true", default=True, help="Đơn giản hóa đồ thị tính toán với onnxsim")

    args = parser.parse_args()
    export_model(args.weights, args.imgsz, args.half, args.simplify)
