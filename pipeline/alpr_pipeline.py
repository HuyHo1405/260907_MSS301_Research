import os
import time
import cv2
import numpy as np

try:
    from .detector import YOLOPlateDetector
except ImportError:
    YOLOPlateDetector = None

from .detector_onnx import ONNXPlateDetector
from .recognizer_rapid import RapidOCRRecognizer

try:
    from .recognizer_lprnet import LPRNetRecognizer
except ImportError:
    LPRNetRecognizer = None

from .postprocess import format_vietnamese_plate


class ALPRPipeline:
    """
    Core Pipeline nghiên cứu nhận diện biển số xe (Two-Stage: Detection + Recognition).
    Thiết kế dạng Plug-and-Play để dễ dàng hoán đổi giữa các engine OCR và đo đạc benchmark.
    """
    def __init__(self, detector_weights='weights/best.pt', engine='paddle', lpr_weights='weights/lprnet.pth', conf_threshold=0.25, device=None):
        print(f"🔧 Khởi tạo ALPRPipeline [Detector: YOLOv8 | OCR Engine: {engine.upper()}]...")
        if detector_weights.endswith('.onnx'):
            self.detector = ONNXPlateDetector(weights_path=detector_weights, conf_threshold=conf_threshold)
        else:
            if YOLOPlateDetector is None:
                raise ImportError("Cần cài đặt ultralytics và torch để chạy mô hình PyTorch .pt. Trong môi trường Edge/Docker siêu nhẹ, vui lòng nạp weights dạng .onnx (VD: weights/best_int8.onnx)!")
            self.detector = YOLOPlateDetector(weights_path=detector_weights, conf_threshold=conf_threshold, device=device)
        self.engine_type = engine.lower()

        if self.engine_type == 'paddle':
            self.recognizer = RapidOCRRecognizer()
        elif self.engine_type == 'lprnet':
            if LPRNetRecognizer is None:
                raise ImportError("Cần cài đặt PyTorch để chạy engine LPRNet. Trong môi trường Edge/Docker siêu nhẹ, hãy dùng engine='paddle'!")
            self.recognizer = LPRNetRecognizer(weights_path=lpr_weights, device=device)
        else:
            raise ValueError(f"Engine không hỗ trợ: '{engine}'. Chọn 'paddle' hoặc 'lprnet'.")

        print("✅ ALPRPipeline đã sẵn sàng thực nghiệm!")

    def predict(self, image_input):
        """
        Thực hiện dự đoán end-to-end trên 1 ảnh.
        Input: image_input (str đường dẫn ảnh hoặc numpy.ndarray BGR)
        Output:
          - annotated_img: Ảnh đã vẽ bounding box và biển số
          - results: Danh sách chi tiết từng biển số phát hiện được
          - latency: Dict thống kê thời gian từng giai đoạn và FPS
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Không thể đọc ảnh từ: {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise TypeError("Đầu vào phải là đường dẫn ảnh (str) hoặc cv2 image array (numpy.ndarray).")

        annotated_img = img.copy()
        results = []

        # 1. Detection Stage
        boxes, confs, rois, det_time_ms = self.detector.detect(img)

        total_ocr_time_ms = 0.0

        if len(boxes) > 0:
            for idx, (box, conf, roi) in enumerate(zip(boxes, confs, rois)):
                # 2. Recognition Stage
                raw_ocr, formatted_plate, ocr_time_ms = self.recognizer.recognize(roi)
                total_ocr_time_ms += ocr_time_ms

                results.append({
                    'plate_id': idx + 1,
                    'box': box,
                    'detector_conf': conf,
                    'raw_ocr': raw_ocr,
                    'formatted_plate': formatted_plate,
                    'ocr_time_ms': ocr_time_ms
                })

                # 3. Vẽ nhãn lên ảnh
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = formatted_plate if formatted_plate else "PLATE"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cv2.rectangle(annotated_img, (x1, max(0, y1 - 28)), (x1 + tw + 10, y1), (0, 255, 0), -1)
                cv2.putText(annotated_img, label, (x1 + 5, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        else:
            # Fallback: Nếu không bắt được khung, thử OCR toàn cảnh
            raw_ocr, formatted_plate, ocr_time_ms = self.recognizer.recognize(img)
            total_ocr_time_ms += ocr_time_ms
            if formatted_plate:
                results.append({
                    'plate_id': 1,
                    'box': [0, 0, img.shape[1], img.shape[0]],
                    'detector_conf': 0.0,
                    'raw_ocr': raw_ocr,
                    'formatted_plate': formatted_plate,
                    'ocr_time_ms': ocr_time_ms
                })
                cv2.putText(annotated_img, f"Biển số: {formatted_plate}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        total_time_ms = det_time_ms + total_ocr_time_ms
        fps = (1000.0 / total_time_ms) if total_time_ms > 0 else 0.0

        latency = {
            'det_time_ms': det_time_ms,
            'ocr_time_ms': total_ocr_time_ms,
            'total_time_ms': total_time_ms,
            'fps': fps
        }

        return annotated_img, results, latency
