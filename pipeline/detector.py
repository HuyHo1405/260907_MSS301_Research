import os
import time
import cv2
from ultralytics import YOLO


class YOLOPlateDetector:
    """
    Detector phát hiện vị trí biển số xe sử dụng mô hình YOLOv8.
    """
    def __init__(self, weights_path='weights/best.pt', conf_threshold=0.25, device=None):
        resolved_path = weights_path
        if not os.path.exists(resolved_path):
            alt_path = os.path.basename(weights_path)
            if os.path.exists(alt_path):
                resolved_path = alt_path
            else:
                raise FileNotFoundError(f"Không tìm thấy file weights YOLO: '{weights_path}'")

        self.weights_path = resolved_path
        self.conf_threshold = conf_threshold
        self.model = YOLO(resolved_path)
        if device is not None:
            self.model.to(device)

    def detect(self, img):
        """
        Phát hiện khung biển số trong ảnh.
        Input: img (numpy.ndarray BGR)
        Output:
          - boxes: List các toạ độ [x1, y1, x2, y2]
          - confidences: List các độ tin cậy float
          - rois: List các ảnh con cắt vùng biển số
          - duration_ms: Thời gian suy luận (ms)
        """
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        results = self.model(img, conf=self.conf_threshold, verbose=False)
        duration_ms = (time.perf_counter() - t0) * 1000

        boxes = []
        confidences = []
        rois = []

        if results and len(results[0].boxes) > 0:
            for b in results[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                # Giới hạn trong kích thước ảnh
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                conf = float(b.conf[0].cpu().numpy())

                roi = img[y1:y2, x1:x2]
                if roi.size > 0:
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(conf)
                    rois.append(roi)

        return boxes, confidences, rois, duration_ms
