"""
==============================================================================
⚡ PURE ONNX RUNTIME DETECTOR (ZERO-PYTORCH / ZERO-ULTRALYTICS)
==============================================================================
Detector độc lập siêu nhẹ chuyên dụng cho Edge / Raspberry Pi (ARM):
  - KHÔNG cần cài đặt PyTorch (~1.5GB) hay Ultralytics.
  - Chỉ phụ thuộc vào: onnxruntime, opencv-python, numpy.
  - Hỗ trợ mô hình FP32 (.onnx) và INT8 Quantized (.onnx).
  - Tối ưu đa luồng cho CPU 4 nhân của Raspberry Pi 4.
==============================================================================
"""

import os
import time
import cv2
import numpy as np
import onnxruntime as ort


class ONNXPlateDetector:
    """
    YOLOv8 Plate Detector chạy thuần bằng ONNX Runtime.
    Không phụ thuộc vào PyTorch hay Ultralytics.
    """
    def __init__(self, weights_path='weights/best_int8.onnx', conf_threshold=0.25, iou_threshold=0.45, num_threads=4, imgsz=640):
        resolved_path = weights_path
        if not os.path.exists(resolved_path):
            alt_path = os.path.basename(weights_path)
            if os.path.exists(alt_path):
                resolved_path = alt_path
            else:
                raise FileNotFoundError(f"Không tìm thấy file ONNX weights: '{weights_path}'")

        self.weights_path = resolved_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_shape = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz

        # Cấu hình tối ưu đa luồng cho CPU Edge / Raspberry Pi
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Khởi tạo Inference Session
        self.session = ort.InferenceSession(resolved_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
        """
        Resize và padding giữ nguyên tỉ lệ aspect ratio theo chuẩn YOLO.
        """
        shape = im.shape[:2]  # [h, w]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2.0, (new_shape[0] - new_unpad[1]) / 2.0

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

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
        if img is None or img.size == 0:
            return [], [], [], 0.0

        h, w = img.shape[:2]
        t0 = time.perf_counter()

        # 1. Preprocessing: Letterbox + Normalization (BGR -> RGB -> 0..1)
        lb_img, r, (dw, dh) = self.letterbox(img, self.input_shape)
        rgb_img = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB)
        input_tensor = rgb_img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0

        # 2. Forward inference ONNX
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

        # 3. Postprocessing & NMS:
        # outputs shape: [1, 5, 8400] -> transpose to [8400, 5]
        predictions = np.squeeze(outputs).T
        scores = predictions[:, 4]

        mask = scores >= self.conf_threshold
        filtered_preds = predictions[mask]
        filtered_scores = scores[mask]

        boxes = []
        confidences = []
        rois = []

        if len(filtered_preds) > 0:
            # cx, cy, w_box, h_box
            cx = filtered_preds[:, 0]
            cy = filtered_preds[:, 1]
            bw = filtered_preds[:, 2]
            bh = filtered_preds[:, 3]

            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0

            boxes_wh = np.stack([x1, y1, bw, bh], axis=1)

            # Áp dụng OpenCV NMSBoxes
            indices = cv2.dnn.NMSBoxes(
                boxes_wh.tolist(),
                filtered_scores.tolist(),
                float(self.conf_threshold),
                float(self.iou_threshold)
            )

            if len(indices) > 0:
                for idx in indices:
                    i = idx[0] if isinstance(idx, (list, np.ndarray)) else idx
                    b = boxes_wh[i]
                    conf = float(filtered_scores[i])

                    # Scale ngược về toạ độ ảnh gốc
                    orig_x1 = max(0, min(w, int(round((b[0] - dw) / r))))
                    orig_y1 = max(0, min(h, int(round((b[1] - dh) / r))))
                    orig_x2 = max(0, min(w, int(round((b[0] + b[2] - dw) / r))))
                    orig_y2 = max(0, min(h, int(round((b[1] + b[3] - dh) / r))))

                    roi = img[orig_y1:orig_y2, orig_x1:orig_x2]
                    if roi.size > 0:
                        boxes.append([orig_x1, orig_y1, orig_x2, orig_y2])
                        confidences.append(conf)
                        rois.append(roi)

        duration_ms = (time.perf_counter() - t0) * 1000.0
        return boxes, confidences, rois, duration_ms
