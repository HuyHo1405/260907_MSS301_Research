import time
import cv2
from rapidocr_onnxruntime import RapidOCR
from .postprocess import format_vietnamese_plate


class RapidOCRRecognizer:
    """
    OCR Recognizer Engine dựa trên RapidOCR (PaddleOCR chạy trên ONNX Runtime).
    Đã được tối ưu hóa sâu cho Edge / ARM nhúng (Raspberry Pi):
      1. Chuẩn hóa ROI height (target_height=96) tránh tính toán ma trận lớn.
      2. det_limit_side_len = 320, det_limit_type = 'max' (chống upscale ROI giả tạo).
      3. det_donot_use_dilation = True (loại bỏ phép giãn nở hình thái học thừa).
      4. use_cls = False (loại bỏ mạng phân loại góc xoay 180 độ).
    """
    def __init__(self, target_height=96):
        self.target_height = target_height
        self.engine = RapidOCR(
            det_limit_side_len=320,
            det_limit_type='max',
            det_donot_use_dilation=True
        )

    def recognize(self, roi):
        """
        Nhận diện ký tự biển số từ vùng ảnh ROI.
        Input: roi (numpy.ndarray BGR)
        Output:
          - raw_texts: Danh sách các chuỗi text OCR thô đọc được
          - formatted_plate: Biển số chuẩn hóa theo quy chuẩn Việt Nam
          - duration_ms: Thời gian suy luận OCR (ms)
        """
        if roi is None or roi.size == 0:
            return [], "", 0.0

        # Tiền xử lý: Downscale vùng ROI nếu kích thước quá lớn (> target_height)
        h, w = roi.shape[:2]
        if self.target_height and h > self.target_height:
            new_w = max(16, int(w * (self.target_height / h)))
            processed_roi = cv2.resize(roi, (new_w, self.target_height), interpolation=cv2.INTER_AREA)
        else:
            processed_roi = roi

        t0 = time.perf_counter()
        ocr_res, _ = self.engine(processed_roi, use_cls=False)
        duration_ms = (time.perf_counter() - t0) * 1000

        if not ocr_res:
            return [], "", duration_ms

        raw_texts = [item[1] for item in ocr_res]
        formatted_plate = format_vietnamese_plate(raw_texts)
        return raw_texts, formatted_plate, duration_ms
