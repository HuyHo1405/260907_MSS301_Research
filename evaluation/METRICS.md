# Bộ chỉ số đánh giá ALPR — phần của Tuấn

Module `evaluation/metrics.py` chỉ dùng thư viện chuẩn Python, không tải model,
không cần GPU và không import pipeline. Chạy từ thư mục gốc dự án:

```powershell
python -m unittest discover -s tests -p "test_metrics.py" -v
python -m evaluation.demo_metrics
```

Demo dùng dữ liệu giả: hai biển thật, một biển được nhận diện đúng và một biển
bị bỏ sót. Kết quả dự kiến: TP=1, FP=0, FN=1, Precision=1, Recall=0.5,
mAP@50=0.5, EM tổng=50%, CER tổng=9/17 và suy giảm EM từ 100% xuống 50% là 50%.
Đây là ví dụ kiểm tra phép tính, không phải kết quả thực nghiệm của model.

## Giao diện cho runner

Mỗi phần tử biểu diễn một biển trong một ảnh. Ảnh có nhiều biển thì có nhiều
phần tử chung `image_id`. Dùng ID ảnh duy nhất trong mỗi lần đánh giá; thêm tên
nguồn dữ liệu/kịch bản vào ID nếu có thể trùng tên file.

```python
from evaluation.metrics import (
    evaluate_detection, evaluate_ocr, pair_ocr_records,
    predictions_from_pipeline, robustness_degradation,
)

ground_truth = [
    {"image_id": "source_a/001.jpg", "bbox": [10, 20, 110, 60],
     "class_id": 1, "text": "30A-123.45", "layout": "one_line"},
]
predictions = [
    {"image_id": "source_a/001.jpg", "bbox": [10, 20, 110, 60],
     "class_id": 1, "score": 0.95, "text": "30A12345"},
]
detection = evaluate_detection(ground_truth, predictions, confidence_threshold=0.25)
pairs = pair_ocr_records(ground_truth, predictions, confidence_threshold=0.25)
ocr = evaluate_ocr(pairs)
```

`class_id` mặc định là `1` (biển số theo tài liệu datasets hiện tại). Nếu model
thực tế dùng lớp `0` cho biển số, truyền đúng cùng class ID cho nhãn và dự đoán.
Chỉ truyền bản ghi biển số khi đánh giá OCR; module không tự lọc lớp phương tiện.

Khi tích hợp ALPRPipeline đã được runner khởi tạo:

```python
# _, results, latency = pipeline.predict(image_path)
# predictions.extend(predictions_from_pipeline(image_id, results, class_id=1))
```

Adapter chuyển `box`, `detector_conf`, `formatted_plate` thành các trường chuẩn.
Trong pipeline hiện tại, bản ghi có `detector_conf=0` là OCR fallback toàn ảnh;
adapter loại bỏ bản ghi đó vì đây không phải phát hiện biển số thực sự.

## Quy ước Detection

- BBox: `[x1, y1, x2, y2]`, tọa độ liên tục trong cùng hệ tọa độ ảnh, không cộng
  thêm 1 vào chiều rộng/cao. Box diện tích 0 có IoU=0; box đảo tọa độ hoặc chứa
  NaN/Infinity bị từ chối. Chuyển box từ ảnh resize về hệ tọa độ GT trước khi gọi.
- Confidence thuộc [0,1]. Sắp giảm dần confidence, ghép greedy với GT còn trống
  có IoU lớn nhất trong cùng ảnh và lớp. IoU >= 0.5 là TP; dự đoán dư là FP;
  GT chưa ghép là FN. Confidence bằng nhau giữ thứ tự đầu vào để tái lập kết quả.
- Precision = TP/(TP+FP); Recall = TP/(TP+FN). Precision bằng 0 khi không có dự
  đoán; Recall là `None` khi không có GT. P/R tổng dùng cộng TP/FP/FN giữa các lớp.
- AP@50 dùng **all-points interpolated precision envelope**: tính diện tích
  dưới đường precision bao trên theo các mức recall thay đổi. mAP@50 là trung
  bình AP của các lớp có GT. Một lớp biển số thì mAP@50 = AP@50.
  Đây không phải phép nội suy COCO 101 điểm hay mặc định của một phiên bản YOLO;
  khi đối chiếu công cụ khác phải chọn cùng quy ước AP.
- Lớp có GT nhưng không có dự đoán: AP=0. Lớp không có GT: AP=`None`, không vào
  trung bình; FP của lớp đó vẫn xuất hiện trong P/R tổng. Không có GT ở tất cả
  các lớp: mAP=`None`. Kết quả chứa `per_class` và `ap_method` để kiểm tra.
- Ngưỡng confidence chỉ áp dụng P/R, không cắt danh sách trước khi tính AP.
  AP chỉ dùng được những dự đoán runner cung cấp: khi benchmark AP, cấu hình
  detector giữ các ứng viên confidence thấp và ghi lại ngưỡng lọc đó. Module
  không khôi phục được box đã bị detector loại bỏ.

## Quy ước OCR

`evaluate_ocr` nhận các cặp `gt_text`, `pred_text`, `layout`, trả `overall`,
`one_line`, `two_line`. `layout` lấy từ GT và phải là `one_line` hoặc `two_line`.

- Mặc định chuẩn hóa Unicode NFC, chữ hoa; bỏ khoảng trắng, dấu chấm, dấu gạch
  ngang. Giữ nguyên ký tự lạ khác để tránh che lỗi OCR. Dùng `normalize=False`
  nếu cần đánh giá chuỗi gốc. Không tự thay O/0, I/1 hoặc sửa lỗi theo biển hợp lệ.
- EM (%) = số chuỗi đúng hoàn toàn / số biển GT × 100.
- Levenshtein dùng chi phí 1 cho thêm, xóa, thay ký tự.
- CER = tổng edit distance / tổng ký tự GT (trung bình có trọng số theo độ dài,
  không phải trung bình CER từng biển). CER là tỷ lệ, có thể >1 khi thêm nhiều ký tự.
- GT rỗng sau chuẩn hóa là nhãn không hợp lệ và báo lỗi. Dự đoán rỗng hợp lệ:
  tính thành mất toàn bộ ký tự. Nhóm không có mẫu trả EM/CER=`None`.
- `pair_ocr_records` ghép box một-một và tạo một cặp cho **mỗi biển GT**; biển
  bị bỏ sót có `pred_text=""`. Đó là đánh giá OCR **end-to-end**, bao gồm lỗi
  detection. FP không tạo thêm biển GT giả trong mẫu số OCR; đọc kèm detection.
  Nếu chỉ đo OCR trên ROI GT chuẩn, runner tự tạo cặp tương ứng, ghi rõ chế độ
  `ocr_on_gt_roi` và không trộn hai chế độ trong cùng bảng kết quả.

## Độ bền bỉ

`iou_drop_rate(clean_box, corrupted_box)` = 1 − IoU, đơn vị tỷ lệ [0,1].
Caller ghép đúng cùng một biển giữa hai ảnh, không tự ghép bằng vị trí phần tử.
Nếu thiếu một hoặc cả hai box, module quy ước drop=1; runner cần báo riêng số
biển thiếu baseline, tránh diễn giải tất cả là suy giảm do nhiễu. Sau biến đổi
phối cảnh, runner phải đưa box về cùng hệ tọa độ hoặc áp phép biến đổi đã biết
lên box clean trước khi so sánh với corrupted. Hàm không tự biến đổi hình học.

`robustness_degradation(clean_em_percent, corrupted_em_percent)` nhận EM theo
thang **0–100**, trả `(clean-corrupted)/clean × 100` theo phần trăm tương đối.
Ví dụ 80% xuống 60% là suy giảm tương đối 25%, giảm tuyệt đối 20 điểm phần trăm.
Baseline bằng 0 trả `None` vì không xác định; kết quả âm thể hiện cải thiện.
Nhóm không có mẫu có EM=`None`: bỏ qua hoặc xuất `null`, không truyền vào hàm.

So sánh clean/corrupted trên cùng tập ID biển, cùng cấu hình chuẩn hóa, ngưỡng
confidence và chế độ OCR; đánh giá riêng mỗi model, loại nhiễu và mức độ.
Kết quả là dict/list/số/`None`, dùng được với `json.dumps(..., allow_nan=False)`.

## Phạm vi bàn giao

- `metrics.py`: phép tính và adapter; không chạy model, sinh nhiễu, đo tài nguyên.
- `../tests/test_metrics.py`: dữ liệu nhỏ có đáp án tính tay và kiểm thử tích hợp.
- `demo_metrics.py`: ví dụ chạy độc lập, không cần dataset hay weights.

Nhãn Source A/B và runner chưa có schema JSON thực tế trong repo tại thời điểm
triển khai. Runner cần chuyển schema của nhóm sang giao diện trên trước khi gọi.
