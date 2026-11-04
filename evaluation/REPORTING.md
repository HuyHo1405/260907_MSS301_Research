# Phần bàn giao của Triết

Triết nhận kết quả runner của Khôi và nhãn từ Huy/Khánh; dùng metrics của Tuấn
để xuất biểu đồ, ca lỗi, báo cáo. Không cần huấn luyện lại model hay thu thập
thay phần dữ liệu của các bạn khác.

## Chạy ngay

Từ thư mục gốc repo:

```powershell
python -m pip install -r requirements_report.txt
python -m evaluation.demo_report
python -m unittest discover -s tests -v
```

Mở `output/demo_report/report.html`. Toàn bộ số liệu, nhãn, box, latency trong
demo là dữ liệu giả để kiểm tra công cụ. Các ảnh samples chỉ minh họa cách vẽ box;
box demo không phải nhãn thật. Không đưa số liệu demo vào bài báo.

Khi Khôi có kết quả thật:

```powershell
python -m evaluation.visualizer --input output/benchmark.json --output output/report --image-root .
python -m evaluation.error_analysis --input output/benchmark.json --output output/failures --image-root .
```

## Giao diện JSON thống nhất với Khôi

Đây là schema đề xuất vì repo chưa có `run_benchmark.py`. Runner chuyển dữ liệu
sang schema này; không cần sửa core pipeline. CLI dùng `python -m evaluation...`
từ gốc repo. Input là JSON; báo cáo tự xuất cả JSON và CSV.

```json
{
  "schema_version": 1,
  "runs": [{
    "run_id": "FP32-source_a-480-clean",
    "model": "FP32",
    "source": "source_a",
    "resolution": 480,
    "profile": "PC CPU limits 4 cores 2GiB",
    "corruption": "clean",
    "severity": 0,
    "confidence_threshold": 0.25,
    "ground_truth": [{
      "image_id": "source_a/001.jpg", "plate_id": "source_a/001/plate1",
      "bbox": [10, 20, 110, 60], "class_id": 1,
      "text": "30A12345", "layout": "one_line"
    }],
    "predictions": [{
      "image_id": "source_a/001.jpg", "bbox": [10, 20, 110, 60],
      "class_id": 1, "score": 0.95, "text": "30A12345"
    }],
    "images": {"source_a/001.jpg": "datasets/source_a/001.jpg"},
    "latency_ms": [125.2, 124.8],
    "peak_ram_mb": 180.5
  }]
}
```

- `model`: dùng đúng `FP32`/`INT8` cho radar đối chiếu. Mỗi run là một cấu hình,
  một nguồn, một nhiễu, một severity. `clean` dùng 0; ảnh nhiễu dùng 1, 2, 3.
- `plate_id` phải duy nhất trong run, giữ ổn định qua clean/corrupted. `image_id`
  được phép đổi theo ảnh sinh nhiễu. Mọi plate ID nên là chuỗi.
- Box xyxy ở tọa độ ảnh hiển thị. Cập nhật GT khi biến đổi phối cảnh. Class ID
  phải thống nhất giữa GT và dự đoán. Chỉ truyền các bản ghi biển số.
- `predictions_from_pipeline` của metrics loại bỏ OCR fallback có confidence=0.
  Runner cần dùng adapter này; không coi OCR toàn ảnh là detection thành công.
- `latency_ms` là thời gian end-to-end thực đo mỗi ảnh, sau warmup, gồm các ảnh
  không phát hiện biển. Không truyền thời gian detection như end-to-end.
- `images` ánh xạ ID ảnh sang đường dẫn tương đối trong `--image-root`. Ảnh thiếu
  hoặc nằm ngoài root vẫn có ca lỗi JSON và cảnh báo, không làm mất ca lỗi.
- Không có latency/RAM thì bỏ trường tương ứng. Không thay số thiếu bằng 0.
- Chỉ hỗ trợ OCR end-to-end, không trộn OCR trên GT ROI. Gộp repetitions trong
  runner trước khi xuất; mỗi cell cấu hình chỉ xuất một run.

## Sản phẩm xuất ra

- `degradation_*.png/.svg`: suy giảm EM tương đối theo level, chia riêng nguồn,
  độ phân giải, profile phần cứng và confidence. Cần clean cùng cấu hình/cùng
  plate IDs, text, layout và class; baseline EM=0 trả null.
- `radar_*.png/.svg`: EM theo từng nhiễu, FP32/INT8 trên cùng cohort và level;
  chỉ vẽ khi có ít nhất ba loại nhiễu chung. Không điền 0 cho dữ liệu thiếu.
- `accuracy_latency.png/.svg`: EM so với latency trung bình, nhãn run; điểm trùng
  tọa độ ghi số run để tránh chồng chữ, tra chi tiết từng run trong summary.
  Đây là đồ thị tradeoff, không suy luận nhân quả hay hệ số tương quan gộp.
- `summary.json/.csv`: metrics detection, EM/CER, latency/RAM và degradation.
  JSON có EM/CER riêng biển một dòng/hai dòng và SHA256 của input.
- `report.md`, `report.html`: bảng tổng hợp, đồ thị và liên kết gallery.
- `failures/index.html`, `cases.json`, `case_*.jpg`: bỏ sót, phát hiện dư, OCR sai;
  GT xanh, dự đoán đỏ. Ghép một-một theo confidence/IoU giống module metrics.
  Phân loại mô tả triệu chứng; muốn kết luận nguyên nhân cần xem ảnh/metadata.

## Docker

```powershell
docker compose -f docker-compose.benchmark.yml run --build --rm benchmark
docker compose -f docker-compose.benchmark.yml run --rm benchmark python -m evaluation.visualizer --input output/benchmark.json --output output/report --image-root /app
```

Lệnh đầu chạy demo để kiểm tra đóng gói. Docker image có pipeline ONNX, cả
weights FP32/INT8, evaluation và thư viện xuất báo cáo; datasets mount read-only,
output giữ trên máy chủ. Khi runner của Khôi được thêm, build lại image rồi
override command bằng CLI runner đã thống nhất với Khôi.

Giới hạn 4 CPU/2GiB chỉ mô phỏng ngân sách tài nguyên. Trên x86, không gọi kết
quả là tốc độ Raspberry Pi 4. Muốn đo Pi thật, dùng Pi OS 64-bit và chạy trên Pi.
Không ép ARM64 trên PC vì QEMU làm thay đổi latency. Ghi CPU, kiến trúc, OS,
Docker version, seed sinh nhiễu, warmup, threads, model/dataset hash và thời điểm
đo trong log runner. Lưu `pip freeze` từ image cùng kết quả: requirements edge
hiện còn dùng khoảng phiên bản nên rebuild không bảo đảm bit-for-bit.

## Đối chiếu tài liệu

Phân công nằm ở `doc/phase_6_robustness_benchmark_and_data_preparation.md`, mục 5,
khớp ảnh người dùng gửi. `23.pdf` là bài tham khảo về YOLO, OCR và Pi; các mục
5.2-5.5, bảng 2-4 gợi ý cách trình bày accuracy/latency/điều kiện môi trường.
Không nhận số FPS/mAP của bài làm số đo của nhóm, cũng không phát sinh nhiệm vụ
nhận diện mũ bảo hiểm từ phạm vi bài tham khảo.

Phần cần chờ nhóm: nhãn Source A/B thật, ảnh nhiễu và kết quả runner Phase 6.
Khi có chúng, chạy lại lệnh report trên để tạo kết quả dùng cho bài báo.
