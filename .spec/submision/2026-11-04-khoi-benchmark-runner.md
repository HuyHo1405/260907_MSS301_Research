# Chi tiết thay đổi — Benchmark Runner & Profiler

Ngày: 2026-11-04. Thành viên: Khôi.

Spec: [2026-11-04-khoi-benchmark-runner.md](../2026-11-04-khoi-benchmark-runner.md).
Biên bản: [2026-11-04-khoi-benchmark-implementation.md](../2026-11-04-khoi-benchmark-implementation.md).

## 1. Phạm vi bàn giao

Triển khai runner batch qua ALPRPipeline, matrix FP32/INT8 × 640/480/320,
profiling từng công đoạn và Peak RAM, xuất dữ liệu thực nghiệm JSON/CSV.
Code sử dụng các thư mục evaluation/, pipeline/, tests/ và các module
metrics/reporting hiện có.

| Yêu cầu | Thực hiện | Kiểm chứng |
|---|---|---|
| K01 — Batch toàn dataset | Inventory ảnh, loader, vòng lặp inference theo ảnh/repetition | Đã chạy toàn 60 ảnh Source B; Source A/corrupted còn phụ thuộc dữ liệu |
| K02 — Model/resolution | CLI/config và truyền tham số xuống ONNX detector | Chạy thật đủ 6 cấu hình |
| K03 — Profiling | Stage timers, wall-clock Total, sampled RSS | Đã kiểm thử và thu mẫu thực nghiệm |
| K04 — Raw JSON/CSV | GT/predictions, timing từng mẫu, cấu hình và tổng hợp | Đã xuất và kiểm tra với reporting |

## 2. File được tạo

### evaluation/run_benchmark.py

Runner chính, chạy bằng python -m evaluation.run_benchmark.

- Đọc config và CLI; kiểm tra source/model/resolution/severity, confidence,
  số luồng, warmup, repeats và lựa chọn matrix bị trùng.
- Chạy preflight trước khi import pipeline hoặc load model.
- Lập matrix theo source, corruption, severity, model và resolution.
- Chạy các cấu hình tuần tự trong cùng process.
- Khởi tạo ALPRPipeline với weights, resolution, threads và NMS tương ứng.
- Warmup riêng; ảnh nhiều biển chỉ inference một lần mỗi repetition.
- Accuracy dùng predictions repetition đầu; latency giữ toàn bộ repetitions
  sau warmup, bao gồm ảnh không có detection.
- Chuyển predictions bằng predictions_from_pipeline; dùng evaluate_detection,
  pair_ocr_records và evaluate_ocr.
- Ghi cấu hình effective, môi trường, phiên bản thư viện, hash model/annotation.
- Xuất kết quả hoặc ghi lỗi và checkpoint partial.

Hàm chính: parser(), main(), execute_cell(), write_json(), write_csv().
execute_cell() nhận factory tùy chọn để kiểm thử điều phối bằng pipeline giả.

### evaluation/benchmark_data.py

Adapter đầu vào dùng thư viện chuẩn:

- Đọc COCO inventory và GT; Source B lấy danh sách file ảnh trong thư mục.
- Chuẩn hóa đường dẫn Windows/Linux và giới hạn đường dẫn trong root.
- Tạo image_id/plate_id từ source, tên ảnh và annotation ID.
- Kiểm tra ảnh thiếu, ID trùng, bbox không hợp lệ/ngoài ảnh, class và layout.
- Bao gồm ảnh âm tính trong COCO inventory; không tự coi ảnh Source B thiếu
  nhãn là ảnh âm tính đã được xác nhận.
- Phát hiện text UNKNOWN/rỗng và trạng thái labels_verified.
- Lấy mẫu theo seed để dùng cùng quy tắc giữa các cấu hình.
- Chuyển bbox COCO xywh của ảnh nhiễu sang xyxy.
- Yêu cầu original_annotation_id và tên ảnh khớp để nối nhãn sạch/nhiễu.
- Tạo cohort ID và hash file nhãn phục vụ truy vết.

Hàm chính: read_json(), resolve(), load_dataset().

### evaluation/benchmark_profiler.py

- RSSSampler lấy process RSS bằng psutil mỗi 10ms, quy đổi MiB.
- Thu peak từ load model, warmup đến đo batch.
- Context manager dừng thread lấy mẫu khi hoàn tất hoặc exception.
- summarize() tính count, mean, median, population std, P95/P99 nội suy
  tuyến tính và FPS = 1000 / mean(total_ms).

### evaluation/benchmark_config.json

Khai báo project_root, model FP32/INT8, Source A/B và corruption manifest.
Source A trỏ datasets/eval_set; Source B trỏ datasets/source_b.
labels_verified mặc định false. Việc chạy không tự thay trạng thái duyệt nhãn.

### evaluation/BENCHMARK.md

Hướng dẫn preflight, smoke/full matrix, dữ liệu nhiễu, phạm vi đo,
JSON/CSV, kiểm thử và command override cho Docker hiện có.

### Tests

| File | Các tình huống kiểm tra |
|---|---|
| tests/test_benchmark_runner.py | Inventory nhiều biển/ảnh âm tính; UNKNOWN; matrix trùng; schema report; thiếu lineage; path thoát root; lỗi thực thi; thống kê latency |
| tests/test_benchmark_profiling.py | Không detection/fallback/vẽ; giữ box confidence thấp cho AP; tương thích API cũ; loại warmup; truyền resolution/thread |

## 3. File được sửa

### pipeline/alpr_pipeline.py

| Giao diện | Thay đổi |
|---|---|
| __init__ | Thêm keyword-only imgsz=640, num_threads=4, iou_threshold=0.45 và truyền xuống ONNX detector |
| predict | Thêm profile=False, annotate=True, fallback_ocr=True, ocr_conf_threshold=None |
| latency profiling | Thêm end_to_end_ms và stages_ms khi profile=True |

Giữ tuple (annotated_img, results, latency) và các key latency cũ.
annotate=False bỏ copy/vẽ và trả ảnh input. Benchmark tắt fallback toàn cảnh.
Box dưới ngưỡng OCR giữ trong predictions cho AP với text rỗng.
Timer OCR profiling đo toàn lời gọi recognize, gồm resize và format chuỗi.

### pipeline/detector_onnx.py

- Xác thực shape ONNX với resolution yêu cầu; báo lỗi nếu model static
  không tương thích, không đổi resolution ngầm.
- Bổ sung last_profile: preprocess_ms, detection_ms, postprocess_ms,
  crop_roi_ms; reset mỗi lần detect.
- Giữ tuple detect() bốn phần tử.

### Tài liệu

- evaluation/README.md: thêm điểm vào runner và liên kết hướng dẫn.
- evaluation/REPORTING.md: cập nhật runner đã có, liên kết BENCHMARK.md.
- evaluation/METRICS.md: cập nhật tích hợp runner với adapter metrics.
- Spec gốc: cập nhật trạng thái và liên kết biên bản triển khai.
- Biên bản triển khai: ghi kiểm chứng và điều kiện còn thiếu.

## 4. Quy ước timing và RAM

| Chỉ số | Phạm vi |
|---|---|
| io_ms | Đọc/decode ảnh trước predict, đo riêng |
| preprocess_ms | Letterbox, đổi màu, chuẩn hóa tensor |
| detection_ms | ONNX session.run |
| postprocess_ms | Confidence filter, NMS, chuyển box về ảnh gốc |
| crop_roi_ms | Cắt các ROI theo box |
| ocr_ms | Tổng lời gọi recognize cho ROI đạt ngưỡng |
| overhead_ms | Total trừ các stage đã đo |
| total_ms | Wall-clock predict trên ndarray, gồm overhead Python |
| peak_ram_mb | Sampled process RSS, giá trị theo MiB |

Total trong chế độ benchmark không gồm I/O, load, warmup, vẽ hoặc export.
Các cell dùng lại process nên allocator/cache có thể ảnh hưởng RAM cell sau.
Peak lấy mẫu có thể bỏ sót spike và không đại diện RAM toàn container.

## 5. Cấu hình và đầu ra

Mặc định: sources=source_a,source_b; models=FP32,INT8;
resolutions=640,480,320; corruptions=clean; severities=1,2,3;
threads=4; detector-confidence=0.001; confidence-threshold=0.25;
nms-iou=0.45; warmup=20; repeats=3; seed=42.
Không có sample-size thì chạy toàn inventory.

--output bắt buộc. --preflight-only không load model.
--allow-pseudo-labels cho phép exploratory với nhãn chưa xác nhận, nhưng vẫn
chặn UNKNOWN. --overwrite thay các file output do runner quản lý.
Exit code: 0 hoàn tất, 2 lỗi đầu vào, 1 lỗi thực thi.

| File output | Nội dung |
|---|---|
| preflight.json | Inventory, matrix dự kiến, lỗi và cảnh báo overlap |
| manifest.json | Effective config, môi trường, thư viện và trạng thái cell |
| benchmark.json | Schema v1, GT/predictions, latency, RAM và metrics |
| samples.csv | Timing theo ảnh/repetition, có run_id/image_id |
| runs.csv | Cấu hình, số ảnh, metrics và thống kê latency/RAM |
| failures.json | Lỗi runtime và run_id |
| checkpoint.partial.json | Run hoàn tất trước lỗi, complete=false |

JSON/CSV ghi qua temp rồi replace; JSON từ chối NaN/Infinity.
benchmark.json chính xuất sau khi toàn matrix hoàn tất.
Output thực nghiệm được lưu trong output/ đã được Git ignore.

## 6. Kết quả kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| Unit/integration | 62 tests đạt, gồm 12 tests mới |
| Smoke ONNX/RapidOCR | 2 ảnh Source B × 6 cấu hình; warmup 1, repeats 1 |
| Source B đầy đủ | 60 ảnh × 6 cấu hình = 360 mẫu; warmup 2 mỗi cell, repeats 1 |
| Reporting | visualizer xuất report từ smoke JSON; error_analysis xuất ca lỗi từ Source B full |
| Source A preflight | 322 ảnh/324 biển; chặn 205 nhãn OCR không hợp lệ |
| Corruption preflight | Chặn do thiếu original_annotation_id |
| Docker | Compose validate được; chưa chạy container vì Engine chưa hoạt động |

Môi trường: Windows x86_64, CPU, Python 3.13.
Source B gắn exploratory, chưa dùng làm accuracy với nhãn đã duyệt.
Warmup/repeats thực nghiệm được ghi trong bảng, khác mặc định 20/3.

Output: output/khoi-smoke, output/khoi-source-b-full,
output/khoi-preflight, output/khoi-preflight-corrupted.

Lần Source B full chạy trước khi bổ sung cột metrics vào runs.csv.
Raw benchmark.json và samples.csv của lần đó vẫn đầy đủ; runner bàn giao
xuất thêm các cột metrics cho lần chạy tiếp theo.

## 7. Phần còn cần kiểm chứng

- Full Source A sau khi nhãn OCR đủ điều kiện.
- Full corrupted khi có lineage và cohort clean/corrupted hợp lệ.
- Accuracy chính thức sau khi nhãn được duyệt.
- Container khi Docker Engine hoạt động và đo trên thiết bị đích nếu cần.

Subprocess isolation và resume từ checkpoint chưa triển khai.
Full benchmark trên toàn dataset và toàn bộ Phase 6 chưa được kết luận hoàn tất.
