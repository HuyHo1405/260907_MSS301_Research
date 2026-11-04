# Phase 6 — Batch Benchmark Runner

Chạy từ thư mục gốc project bằng Python 3.10 trở lên. Dependency dùng chung
requirements_edge.txt; xuất biểu đồ cần thêm requirements_report.txt.

## Cấu trúc

- run_benchmark.py: CLI, matrix, gọi ALPRPipeline, metrics và xuất kết quả.
- benchmark_data.py: adapter COCO/GT, inventory ảnh và kiểm tra dữ liệu.
- benchmark_profiler.py: lấy mẫu RSS và thống kê latency.
- benchmark_config.json: đường dẫn model/dataset tương đối project_root.

Không cần đổi cấu trúc datasets, weights hoặc các module báo cáo hiện có.
Config mặc định nằm cạnh runner; project_root được resolve từ vị trí config.

## Kiểm tra dữ liệu

~~~powershell
python -m evaluation.run_benchmark --preflight-only --output output/preflight
~~~

Preflight không import pipeline hoặc khởi tạo ONNX. preflight.json chứa inventory,
matrix dự kiến, lỗi nhãn và số tên file trùng giữa nguồn.
Exit code: 0 hoàn tất, 2 đầu vào không hợp lệ, 1 lỗi thực thi.
Output đã tồn tại cần --overwrite; cờ này thay các file do runner quản lý.

Config mặc định ghi labels_verified=false vì repo chưa có bằng chứng duyệt nhãn.
Chỉ chuyển true sau khi xác nhận nhãn thực tế; không dùng cờ này để bỏ qua UNKNOWN.
--allow-pseudo-labels cho phép chạy thử với text không UNKNOWN, gắn exploratory.
UNKNOWN/rỗng vẫn bị chặn để không đưa thành ground truth OCR giả.

## Chạy nhanh và chạy toàn bộ ảnh

~~~powershell
python -m evaluation.run_benchmark --sources source_b --models FP32,INT8 --resolutions 640,480,320 --sample-size 2 --warmup 1 --repeats 1 --allow-pseudo-labels --output output/smoke
python -m evaluation.run_benchmark --sources source_b --models FP32,INT8 --resolutions 640,480,320 --warmup 20 --repeats 3 --allow-pseudo-labels --output output/source_b
~~~

Bỏ --sample-size để chạy toàn inventory. Ảnh nhiều biển chỉ inference một lần
mỗi repetition. Accuracy dùng prediction repetition đầu; latency dùng tất cả
repetitions sau warmup. Không nhân GT theo repetitions.

Mặc định detector-confidence=0.001, confidence-threshold=0.25, nms-iou=0.45,
threads=4 (ONNX detector intra-op), seed=42, warmup=20, repeats=3.
--profile đặt tên môi trường, không thay ngân sách phần cứng.

## Dữ liệu nhiễu

~~~powershell
python -m evaluation.run_benchmark --sources source_a --corruptions all --severities 1,2,3 --output output/robustness
~~~

Manifest dùng đường dẫn tương đối project root, nhận cả separator Windows/Linux.
Mỗi corrupted annotation phải có original_annotation_id trỏ tới ann_id sạch,
file_name phải khớp; bbox dùng tọa độ đã biến đổi. Runner chặn dữ liệu hiện tại
thiếu lineage. Cần tái sinh hoặc bổ sung mapping đã xác thực; không suy diễn từ
thứ tự list. Clean/corrupted phải cùng image/plate IDs để xuất chung một báo cáo.
Runner không tự lọc ảnh để làm hai cohort khớp và không sửa nhãn/generator.

## Thời gian và bộ nhớ

Stage: preprocess, detection (ONNX inference), postprocess (NMS/scale box),
crop ROI, toàn lời gọi OCR và overhead. Total là wall-clock predict trên ndarray,
không đọc ảnh, vẽ, load model, warmup hoặc export. io_ms được đo riêng.
Fallback OCR toàn cảnh tắt; box dưới confidence-threshold giữ cho AP, không chạy OCR.
API predict cũ vẫn có tuple/latency keys cũ; benchmark dùng end_to_end_ms mới.

Peak RAM là process RSS lấy mẫu 10ms, số báo theo MiB; gồm load model/warmup.
Các cell chạy tuần tự trong process dùng lại; allocator/cache có thể ảnh hưởng
RAM cell sau. Đây là peak quan sát, không phải peak toàn hệ thống/container.
manifest.json lưu Python, hệ điều hành/kiến trúc và phiên bản thư viện;
metadata mỗi run lưu hash model/annotation và trạng thái chất lượng nhãn.

## Đầu ra và báo cáo

- benchmark.json: schema_version=1, raw GT/predictions/latency và metrics.
- samples.csv: mỗi ảnh/repetition và thời gian từng stage.
- runs.csv: thống kê latency/RAM và metrics theo run.
- preflight.json, manifest.json: validation, cấu hình effective và trạng thái.
- failures.json, checkpoint.partial.json: lỗi runtime và các cell hoàn tất trước lỗi.

benchmark.json chỉ xuất sau khi toàn ma trận thành công; partial không dùng làm
báo cáo cuối. File được ghi qua temp rồi replace; JSON từ chối NaN/Infinity.

~~~powershell
python -m evaluation.visualizer --input output/source_b/benchmark.json --output output/source_b/report --image-root .
python -m evaluation.error_analysis --input output/source_b/benchmark.json --output output/source_b/failures --image-root .
~~~

Docker dùng image/config hiện có:

~~~powershell
docker compose -f docker-compose.benchmark.yml run --build --rm benchmark python -m evaluation.run_benchmark --sources source_b --models INT8 --resolutions 480 --sample-size 2 --warmup 1 --repeats 1 --allow-pseudo-labels --output output/docker-smoke
~~~

Compose giữ demo_report mặc định. CPU/RAM quota trên x86 không đại diện tốc độ Pi.

## Kiểm thử

~~~powershell
python -m unittest discover -s tests -v
~~~
