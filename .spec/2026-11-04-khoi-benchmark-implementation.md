# Benchmark Runner — Biên bản triển khai

Ngày: 2026-11-04.

## Thay đổi

Đã thêm evaluation/run_benchmark.py, benchmark_data.py, benchmark_profiler.py,
benchmark_config.json, BENCHMARK.md và tests. Runner dùng ALPRPipeline, metrics
và schema reporting hiện có; không đổi cấu trúc datasets/weights/Docker.

Pipeline nhận resolution/thread/NMS; predict hỗ trợ profiling, tắt vẽ/fallback.
Tuple và latency keys cũ được giữ cho caller hiện có.

## Quyết định triển khai

- Chạy các cell tuần tự trong process dùng lại; sampled RSS ghi rõ đơn vị MiB
  và giới hạn ảnh hưởng allocator/cache từ các cell trước.
- UNKNOWN chặn accuracy run. Nhãn chưa xác nhận chỉ chạy exploratory khi có
  --allow-pseudo-labels; config không tự đánh dấu nhãn đã duyệt.
- Corrupted thiếu original_annotation_id hoặc khác cohort bị báo lỗi; không
  suy đoán mapping, sửa generator hoặc tự bỏ ảnh khỏi full dataset.
- Subprocess isolation và resume là phần nâng cao chưa triển khai.
- Output được ghi qua file temp; JSON chính xuất khi toàn matrix thành công.

## Kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| Unit/integration | 62 tests đạt, gồm 12 tests mới cho runner/profiling |
| ONNX/RapidOCR smoke | Đủ FP32/INT8 × 640/480/320 trên cùng 2 ảnh Source B, warmup 1, repeats 1 |
| Source B full | 6 cell × 60 ảnh = 360 mẫu; warmup 2 mỗi cell, repeats 1; exploratory |
| Reporting | visualizer xuất report từ smoke JSON; error_analysis xuất ca lỗi từ Source B full JSON |
| Source A preflight | 322 ảnh, 324 biển; 205 nhãn OCR không hợp lệ |
| Corruption preflight | Annotation thiếu original_annotation_id |
| Docker | Compose config validate được; chưa chạy container vì Engine chưa hoạt động |

Output: output/khoi-smoke, output/khoi-source-b-full, output/khoi-preflight,
output/khoi-preflight-corrupted. Output đã được Git ignore.
Môi trường đo: CPU Windows x86_64, Python 3.13; không phải tốc độ Raspberry Pi.
Manifest lưu cấu hình và phiên bản thư viện.

## Trạng thái nghiệm thu

K01 đã kiểm chứng batch toàn Source B; chưa kiểm chứng full Source A/corrupted.
K02 đã chạy thật đủ 6 cấu hình. K03 có stage/Total/RSS và giới hạn đo.
K04 đã xuất JSON/CSV và kiểm tra reporting.

Full benchmark trên toàn dataset, accuracy với nhãn đã duyệt và smoke Docker
còn cần kiểm chứng. Chưa kết luận hoàn tất toàn bộ Phase 6.
