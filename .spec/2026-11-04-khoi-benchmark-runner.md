# Spec phần việc Khôi: Benchmark Runner & Profiler — Phase 6

Ngày: 2026-11-04. Chủ nhiệm: Khôi.
Trạng thái: Đã triển khai runner và kiểm chứng trên Source B; full Source A/corrupted còn phụ thuộc dữ liệu. Xem [biên bản triển khai](2026-11-04-khoi-benchmark-implementation.md).
Phạm vi: runner batch qua ALPRPipeline, ma trận FP32/INT8 × resolution, profiling và xuất dữ liệu cho Tuấn/Triết.

Tài liệu mô tả thiết kế runner, cấu hình benchmark, phương pháp profiling, định dạng đầu ra và tiêu chí nghiệm thu phần việc Khôi trong Phase 6.

## 0. Truy vết công việc được giao → triển khai → kiểm chứng

| Mã | Đầu việc trong bảng phân công | Phần thực hiện trong spec | Bằng chứng hoàn thành |
|---|---|---|---|
| K01 | Viết evaluation/run_benchmark.py chạy batch toàn dataset qua ALPRPipeline | Mục 3–5, 9: CLI, loader, inventory ảnh và vòng lặp inference; mặc định full dataset, sample-size chỉ dùng thử | Có runner thực; đối chiếu số ảnh đầu vào/đã xử lý/lỗi; ảnh nhiều biển không chạy lặp theo GT, ảnh không detection vẫn có kết quả và timing. Không âm thầm bỏ ảnh. |
| K02 | Hoán đổi FP32 best.onnx / INT8 best_int8.onnx và input 640/480/320 | Mục 4, 6: cấu hình model/resolution, truyền xuống ONNX detector, xác thực shape | Chạy thật đủ 6 cấu hình trên cùng tập kiểm tra và lưu effective config; không thay resolution ngầm. Shape/runtime không hỗ trợ là phần còn chưa đạt, không phải ngoại lệ để đánh dấu hoàn thành. |
| K03 | Đo Preprocess, Detection, Crop ROI, OCR, Total và Peak RAM | Mục 6–7: timer theo stage, end-to-end, lấy mẫu RSS và phạm vi đo | Mỗi ảnh/repetition có đủ stage; mỗi run có Peak RAM, đơn vị và phương pháp đo; warmup không trộn vào mẫu latency; kiểm tra không cộng trùng stage. |
| K04 | Xuất dữ liệu thực nghiệm thô JSON/CSV | Mục 8: benchmark.json, samples.csv và bảng cấu hình/kết quả | JSON có GT/predictions và timing; CSV có từng mẫu đo cùng run/image/repeat ID; hai định dạng truy vết được về cùng run; có output inference thật, không chỉ dữ liệu demo. |

Sản phẩm bắt buộc: evaluation/run_benchmark.py, cấu hình CLI/matrix, dữ liệu JSON/CSV và hướng dẫn tái chạy. Các module phụ là cách tổ chức code; không thay thế runner chính. Mục 10 quy định nghiệm thu theo K01–K04.

## 1. Đối chiếu nhiệm vụ và hiện trạng

Nguồn phân công: doc/phase_6_robustness_benchmark_and_data_preparation.md, mục 5–6.
Khôi phải bàn giao evaluation/run_benchmark.py, CLI/matrix runner, latency từng công đoạn, peak RAM và dữ liệu JSON/CSV.

| Yêu cầu | Hiện trạng kiểm tra trực tiếp | Thay đổi cần thiết |
|---|---|---|
| Runner tổng hợp | Chưa có evaluation/run_benchmark.py | Thêm runner điều phối và cấu hình |
| Benchmark Source B | benchmark_source_b.py cố định INT8, duyệt từng record GT, in metrics ra console | Duyệt ảnh duy nhất; xuất schema báo cáo; hỗ trợ ma trận |
| FP32/INT8, 640/480/320 | Detector ONNX đã nhận imgsz/num_threads; ALPRPipeline chưa truyền chúng | Bổ sung tham số keyword, giữ tương thích caller cũ |
| Profiling | det_time_ms gồm preprocess, inference, NMS, crop; total là det+ocr | Đo từng đoạn và wall-clock end-to-end riêng |
| OCR | Timer hiện bỏ phần resize ROI và định dạng chuỗi | Đo toàn bộ lời gọi recognize ở pipeline |
| Metrics | metrics.py và METRICS.md đã có | Dùng adapter/functions hiện có |
| Report/Docker | REPORTING.md có schema_version=1; Docker mặc định chạy demo_report | Xuất schema tương thích và thêm lệnh chạy thật |
| Source A | datasets/eval_set thay cho datasets/source_a; 324 record biển, 205 UNKNOWN | Alias source_a; preflight nhãn OCR |
| Source B | 60 record, không UNKNOWN; cả 60 tên file có trong Source A | Báo overlap; chưa gọi là tập độc lập |
| Nhiễu | manifest chỉ processed_sample_size=5 trên total_source_images=322 | Xác định cohort sạch tương ứng; không coi là full benchmark |

scripts/process_ground_truth.py dùng RapidOCR sinh pseudo-label; việc có text không chứng minh đã kiểm duyệt thủ công. Cần Huy/Khánh xác nhận provenance và nhãn đã duyệt trước khi công bố accuracy.
Các con số trên là kiểm tra file JSON, không phải kết quả chạy mô hình.

## 2. Mục tiêu và giới hạn

- Một CLI chạy toàn bộ cấu hình đã chọn, không cần sửa source code cho mỗi lần đo.
- Mỗi run = model × resolution × source/cohort × corruption × severity × cấu hình runtime.
- Xuất JSON dùng ngay với visualizer/error_analysis, CSV mẫu đo và manifest môi trường.
- Không huấn luyện, đổi trọng số, tự sửa nhãn OCR, viết lại công thức metrics hoặc thiết kế lại biểu đồ.
- Phạm vi benchmark chính: ONNX CPU + RapidOCR (engine hiện gọi là paddle).
- Source A sạch dùng đường dẫn eval_set hiện có, không bắt buộc di chuyển dataset.
- Các đường dẫn và giao diện trong phần tiếp theo là thiết kế mới, chưa phải CLI đã có.

### 2.1. Ranh giới trách nhiệm và mức ưu tiên

- **Cốt lõi của Khôi:** runner batch, CLI/matrix FP32/INT8 × 640/480/320, truyền cấu hình pipeline, đo các stage/Peak RAM, xuất raw JSON/CSV và kiểm tra tính đúng của công cụ.
- **Tích hợp:** Khôi đọc dữ liệu Huy/Khánh/Hy bàn giao, gọi metrics của Tuấn, xuất schema cho Triết; kiểm tra và báo đầu vào không hợp lệ.
- **Chủ sở hữu ngoài phần Khôi:** Huy/Khánh sửa và duyệt nhãn; Hy sửa generator/lineage hình học; Tuấn quyết định quy ước metrics; Triết đóng gói Docker, biểu đồ và báo cáo. Khôi cung cấp CLI và phối hợp kiểm tra.
- **Nâng cao, không bắt buộc nghiệm thu công cụ:** subprocess riêng mỗi cell, hash toàn bộ ảnh, checkpoint khôi phục, phát hiện prediction khác nhau giữa repeats. Chỉ bổ sung khi cần độ cô lập/tái lập cao hơn.
- Fixture/tập nhỏ dùng kiểm tra trong quá trình phát triển. Để xác nhận đầy đủ K01 phải có lần chạy toàn bộ dataset đã xác định; nếu dữ liệu chưa sẵn sàng thì ghi rõ phần đã kiểm chứng và phần còn chờ, không kết luận hoàn tất toàn bộ công việc.

### 2.2. Trạng thái quyết định tham số

Confidence, warmup, repeats và chu kỳ lấy mẫu RAM được cấu hình qua runner. Output phải lưu cấu hình thực tế của từng lần chạy. Phạm vi dataset, quy ước đo và hợp đồng đầu ra cần thống nhất được liệt kê tại [mục 7 của bản phân công](../doc/phase_6_robustness_benchmark_and_data_preparation.md#7-các-điểm-cần-làm-rõ-trước-khi-chạy-benchmark-chính-thức).
Các lần chạy thử được đánh dấu thử nghiệm. Nghiệm thu chức năng bao gồm FP32/INT8, ba resolution 640/480/320, latency từng công đoạn, Peak RAM và dữ liệu JSON/CSV.

## 3. Danh sách thay đổi dự kiến

| File | Thao tác | Trách nhiệm |
|---|---|---|
| evaluation/run_benchmark.py | Tạo | CLI, preflight, lập ma trận, điều phối worker, tổng hợp output |
| evaluation/benchmark_data.py | Tạo | Đọc COCO/GT/manifest, chuẩn hóa ảnh/plate IDs, cohort và nhãn |
| evaluation/benchmark_profiler.py | Tạo | Đồng hồ monotonic, sampled RSS, thống kê latency |
| evaluation/benchmark_worker.py | Tùy chọn nâng cao | Cô lập process từng cell nếu cần; MVP có thể chạy tuần tự trong runner |
| evaluation/benchmark_config.json | Tạo | Đường dẫn Source A/B, weights và tùy chọn mặc định |
| pipeline/alpr_pipeline.py | Sửa | Cấu hình ONNX, profile chi tiết, tùy chọn tắt vẽ/fallback |
| pipeline/detector_onnx.py | Sửa | Ghi timing từng đoạn, xác thực shape input ONNX |
| evaluation/README.md | Sửa | Lệnh runner, preflight, smoke và full matrix |
| evaluation/REPORTING.md | Phối hợp Triết | Khôi cung cấp schema đầu ra; Triết xác nhận consumer và cập nhật tài liệu báo cáo |
| docker-compose.benchmark.yml | Triết phụ trách | Khôi bàn giao lệnh CLI và phối hợp smoke; không bắt buộc Khôi sửa Docker |
| tests/test_benchmark_data.py | Tạo khi triển khai | Chuẩn hóa, ID, nhãn thiếu, cohort |
| tests/test_benchmark_runner.py | Tạo khi triển khai | Matrix, export, repetitions, lỗi/worker |
| tests/test_benchmark_profiling.py | Tạo khi triển khai | Timer/RSS bằng clock/sampler giả, API compatibility |

Không cần thêm psutil vì requirements_edge.txt đã có.
Dockerfile.benchmark đã COPY evaluation/pipeline/weights; nếu cần thay đổi đóng gói, chuyển yêu cầu cho Triết.
Thay đổi corruptions.py về provenance/geometry phải phối hợp Hy; ghi rõ là phụ thuộc tích hợp, không âm thầm suy đoán ID.

## 4. CLI đề xuất

~~~powershell
# Kiểm tra đầu vào, xuất kế hoạch, không load model
python -m evaluation.run_benchmark --config evaluation/benchmark_config.json --sources source_a,source_b --models FP32,INT8 --resolutions 640,480,320 --preflight-only --output output/preflight

# Smoke test công cụ: không dùng để công bố accuracy
python -m evaluation.run_benchmark --config evaluation/benchmark_config.json --sources source_b --models INT8 --resolutions 480 --sample-size 5 --warmup 2 --repeats 1 --allow-pseudo-labels --output output/smoke

# Full run sau khi nhãn/provenance đã hợp lệ
python -m evaluation.run_benchmark --config evaluation/benchmark_config.json --sources source_a,source_b --models FP32,INT8 --resolutions 640,480,320 --corruptions all --severities 1,2,3 --warmup 20 --repeats 3 --threads 4 --output output/phase6
~~~

Tham số cấu hình dự kiến:
- --config: JSON; CLI ghi đè config. Relative path resolve từ project_root trong config, không phụ thuộc cwd.
- --sources: source_a/source_b; --models: FP32/INT8; --resolutions: 640,480,320.
- --corruptions: clean hoặc danh sách scenario/all; chỉ áp dụng source_a. Source B là ảnh thật, không tự gán severity nhân tạo.
- --severities: 1–3; clean=0. Luôn tạo clean cohort đối chứng khi chạy corrupted.
- --threads: detector ONNX intra-op (mặc định 4); inter-op=1. Không tuyên bố cờ này điều khiển RapidOCR.
- --detector-confidence: mặc định 0.001 để giữ ứng viên AP; --confidence-threshold: 0.25 cho P/R và OCR; --nms-iou: 0.45.
- --warmup: mặc định 20 lần gọi trên tập ảnh chọn theo thứ tự cố định; --repeats: 3, đều phải >=1, warmup >=0.
- --sample-size: lấy mẫu bằng seed=42, cùng danh sách cho mọi model/resolution; không chọn theo kết quả prediction.
- --allow-pseudo-labels: chỉ dành exploratory/smoke, gắn quality_status=exploratory trong output.
- --preflight-only: không inference; --output: thư mục mới. Có dữ liệu thì fail, cần --overwrite rõ ràng; chỉ thay file output do runner quản lý.
- --profile: tên mô tả phần cứng; metadata tự ghi kiến trúc thực tế.
- Lỗi cấu hình/nhãn/provenance: exit 2; lỗi runtime/run chưa hoàn tất: exit 1; hoàn tất: exit 0.

Khả năng công bố kết quả là trạng thái riêng với việc lệnh chạy thành công.
Không tự bật --allow-pseudo-labels hoặc tự loại nhãn lỗi để làm lệnh full chạy qua.

## 5. Hợp đồng dữ liệu và preflight

Giao diện đề xuất:
- load_dataset(config, source, scenario, severity) -> DatasetBundle.
- DatasetBundle gồm image_records, ground_truth, provenance, cohort_id và validation_errors.
- ImageRecord gồm image_id, original_image_id, path, width, height, plate_records.
- plate_id = source + original_image_id + original_annotation_id, chuỗi ổn định qua clean/corrupted.
- Tất cả bbox chuyển xywh COCO sang xyxy, tọa độ ảnh gốc; chỉ class biển số đã mapping rõ ràng.
- Không suy luận class=0 là vehicle cho mọi file: categories hiện có cả 0/1 mang tên license-plate. Config mapping phải xác nhận category thực sự dùng.
- Dùng danh sách images trong COCO làm inventory, kể cả ảnh không GT. Source B cần inventory ảnh và kiểm tra thiếu nhãn; không tự coi ảnh chưa gắn nhãn là ảnh âm tính.
- Ảnh nhiều biển chạy inference một lần mỗi repetition, không một lần mỗi record GT.
- File hỏng/thiếu, ID trùng, box đảo/ngoài ảnh, NaN/Infinity, GT text rỗng/UNKNOWN, layout ngoài one_line/two_line: báo lỗi cụ thể.
- Pseudo-label có text hợp lệ vẫn cần provenance đã duyệt. Không dùng regex biển hợp lệ thay kiểm duyệt của người.
- Mặc định OCR strict: chặn run có UNKNOWN. Nếu cần nghiên cứu detection-only, đó là extension riêng; không gửi UNKNOWN sang report v1 rồi gọi là accuracy OCR.
- Báo overlap filename A/B; nếu cần xác nhận trùng nội dung, phối hợp Huy/Khánh đối chiếu SHA256. Không kết luận trùng ảnh chỉ từ tên; không cộng hai nguồn thành mẫu độc lập khi provenance chưa được xác nhận.
- Lưu cấu hình effective, đường dẫn model/dataset và danh sách cohort. Hash weights/annotation là đề xuất tái lập; hash toàn bộ ảnh là nâng cao, không bắt buộc nghiệm thu runner.
- Manifest có dấu gạch chéo Windows phải đọc được trên Linux; chuẩn hóa separator và cấm path thoát dataset root.

### Lineage của corruption

Generator hiện đánh lại annotation id; perspective có thể bỏ box nhưng code batch còn ghép updated_boxes bằng vị trí. Vì vậy không join GT OCR bằng chỉ số list hay annotation id mới.

Yêu cầu phối hợp Hy:
1. Ghi original_image_id, original_annotation_id/source_plate_id vào mỗi annotation.
2. Biến đổi geometry trả mapping theo annotation ID, trạng thái retained/dropped và ma trận biến đổi.
3. Khi box bị loại, ghi lý do; không giữ bbox cũ hoặc gán bbox của biển khác.
4. Dữ liệu cũ thiếu mapping chỉ migrate nếu ánh xạ duy nhất đã được kiểm chứng; nếu không, tái sinh bằng generator sửa rồi.
5. Runner kiểm tra lineage trước benchmark corrupted; geometry thiếu mapping phải fail preflight.

Clean và corrupted phải cùng cohort ảnh/biển dùng để tính degradation. Chính sách tạm đề xuất: nếu một biến thể mất/không xác định được GT của một biển, loại cả ảnh tương ứng khỏi cohort so sánh ở clean và mọi severity; ghi danh sách ảnh/biển bị loại và lý do. Không chỉ xóa một GT trong ảnh còn giữ lại vì prediction của biển đó có thể bị tính nhầm FP. Đây là phân tích trên cohort đã lọc, phải báo coverage; đánh giá toàn dataset vẫn cần quy tắc nhãn/ignore được Hy và Tuấn xác nhận. Không tự triển khai ignore-region nếu metrics chưa hỗ trợ. Box thực sự ra khỏi khung do biến đổi không phải detector FN.
Manifest hiện chỉ 5 ảnh; lấy baseline đúng 5 ảnh đó, không so toàn bộ 322 ảnh sạch với 5 ảnh nhiễu. Với nhiều cohort khác nhau, tạo baseline riêng theo cohort_id.

## 6. Thay đổi pipeline và thời gian đo

ALPRPipeline.__init__ thêm keyword-only:
imgsz=640, num_threads=4, iou_threshold=0.45.
Giữ positional arguments cũ; truyền các cờ này tới ONNXPlateDetector.
ALPRPipeline.predict thêm keyword-only:
profile=False, annotate=True, fallback_ocr=True, ocr_conf_threshold=None.
Caller cũ vẫn nhận tuple (annotated_img, results, latency), giữ nguyên các key latency cũ.
Benchmark gọi profile=True, annotate=False, fallback_ocr=False, ocr_conf_threshold=0.25.
Khi annotate=False trả ảnh gốc ở vị trí đầu, không copy/vẽ và không sửa input.

ONNXPlateDetector.detect giữ tuple 4 phần tử để không phá caller cũ.
Có last_profile chứa preprocess_ms, inference_ms, postprocess_ms, crop_roi_ms; reset mỗi lần gọi, kể cả ảnh không detection.
Pipeline chuyển snapshot đó vào latency["stages_ms"] khi profile=True.
Đọc shape từ ONNX session; dynamic nhận resolution yêu cầu; static không tương thích thì báo lỗi, không âm thầm resize sang kích thước khác.

Quy ước timing dùng time.perf_counter_ns:
- io_ms: decode ảnh, đo riêng.
- preprocess_ms: letterbox, đổi màu, tensor/normalize.
- detection_ms: ONNX session.run.
- postprocess_ms: confidence filter, NMS, ánh xạ tọa độ.
- crop_roi_ms: cắt các ROI sau box hợp lệ.
- ocr_ms: tổng wall-clock lời gọi recognize, gồm resize ROI, OCR engine, format chuỗi.
- total_ms: từ ndarray vào predict đến trả results; gồm overhead Python, không disk I/O, load model, warmup, export hay vẽ.
- overhead_ms: total trừ tổng stage, cho phép dung sai clock nhỏ, không che sai lệch lớn.
- pipeline total_time_ms cũ giữ semantics cũ để tương thích; key mới end_to_end_ms dùng cho runner.
- Ảnh không detection vẫn có total_ms, prediction list rỗng và OCR=0 khi fallback tắt.
- Giữ ứng viên confidence thấp cho AP; chỉ chạy OCR trên box >= ocr_conf_threshold, predictions thấp ngưỡng có text rỗng.
- Ghi detector-confidence, eval confidence và OCR threshold vào metadata; so sánh latency chỉ giữa cấu hình ngưỡng giống nhau.

## 7. RAM, worker và repetitions

MVP chạy các cell tuần tự, load một cấu hình tại một thời điểm, giải phóng model sau mỗi cell và ghi rõ RAM đo trong process dùng lại có thể chịu ảnh hưởng allocator/cache trước đó. Không khẳng định Peak RAM là mức riêng tuyệt đối của từng model.
Phương án nâng cao: subprocess mới mỗi cell để cô lập RSS; Windows dùng spawn và main guard. Khi chọn phương án này, worker khởi động sampler trước load, warmup, đo batch rồi thoát. Cần chốt phương án đo trước khi so sánh RAM chính thức.

- RSS lấy psutil.Process(...).memory_info().rss / 1024**2, chu kỳ 10ms.
- peak_ram_mb giữ tên schema v1 nhưng metadata ghi unit=MiB, method=sampled_rss, sampling_interval_ms=10.
- Ghi baseline RSS, load_peak, measured_peak và sampled toàn run; peak_ram_mb = peak toàn worker gồm load/warmup.
- Đây là peak quan sát qua lấy mẫu, không khẳng định bắt được mọi spike hay là peak toàn hệ thống/container.
- Dừng sampler trong finally; worker crash/OOM phải được parent ghi trạng thái; không biến run lỗi thành kết quả 0.

Accuracy dùng prediction của repetition đầu tiên; không nhân GT/prediction theo repeats, không chọn repetition tốt nhất.
latency_ms chứa mọi mẫu sau warmup, có bảng image_id/repeat_index đi kèm; ghi prediction mismatch giữa repeats nếu có.
Warmup loại hoàn toàn khỏi latency/accuracy. Sắp ảnh cố định, lưu run order, không khẳng định số đo hoàn toàn xác định giữa các máy.
Thống kê count, mean, median, std population, P95/P99 (linear interpolation), throughput=1000/mean(total_ms).
Ghi OCR settings, ONNX/provider, CPU/OS/architecture, threads thực tế/configured, package versions, Git revision nếu đọc được và timestamp UTC.

## 8. Đầu ra và tương thích report

Thư mục output gồm:
- benchmark.json: schema_version=1, runs hoàn chỉnh tương thích REPORTING.md.
- samples.csv: run_id,image_id,repeat_index,io_ms,preprocess_ms,detection_ms,postprocess_ms,crop_roi_ms,ocr_ms,total_ms.
- runs.csv: cấu hình, số ảnh/biển, latency summary, peak RAM, metrics summary.
- manifest.json: effective config, hashes, provenance, môi trường, cohort, warnings và trạng thái từng cell.
- preflight.json: lỗi/chất lượng dữ liệu/overlap; có cả khi preflight fail.
- failures.json: lỗi thực thi/ảnh, khác failure cases về nhận diện của error_analysis.py.

Mỗi run có các trường v1: run_id,model,source,resolution,profile,corruption,severity,confidence_threshold,ground_truth,predictions,images,latency_ms,peak_ram_mb.
Bổ sung metadata/cohort_id/quality_status là extension; không đổi schema fields mà consumer đang dùng.
model viết đúng FP32/INT8; image paths tương đối project root, dùng slash; plate IDs ổn định.
Dùng predictions_from_pipeline để chuyển box/detector_conf/formatted_plate và loại fallback confidence=0.
Tính metrics qua evaluate_detection, pair_ocr_records, evaluate_ocr; không viết lại công thức.
JSON allow_nan=False, số numpy chuyển sang Python scalars. Số thiếu là null hoặc bỏ trường, không thay bằng 0.
Atomic write bằng temp + replace cùng thư mục. Chỉ xuất benchmark.json chính khi tất cả cell yêu cầu hoàn tất; lỗi giữa chừng ghi trạng thái/lý do vào manifest. Checkpoint.partial.json và resume là nâng cao, không bắt buộc MVP.
Không đưa partial/failed runs vào báo cáo cuối mà không gắn rõ trạng thái.

Report hiện nhóm baseline theo cấu hình nhưng chưa chắc hỗ trợ nhiều cohort cùng group. Trước bàn giao cần kiểm tra consumer; lựa chọn MVP: mỗi file benchmark.json chỉ có một cohort source_a nhất quán cho toàn bộ severity. Chạy cohort khác xuất thư mục khác.
Source B không có cặp clean/corrupted tương ứng nên không suy ra degradation từ source_a.
Delta IoU phối cảnh chỉ tính khi có phép biến đổi chung hợp lệ; nếu thiếu metadata thì null + lý do, không so trực tiếp hai hệ tọa độ khác nhau.

## 9. Trình tự thực hiện và nghiệm thu

- [ ] Bước 1 — Khôi xác định data contract, dựng fixture hợp lệ để phát triển. Ghi các quyết định chờ nhóm ở mục 7 bản phân công; nhãn/lineage thật chưa sẵn sàng không chặn phát triển runner.
- [ ] Bước 2 — Implement benchmark_data và preflight. Test multi-plate, ảnh âm tính, UNKNOWN, ID trùng, path Windows/Linux, box/class mapping và cohort 5/322. Preflight không import model nặng.
- [ ] Bước 3 — Bổ sung cấu hình/timing pipeline. Test fake detector/recognizer, ảnh không detection, giữ tuple cũ, truyền resolution/thread, OCR threshold và không vẽ.
- [ ] Bước 4 — Implement profiler. Test fake clock/RSS, loại warmup, cleanup khi exception và repeats không nhân accuracy. Nếu chọn worker nâng cao thì bổ sung kiểm tra subprocess exit.
- [ ] Bước 5 — Implement CLI/matrix/export. Test 2 models × 3 resolutions = 6 cell mỗi dataset slice; unique run IDs; JSON/CSV đồng nhất; strict failure và atomic output.
- [ ] Bước 6 — Test tích hợp runner với metrics/report bằng fixture có đáp án biết trước, đánh dấu synthetic. Kiểm tra không xuất UNKNOWN thành nhãn hợp lệ, fallback không là TP.
- [ ] Bước 7 — Smoke ONNX thật trên dữ liệu đủ điều kiện: cả FP32/INT8 tại 640/480/320; xác nhận input shape/runtime; lưu logs. Smoke không thay full benchmark.
- [ ] Bước 8 — Bàn giao CLI cho Triết và phối hợp smoke Docker. Triết chịu trách nhiệm cấu hình/container; không lấy việc Docker chưa hoàn tất làm lỗi bàn giao công cụ Khôi. CPU quota x86 không được gọi là hiệu năng Raspberry Pi thật.
- [ ] Bước 9 — Tích hợp toàn nhóm: Khôi chạy full benchmark sau khi dữ liệu và tham số đã chốt; chuyển output cho Triết. Theo dõi riêng với nghiệm thu công cụ.
- [ ] Bước 10 — Review diff, tách commit theo data adapter / profiling / orchestration / docs; không commit output ảnh lớn.

Lệnh kiểm tra dự kiến khi triển khai:
~~~powershell
python -m unittest discover -s tests -v
python -m evaluation.run_benchmark --config evaluation/benchmark_config.json --preflight-only --output output/preflight
python -m evaluation.visualizer --input output/phase6/benchmark.json --output output/phase6/report --image-root .
python -m evaluation.error_analysis --input output/phase6/benchmark.json --output output/phase6/failures --image-root .
docker compose -f docker-compose.benchmark.yml run --build --rm benchmark python -m evaluation.run_benchmark --config evaluation/benchmark_config.json --sources source_b --models INT8 --resolutions 480 --sample-size 5 --warmup 2 --repeats 1 --allow-pseudo-labels --output output/docker-smoke
~~~

Các tests và CLI trên đã được triển khai; kết quả kiểm chứng được ghi trong biên bản triển khai đi kèm.
Không tự commit, push hoặc chỉnh dữ liệu nhóm khi chỉ đang lập spec.

## 10. Nghiệm thu theo công việc được giao

### 10.1. Bàn giao công cụ — trách nhiệm Khôi

1. Có run_benchmark.py và CLI/matrix FP32/INT8 × 640/480/320, batch qua ALPRPipeline; ảnh nhiều biển không bị chạy lặp theo số GT.
2. Đo và xuất Preprocess, Detection, Crop ROI, OCR, Total và Peak RAM, kèm phạm vi đo/đơn vị/cấu hình effective.
3. Xuất raw JSON/CSV từ inference thật, truy vết từng ảnh/cấu hình và tương thích schema nhận của metrics/report. Fixture chỉ kiểm chứng hợp đồng và phép tính, không thay sản phẩm dữ liệu thực nghiệm.
4. Kiểm tra đầu vào, lỗi cấu hình và dữ liệu; báo rõ trường hợp không tính được accuracy, không tự sửa hoặc giả lập nhãn thật.
5. Tests công cụ đạt và kiểm tra ONNX thật đủ FP32/INT8 × 640/480/320. Nếu shape/runtime chưa hỗ trợ thì K02 chưa hoàn thành; ghi rõ trở ngại và xử lý trước khi nghiệm thu yêu cầu này.
6. Bàn giao code, cấu hình effective, hướng dẫn CLI, logs và output thật. K01 cần bằng chứng chạy full dataset: số ảnh dự kiến/đã xử lý/lỗi và lý do. Sample-size hoặc cohort đã lọc phải được ghi rõ, không thay thế tuyên bố hoàn thành toàn dataset.
7. Subprocess isolation, full-image hashes và checkpoint/resume không phải điều kiện bắt buộc của MVP.

### 10.2. Tích hợp thực nghiệm — phụ thuộc phối hợp

- Huy/Khánh xác nhận và duyệt nhãn, nguồn dữ liệu và overlap.
- Hy cung cấp ảnh nhiễu, lineage và box hình học đúng; Tuấn xác nhận quy tắc cohort/ignore và metrics.
- Khôi cùng nhóm chốt tham số ở mục 7 bản phân công, rồi chạy toàn bộ dữ liệu và xuất kết quả thật.
- Triết xác nhận schema consumer, đóng gói Docker và sinh báo cáo.
- Nếu dữ liệu hoặc môi trường chưa sẵn sàng, ghi trạng thái từng K01–K04: đã triển khai, đã kiểm chứng, hoặc chưa kiểm chứng do phụ thuộc. Có thể bàn giao phần code đã làm, nhưng không tuyên bố hoàn tất yêu cầu full dataset/đủ ma trận khi chưa có bằng chứng. Việc sửa nhãn, generator, Docker và biểu đồ không chuyển thành đầu việc triển khai của Khôi.
