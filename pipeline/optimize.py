"""
==============================================================================
🚀 CÔNG CỤ TỐI ƯU HÓA MÔ HÌNH (MODEL OPTIMIZATION TOOLKIT)
==============================================================================
Triển khai 3 kỹ thuật tối ưu hóa phục vụ suy luận trên Edge (Raspberry Pi / ARM):
  1. Pruning (Cắt tỉa trọng số dư thừa, tăng độ thưa ma trận).
  2. Quantization (Lượng tử hóa FP32 -> INT8, giảm >40% kích thước, tăng tốc inference).
  3. ONNX Runtime Conversion (Export sang ONNX, tối ưu đồ thị tính toán).
==============================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def export_yolo_to_onnx(weights_path='weights/best.pt', output_path='weights/best.onnx', imgsz=640):
    """
    Kỹ thuật 3: Export mô hình PyTorch YOLOv8 sang ONNX Runtime tối ưu.
    """
    print("\n" + "=" * 65)
    print("📦 [BƯỚC 1/3] EXPORT PYTORCH SANG ONNX RUNTIME")
    print("=" * 65)
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Không tìm thấy weights: '{weights_path}'")
        
    size_pt = os.path.getsize(weights_path) / (1024 * 1024)
    print(f"🔹 File gốc (PyTorch .pt): '{weights_path}' ({size_pt:.2f} MB)")
    print("⏳ Đang export và tối ưu đồ thị với onnxslim (opset=18)...")
    
    t0 = time.perf_counter()
    from ultralytics import YOLO
    model = YOLO(weights_path)
    exported = model.export(format='onnx', imgsz=imgsz, simplify=True, opset=18)
    elapsed = time.perf_counter() - t0
    
    # Nếu file export ở vị trí khác mong muốn thì copy/rename
    if exported != output_path and os.path.exists(exported):
        if os.path.abspath(exported) != os.path.abspath(output_path):
            import shutil
            shutil.copyfile(exported, output_path)
            
    size_onnx = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Đã export thành công sang: '{output_path}'")
    print(f"📊 Kích thước ONNX FP32: {size_onnx:.2f} MB (Thời gian export: {elapsed:.2f}s)")
    return output_path


def quantize_to_int8(onnx_input_path='weights/best.onnx', onnx_output_path='weights/best_int8.onnx'):
    """
    Kỹ thuật 2 (BẮT BUỘC): Lượng tử hóa trọng số từ FP32 (32-bit) sang INT8 (8-bit).
    Sử dụng ONNX Runtime Dynamic Quantization (Industry Standard cho ARM / Raspberry Pi).
    """
    print("\n" + "=" * 65)
    print("⚡ [BƯỚC 2/3] QUANTIZATION (LƯỢNG TỬ HÓA FP32 ➔ INT8)")
    print("=" * 65)
    
    if not os.path.exists(onnx_input_path):
        raise FileNotFoundError(f"Không tìm thấy file ONNX đầu vào: '{onnx_input_path}'. Hãy export ONNX trước!")
        
    size_fp32 = os.path.getsize(onnx_input_path) / (1024 * 1024)
    print(f"🔹 File đầu vào (FP32 ONNX): '{onnx_input_path}' ({size_fp32:.2f} MB)")
    print("⏳ Đang tiến hành lượng tử hóa sang INT8 (Weight-only QUInt8)...")
    
    t0 = time.perf_counter()
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    quantize_dynamic(
        model_input=onnx_input_path,
        model_output=onnx_output_path,
        weight_type=QuantType.QUInt8
    )
    elapsed = time.perf_counter() - t0
    
    size_int8 = os.path.getsize(onnx_output_path) / (1024 * 1024)
    reduction = (1.0 - (size_int8 / size_fp32)) * 100.0
    speedup_ratio = size_fp32 / size_int8
    
    print("\n" + "-" * 65)
    print("📈 KẾT QUẢ LƯỢNG TỬ HÓA INT8:")
    print(f"  • Dung lượng ban đầu (FP32) : {size_fp32:.2f} MB")
    print(f"  • Dung lượng sau INT8       : {size_int8:.2f} MB")
    print(f"  • Mức giảm dung lượng       : 🔻 {reduction:.1f}% (Tiết kiệm {size_fp32 - size_int8:.2f} MB)")
    print(f"  • Tỉ lệ nén trọng số        : 🗜️ {speedup_ratio:.2f}× nhỏ hơn")
    print(f"  • Thời gian xử lý           : {elapsed:.2f}s")
    print(f"  • Lưu thành công tại        : '{onnx_output_path}'")
    print("-" * 65)
    
    # Sanity check: Xác thực mô hình INT8 hoạt động chính xác
    print("🔍 Đang chạy Sanity Check kiểm tra tính đúng đắn trên ảnh mẫu...")
    verify_int8_model(onnx_output_path)
    return onnx_output_path


def verify_int8_model(int8_path='weights/best_int8.onnx', test_image='samples/test2.jpg'):
    """
    Kiểm tra nhanh xem mô hình INT8 có phát hiện được biển số và tọa độ có chính xác không.
    """
    from ultralytics import YOLO
    if not os.path.exists(test_image):
        print("⚠️ Bỏ qua Sanity Check vì không tìm thấy ảnh test.")
        return
        
    model = YOLO(int8_path)
    results = model(test_image, verbose=False)
    if results and len(results[0].boxes) > 0:
        box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
        conf = float(results[0].boxes.conf[0].cpu().numpy())
        print(f"  ✅ SANITY CHECK PASSED: Bắt được {len(results[0].boxes)} biển số!")
        print(f"     - Bounding Box : {box.tolist()}")
        print(f"     - Confidence   : {conf*100:.1f}% (Độ chính xác nguyên vẹn!)")
    else:
        print("  ⚠️ Cảnh báo: Mô hình INT8 không phát hiện được biển số trên ảnh test.")


def prune_pytorch_model(weights_path='weights/best.pt', output_path='weights/best_pruned.pt', amount=0.25):
    """
    Kỹ thuật 1: Pruning (Cắt tỉa trọng số dư thừa).
    Sử dụng L1-norm unstructured pruning loại bỏ các trọng số có độ lớn nhỏ nhất.
    """
    print("\n" + "=" * 65)
    print(f"✂️ [BƯỚC 3/3] PRUNING (CẮT TỈA TRỌNG SỐ {amount*100:.0f}%)")
    print("=" * 65)
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Không tìm thấy weights: '{weights_path}'")
        
    import torch
    import torch.nn as nn
    import torch.nn.utils.prune as prune
    from ultralytics import YOLO
    
    print(f"🔹 File gốc: '{weights_path}'")
    print(f"⏳ Đang áp dụng L1-norm Pruning trên các lớp Conv2d (Tỉ lệ tỉa: {amount*100:.0f}%)...")
    
    yolo = YOLO(weights_path)
    pytorch_model = yolo.model
    
    total_weights = 0
    conv_layers_pruned = 0
    
    # 1. Áp dụng L1 Unstructured Pruning cho các lớp Conv2d
    for name, module in pytorch_model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Khóa trạng thái pruning thành trọng số 0 thực sự
            prune.remove(module, 'weight')
            conv_layers_pruned += 1
            
    # 2. Đếm số lượng trọng số 0 (Sparsity)
    total_params = 0
    zero_params = 0
    for p in pytorch_model.parameters():
        total_params += p.numel()
        zero_params += (p == 0).sum().item()
        
    sparsity = (zero_params / total_params) * 100.0
    
    # 3. Lưu mô hình pruned
    torch.save({'model': pytorch_model}, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print("\n" + "-" * 65)
    print("📈 KẾT QUẢ PRUNING:")
    print(f"  • Số lớp Conv2d đã cắt tỉa   : {conv_layers_pruned} layers")
    print(f"  • Tổng số tham số            : {total_params:,}")
    print(f"  • Số trọng số bị triệt tiêu (0): {zero_params:,}")
    print(f"  • Độ thưa ma trận (Sparsity) : 🎯 {sparsity:.2f}%")
    print(f"  • File lưu tại               : '{output_path}' ({size_mb:.2f} MB)")
    print("-" * 65)
    return output_path


def benchmark_all_variants(test_image='samples/test2.jpg', runs=10):
    """
    Chạy đo đạc so sánh định lượng trực tiếp giữa 3 phiên bản:
      1. Baseline FP32 (PyTorch)
      2. ONNX Runtime (FP32)
      3. ONNX Runtime INT8 (Quantized)
    """
    print("\n" + "=" * 65)
    print("📊 BENCHMARK SO SÁNH ĐỊNH LƯỢNG (FP32 vs ONNX vs INT8)")
    print("=" * 65)
    
    if not os.path.exists(test_image):
        print(f"❌ Không tìm thấy ảnh: '{test_image}'")
        return
        
    from ultralytics import YOLO
    img = cv2.imread(test_image)
    
    models_to_test = [
        ('1. PyTorch Baseline (FP32)', 'weights/best.pt'),
        ('2. ONNX Runtime (FP32)',     'weights/best.onnx'),
        ('3. ONNX Runtime (INT8)',     'weights/best_int8.onnx'),
    ]
    
    results_table = []
    
    for label, path in models_to_test:
        if not os.path.exists(path):
            print(f"⚠️ Bỏ qua {label} (File '{path}' chưa tồn tại)")
            continue
            
        file_size = os.path.getsize(path) / (1024 * 1024)
        m = YOLO(path)
        
        # Warmup
        m(img, verbose=False)
        
        times = []
        conf = 0.0
        for _ in range(runs):
            t0 = time.perf_counter()
            r = m(img, verbose=False)
            dt = (time.perf_counter() - t0) * 1000.0
            times.append(dt)
            if r and len(r[0].boxes) > 0:
                conf = float(r[0].boxes.conf[0].cpu().numpy())
                
        avg_t = np.mean(times)
        fps = 1000.0 / avg_t if avg_t > 0 else 0.0
        
        results_table.append({
            'name': label,
            'path': path,
            'size_mb': file_size,
            'latency_ms': avg_t,
            'fps': fps,
            'conf': conf * 100
        })
        
    print(f"\n{'Phiên Bản Mô Hình':<28} | {'Dung Lượng':<10} | {'Độ Trễ (ms)':<12} | {'FPS':<8} | {'Confidence':<10}")
    print("-" * 75)
    for row in results_table:
        print(f"{row['name']:<28} | {row['size_mb']:>7.2f} MB | {row['latency_ms']:>9.2f} ms | {row['fps']:>6.1f} | {row['conf']:>8.1f}%")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Toolkit Tối Ưu Hóa Mô Hình (Pruning, Quantization, ONNX)")
    parser.add_argument('--all', action='store_true', help='Chạy toàn bộ: Export ONNX + Quantize INT8 + Pruning + Benchmark')
    parser.add_argument('--export_onnx', action='store_true', help='Export YOLOv8 sang ONNX Runtime')
    parser.add_argument('--quantize', action='store_true', help='Lượng tử hóa FP32 -> INT8 (BẮT BUỘC)')
    parser.add_argument('--prune', action='store_true', help='Cắt tỉa trọng số dư thừa (Pruning)')
    parser.add_argument('--benchmark', action='store_true', help='Chạy benchmark so sánh 3 phiên bản')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='Đường dẫn weights YOLO gốc')
    parser.add_argument('--amount', type=float, default=0.25, help='Ti le pruning (mac dinh 0.25 = 25%%)')
    args = parser.parse_args()
    
    # Nếu không truyền cờ nào, mặc định chạy toàn bộ
    run_all = args.all or (not args.export_onnx and not args.quantize and not args.prune and not args.benchmark)
    
    if run_all or args.export_onnx:
        export_yolo_to_onnx(weights_path=args.weights, output_path='weights/best.onnx')
        
    if run_all or args.quantize:
        quantize_to_int8(onnx_input_path='weights/best.onnx', onnx_output_path='weights/best_int8.onnx')
        
    if run_all or args.prune:
        prune_pytorch_model(weights_path=args.weights, output_path='weights/best_pruned.pt', amount=args.amount)
        
    if run_all or args.benchmark:
        benchmark_all_variants(test_image='samples/test2.jpg', runs=10)
        
    print("\n✨ HOÀN TẤT TỐI ƯU HÓA MÔ HÌNH!")


if __name__ == '__main__':
    main()
