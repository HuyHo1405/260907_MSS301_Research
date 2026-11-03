# 📂 Source B: Real-World Challenging Test Dataset
## Phụ trách: Khánh (Field Data & QA Lead) - Phase 6

Thư mục này lưu trữ tập dữ liệu kiểm thử thực địa điều kiện bất lợi tại cổng an ninh (60 ảnh chọn lọc) dùng làm tập thẩm định đối chiếu độc lập (**Empirical Sanity Check**).

---

## 🗂️ Cấu Trúc Thư Mục

```text
datasets/source_b/
├── images/                         # 60 ảnh thực tế điều kiện bất lợi
├── ground_truth_source_b.json      # File nhãn Ground Truth chuẩn (tương thích 100% metrics.py)
├── metadata_source_b.json          # File metadata gắn thẻ điều kiện môi trường
└── README.md                       # Tài liệu hướng dẫn này
```

---

## 📊 Phân Phối Dữ Liệu 4 Nhóm Thử Thách

* **Low Light (15 ảnh):** Điều kiện ban đêm, hầm xe thiếu sáng (`mean(gray) < 65`).
* **Overexposure / Glare (15 ảnh):** Đèn pha chiếu rọi, chói lóa (`mean(gray) > 175`).
* **Defocus / Motion Blur (15 ảnh):** Phương tiện đang di chuyển, lệch nét (`Var(Laplacian) < 120`).
* **Perspective Tilt / Two-line (15 ảnh):** Góc máy nghiêng và biển vuông xe máy 2 dòng.

---

## ⚡ Lệnh Kiểm Toán QA Nhanh

```powershell
python evaluation/validate_dataset_qa.py
```

Chi tiết báo cáo khoa học và kịch bản bảo vệ: xem tại **[`doc/phase_6_source_b_field_data_and_qa.md`](../../doc/phase_6_source_b_field_data_and_qa.md)**.

