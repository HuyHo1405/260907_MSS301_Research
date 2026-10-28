# 📂 Quản Lý Tập Dữ Liệu (Datasets)

Thư mục lưu trữ hướng dẫn cấu trúc dataset, metadata nhãn và scripts tiền xử lý dữ liệu.

> **Quy định:** Không commit các file ảnh hoặc nén dữ liệu dung lượng lớn (`.zip`, `.tar.gz`) vào Git. Sử dụng đường dẫn symbolic link hoặc download script từ cloud storage.

---

## 🗂️ Cấu Trúc Dataset Chuẩn

```text
datasets/
├── license_plates/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── data.yaml                 # File cấu hình định dạng YOLO
```

---

## 🔖 Các Lớp Đối Tượng (Classes)
- `0: vehicle` (Ô tô, Xe máy)
- `1: license_plate` (Biển số xe Việt Nam)
