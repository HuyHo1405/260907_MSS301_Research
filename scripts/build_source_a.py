"""Build clean, balanced Source A dataset (150 cars + 150 motorbikes) for Phase 6.

Reads clean, non-augmented images and COCO BBoxes from
`datasets/vietnamese license plate.v1i.coco.zip`, runs RapidOCR (upscaling small
ROIs so DBNet does not miss characters), filters strictly for valid unique
Vietnamese license plates with zero UNKNOWN entries, exports:
  - datasets/source_a/images/
  - datasets/source_a/_annotations.coco.json
  - datasets/source_a/ground_truth_source_a.json
  - datasets/source_a/metadata_source_a.json
  - output/source_a_review/index.html (cropped ROI review sheet for human QA)
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from evaluation.metrics import (
    evaluate_detection,
    evaluate_ocr,
    normalize_plate,
    pair_ocr_records,
)
from pipeline.recognizer_rapid import RapidOCRRecognizer
from scripts.process_ground_truth import clean_plate_string

DEFAULT_ZIP = Path("datasets/vietnamese license plate.v1i.coco.zip")
DEFAULT_OUT = Path("datasets/source_a")

# Strict Vietnamese plate pattern: 2-digit province (11..99) + 1 letter + optional char + 4-5 digits.
VN_PLATE_RE = re.compile(r"^([1-9][0-9])([A-Z])([A-Z0-9]?)([0-9]{4,5})$")


def is_strict_vn_plate(text: str) -> bool:
    """Return True if text normalizes to a valid Vietnamese license plate."""
    if not text or text.upper() == "UNKNOWN":
        return False
    norm = re.sub(r"[\s.\-]+", "", text.upper())
    match = VN_PLATE_RE.match(norm)
    if not match:
        return False
    return 11 <= int(match.group(1)) <= 99


def classify_image_group(file_name: str) -> str | None:
    """Return 'car' or 'moto' for clean un-augmented filenames, else None."""
    if file_name.startswith("CarLongPlate") and not file_name.startswith("CarLongPlateGen"):
        return "car"
    if file_name.startswith("xemay"):
        return "moto"
    return None


def recognize_roi(recognizer: RapidOCRRecognizer, roi: np.ndarray) -> tuple[list[str], str]:
    """Run RapidOCR on a plate ROI, upscaling small crops so DBNet detects text."""
    if roi.shape[0] < 64:
        scale = 96.0 / max(roi.shape[0], 1)
        roi_in = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        roi_in = roi
    raw_texts, formatted, _ = recognizer.recognize(roi_in)
    cleaned = formatted if len(raw_texts) > 2 else clean_plate_string(formatted, raw_texts)
    return raw_texts, cleaned


def collect_clean_candidates(
    zip_path: Path,
    recognizer: RapidOCRRecognizer,
    target_per_group: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Collect up to `target_per_group` unique-plate clean candidates per group."""
    buckets = {"car": [], "moto": []}
    seen_plates = {"car": set(), "moto": set()}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for split in ("test", "valid", "train"):
            if all(len(buckets[g]) >= target_per_group for g in ("car", "moto")):
                break
            coco_member = f"{split}/_annotations.coco.json"
            if coco_member not in zf.namelist():
                continue
            coco = json.loads(zf.read(coco_member).decode("utf-8"))
            anns_by_img = defaultdict(list)
            for ann in coco.get("annotations", []):
                anns_by_img[ann["image_id"]].append(ann)

            images = list(coco.get("images", []))
            random.Random(seed).shuffle(images)

            for im in images:
                grp = classify_image_group(im["file_name"])
                if grp is None or len(buckets[grp]) >= target_per_group:
                    continue
                anns = anns_by_img.get(im["id"], [])
                if not anns:
                    continue
                member = f"{split}/{im['file_name']}"
                if member not in zf.namelist():
                    continue
                raw_bytes = zf.read(member)
                img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                h, w = img.shape[:2]
                plate_items = []
                plate_keys = []
                valid = True
                for ann in anns:
                    bx, by, bw, bh = [float(v) for v in ann["bbox"]]
                    x1, y1 = max(0.0, bx), max(0.0, by)
                    x2, y2 = min(float(w), bx + bw), min(float(h), by + bh)
                    cw, ch = x2 - x1, y2 - y1
                    if cw < 10.0 or ch < 8.0:
                        valid = False
                        break
                    roi = img[int(y1):int(y2), int(x1):int(x2)]
                    if roi.size == 0:
                        valid = False
                        break
                    raw_texts, text = recognize_roi(recognizer, roi)
                    if not is_strict_vn_plate(text):
                        valid = False
                        break
                    ar = cw / max(ch, 1.0)
                    layout = "two_line" if (len(raw_texts) >= 2 or ar < 1.85) else "one_line"
                    plate_items.append({
                        "bbox_coco": [round(x1, 2), round(y1, 2), round(cw, 2), round(ch, 2)],
                        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "text": text,
                        "raw_ocr": raw_texts,
                        "layout": layout,
                        "aspect_ratio": round(ar, 2),
                    })
                    plate_keys.append(normalize_plate(text))

                if not valid or any(k in seen_plates[grp] for k in plate_keys):
                    continue
                seen_plates[grp].update(plate_keys)
                buckets[grp].append({
                    "split": split,
                    "file_name": im["file_name"],
                    "width": w,
                    "height": h,
                    "raw_bytes": raw_bytes,
                    "plates": plate_items,
                    "vehicle_type": grp,
                })

    return buckets["car"], buckets["moto"]


def export_source_a(
    selected: list[dict],
    out_dir: Path,
    review_dir: Path | None,
) -> dict:
    """Write clean Source A images, COCO JSON, GT JSON, metadata, and QA review HTML."""
    images_dir = out_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    crops_dir = review_dir / "crops" if review_dir else None
    if review_dir:
        if review_dir.exists():
            shutil.rmtree(review_dir)
        crops_dir.mkdir(parents=True, exist_ok=True)

    coco_images = []
    coco_annotations = []
    gt_records = []
    review_cards = []
    ann_id = 0

    for img_id, item in enumerate(selected):
        fn = item["file_name"]
        (images_dir / fn).write_bytes(item["raw_bytes"])
        coco_images.append({
            "id": img_id,
            "file_name": fn,
            "width": item["width"],
            "height": item["height"],
            "source_split": item["split"],
            "vehicle_type": item["vehicle_type"],
        })

        img = cv2.imdecode(np.frombuffer(item["raw_bytes"], np.uint8), cv2.IMREAD_COLOR) if review_dir else None

        for p in item["plates"]:
            bx, by, bw, bh = p["bbox_coco"]
            x1, y1, x2, y2 = p["bbox"]
            coco_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [bx, by, bw, bh],
                "bbox_xyxy": [x1, y1, x2, y2],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
                "segmentation": [],
                "text": p["text"],
                "layout": p["layout"],
                "raw_ocr": p["raw_ocr"],
            })
            gt_records.append({
                "image_id": f"source_a/{fn}",
                "file_name": fn,
                "ann_id": ann_id,
                "width": item["width"],
                "height": item["height"],
                "class_id": 1,
                "bbox": [x1, y1, x2, y2],
                "bbox_coco": [bx, by, bw, bh],
                "text": p["text"],
                "raw_ocr": p["raw_ocr"],
                "layout": p["layout"],
                "aspect_ratio": p["aspect_ratio"],
                "vehicle_type": item["vehicle_type"],
            })
            if review_dir and img is not None:
                crop_name = f"ann_{ann_id:04d}.jpg"
                roi = img[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                cv2.imwrite(str(crops_dir / crop_name), roi)
                review_cards.append(
                    f'<div class="card"><img src="crops/{crop_name}" alt="{html.escape(p["text"])}">'
                    f'<div class="meta"><b>#{ann_id}</b> [{item["vehicle_type"]} | {p["layout"]}]<br>'
                    f'<code>{html.escape(p["text"])}</code><br>'
                    f'<small>{html.escape(fn)}</small></div></div>'
                )
            ann_id += 1

    coco_doc = {
        "info": {"description": "Phase 6 Source A Clean Baseline (school-fuhih/vietnamese-license-plate-tptd0)"},
        "licenses": [{"id": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}],
        "categories": [
            {"id": 0, "name": "license-plate", "supercategory": "none"},
            {"id": 1, "name": "license-plate", "supercategory": "license-plate"},
        ],
        "images": coco_images,
        "annotations": coco_annotations,
    }
    (out_dir / "_annotations.coco.json").write_text(
        json.dumps(coco_doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "ground_truth_source_a.json").write_text(
        json.dumps(gt_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = {
        "total_images": len(coco_images),
        "total_plates": len(gt_records),
        "vehicle_distribution": {
            "car": sum(1 for item in selected if item["vehicle_type"] == "car"),
            "moto": sum(1 for item in selected if item["vehicle_type"] == "moto"),
        },
        "layout_distribution": {
            "one_line": sum(1 for g in gt_records if g["layout"] == "one_line"),
            "two_line": sum(1 for g in gt_records if g["layout"] == "two_line"),
        },
        "splits": {
            s: sum(1 for item in selected if item["split"] == s)
            for s in ("test", "valid", "train")
        },
        "labels_verified": False,
        "note": "OCR pseudo-labels generated by RapidOCR; human review required before setting labels_verified=true.",
    }
    (out_dir / "metadata_source_a.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if review_dir:
        page = (
            "<!doctype html><html><meta charset='utf-8'><title>Source A Label Review</title>"
            "<style>body{font:14px system-ui;margin:24px;background:#f7f7f7}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}"
            ".card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px}"
            "img{max-width:100%;max-height:110px;display:block;margin-bottom:6px;border:1px solid #eee}"
            "code{font-size:16px;font-weight:700;color:#0a5}</style>"
            f"<h1>Source A Review ({len(gt_records)} plates / {len(coco_images)} images)</h1>"
            "<div class='grid'>" + "".join(review_cards) + "</div></html>"
        )
        (review_dir / "index.html").write_text(page, encoding="utf-8")

    # Self-check with evaluation/metrics.py
    mock_preds = [
        {"image_id": g["image_id"], "class_id": 1, "bbox": g["bbox"], "score": 0.99, "text": g["text"]}
        for g in gt_records
    ]
    det = evaluate_detection(gt_records, mock_preds, confidence_threshold=0.25)
    ocr = evaluate_ocr(pair_ocr_records(gt_records, mock_preds, confidence_threshold=0.25))
    assert det["tp"] == len(gt_records) and det["fp"] == 0 and det["fn"] == 0
    assert det["map50"] == 1.0
    assert ocr["overall"]["em_percent"] == 100.0 and ocr["overall"]["cer"] == 0.0
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--review-dir", type=Path, default=None)
    parser.add_argument("--per-group", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    recognizer = RapidOCRRecognizer(target_height=96)
    cars, motos = collect_clean_candidates(args.zip_path, recognizer, args.per_group, args.seed)
    if len(cars) < args.per_group or len(motos) < args.per_group:
        raise RuntimeError(
            f"Insufficient clean candidates: car={len(cars)}, moto={len(motos)}, requested={args.per_group}"
        )
    selected = cars[:args.per_group] + motos[:args.per_group]
    random.Random(args.seed).shuffle(selected)

    summary = export_source_a(selected, args.output_dir, args.review_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
