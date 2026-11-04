"""Dataset adapters for the Phase 6 runner; no inference dependencies."""
import hashlib
import json
import math
import random
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(root, value):
    path = (root / Path(str(value).replace("\\", "/"))).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside project root: {value}")
    return path


def load_dataset(root, config, source, sample_size=None, seed=42, variant=None):
    entry = config["datasets"][source]
    gt_path = resolve(root, entry["ground_truth"])
    labels = read_json(gt_path)
    image_dir = resolve(root, entry["images"])
    coco = read_json(resolve(root, entry["annotations"])) if entry.get("annotations") else None
    names = [i["file_name"] for i in coco["images"]] if coco else [
        p.relative_to(image_dir).as_posix() for p in image_dir.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image file names")
    original = {g["ann_id"]: g for g in labels}
    if len(original) != len(labels):
        raise ValueError("Duplicate annotation IDs")
    if variant:
        image_dir = resolve(root, variant["image_dir"])
        coco = read_json(resolve(root, variant["annotations_file"]))
        names = [i["file_name"] for i in coco["images"]]
        by_id = {i["id"]: i for i in coco["images"]}
        mapped = []
        for ann in coco["annotations"]:
            if "original_annotation_id" not in ann:
                raise ValueError("Corrupted annotations lack original_annotation_id; regenerate with lineage")
            clean = original[ann["original_annotation_id"]]
            name = by_id[ann["image_id"]]["file_name"]
            if clean["file_name"] != name:
                raise ValueError("Corruption lineage image mismatch")
            x, y, w, h = ann["bbox"]
            mapped.append(dict(clean, bbox=[x, y, x+w, y+h]))
        labels = mapped
    selected = sorted(names)
    if sample_size:
        selected = sorted(random.Random(seed).sample(selected, min(sample_size, len(selected))))
    if not selected:
        raise ValueError("Empty dataset")
    records, gt, ids = [], [], set()
    for name in selected:
        path = resolve(image_dir, name)
        if not path.is_file():
            raise ValueError(f"Missing image: {path}")
        image_id = f"{source}/{name}"
        records.append({"image_id": image_id, "path": path})
        for g in labels:
            if g["file_name"] != name:
                continue
            box = g["bbox"]
            if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
                raise ValueError(f"Invalid bbox: {name}")
            if not (0 <= box[0] < box[2] <= g["width"] and 0 <= box[1] < box[3] <= g["height"]):
                raise ValueError(f"Out-of-bounds bbox: {name}")
            if g["class_id"] != 1:
                raise ValueError("Explicit class mapping required; supported plate class is 1")
            if g["layout"] not in ("one_line", "two_line"):
                raise ValueError(f"Invalid layout: {name}")
            plate_id = f"{source}/{name}/{g['ann_id']}"
            if plate_id in ids:
                raise ValueError("Duplicate plate ID")
            ids.add(plate_id)
            gt.append(dict(image_id=image_id, plate_id=plate_id, bbox=box,
                           class_id=1, text=g["text"], layout=g["layout"]))
    if coco:
        expected = {i["id"]: i["file_name"] for i in coco["images"]}
        for ann in coco["annotations"]:
            name = expected[ann["image_id"]]
            if name not in selected:
                continue
            original_id = ann.get("original_annotation_id", ann["id"])
            if f"{source}/{name}/{original_id}" not in ids:
                raise ValueError(f"Missing GT annotation: {name}/{original_id}")
    elif set(selected) != {g["file_name"] for g in labels if g["file_name"] in selected}:
        raise ValueError("Images without annotation inventory; cannot assume negative images")
    invalid = [g["plate_id"] for g in gt if not str(g["text"]).strip() or g["text"].upper() == "UNKNOWN"]
    digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]
    return {"records": records, "ground_truth": gt, "invalid_ocr": invalid,
            "labels_verified": entry.get("labels_verified", False), "cohort_id": digest,
            "annotation_sha256": hashlib.sha256(gt_path.read_bytes()).hexdigest()}
