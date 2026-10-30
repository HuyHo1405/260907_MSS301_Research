"""Independent ALPR metrics. Standard library only; see METRICS.md for conventions."""

import math
import re
import unicodedata


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _unit(value, name):
    value = _finite(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _box(box):
    if len(box) != 4:
        raise ValueError("bbox must contain [x1, y1, x2, y2]")
    x1, y1, x2, y2 = [_finite(v, "bbox coordinate") for v in box]
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox must satisfy x2 >= x1 and y2 >= y1")
    return x1, y1, x2, y2


def compute_iou(box_a, box_b):
    """Continuous xyxy coordinates (no +1). Zero-area boxes have IoU 0."""
    ax1, ay1, ax2, ay2 = _box(box_a)
    bx1, by1, bx2, by2 = _box(box_b)
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def iou_drop_rate(clean_box, corrupted_box):
    """1 - IoU in [0,1]; None for either box means missing detection (drop 1).

    Caller must align boxes into the same coordinate frame before calling.
    """
    if clean_box is not None:
        _box(clean_box)
    if corrupted_box is not None:
        _box(corrupted_box)
    if clean_box is None or corrupted_box is None:
        return 1.0
    return 1.0 - compute_iou(clean_box, corrupted_box)


def _detections(records, predictions=False):
    checked = []
    for record in records:
        image_id = record["image_id"]
        class_id = record.get("class_id", 1)
        # These identifiers are dictionary keys during matching.
        hash(image_id)
        hash(class_id)
        item = dict(record, image_id=image_id, class_id=class_id,
                    bbox=_box(record["bbox"]))
        if predictions:
            item["score"] = _unit(record["score"], "score")
        checked.append(item)
    return checked


def _match(predictions, ground_truth, threshold):
    """Confidence-greedy, one-to-one matching within each image and class."""
    groups = {}
    for index, gt in enumerate(ground_truth):
        groups.setdefault((gt["image_id"], gt["class_id"]), []).append(index)
    used = set()
    outcomes = []
    for prediction in sorted(predictions, key=lambda p: p["score"], reverse=True):
        candidates = groups.get((prediction["image_id"], prediction["class_id"]), [])
        best_index, best_iou = None, -1.0
        for index in candidates:
            if index not in used:
                overlap = compute_iou(prediction["bbox"], ground_truth[index]["bbox"])
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
        matched = best_index is not None and best_iou >= threshold
        if matched:
            used.add(best_index)
        outcomes.append((prediction, best_index if matched else None))
    return outcomes


def _average_precision(outcomes, gt_count):
    """All-points interpolated precision-envelope area, NOT COCO 101-point AP."""
    if gt_count == 0:
        return None
    precisions, recalls = [0.0], [0.0]
    true_positives = 0
    for rank, (_, gt_index) in enumerate(outcomes, 1):
        true_positives += gt_index is not None
        precisions.append(true_positives / rank)
        recalls.append(true_positives / gt_count)
    precisions.append(0.0)
    recalls.append(1.0)
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    return sum((recalls[i] - recalls[i - 1]) * precisions[i]
               for i in range(1, len(recalls)) if recalls[i] != recalls[i - 1])


def evaluate_detection(ground_truth, predictions, confidence_threshold=0.25):
    """P/R at fixed confidence, AP50 on ALL supplied predictions.

    Records: image_id, bbox, optional class_id (default 1: license plate).
    Predictions additionally require score. AP is averaged over classes with GT.
    Call with plate-only records if only license-plate evaluation is intended.
    """
    confidence_threshold = _unit(confidence_threshold, "confidence_threshold")
    ground_truth = _detections(ground_truth)
    predictions = _detections(predictions, predictions=True)
    classes = list(dict.fromkeys(r["class_id"] for r in ground_truth + predictions))
    per_class = []
    for class_id in classes:
        targets = [r for r in ground_truth if r["class_id"] == class_id]
        detected = [r for r in predictions if r["class_id"] == class_id]
        outcomes = _match(detected, targets, 0.5)
        selected = [(p, index) for p, index in outcomes if p["score"] >= confidence_threshold]
        tp = sum(index is not None for _, index in selected)
        fp, fn = len(selected) - tp, len(targets) - tp
        per_class.append({
            "class_id": class_id, "gt_count": len(targets),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / len(targets) if targets else None,
            "ap50": _average_precision(outcomes, len(targets)),
        })
    tp = sum(r["tp"] for r in per_class)
    fp = sum(r["fp"] for r in per_class)
    fn = sum(r["fn"] for r in per_class)
    aps = [r["ap50"] for r in per_class if r["ap50"] is not None]
    return {
        "iou_threshold": 0.5, "confidence_threshold": confidence_threshold,
        "ap_method": "all_points_precision_envelope",
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else None,
        "map50": sum(aps) / len(aps) if aps else None,
        "per_class": per_class,
    }


def normalize_plate(text):
    """Uppercase/NFC; remove whitespace, periods and hyphens only.

    Preserve other characters so an OCR error is not silently discarded.
    """
    if not isinstance(text, str):
        raise TypeError("plate text must be a string")
    return re.sub(r"[\s.\-]+", "", unicodedata.normalize("NFC", text).upper())


def levenshtein_distance(reference, prediction):
    """Edit distance with O(min(len(reference), len(prediction))) space."""
    if not isinstance(reference, str) or not isinstance(prediction, str):
        raise TypeError("texts must be strings")
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i]
        for j, pred_char in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref_char != pred_char)))
        previous = current
    return previous[-1]


def evaluate_ocr(records, normalize=True):
    """Paired plate records: gt_text, pred_text, layout ('one_line'/'two_line').

    Use pred_text='' for a missed plate. Empty normalized GT is invalid.
    CER is corpus-weighted and may exceed 1 due to insertions.
    """
    groups = {name: {"count": 0, "exact_matches": 0, "edit_distance": 0,
                     "reference_characters": 0}
              for name in ("overall", "one_line", "two_line")}
    for record in records:
        layout = record["layout"]
        if layout not in ("one_line", "two_line"):
            raise ValueError("layout must be 'one_line' or 'two_line'")
        reference, prediction = record["gt_text"], record["pred_text"]
        if not isinstance(reference, str) or not isinstance(prediction, str):
            raise TypeError("gt_text and pred_text must be strings")
        if normalize:
            reference, prediction = normalize_plate(reference), normalize_plate(prediction)
        if not reference:
            raise ValueError("gt_text must not be empty after normalization")
        distance = levenshtein_distance(reference, prediction)
        for key in ("overall", layout):
            group = groups[key]
            group["count"] += 1
            group["exact_matches"] += reference == prediction
            group["edit_distance"] += distance
            group["reference_characters"] += len(reference)
    for group in groups.values():
        group["em_percent"] = (100 * group["exact_matches"] / group["count"]
                               if group["count"] else None)
        group["cer"] = (group["edit_distance"] / group["reference_characters"]
                        if group["reference_characters"] else None)
    return groups


def pair_ocr_records(ground_truth, predictions, iou_threshold=0.5,
                     confidence_threshold=0.25):
    """Build end-to-end OCR pairs via box matching; GT supplies text/layout.

    Each GT plate is included, even if missed. Unmatched predictions are counted
    by detection FP, not added as imaginary GT plates to the OCR denominator.
    """
    threshold = _unit(iou_threshold, "iou_threshold")
    if threshold == 0:
        raise ValueError("iou_threshold must be > 0")
    confidence = _unit(confidence_threshold, "confidence_threshold")
    targets = _detections(ground_truth)
    detected = [p for p in _detections(predictions, predictions=True)
                if p["score"] >= confidence]
    matched = {index: prediction for prediction, index in _match(detected, targets, threshold)
               if index is not None}
    return [{"image_id": gt["image_id"], "gt_text": gt["text"],
             "pred_text": matched[index]["text"] if index in matched else "",
             "layout": gt["layout"]}
            for index, gt in enumerate(targets)]


def predictions_from_pipeline(image_id, results, class_id=1):
    """Convert ALPRPipeline.predict results; exclude full-image OCR fallback.

    The current pipeline marks fallback with detector_conf=0. It is not a real
    detection and must not become an AP candidate.
    """
    converted = []
    for result in results:
        score = _unit(result["detector_conf"], "detector_conf")
        if score == 0:
            continue
        converted.append({"image_id": image_id, "class_id": class_id,
                          "bbox": list(_box(result["box"])), "score": score,
                          "text": result["formatted_plate"]})
    return converted


def robustness_degradation(clean_em_percent, corrupted_em_percent):
    """Relative EM degradation in percent. None if clean baseline is zero.

    Negative values indicate an improvement; do not clamp them to zero.
    """
    clean = _finite(clean_em_percent, "clean_em_percent")
    corrupted = _finite(corrupted_em_percent, "corrupted_em_percent")
    if not (0 <= clean <= 100 and 0 <= corrupted <= 100):
        raise ValueError("EM values must be percentages in [0, 100]")
    return (clean - corrupted) / clean * 100 if clean else None
