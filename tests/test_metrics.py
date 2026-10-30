"""Hand-computable fixtures; run with unittest, without model dependencies."""

import json
import unittest

from evaluation.metrics import (
    compute_iou, evaluate_detection, evaluate_ocr, iou_drop_rate,
    levenshtein_distance, normalize_plate, pair_ocr_records,
    predictions_from_pipeline, robustness_degradation,
)


def target(image="a", box=(0, 0, 10, 10), class_id=1, **extra):
    return dict(image_id=image, bbox=list(box), class_id=class_id, **extra)


def prediction(image="a", box=(0, 0, 10, 10), score=0.9, class_id=1, **extra):
    return dict(target(image, box, class_id, **extra), score=score)


class IoUTests(unittest.TestCase):
    def test_identical_and_disjoint(self):
        self.assertEqual(compute_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(compute_iou([0, 0, 10, 10], [10, 0, 20, 10]), 0)

    def test_partial_overlap_and_containment(self):
        self.assertAlmostEqual(compute_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        self.assertEqual(compute_iou([0, 0, 10, 10], [0, 0, 5, 5]), 0.25)

    def test_zero_area(self):
        self.assertEqual(compute_iou([0, 0, 0, 10], [0, 0, 0, 10]), 0)

    def test_invalid_boxes(self):
        for box in ([0, 0, -1, 1], [0, 0, 1], [0, 0, float("nan"), 1],
                    [0, 0, float("inf"), 1]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                compute_iou(box, [0, 0, 10, 10])

    def test_drop_and_missing_detection(self):
        self.assertAlmostEqual(iou_drop_rate([0, 0, 10, 10], [5, 0, 15, 10]), 2 / 3)
        self.assertEqual(iou_drop_rate([0, 0, 10, 10], None), 1)
        self.assertEqual(iou_drop_rate(None, None), 1)


class DetectionTests(unittest.TestCase):
    def test_perfect_detection(self):
        result = evaluate_detection([target()], [prediction()])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))
        self.assertEqual((result["precision"], result["recall"], result["map50"]), (1, 1, 1))

    def test_duplicate_is_false_positive(self):
        result = evaluate_detection([target()], [prediction(), prediction(score=0.8)])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertEqual(result["precision"], 0.5)
        # A lower-ranked duplicate after full recall does not lower AP.
        self.assertEqual(result["map50"], 1)

    def test_no_predictions(self):
        result = evaluate_detection([target()], [])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 1))
        self.assertEqual(result["map50"], 0)
        self.assertEqual(result["recall"], 0)

    def test_no_ground_truth(self):
        result = evaluate_detection([], [prediction()])
        self.assertEqual(result["fp"], 1)
        self.assertIsNone(result["map50"])
        self.assertIsNone(result["recall"])
        self.assertIsNone(evaluate_detection([], [])["map50"])

    def test_iou_boundary_inclusive(self):
        result = evaluate_detection([target()], [prediction(box=(0, 0, 5, 10))])
        self.assertEqual(result["tp"], 1)
        self.assertEqual(evaluate_detection([target()], [prediction(box=(0, 0, 4, 10))])["tp"], 0)

    def test_matching_requires_same_image_and_class(self):
        result = evaluate_detection([target()], [prediction(image="b"), prediction(class_id=0)])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 2, 1))

    def test_ap_ranking_and_envelope(self):
        # Ranked TP, FP, TP: AP = 1/2 * 1 + 1/2 * 2/3 = 5/6.
        preds = [prediction("b", score=0.7), prediction("missing", score=0.8),
                 prediction("a", score=0.9)]
        result = evaluate_detection([target("a"), target("b")], preds)
        self.assertAlmostEqual(result["map50"], 5 / 6)
        self.assertAlmostEqual(result["precision"], 2 / 3)

    def test_false_positive_before_true_positive(self):
        result = evaluate_detection([target()], [prediction("wrong", score=0.99), prediction()])
        self.assertEqual(result["map50"], 0.5)

    def test_partial_recall_limits_ap(self):
        result = evaluate_detection([target("a"), target("b")], [prediction()])
        self.assertEqual(result["map50"], 0.5)

    def test_confidence_filters_pr_but_not_ap(self):
        result = evaluate_detection([target()], [prediction(score=0.1)], confidence_threshold=0.5)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 1))
        self.assertEqual(result["map50"], 1)

    def test_macro_average_and_absent_class(self):
        result = evaluate_detection(
            [target(class_id=0), target(class_id=1)],
            [prediction(class_id=0), prediction(class_id=2)],
        )
        self.assertEqual(result["map50"], 0.5)
        self.assertEqual(result["fp"], 1)
        self.assertIsNone(result["per_class"][2]["ap50"])

    def test_equal_scores_keep_input_order(self):
        result = evaluate_detection([target()], [prediction("wrong"), prediction()])
        self.assertEqual(result["map50"], 0.5)

    def test_two_targets_cannot_share_prediction(self):
        result = evaluate_detection([target(), target()], [prediction()])
        self.assertEqual((result["tp"], result["fn"]), (1, 1))

    def test_prediction_uses_best_available_target(self):
        result = evaluate_detection(
            [target(box=(0, 0, 10, 10)), target(box=(2, 0, 12, 10))],
            [prediction(box=(2, 0, 12, 10)), prediction(score=0.8)],
        )
        self.assertEqual(result["tp"], 2)

    def test_invalid_confidence(self):
        for bad in (-0.1, 1.1, float("nan")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                evaluate_detection([target()], [prediction(score=bad)])
        with self.assertRaises(ValueError):
            evaluate_detection([], [], confidence_threshold=2)


class OCRTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_plate(" 30a-123.45\n"), "30A12345")
        self.assertEqual(normalize_plate("30A?123"), "30A?123")

    def test_edit_distance(self):
        for reference, predicted, expected in (
            ("", "", 0), ("ABC", "ABC", 0), ("ABC", "ADC", 1),
            ("ABC", "AB", 1), ("ABC", "ABCD", 1),
            ("kitten", "sitting", 3), ("", "ABC", 3), ("ABC", "", 3),
        ):
            with self.subTest(reference=reference, predicted=predicted):
                self.assertEqual(levenshtein_distance(reference, predicted), expected)

    def test_layouts_and_weighted_cer(self):
        result = evaluate_ocr([
            {"gt_text": "AB", "pred_text": "AX", "layout": "one_line"},
            {"gt_text": "ABCD", "pred_text": "ABCD", "layout": "two_line"},
        ])
        self.assertEqual(result["overall"]["em_percent"], 50)
        self.assertAlmostEqual(result["overall"]["cer"], 1 / 6)
        self.assertEqual(result["one_line"]["cer"], 0.5)
        self.assertEqual(result["two_line"]["em_percent"], 100)

    def test_missed_plate_counts_as_deletions(self):
        result = evaluate_ocr([{"gt_text": "ABC", "pred_text": "", "layout": "one_line"}])
        self.assertEqual(result["overall"]["cer"], 1)
        self.assertEqual(result["overall"]["em_percent"], 0)
        self.assertIsNone(result["two_line"]["em_percent"])

    def test_insertions_can_exceed_one(self):
        result = evaluate_ocr([{"gt_text": "A", "pred_text": "ABCD", "layout": "one_line"}])
        self.assertEqual(result["overall"]["cer"], 3)

    def test_empty_dataset(self):
        self.assertIsNone(evaluate_ocr([])["overall"]["cer"])

    def test_raw_mode(self):
        records = [{"gt_text": "30a-123", "pred_text": "30A123", "layout": "one_line"}]
        self.assertEqual(evaluate_ocr(records)["overall"]["em_percent"], 100)
        self.assertEqual(evaluate_ocr(records, normalize=False)["overall"]["em_percent"], 0)

    def test_invalid_reference_layout_and_type(self):
        for record in (
            {"gt_text": " .- ", "pred_text": "", "layout": "one_line"},
            {"gt_text": "A", "pred_text": "A", "layout": "unknown"},
        ):
            with self.subTest(record=record), self.assertRaises(ValueError):
                evaluate_ocr([record])
        with self.assertRaises(TypeError):
            evaluate_ocr([{"gt_text": "A", "pred_text": None, "layout": "one_line"}])


class IntegrationTests(unittest.TestCase):
    def test_pipeline_adapter_and_end_to_end_missed_plate(self):
        targets = [target("a", text="30A12345", layout="one_line"),
                   target("b", text="59B112345", layout="two_line")]
        preds = predictions_from_pipeline("a", [
            {"box": [0, 0, 10, 10], "detector_conf": 0.9, "formatted_plate": "30A-123.45"},
            {"box": [0, 0, 100, 100], "detector_conf": 0, "formatted_plate": "FALLBACK"},
        ])
        self.assertEqual(len(preds), 1)
        result = evaluate_ocr(pair_ocr_records(targets, preds))
        self.assertEqual(result["overall"]["em_percent"], 50)
        self.assertEqual(result["two_line"]["cer"], 1)
        # Valid JSON for runner/visualizer, including undefined values as null.
        json.dumps(result, allow_nan=False)

    def test_wrong_image_and_low_confidence_cannot_supply_ocr(self):
        targets = [target(text="ABC", layout="one_line")]
        preds = [prediction("b", text="ABC"), prediction(score=0.1, text="ABC")]
        self.assertEqual(pair_ocr_records(targets, preds)[0]["pred_text"], "")

    def test_duplicate_ocr_uses_confidence_not_text_correctness(self):
        targets = [target(text="ABC", layout="one_line")]
        preds = [prediction(score=0.9, text="WRONG"), prediction(score=0.8, text="ABC")]
        self.assertEqual(pair_ocr_records(targets, preds)[0]["pred_text"], "WRONG")

    def test_invalid_matching_threshold(self):
        with self.assertRaises(ValueError):
            pair_ocr_records([], [], iou_threshold=0)


class RobustnessTests(unittest.TestCase):
    def test_relative_percentage(self):
        self.assertEqual(robustness_degradation(80, 60), 25)
        self.assertEqual(robustness_degradation(80, 80), 0)
        self.assertEqual(robustness_degradation(80, 0), 100)

    def test_zero_baseline_and_improvement(self):
        self.assertIsNone(robustness_degradation(0, 0))
        self.assertIsNone(robustness_degradation(0, 50))
        self.assertEqual(robustness_degradation(50, 75), -50)

    def test_invalid_percentages(self):
        for clean, corrupted in ((101, 50), (50, -1), (float("nan"), 1)):
            with self.subTest(clean=clean, corrupted=corrupted), self.assertRaises(ValueError):
                robustness_degradation(clean, corrupted)


if __name__ == "__main__":
    unittest.main()
