"""Unit tests for synthetic corruptions and perturbation generator.

Tests 10 scenarios, 3 severity levels, BBox transformations, and determinism.
Author: Hy (Perturbation Engineer) - Phase 6
"""

import json
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from evaluation.corruptions import (
    ALL_SCENARIOS,
    SCENARIOS_BY_GROUP,
    SEVERITY_LEVELS,
    CorruptionBatchGenerator,
    apply_corruption,
    apply_defocus_blur,
    apply_jpeg_compression,
    apply_low_light,
    apply_motion_blur,
    apply_overexposure,
    apply_partial_occlusion,
    apply_perspective_tilt,
    apply_rain_streaks,
    apply_sensor_noise,
    apply_uneven_shadows,
    transform_bbox_perspective,
)


class TestCorruptions(unittest.TestCase):
    def setUp(self):
        # Create a standard synthetic test image (H=200, W=300, 3)
        self.img = np.full((200, 300, 3), 128, dtype=np.uint8)
        # Add some patterns/edges
        cv2.rectangle(self.img, (50, 50), (250, 150), (200, 200, 200), -1)
        cv2.putText(
            self.img,
            "51A-123.45",
            (70, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (20, 20, 20),
            2,
        )
        self.bboxes = [[70.0, 75.0, 160.0, 50.0]]

    def test_all_10_scenarios_execute(self):
        """Ensure all 10 scenarios run across all 3 severity levels without error."""
        self.assertEqual(len(ALL_SCENARIOS), 10)
        for scenario in ALL_SCENARIOS:
            for severity in SEVERITY_LEVELS:
                with self.subTest(scenario=scenario, severity=severity):
                    corrupted, out_boxes = apply_corruption(
                        scenario=scenario,
                        severity=severity,
                        image=self.img,
                        bboxes=self.bboxes,
                        seed=42,
                    )
                    self.assertIsInstance(corrupted, np.ndarray)
                    self.assertEqual(corrupted.shape, self.img.shape)
                    self.assertEqual(corrupted.dtype, np.uint8)
                    self.assertTrue(np.all(corrupted >= 0))
                    self.assertTrue(np.all(corrupted <= 255))
                    self.assertIsInstance(out_boxes, list)
                    self.assertGreater(len(out_boxes), 0)

    def test_determinism_with_seed(self):
        """Same seed must produce identical outputs for non-deterministic transforms."""
        for scenario in ["sensor_noise", "rain_streaks", "partial_occlusion", "overexposure"]:
            with self.subTest(scenario=scenario):
                out1, _ = apply_corruption(scenario, 2, self.img, self.bboxes, seed=12345)
                out2, _ = apply_corruption(scenario, 2, self.img, self.bboxes, seed=12345)
                self.assertTrue(
                    np.array_equal(out1, out2),
                    f"Output not deterministic for {scenario}",
                )

    def test_low_light_monotonicity(self):
        """Higher severity low_light must yield lower average luminance."""
        l1 = apply_low_light(self.img, severity=1)
        l2 = apply_low_light(self.img, severity=2)
        l3 = apply_low_light(self.img, severity=3)

        mean1 = float(np.mean(l1))
        mean2 = float(np.mean(l2))
        mean3 = float(np.mean(l3))

        self.assertGreater(mean1, mean2)
        self.assertGreater(mean2, mean3)

    def test_sensor_noise_monotonicity(self):
        """Higher severity sensor noise must yield higher difference from base image."""
        flat = np.full((100, 100, 3), 128, dtype=np.uint8)
        n1 = apply_sensor_noise(flat, severity=1, seed=42)
        n2 = apply_sensor_noise(flat, severity=2, seed=42)
        n3 = apply_sensor_noise(flat, severity=3, seed=42)

        std1 = np.std(n1.astype(np.float32) - 128.0)
        std2 = np.std(n2.astype(np.float32) - 128.0)
        std3 = np.std(n3.astype(np.float32) - 128.0)

        self.assertGreater(std2, std1)
        self.assertGreater(std3, std2)

    def test_defocus_blur_sharpness_decrease(self):
        """Higher severity defocus blur must reduce Laplacian variance (sharpness)."""
        b1 = apply_defocus_blur(self.img, severity=1)
        b2 = apply_defocus_blur(self.img, severity=2)
        b3 = apply_defocus_blur(self.img, severity=3)

        def laplacian_var(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()

        sharp1 = laplacian_var(b1)
        sharp2 = laplacian_var(b2)
        sharp3 = laplacian_var(b3)

        self.assertGreater(sharp1, sharp2)
        self.assertGreater(sharp2, sharp3)

    def test_perspective_tilt_bbox_reprojection(self):
        """Perspective tilt must accurately re-project bboxes and clamp within boundary."""
        box = [50.0, 60.0, 100.0, 40.0]
        t_img, t_boxes = apply_perspective_tilt(
            self.img, bboxes=[box], severity=2, seed=42
        )
        self.assertEqual(len(t_boxes), 1)
        t_box = t_boxes[0]
        tx, ty, tw, th = t_box

        # Must have valid positive dimensions
        self.assertGreater(tw, 0)
        self.assertGreater(th, 0)

        # Must be bounded by image dimensions
        h, w = self.img.shape[:2]
        self.assertGreaterEqual(tx, 0)
        self.assertGreaterEqual(ty, 0)
        self.assertLessEqual(tx + tw, w)
        self.assertLessEqual(ty + th, h)

    def test_perspective_tilt_extreme_clamping(self):
        """A box touching the outer edge must stay clamped within frame without throwing."""
        h, w = self.img.shape[:2]
        edge_box = [w - 20.0, h - 20.0, 40.0, 40.0]
        # Direct transform with homography
        matrix = np.array(
            [[1.2, 0.2, -10.0], [-0.1, 1.1, 5.0], [0.0005, 0.0002, 1.0]],
            dtype=np.float32,
        )
        clamped_box = transform_bbox_perspective(edge_box, matrix, w, h)
        if clamped_box is not None:
            cx, cy, cw, ch = clamped_box
            self.assertGreaterEqual(cx, 0)
            self.assertGreaterEqual(cy, 0)
            self.assertLessEqual(cx + cw, w)
            self.assertLessEqual(cy + ch, h)

    def test_invalid_scenario_or_severity(self):
        with self.assertRaises(ValueError):
            apply_corruption("unknown_corruption", 1, self.img)
        with self.assertRaises(ValueError):
            apply_corruption("low_light", 5, self.img)


class TestBatchGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, "source_images")
        os.makedirs(self.input_dir, exist_ok=True)

        # Create two sample images
        self.img1_name = "car1.jpg"
        self.img2_name = "bike1.jpg"
        img1 = np.full((100, 150, 3), 150, dtype=np.uint8)
        img2 = np.full((120, 160, 3), 120, dtype=np.uint8)
        cv2.imwrite(os.path.join(self.input_dir, self.img1_name), img1)
        cv2.imwrite(os.path.join(self.input_dir, self.img2_name), img2)

        # Sample COCO JSON
        self.coco_file = os.path.join(self.temp_dir, "_annotations.coco.json")
        coco_data = {
            "images": [
                {"id": 1, "file_name": self.img1_name, "width": 150, "height": 100},
                {"id": 2, "file_name": self.img2_name, "width": 160, "height": 120},
            ],
            "annotations": [
                {"id": 0, "image_id": 1, "category_id": 1, "bbox": [20, 30, 50, 25]},
                {"id": 1, "image_id": 2, "category_id": 1, "bbox": [30, 40, 60, 30]},
            ],
            "categories": [{"id": 1, "name": "license-plate"}],
        }
        with open(self.coco_file, "w") as f:
            json.dump(coco_data, f)

        self.output_dir = os.path.join(self.temp_dir, "corrupted_output")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_batch_generator_pipeline(self):
        generator = CorruptionBatchGenerator(
            input_dir=self.input_dir,
            annotations_file=self.coco_file,
            output_dir=self.output_dir,
            seed=42,
        )

        manifest = generator.generate(
            scenarios=["low_light", "perspective_tilt"],
            severities=[1, 2],
            sample_size=None,
            visualize=True,
        )

        self.assertEqual(len(manifest["datasets"]), 4)  # 2 scenarios x 2 severities

        manifest_file = os.path.join(self.output_dir, "manifest.json")
        self.assertTrue(os.path.exists(manifest_file))

        # Check low_light level 1
        l1_coco_path = os.path.join(
            self.output_dir, "low_light", "level_1", "_annotations.coco.json"
        )
        self.assertTrue(os.path.exists(l1_coco_path))
        with open(l1_coco_path, "r") as f:
            l1_coco = json.load(f)
        self.assertEqual(len(l1_coco["images"]), 2)
        self.assertEqual(len(l1_coco["annotations"]), 2)
        self.assertIn("bbox_xyxy", l1_coco["annotations"][0])

        # Check perspective_tilt level 2
        tilt_coco_path = os.path.join(
            self.output_dir, "perspective_tilt", "level_2", "_annotations.coco.json"
        )
        self.assertTrue(os.path.exists(tilt_coco_path))
        with open(tilt_coco_path, "r") as f:
            tilt_coco = json.load(f)
        self.assertEqual(len(tilt_coco["images"]), 2)
        self.assertEqual(len(tilt_coco["annotations"]), 2)


if __name__ == "__main__":
    unittest.main()

