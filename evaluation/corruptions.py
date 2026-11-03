"""Robustness Evaluation & Synthetic Perturbations Module.

Provides 4 corruption groups across 10 realistic scenarios with 3 severity levels:
1. Lighting & Optics: low_light, overexposure, uneven_shadows
2. Blur & Motion: motion_blur, defocus_blur
3. Environment & Transmission: rain_streaks, sensor_noise, jpeg_compression
4. Geometry & Physical: perspective_tilt (with BBox re-projection), partial_occlusion

Author: Hy (Perturbation Engineer) - Phase 6
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants & Metadata
# ---------------------------------------------------------------------------

SCENARIOS_BY_GROUP = {
    "lighting": ["low_light", "overexposure", "uneven_shadows"],
    "blur": ["motion_blur", "defocus_blur"],
    "weather_transmission": ["rain_streaks", "sensor_noise", "jpeg_compression"],
    "geometry_physical": ["perspective_tilt", "partial_occlusion"],
}

ALL_SCENARIOS = [
    scenario for scenarios in SCENARIOS_BY_GROUP.values() for scenario in scenarios
]

SEVERITY_LEVELS = [1, 2, 3]


def _get_rng(seed: Optional[int] = None) -> np.random.RandomState:
    """Return a deterministic RandomState if seed is provided, else unseeded."""
    return np.random.RandomState(seed)


# ---------------------------------------------------------------------------
# Group 1: Lighting & Optical Corruptions
# ---------------------------------------------------------------------------


def apply_low_light(
    image: np.ndarray,
    severity: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate low-light / night conditions via HSV value scaling and gamma reduction.

    Level 1: Mild darkness (V * 0.70, gamma 0.85)
    Level 2: Underground parking / dusk (V * 0.45, gamma 0.65)
    Level 3: Deep night / poorly lit street (V * 0.25, gamma 0.40)
    """
    params = {
        1: (0.70, 0.85),
        2: (0.45, 0.65),
        3: (0.25, 0.40),
    }
    v_scale, gamma = params.get(severity, params[1])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Gamma correction table
    inv_gamma = 1.0 / max(gamma, 1e-4)
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
        np.uint8
    )
    return cv2.LUT(bgr, lut)


def apply_overexposure(
    image: np.ndarray,
    bboxes: Optional[List[List[float]]] = None,
    severity: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate intense glare or direct headlight reflection on license plate.

    Level 1: Mild glare (intensity +70, radius 25% of plate)
    Level 2: Moderate headlight glare (intensity +150, radius 45% of plate)
    Level 3: Severe blinding flare (intensity +240, radius 80% of plate)
    """
    rng = _get_rng(seed)
    h, w = image.shape[:2]
    out = image.astype(np.float32)

    params = {
        1: (70.0, 0.25),
        2: (150.0, 0.45),
        3: (240.0, 0.80),
    }
    max_intensity, rad_factor = params.get(severity, params[1])

    centers_and_radii = []
    if bboxes and len(bboxes) > 0:
        for box in bboxes:
            bx, by, bw, bh = box
            cx = bx + bw / 2.0 + rng.uniform(-0.15, 0.15) * bw
            cy = by + bh / 2.0 + rng.uniform(-0.15, 0.15) * bh
            radius = max(bw, bh) * rad_factor
            centers_and_radii.append((cx, cy, radius))
    else:
        # Fallback to image center if no bbox provided
        centers_and_radii.append((w / 2.0, h / 2.0, min(w, h) * rad_factor * 0.4))

    y_grid, x_grid = np.ogrid[:h, :w]
    for cx, cy, radius in centers_and_radii:
        sigma = max(radius / 2.0, 1.0)
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        glare_blob = np.exp(-dist_sq / (2.0 * sigma * sigma))
        glare_blob = np.clip(glare_blob * max_intensity, 0, 255)
        for c in range(3):
            out[:, :, c] += glare_blob

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_uneven_shadows(
    image: np.ndarray,
    bboxes: Optional[List[List[float]]] = None,
    severity: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate harsh shadows cast by roof barriers, tree branches, or gate edges.

    Level 1: Light shadow (darken 35%, soft boundary)
    Level 2: Strong shadow (darken 55%, across half of plate/vehicle)
    Level 3: Pitch shadow (darken 75%, multiple sharp shadow bands)
    """
    rng = _get_rng(seed)
    h, w = image.shape[:2]

    params = {
        1: (0.65, 1),
        2: (0.45, 1),
        3: (0.25, 2),
    }
    attenuation, num_bands = params.get(severity, params[1])

    shadow_mask = np.ones((h, w), dtype=np.float32)

    for _ in range(num_bands):
        # Create a polygonal shadow band crossing the image
        angle = rng.uniform(-math.pi / 4, math.pi / 4)
        offset_x = rng.uniform(0.1, 0.9) * w
        p1 = (int(offset_x - math.tan(angle) * h), 0)
        p2 = (int(offset_x + math.tan(angle) * h), h)
        p3 = (w, h)
        p4 = (w, 0)
        poly = np.array([p1, p2, p3, p4], dtype=np.int32)

        band_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(band_mask, [poly], 1.0)

        # Soften shadow edge
        ksize = int(min(w, h) * 0.05) | 1
        band_mask = cv2.GaussianBlur(band_mask, (ksize, ksize), 0)
        shadow_mask = np.minimum(shadow_mask, 1.0 - band_mask * (1.0 - attenuation))

    out = image.astype(np.float32)
    for c in range(3):
        out[:, :, c] *= shadow_mask

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Group 2: Blur & Motion Corruptions
# ---------------------------------------------------------------------------


def apply_motion_blur(
    image: np.ndarray,
    severity: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate motion blur caused by vehicles in motion past security cameras.

    Level 1: Mild motion blur (k=7)
    Level 2: Moderate motion blur (k=15)
    Level 3: Severe motion blur (k=25)
    """
    rng = _get_rng(seed)
    params = {1: 7, 2: 15, 3: 25}
    k = params.get(severity, 7)

    # Random angle between -20 and 20 degrees representing vehicle motion
    angle = rng.uniform(-20.0, 20.0)

    # Construct motion blur kernel
    kernel = np.zeros((k, k), dtype=np.float32)
    center = k // 2
    cv2.line(
        kernel,
        (0, int(center - math.tan(math.radians(angle)) * center)),
        (k - 1, int(center + math.tan(math.radians(angle)) * center)),
        1.0,
        thickness=1,
    )
    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum
    else:
        kernel[center, :] = 1.0 / k

    return cv2.filter2D(image, -1, kernel)


def apply_defocus_blur(
    image: np.ndarray,
    severity: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate out-of-focus camera optics via broad Gaussian blurring.

    Level 1: k=5, sigma=1.2
    Level 2: k=9, sigma=2.5
    Level 3: k=15, sigma=4.5
    """
    params = {
        1: (5, 1.2),
        2: (9, 2.5),
        3: (15, 4.5),
    }
    ksize, sigma = params.get(severity, params[1])
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


# ---------------------------------------------------------------------------
# Group 3: Environment & Transmission Corruptions
# ---------------------------------------------------------------------------


def apply_rain_streaks(
    image: np.ndarray,
    severity: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate rain streaks and weather haze across the camera field.

    Level 1: 100 thin rain streaks, light opacity (alpha=0.30)
    Level 2: 250 rain streaks, moderate opacity (alpha=0.50)
    Level 3: 500 thick rain streaks + atmospheric rain haze (alpha=0.70)
    """
    rng = _get_rng(seed)
    h, w = image.shape[:2]

    params = {
        1: (100, 15, 25, 1, 0.30, False),
        2: (250, 25, 40, 1, 0.50, False),
        3: (500, 35, 60, 2, 0.70, True),
    }
    count, min_len, max_len, thick, alpha, add_haze = params.get(severity, params[1])

    rain_layer = np.zeros_like(image, dtype=np.uint8)

    # Slant angle around 70 degrees
    slant_deg = rng.uniform(65.0, 75.0)
    slant_rad = math.radians(slant_deg)
    dx = math.cos(slant_rad)
    dy = math.sin(slant_rad)

    for _ in range(count):
        x1 = rng.randint(0, w)
        y1 = rng.randint(0, h)
        length = rng.uniform(min_len, max_len)
        x2 = int(x1 + dx * length)
        y2 = int(y1 + dy * length)
        brightness = rng.randint(190, 255)
        cv2.line(
            rain_layer,
            (x1, y1),
            (x2, y2),
            (brightness, brightness, brightness),
            thickness=thick,
        )

    # Blur rain streaks slightly for natural motion
    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)

    out = cv2.addWeighted(image, 1.0, rain_layer, alpha, 0)

    if add_haze:
        # Gray fog/mist layer for severe downpour
        haze = np.full_like(image, 180, dtype=np.uint8)
        out = cv2.addWeighted(out, 0.85, haze, 0.15, 0)

    return out


def apply_sensor_noise(
    image: np.ndarray,
    severity: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate high ISO sensor noise in camera sensor hardware.

    Level 1: Gaussian noise sigma=15
    Level 2: Gaussian noise sigma=30
    Level 3: Gaussian noise sigma=50
    """
    rng = _get_rng(seed)
    params = {1: 15.0, 2: 30.0, 3: 50.0}
    sigma = params.get(severity, 15.0)

    noise = rng.normal(0.0, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_jpeg_compression(
    image: np.ndarray,
    severity: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate low-bitrate RTSP stream compression artifacts.

    Level 1: JPEG Quality Q=30
    Level 2: JPEG Quality Q=15
    Level 3: JPEG Quality Q=5
    """
    params = {1: 30, 2: 15, 3: 5}
    q = params.get(severity, 30)

    success, enc = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not success:
        return image.copy()
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Group 4: Geometry & Physical Corruptions
# ---------------------------------------------------------------------------


def transform_bbox_perspective(
    box: List[float],
    matrix: np.ndarray,
    img_w: int,
    img_h: int,
) -> Optional[List[float]]:
    """Transform a single COCO bbox [x, y, w, h] using homography matrix M.

    Returns the new enclosing axis-aligned bbox [x_min, y_min, w_new, h_new],
    clamped to image boundary, or None if the box falls entirely out of view.
    """
    bx, by, bw, bh = box
    corners = np.array(
        [
            [bx, by, 1.0],
            [bx + bw, by, 1.0],
            [bx + bw, by + bh, 1.0],
            [bx, by + bh, 1.0],
        ],
        dtype=np.float32,
    )

    # Matrix multiplication: [4, 3] @ [3, 3].T -> [4, 3]
    projected = corners @ matrix.T
    z = projected[:, 2:3]
    # Guard against division by near-zero or negative projective scale
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    pts = projected[:, :2] / z

    x_min = float(np.min(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    x_max = float(np.max(pts[:, 0]))
    y_max = float(np.max(pts[:, 1]))

    # Clamp to image boundaries
    x_min_clamped = max(0.0, min(float(img_w), x_min))
    y_min_clamped = max(0.0, min(float(img_h), y_min))
    x_max_clamped = max(0.0, min(float(img_w), x_max))
    y_max_clamped = max(0.0, min(float(img_h), y_max))

    w_new = x_max_clamped - x_min_clamped
    h_new = y_max_clamped - y_min_clamped

    if w_new < 1.0 or h_new < 1.0:
        return None

    return [
        round(x_min_clamped, 2),
        round(y_min_clamped, 2),
        round(w_new, 2),
        round(h_new, 2),
    ]


def apply_perspective_tilt(
    image: np.ndarray,
    bboxes: Optional[List[List[float]]] = None,
    severity: int = 1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    """Simulate oblique camera angle / lane tilt (15, 30, 45 degrees).

    Accurately re-projects all BBoxes using the 3x3 homography matrix.
    Level 1: 15 deg tilt (~0.08 displacement)
    Level 2: 30 deg tilt (~0.16 displacement)
    Level 3: 45 deg tilt (~0.25 displacement)
    """
    rng = _get_rng(seed)
    h, w = image.shape[:2]

    params = {
        1: 0.08,
        2: 0.16,
        3: 0.25,
    }
    scale = params.get(severity, params[1])

    # Direction: -1 (left tilt) or 1 (right tilt)
    direction = rng.choice([-1.0, 1.0])

    # Original 4 corners of the full frame
    src = np.array(
        [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
        dtype=np.float32,
    )

    dx = scale * w * direction
    dy = scale * h * 0.5

    # Move corners to create trapezoidal perspective tilt
    if direction > 0:
        dst = np.array(
            [
                [dx, dy],
                [w - 1.0, 0.0],
                [w - 1.0, h - 1.0],
                [dx, h - 1.0 - dy],
            ],
            dtype=np.float32,
        )
    else:
        dst = np.array(
            [
                [0.0, 0.0],
                [w - 1.0 + dx, dy],
                [w - 1.0 + dx, h - 1.0 - dy],
                [0.0, h - 1.0],
            ],
            dtype=np.float32,
        )

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(
        image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT
    )

    transformed_bboxes: List[List[float]] = []
    if bboxes:
        for box in bboxes:
            new_box = transform_bbox_perspective(box, matrix, w, h)
            if new_box is not None:
                transformed_bboxes.append(new_box)

    return warped_img, transformed_bboxes


def apply_partial_occlusion(
    image: np.ndarray,
    bboxes: Optional[List[List[float]]] = None,
    severity: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate mud splatters, screws, scratches, or stickers occluding characters.

    Level 1: 1-2 small dirt spots covering ~5% plate area
    Level 2: 2-3 mud patches or scratches covering ~10% plate area
    Level 3: Severe mud blotches covering ~18% plate area
    """
    rng = _get_rng(seed)
    h, w = image.shape[:2]
    out = image.copy()

    params = {
        1: (2, 0.05),
        2: (4, 0.10),
        3: (7, 0.18),
    }
    num_spots, area_fraction = params.get(severity, params[1])

    targets = []
    if bboxes and len(bboxes) > 0:
        for box in bboxes:
            targets.append(box)
    else:
        # Fallback target region in the lower middle
        targets.append([w * 0.25, h * 0.4, w * 0.5, h * 0.3])

    mud_colors = [
        (25, 35, 45),  # Dark mud
        (35, 50, 65),  # Brown clay
        (15, 15, 15),  # Black dirt / grease
        (60, 70, 80),  # Dry gray dirt
    ]

    for bx, by, bw, bh in targets:
        for _ in range(num_spots):
            color = mud_colors[rng.randint(0, len(mud_colors))]
            spot_type = rng.choice(["ellipse", "scratch", "circle"])

            # Location within plate BBox
            cx = int(bx + rng.uniform(0.1, 0.9) * bw)
            cy = int(by + rng.uniform(0.2, 0.8) * bh)

            if spot_type == "circle":
                radius = max(2, int(rng.uniform(0.04, 0.08) * max(bw, bh)))
                cv2.circle(out, (cx, cy), radius, color, -1)
            elif spot_type == "ellipse":
                axes = (
                    max(3, int(rng.uniform(0.06, 0.12) * bw)),
                    max(2, int(rng.uniform(0.04, 0.08) * bh)),
                )
                angle = rng.uniform(0, 180)
                cv2.ellipse(out, (cx, cy), axes, angle, 0, 360, color, -1)
            else:
                # Scratch line
                length = rng.uniform(0.15, 0.35) * max(bw, bh)
                scratch_angle = rng.uniform(0, math.pi)
                x2 = int(cx + length * math.cos(scratch_angle))
                y2 = int(cy + length * math.sin(scratch_angle))
                thickness = rng.randint(2, 4)
                cv2.line(out, (cx, cy), (x2, y2), color, thickness)

    return out


# ---------------------------------------------------------------------------
# Master Dispatcher Function
# ---------------------------------------------------------------------------


def apply_corruption(
    scenario: str,
    severity: int,
    image: np.ndarray,
    bboxes: Optional[List[List[float]]] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    """Unified entry point to apply any of the 10 corruption scenarios.

    Args:
        scenario: Name of scenario (e.g. 'low_light', 'perspective_tilt').
        severity: Level 1 (Mild), 2 (Moderate), 3 (Severe).
        image: BGR image numpy array (H, W, 3).
        bboxes: List of COCO bboxes [[x, y, w, h], ...].
        seed: Random seed for 100% reproducible perturbations.

    Returns:
        Tuple of (corrupted_image, transformed_bboxes).
        For all non-geometric scenarios, transformed_bboxes is identical to bboxes.
    """
    if scenario not in ALL_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Available: {ALL_SCENARIOS}"
        )
    if severity not in SEVERITY_LEVELS:
        raise ValueError(
            f"Severity must be in {SEVERITY_LEVELS}, got {severity}"
        )

    current_bboxes = copy.deepcopy(bboxes) if bboxes is not None else []

    if scenario == "low_light":
        return apply_low_light(image, severity, seed), current_bboxes
    elif scenario == "overexposure":
        return apply_overexposure(image, current_bboxes, severity, seed), current_bboxes
    elif scenario == "uneven_shadows":
        return apply_uneven_shadows(image, current_bboxes, severity, seed), current_bboxes
    elif scenario == "motion_blur":
        return apply_motion_blur(image, severity, seed), current_bboxes
    elif scenario == "defocus_blur":
        return apply_defocus_blur(image, severity, seed), current_bboxes
    elif scenario == "rain_streaks":
        return apply_rain_streaks(image, severity, seed), current_bboxes
    elif scenario == "sensor_noise":
        return apply_sensor_noise(image, severity, seed), current_bboxes
    elif scenario == "jpeg_compression":
        return apply_jpeg_compression(image, severity, seed), current_bboxes
    elif scenario == "perspective_tilt":
        return apply_perspective_tilt(image, current_bboxes, severity, seed)
    elif scenario == "partial_occlusion":
        return apply_partial_occlusion(image, current_bboxes, severity, seed), current_bboxes
    else:
        raise NotImplementedError(f"Scenario {scenario} not implemented")


# ---------------------------------------------------------------------------
# Batch Generator & CLI Pipeline
# ---------------------------------------------------------------------------


class CorruptionBatchGenerator:
    """Manages batch generation of corrupted datasets from clean baseline."""

    def __init__(
        self,
        input_dir: Union[str, Path],
        annotations_file: Union[str, Path],
        output_dir: Union[str, Path],
        seed: int = 42,
    ):
        self.input_dir = Path(input_dir)
        self.annotations_file = Path(annotations_file)
        self.output_dir = Path(output_dir)
        self.seed = seed

        with open(self.annotations_file, "r", encoding="utf-8") as f:
            self.coco_data: Dict[str, Any] = json.load(f)

        # Index annotations by image_id
        self.annotations_by_img_id: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.coco_data.get("annotations", []):
            self.annotations_by_img_id.setdefault(ann["image_id"], []).append(ann)

    def generate(
        self,
        scenarios: List[str],
        severities: List[int],
        sample_size: Optional[int] = None,
        visualize: bool = False,
    ) -> Dict[str, Any]:
        """Generate corrupted images and updated annotations across all specified scenarios."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        images_to_process = self.coco_data.get("images", [])
        if sample_size is not None and sample_size > 0:
            images_to_process = images_to_process[:sample_size]

        manifest: Dict[str, Any] = {
            "source_annotations": str(self.annotations_file),
            "source_image_dir": str(self.input_dir),
            "total_source_images": len(self.coco_data.get("images", [])),
            "processed_sample_size": len(images_to_process),
            "seed": self.seed,
            "datasets": [],
        }

        print(
            f"[*] Bat dau sinh batch nhieu: {len(images_to_process)} anh x "
            f"{len(scenarios)} kich ban x {len(severities)} cap do..."
        )

        for scenario in scenarios:
            for severity in severities:
                dataset_name = f"{scenario}_level_{severity}"
                scenario_out_dir = self.output_dir / scenario / f"level_{severity}"
                images_out_dir = scenario_out_dir / "images"
                images_out_dir.mkdir(parents=True, exist_ok=True)

                if visualize:
                    vis_dir = scenario_out_dir / "visualizations"
                    vis_dir.mkdir(parents=True, exist_ok=True)

                new_images = []
                new_annotations = []
                ann_id_counter = 0

                for img_info in images_to_process:
                    img_id = img_info["id"]
                    file_name = img_info["file_name"]
                    src_img_path = self.input_dir / file_name

                    if not src_img_path.exists():
                        # Try searching by basename if filename had directory prefix
                        alt_path = self.input_dir / Path(file_name).name
                        if alt_path.exists():
                            src_img_path = alt_path
                        else:
                            continue

                    image = cv2.imread(str(src_img_path))
                    if image is None:
                        continue

                    # Retrieve GT bboxes for this image
                    orig_anns = self.annotations_by_img_id.get(img_id, [])
                    boxes = [ann["bbox"] for ann in orig_anns]

                    # Unique deterministic seed per image & scenario & severity
                    img_seed = (self.seed + img_id * 10007 + severity * 997) % (2**31 - 1)

                    corrupted_img, updated_boxes = apply_corruption(
                        scenario=scenario,
                        severity=severity,
                        image=image,
                        bboxes=boxes,
                        seed=img_seed,
                    )

                    # Save corrupted image
                    dst_img_path = images_out_dir / file_name
                    dst_img_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dst_img_path), corrupted_img)

                    # Update image metadata
                    new_img_entry = dict(img_info)
                    new_images.append(new_img_entry)

                    # Update annotation entries
                    for idx, ann in enumerate(orig_anns):
                        new_ann = dict(ann)
                        new_ann["id"] = ann_id_counter
                        ann_id_counter += 1

                        if idx < len(updated_boxes):
                            new_box = updated_boxes[idx]
                            new_ann["bbox"] = new_box
                            new_ann["area"] = round(new_box[2] * new_box[3], 2)
                            # Extra helper field for metrics.py [x1, y1, x2, y2]
                            new_ann["bbox_xyxy"] = [
                                new_box[0],
                                new_box[1],
                                round(new_box[0] + new_box[2], 2),
                                round(new_box[1] + new_box[3], 2),
                            ]
                        new_annotations.append(new_ann)

                    # Optional visual verification render
                    if visualize and len(new_images) <= 20:
                        vis_img = corrupted_img.copy()
                        for box in updated_boxes:
                            x, y, w, h = [int(v) for v in box]
                            cv2.rectangle(
                                vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2
                            )
                            cv2.putText(
                                vis_img,
                                f"{scenario} L{severity}",
                                (x, max(15, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                            )
                        vis_path = vis_dir / f"vis_{Path(file_name).name}"
                        cv2.imwrite(str(vis_path), vis_img)

                # Save new COCO annotations json for this specific corruption
                new_coco = {
                    "info": self.coco_data.get("info", {}),
                    "licenses": self.coco_data.get("licenses", []),
                    "categories": self.coco_data.get("categories", []),
                    "images": new_images,
                    "annotations": new_annotations,
                    "corruption": {
                        "scenario": scenario,
                        "severity": severity,
                        "source": "Source A (synthetic)",
                    },
                }

                out_json_path = scenario_out_dir / "_annotations.coco.json"
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(new_coco, f, indent=2, ensure_ascii=False)

                manifest["datasets"].append(
                    {
                        "name": dataset_name,
                        "scenario": scenario,
                        "severity": severity,
                        "image_count": len(new_images),
                        "annotation_count": len(new_annotations),
                        "image_dir": str(images_out_dir),
                        "annotations_file": str(out_json_path),
                    }
                )
                print(
                    f"  [+] Da sinh: {dataset_name} -> {len(new_images)} anh, "
                    f"{len(new_annotations)} nhan BBox."
                )

        # Write manifest file
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"\n[DONE] Hoan tat toan bo batch! Manifest luu tai: {manifest_path}")
        return manifest


# ---------------------------------------------------------------------------
# CLI Parser
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI Cong cu sinh nhieu nhan tao (Perturbation Generator) - Phase 6"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="datasets/eval_set/images",
        help="Thu muc anh goc Source A (mac dinh: datasets/eval_set/images)",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="datasets/eval_set/_annotations.coco.json",
        help="File nhan COCO goc (mac dinh: datasets/eval_set/_annotations.coco.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/corrupted",
        help="Thu muc xuat anh nhieu va nhan moi (mac dinh: datasets/corrupted)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="all",
        help="Danh sach kich ban phan cach bang dau phay, hoac 'all' de chay 10 kich ban.",
    )
    parser.add_argument(
        "--severities",
        type=str,
        default="1,2,3",
        help="Cac cap do nghiem trong, phan cach bang dau phay (mac dinh: 1,2,3)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="So luong anh can xu ly (None de chay toan bo dataset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed dam bao tinh tai lap 100%% (mac dinh: 42)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Luu anh kem BBox ve truc quan de doi chieu kiem tra",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.scenarios == "all":
        selected_scenarios = ALL_SCENARIOS
    else:
        selected_scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        for s in selected_scenarios:
            if s not in ALL_SCENARIOS:
                raise ValueError(
                    f"Kịch bản '{s}' không hợp lệ. Chọn từ: {ALL_SCENARIOS}"
                )

    selected_severities = [
        int(lvl.strip()) for lvl in args.severities.split(",") if lvl.strip()
    ]
    for lvl in selected_severities:
        if lvl not in SEVERITY_LEVELS:
            raise ValueError(f"Cấp độ {lvl} không hợp lệ. Chỉ chấp nhận 1, 2, 3.")

    generator = CorruptionBatchGenerator(
        input_dir=args.input_dir,
        annotations_file=args.annotations,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    generator.generate(
        scenarios=selected_scenarios,
        severities=selected_severities,
        sample_size=args.sample_size,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
