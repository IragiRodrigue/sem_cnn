from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import networkx as nx
import numpy as np
from scipy import interpolate
from scipy.spatial.distance import pdist, squareform
from skimage.filters import frangi
from skimage.morphology import skeletonize


@dataclass
class PipelineConfig:
    sam_checkpoint: str = "sam_checkpoint/sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"
    fast_mode: bool = True
    sam_variants_fast: tuple[str, ...] = ("raw", "clahe")
    sam_variants_full: tuple[str, ...] = ("raw", "clahe", "frangi_blend", "frangi_binary")
    sam_points_per_side_fast: int = 64
    sam_points_per_side_full: int = 160
    sam_crop_n_layers_fast: int = 0
    sam_crop_n_layers_full: int = 1
    sam_crop_downscale_fast: int = 1
    sam_crop_downscale_full: int = 2
    sam_pred_iou_thresh_fast: float = 0.62
    sam_pred_iou_thresh_full: float = 0.55
    sam_stability_thresh_fast: float = 0.90
    sam_stability_thresh_full: float = 0.85
    min_mask_area: int = 120
    target_num_keypoints: int = 40
    mask_iou_dedup_threshold: float = 0.75
    min_fiber_length_px: float = 45.0
    max_width_length_ratio: float = 0.35
    max_branch_nodes: int = 1
    min_main_path_coverage: float = 0.55
    max_border_touch_fraction: float = 0.025
    min_spline_mask_iou: float = 0.12
    spline_interpolation_steps: int = 100
    final_mask_iou_dedup_threshold: float = 0.35
    crop_bottom_percent: float = 15.0
    sam_device: str = "cuda"

    @property
    def active_variants(self) -> tuple[str, ...]:
        return self.sam_variants_fast if self.fast_mode else self.sam_variants_full

    @property
    def points_per_side(self) -> int:
        return self.sam_points_per_side_fast if self.fast_mode else self.sam_points_per_side_full

    @property
    def crop_n_layers(self) -> int:
        return self.sam_crop_n_layers_fast if self.fast_mode else self.sam_crop_n_layers_full

    @property
    def crop_downscale(self) -> int:
        return self.sam_crop_downscale_fast if self.fast_mode else self.sam_crop_downscale_full

    @property
    def pred_iou_thresh(self) -> float:
        return (
            self.sam_pred_iou_thresh_fast if self.fast_mode else self.sam_pred_iou_thresh_full
        )

    @property
    def stability_thresh(self) -> float:
        return self.sam_stability_thresh_fast if self.fast_mode else self.sam_stability_thresh_full


@dataclass
class PreprocessedVariant:
    name: str
    image: np.ndarray


@dataclass
class FiberCandidate:
    fiber_id: int
    source: str
    mask_index: int
    preprocess_variant: str
    binary_mask: np.ndarray
    final_mask: np.ndarray
    spline_mask: np.ndarray
    keypoints_rc: list[tuple[int, int]]
    visibility: list[int]
    visible_keypoints: int
    keypoint_strategy: str
    fiber_width: float
    fiber_length: float
    fiber_curvature: float
    fiber_orientation: float
    spline_mask_iou: float
    skeleton_stats: dict[str, Any]
    bbox: list[int]
    area: float
    polygons: list[list[int]]
    metadata: dict[str, Any] = field(default_factory=dict)
    has_bead: bool = False
    is_blurry: bool = False
    is_crossing: bool = False

    def to_coco_annotation(self, image_id: int, annotation_id: int) -> dict[str, Any]:
        coco_keypoints: list[int] = []
        for point, visibility in zip(self.keypoints_rc, self.visibility):
            coco_keypoints.extend([int(point[1]), int(point[0]), int(visibility)])

        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": self.polygons,
            "keypoints": coco_keypoints,
            "num_keypoints": int(self.visible_keypoints),
            "bbox": self.bbox,
            "area": self.area,
            "iscrowd": 0,
            "has_bead": bool(self.has_bead),
            "is_blurry": bool(self.is_blurry),
            "is_crossing": bool(self.is_crossing),
            "fiber_width": round(self.fiber_width, 3),
            "fiber_length": round(self.fiber_length, 3),
            "fiber_curvature": round(self.fiber_curvature, 6),
            "fiber_orientation": round(self.fiber_orientation, 2),
            "metadata": {
                "source": self.source,
                "order_rule": "TopToBottomLeftToRight",
                "mask_index": self.mask_index,
                "preprocess_variant": self.preprocess_variant,
                "keypoint_strategy": self.keypoint_strategy,
                "visible_keypoints": int(self.visible_keypoints),
                "spline_mask_iou": round(self.spline_mask_iou, 4),
                "skeleton_stats": self.skeleton_stats,
                "mask_type": "spline_reconstructed",
                **self.metadata,
            },
        }


def crop_percentage(img: np.ndarray, percent: float = 15.0) -> np.ndarray:
    height = img.shape[0]
    new_height = int(height * (1 - (percent / 100.0)))
    return img[0:new_height, :]


def preprocess_sem_variants(img_gray: np.ndarray, config: PipelineConfig) -> list[PreprocessedVariant]:
    base = img_gray.astype(np.uint8)
    denoised = cv2.medianBlur(base, 3)
    denoised = cv2.GaussianBlur(denoised, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(denoised)

    vesselness = frangi(
        clahe_img.astype(np.float32) / 255.0,
        sigmas=range(1, 4),
        black_ridges=False,
    )
    vesselness = cv2.normalize(vesselness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, vessel_binary = cv2.threshold(vesselness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_binary = cv2.morphologyEx(vessel_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    vessel_binary = cv2.morphologyEx(vessel_binary, cv2.MORPH_OPEN, kernel, iterations=1)

    frangi_blend = cv2.addWeighted(clahe_img, 0.65, vesselness, 0.35, 0)
    variants = [
        PreprocessedVariant("raw", base),
        PreprocessedVariant("clahe", clahe_img),
        PreprocessedVariant("frangi_blend", frangi_blend),
        PreprocessedVariant("frangi_binary", vessel_binary),
    ]
    enabled = set(config.active_variants)
    return [variant for variant in variants if variant.name in enabled]


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def deduplicate_masks(
    mask_records: list[dict[str, Any]], iou_threshold: float
) -> list[dict[str, Any]]:
    ordered = sorted(
        mask_records,
        key=lambda m: (
            float(m.get("predicted_iou", 0.0)),
            float(m.get("stability_score", 0.0)),
            float(m.get("area", 0.0)),
        ),
        reverse=True,
    )

    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        candidate_mask = candidate["segmentation"].astype(bool)
        duplicate = False
        for existing in kept:
            if mask_iou(candidate_mask, existing["segmentation"].astype(bool)) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    return sorted(kept, key=lambda m: (int(m["bbox"][1]), int(m["bbox"][0])))


def build_skeleton_graph(binary_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, nx.Graph]:
    skeleton = skeletonize(binary_mask > 0)
    points = np.argwhere(skeleton)
    graph = nx.Graph()
    point_set = set(map(tuple, points))

    for point in points:
        point_tuple = tuple(point)
        for dy, dx in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]:
            neighbor = (point[0] + dy, point[1] + dx)
            if neighbor in point_set:
                graph.add_edge(point_tuple, neighbor, weight=1)

    return skeleton, points, graph


def normalize_path_direction(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not path:
        return path
    p_start, p_end = path[0], path[-1]
    if p_end[0] < p_start[0] or (p_end[0] == p_start[0] and p_end[1] < p_start[1]):
        return list(reversed(path))
    return path


def longest_path_from_graph(
    points: np.ndarray, graph: nx.Graph
) -> tuple[list[tuple[int, int]] | None, str]:
    if len(points) < 2 or graph.number_of_nodes() < 2:
        return None, "too_small"

    endpoints = [node for node, degree in graph.degree() if degree == 1]
    candidate_pairs: list[tuple[tuple[int, int], tuple[int, int], str]] = []

    if len(endpoints) >= 2:
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                candidate_pairs.append((endpoints[i], endpoints[j], "endpoints"))
    else:
        distances = squareform(pdist(points))
        idx1, idx2 = np.unravel_index(np.argmax(distances), distances.shape)
        candidate_pairs.append((tuple(points[idx1]), tuple(points[idx2]), "farthest_points"))

    best_path = None
    best_strategy = "unresolved"
    best_len = -1

    for src, dst, strategy in candidate_pairs:
        try:
            path = nx.shortest_path(graph, source=src, target=dst, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        if len(path) > best_len:
            best_path = path
            best_len = len(path)
            best_strategy = strategy

    if best_path is None:
        return None, "disconnected"

    return normalize_path_direction(best_path), best_strategy


def sample_path_to_fixed_keypoints(
    path: list[tuple[int, int]], target_num_keypoints: int
) -> tuple[list[tuple[int, int]] | None, int, str, list[int]]:
    if not path:
        return None, 0, "missing_path", []

    if len(path) == 1:
        sampled = [path[0]] * target_num_keypoints
        visibility = [1] * target_num_keypoints
        return sampled, 1, "single_point", visibility

    indices = np.linspace(0, len(path) - 1, target_num_keypoints)
    sampled = [tuple(path[int(round(i))]) for i in indices]

    visibility: list[int] = []
    visible_unique = 0
    seen: set[tuple[int, int]] = set()
    for point in sampled:
        if point not in seen:
            visibility.append(2)
            seen.add(point)
            visible_unique += 1
        else:
            visibility.append(1)

    strategy = "direct_40" if len(path) >= target_num_keypoints else "adaptive_resample"
    return sampled, visible_unique, strategy, visibility


def compute_border_touch_stats(binary_mask: np.ndarray) -> tuple[float, int]:
    mask_bool = binary_mask.astype(bool)
    if mask_bool.size == 0:
        return 0.0, 0

    top = mask_bool[0, :]
    bottom = mask_bool[-1, :]
    left = mask_bool[:, 0]
    right = mask_bool[:, -1]
    border_pixels = int(top.sum() + bottom.sum() + left.sum() + right.sum())
    total_pixels = int(mask_bool.sum())
    border_touch_fraction = float(border_pixels / total_pixels) if total_pixels else 0.0
    border_sides_touched = int(any(top)) + int(any(bottom)) + int(any(left)) + int(any(right))
    return border_touch_fraction, border_sides_touched


def summarize_skeleton_graph(
    graph: nx.Graph, path: list[tuple[int, int]], binary_mask: np.ndarray
) -> dict[str, Any]:
    node_count = int(graph.number_of_nodes())
    branch_nodes = int(sum(1 for _, degree in graph.degree() if degree > 2))
    endpoints = int(sum(1 for _, degree in graph.degree() if degree == 1))
    path_len = len(path) if path else 0
    main_path_coverage = float(path_len / node_count) if node_count else 0.0
    border_touch_fraction, border_sides_touched = compute_border_touch_stats(binary_mask)
    return {
        "node_count": node_count,
        "branch_nodes": branch_nodes,
        "endpoints": endpoints,
        "path_len": path_len,
        "main_path_coverage": main_path_coverage,
        "border_touch_fraction": border_touch_fraction,
        "border_sides_touched": border_sides_touched,
    }


def extract_ordered_keypoints(
    binary_mask: np.ndarray, config: PipelineConfig
) -> tuple[list[tuple[int, int]] | None, int, str, list[int], dict[str, Any]]:
    _, points, graph = build_skeleton_graph(binary_mask)
    path, path_strategy = longest_path_from_graph(points, graph)
    if path is None:
        return None, 0, path_strategy, [], {}

    sampled, visible_unique, sampling_strategy, visibility = sample_path_to_fixed_keypoints(
        path, config.target_num_keypoints
    )
    stats = summarize_skeleton_graph(graph, path, binary_mask)
    return sampled, visible_unique, f"{path_strategy}+{sampling_strategy}", visibility, stats


def calculate_curvature(points: list[tuple[int, int]]) -> float:
    pts = np.array(points)
    dx = np.gradient(pts[:, 1])
    dy = np.gradient(pts[:, 0])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = np.power(dx**2 + dy**2, 1.5)
    curvature_profile = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )
    return float(np.mean(curvature_profile))


def compute_fiber_metrics(
    points: list[tuple[int, int]], binary_mask: np.ndarray
) -> tuple[float, float, float, float]:
    dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
    widths: list[float] = []
    for y, x in points:
        y = int(np.clip(y, 0, dist_map.shape[0] - 1))
        x = int(np.clip(x, 0, dist_map.shape[1] - 1))
        local_width = float(dist_map[y, x] * 2.0)
        if local_width > 0:
            widths.append(local_width)

    fiber_width = float(np.median(widths)) if widths else 0.0

    fiber_length = 0.0
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        fiber_length += float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))

    orientation = float(
        np.degrees(np.arctan2(points[-1][0] - points[0][0], points[-1][1] - points[0][1]))
    )
    curvature = float(calculate_curvature(points))
    return fiber_width, fiber_length, orientation, curvature


def interpolate_keypoints_xy(
    keypoints_xy: np.ndarray, num_steps: int
) -> np.ndarray:
    keypoints_xy = np.asarray(keypoints_xy, dtype=np.float32)
    if len(keypoints_xy) < 2:
        return keypoints_xy

    keypoints_xy = np.unique(keypoints_xy, axis=0)
    if len(keypoints_xy) < 2:
        return keypoints_xy

    spline_degree = 1 if len(keypoints_xy) < 4 else 3
    try:
        tck, _ = interpolate.splprep(keypoints_xy.T, s=0, k=spline_degree)
        x_new, y_new = interpolate.splev(np.linspace(0, 1, num_steps), tck, der=0)
        return np.stack((x_new, y_new), axis=1).astype(np.float32)
    except ValueError:
        return keypoints_xy


def spline_mask_from_keypoints(
    mask_shape: tuple[int, int], points_rc: list[tuple[int, int]], fiber_width: float, config: PipelineConfig
) -> np.ndarray:
    if not points_rc or fiber_width <= 0:
        return np.zeros(mask_shape, dtype=np.uint8)

    keypoints_xy = np.array([(point[1], point[0]) for point in points_rc], dtype=np.float32)
    smooth_xy = interpolate_keypoints_xy(keypoints_xy, config.spline_interpolation_steps)
    if len(smooth_xy) < 2:
        return np.zeros(mask_shape, dtype=np.uint8)

    thickness = max(1, int(round(fiber_width)))
    canvas = np.zeros(mask_shape, dtype=np.uint8)
    polyline = np.round(smooth_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [polyline], isClosed=False, color=255, thickness=thickness)

    radius = max(1, thickness // 2)
    for x, y in np.round(smooth_xy[1:-1]).astype(np.int32):
        cv2.circle(canvas, (int(x), int(y)), radius, 255, -1)

    return canvas


def compute_mask_fit_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a_bool = mask_a.astype(bool)
    b_bool = mask_b.astype(bool)
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(a_bool, b_bool).sum()
    return float(intersection / union)


def mask_bbox_area(binary_mask: np.ndarray) -> tuple[list[int] | None, float]:
    mask_bool = binary_mask.astype(bool)
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return None, 0.0

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
    return bbox, float(mask_bool.sum())


def evaluate_candidate_fiber(
    skeleton_stats: dict[str, Any],
    fiber_width: float,
    fiber_length: float,
    spline_mask_iou: float,
    config: PipelineConfig,
) -> tuple[bool, str]:
    if fiber_length < config.min_fiber_length_px:
        return False, "too_short"
    if fiber_width <= 0:
        return False, "invalid_width"
    if fiber_width / max(fiber_length, 1.0) > config.max_width_length_ratio:
        return False, "too_blob_like"
    if skeleton_stats["branch_nodes"] > config.max_branch_nodes:
        return False, "too_branchy"
    if skeleton_stats["main_path_coverage"] < config.min_main_path_coverage:
        return False, "low_main_path_coverage"
    if (
        skeleton_stats["border_touch_fraction"] > config.max_border_touch_fraction
        and skeleton_stats["border_sides_touched"] >= 1
    ):
        return False, "border_artifact"
    if spline_mask_iou < config.min_spline_mask_iou:
        return False, "poor_spline_fit"
    return True, "accepted"


def mask_to_polygons(binary_mask: np.ndarray) -> list[list[int]]:
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons: list[list[int]] = []
    for contour in contours:
        if contour.size >= 6:
            polygons.append(contour.flatten().tolist())
    return polygons


def analyze_fiber_complexity(mask: np.ndarray, image_gray: np.ndarray) -> tuple[bool, bool, bool]:
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_width = float(np.max(dist_transform))
    mean_width = float(np.mean(dist_transform[mask > 0])) if np.any(mask > 0) else 0.0
    has_bead = bool(mean_width > 0 and max_width > mean_width * 3.0)

    x, y, w, h = cv2.boundingRect(mask)
    roi = image_gray[y : y + h, x : x + w]
    blur_score = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size else 0.0
    is_blurry = bool(blur_score < 100)

    skeleton = skeletonize(mask > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 10, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton, -1, kernel)
    is_crossing = bool(np.any(neighbors > 12))
    return has_bead, is_blurry, is_crossing


def build_candidate_from_mask(
    binary_mask: np.ndarray,
    image_gray: np.ndarray,
    config: PipelineConfig,
    *,
    fiber_id: int,
    source: str,
    mask_index: int,
    preprocess_variant: str = "prompted",
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[FiberCandidate | None, str]:
    points, visible_keypoints, keypoint_strategy, visibility, skeleton_stats = extract_ordered_keypoints(
        binary_mask, config
    )
    if points is None:
        return None, keypoint_strategy

    fiber_width, fiber_length, orientation, fiber_curvature = compute_fiber_metrics(
        points, binary_mask
    )
    spline_mask = spline_mask_from_keypoints(binary_mask.shape, points, fiber_width, config)
    spline_mask_iou = compute_mask_fit_iou(binary_mask, spline_mask)
    is_valid, reject_reason = evaluate_candidate_fiber(
        skeleton_stats, fiber_width, fiber_length, spline_mask_iou, config
    )
    if not is_valid:
        return None, reject_reason

    final_mask = spline_mask if np.any(spline_mask) else binary_mask
    polygons = mask_to_polygons(final_mask)
    if not polygons:
        return None, "empty_polygon_after_spline"

    bbox, area = mask_bbox_area(final_mask)
    if bbox is None:
        return None, "empty_final_mask"

    has_bead, is_blurry, is_crossing = analyze_fiber_complexity(final_mask, image_gray)
    return (
        FiberCandidate(
            fiber_id=fiber_id,
            source=source,
            mask_index=mask_index,
            preprocess_variant=preprocess_variant,
            binary_mask=binary_mask.copy(),
            final_mask=final_mask.copy(),
            spline_mask=spline_mask.copy(),
            keypoints_rc=[(int(r), int(c)) for r, c in points],
            visibility=[int(v) for v in visibility],
            visible_keypoints=int(visible_keypoints),
            keypoint_strategy=keypoint_strategy,
            fiber_width=float(fiber_width),
            fiber_length=float(fiber_length),
            fiber_curvature=float(fiber_curvature),
            fiber_orientation=float(orientation),
            spline_mask_iou=float(spline_mask_iou),
            skeleton_stats=skeleton_stats,
            bbox=bbox,
            area=area,
            polygons=polygons,
            metadata=extra_metadata or {},
            has_bead=has_bead,
            is_blurry=is_blurry,
            is_crossing=is_crossing,
        ),
        "accepted",
    )


class SamFiberSegmenter:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._sam = None
        self._predictor = None
        self._mask_generator = None

    def _ensure_loaded(self) -> None:
        if self._sam is not None:
            return

        try:
            import torch
            from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "Missing dependencies for SAM. Install torch and segment-anything first."
            ) from exc

        device = self.config.sam_device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        checkpoint = Path(self.config.sam_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

        self._sam = sam_model_registry[self.config.sam_model_type](checkpoint=str(checkpoint))
        self._sam.to(device=device)
        self._mask_generator = SamAutomaticMaskGenerator(
            model=self._sam,
            points_per_side=self.config.points_per_side,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_thresh,
            crop_n_layers=self.config.crop_n_layers,
            crop_n_points_downscale_factor=self.config.crop_downscale,
            min_mask_region_area=self.config.min_mask_area,
        )
        self._predictor = SamPredictor(self._sam)

    @staticmethod
    def _as_rgb(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image.copy()

    def automatic_masks(
        self, img_gray: np.ndarray
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[PreprocessedVariant]]:
        self._ensure_loaded()
        assert self._mask_generator is not None

        all_masks: list[dict[str, Any]] = []
        masks_by_variant: dict[str, list[dict[str, Any]]] = {}
        variants = preprocess_sem_variants(img_gray, self.config)
        for variant in variants:
            masks = self._mask_generator.generate(self._as_rgb(variant.image))
            masks_by_variant[variant.name] = masks
            for mask in masks:
                enriched = dict(mask)
                enriched["preprocess_variant"] = variant.name
                all_masks.append(enriched)

        merged = deduplicate_masks(all_masks, self.config.mask_iou_dedup_threshold)
        return merged, masks_by_variant, variants

    def prompted_mask(
        self,
        image: np.ndarray,
        *,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        multimask_output: bool = False,
    ) -> tuple[np.ndarray, float]:
        self._ensure_loaded()
        assert self._predictor is not None

        rgb_image = self._as_rgb(image)
        self._predictor.set_image(rgb_image)

        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output,
        )
        best_idx = int(np.argmax(scores))
        mask = (masks[best_idx].astype(np.uint8) * 255).astype(np.uint8)
        return mask, float(scores[best_idx])


def labels_from_candidates(shape: tuple[int, int], candidates: list[FiberCandidate]) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int32)
    for candidate in candidates:
        labels[candidate.final_mask > 0] = int(candidate.fiber_id)
    return labels


def labels_from_raw_masks(
    shape: tuple[int, int], masks: list[dict[str, Any]], start_index: int = 1
) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int32)
    for idx, mask_data in enumerate(masks, start=start_index):
        labels[mask_data["segmentation"].astype(bool)] = idx
    return labels


def export_coco(
    image_path: str | Path,
    image: np.ndarray,
    candidates: list[FiberCandidate],
    output_path: str | Path,
) -> Path:
    image_path = Path(image_path)
    output_path = Path(output_path)

    categories = [
        {
            "id": 1,
            "name": "fiber",
            "supercategory": "fiber",
            "keypoints": [f"p{i + 1}" for i in range(40)],
            "skeleton": [[i + 1, i + 2] for i in range(39)],
        }
    ]

    images = [
        {
            "id": 1,
            "file_name": image_path.name,
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
        }
    ]
    annotations = [
        candidate.to_coco_annotation(image_id=1, annotation_id=i + 1)
        for i, candidate in enumerate(candidates)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"images": images, "annotations": annotations, "categories": categories},
            f,
            indent=2,
        )
    return output_path
