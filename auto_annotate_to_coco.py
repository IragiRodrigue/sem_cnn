import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import networkx as nx
import torch
from scipy import interpolate
from scipy.spatial.distance import pdist, squareform
from skimage.filters import frangi
from skimage.morphology import skeletonize
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from fiber_viz import (
    analyze_fiber_complexity,
    calculate_curvature,
    crop_percentage,
    mask_to_polygons,
    save_debug_diagnostic,
    visualize_annotation_overview,
    visualize_mask_summary,
    visualize_preprocessing_overview,
    visualize_single_fiber_debug,
)


IMAGE_DIR = "images"
OUTPUT_JSON = "coco_fiber.json"
SAM_CHECKPOINT = "sam_checkpoint/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
FAST_MODE = True
SAM_VARIANTS_FAST = ("raw", "clahe")
SAM_VARIANTS_FULL = ("raw", "clahe", "frangi_blend", "frangi_binary")
SAM_POINTS_PER_SIDE_FAST = 64
SAM_POINTS_PER_SIDE_FULL = 160
SAM_CROP_N_LAYERS_FAST = 0
SAM_CROP_N_LAYERS_FULL = 1
SAM_CROP_DOWNSCALE_FAST = 1
SAM_CROP_DOWNSCALE_FULL = 2
SAM_PRED_IOU_THRESH_FAST = 0.62
SAM_PRED_IOU_THRESH_FULL = 0.55
SAM_STABILITY_THRESH_FAST = 0.90
SAM_STABILITY_THRESH_FULL = 0.85
MIN_MASK_AREA = 120
TARGET_NUM_KEYPOINTS = 40
MASK_IOU_DEDUP_THRESHOLD = 0.75
DEBUG_VISUALS = True
SAVE_DEBUG_VISUALS = True
SHOW_DEBUG_VISUALS = False
MAX_FIBER_DEBUG_PLOTS = 6
DEBUG_OUTPUT_DIR = "debug_plots"
SAM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_FIBER_LENGTH_PX = 45.0
MAX_WIDTH_LENGTH_RATIO = 0.35
MAX_BRANCH_NODES = 1
MIN_MAIN_PATH_COVERAGE = 0.55
MAX_BORDER_TOUCH_FRACTION = 0.025
MIN_SPLINE_MASK_IOU = 0.12
SPLINE_INTERPOLATION_STEPS = 100
FINAL_MASK_IOU_DEDUP_THRESHOLD = 0.35


@dataclass
class PreprocessedVariant:
    name: str
    image: np.ndarray


def preprocess_sem_variants(img_gray):
    """
    Prétraitements adaptés aux images MEB:
    réduction du bruit, amélioration du contraste, accentuation des fibres.
    """
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

    enabled_names = set(SAM_VARIANTS_FAST if FAST_MODE else SAM_VARIANTS_FULL)
    return [variant for variant in variants if variant.name in enabled_names]


def mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def deduplicate_masks(mask_records, iou_threshold=MASK_IOU_DEDUP_THRESHOLD):
    ordered = sorted(
        mask_records,
        key=lambda m: (
            float(m.get("predicted_iou", 0.0)),
            float(m.get("stability_score", 0.0)),
            float(m.get("area", 0.0)),
        ),
        reverse=True,
    )

    kept = []
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


def build_skeleton_graph(binary_mask):
    skeleton = skeletonize(binary_mask > 0)
    points = np.argwhere(skeleton)
    graph = nx.Graph()
    point_set = set(map(tuple, points))

    for point in points:
        point_tuple = tuple(point)
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            neighbor = (point[0] + dy, point[1] + dx)
            if neighbor in point_set:
                graph.add_edge(point_tuple, neighbor, weight=1)

    return skeleton, points, graph


def normalize_path_direction(path):
    if not path:
        return path

    p_start, p_end = path[0], path[-1]
    if p_end[0] < p_start[0] or (p_end[0] == p_start[0] and p_end[1] < p_start[1]):
        return list(reversed(path))
    return path


def longest_path_from_graph(points, graph):
    if len(points) < 2 or graph.number_of_nodes() < 2:
        return None, "too_small"

    endpoints = [node for node, degree in graph.degree() if degree == 1]
    candidate_pairs = []

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


def sample_path_to_fixed_keypoints(path, target_num_keypoints=TARGET_NUM_KEYPOINTS):
    """
    Retourne toujours 40 points pour rester compatible avec COCO/Detectron2.
    Les petites fibres sont ré-échantillonnées avec répétition contrôlée.
    """
    if not path:
        return None, 0, "missing_path", []

    if len(path) == 1:
        sampled = [path[0]] * target_num_keypoints
        visibility = [1] * target_num_keypoints
        return sampled, 1, "single_point", visibility

    indices = np.linspace(0, len(path) - 1, target_num_keypoints)
    sampled = [path[int(round(i))] for i in indices]

    visibility = []
    visible_unique = 0
    seen = set()
    for point in sampled:
        if point not in seen:
            visibility.append(2)
            seen.add(point)
            visible_unique += 1
        else:
            visibility.append(1)

    strategy = "direct_40" if len(path) >= target_num_keypoints else "adaptive_resample"
    return sampled, visible_unique, strategy, visibility


def extract_40_keypoints_ordered(binary_mask):
    _, points, graph = build_skeleton_graph(binary_mask)
    path, path_strategy = longest_path_from_graph(points, graph)
    if path is None:
        return None, 0, path_strategy, [], {}

    sampled, visible_unique, sampling_strategy, visibility = sample_path_to_fixed_keypoints(path)
    stats = summarize_skeleton_graph(graph, path, binary_mask)
    return sampled, visible_unique, f"{path_strategy}+{sampling_strategy}", visibility, stats


def summarize_skeleton_graph(graph, path, binary_mask):
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


def compute_border_touch_stats(binary_mask):
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


def compute_fiber_metrics(points, binary_mask):
    dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)

    widths = []
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


def interpolate_keypoints_xy(keypoints_xy, num_steps=SPLINE_INTERPOLATION_STEPS):
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


def spline_mask_from_keypoints(mask_shape, points_rc, fiber_width):
    if not points_rc or fiber_width <= 0:
        return np.zeros(mask_shape, dtype=np.uint8)

    keypoints_xy = np.array([(point[1], point[0]) for point in points_rc], dtype=np.float32)
    smooth_xy = interpolate_keypoints_xy(keypoints_xy)
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


def compute_mask_fit_iou(mask_a, mask_b):
    a_bool = mask_a.astype(bool)
    b_bool = mask_b.astype(bool)
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(a_bool, b_bool).sum()
    return float(intersection / union)


def mask_bbox_area(binary_mask):
    mask_bool = binary_mask.astype(bool)
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return None, 0.0

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
    return bbox, float(mask_bool.sum())


def evaluate_candidate_fiber(skeleton_stats, fiber_width, fiber_length, spline_mask_iou):
    if fiber_length < MIN_FIBER_LENGTH_PX:
        return False, "too_short"

    if fiber_width <= 0:
        return False, "invalid_width"

    if fiber_width / max(fiber_length, 1.0) > MAX_WIDTH_LENGTH_RATIO:
        return False, "too_blob_like"

    if skeleton_stats["branch_nodes"] > MAX_BRANCH_NODES:
        return False, "too_branchy"

    if skeleton_stats["main_path_coverage"] < MIN_MAIN_PATH_COVERAGE:
        return False, "low_main_path_coverage"

    if (
        skeleton_stats["border_touch_fraction"] > MAX_BORDER_TOUCH_FRACTION
        and skeleton_stats["border_sides_touched"] >= 1
    ):
        return False, "border_artifact"

    if spline_mask_iou < MIN_SPLINE_MASK_IOU:
        return False, "poor_spline_fit"

    return True, "accepted"


def sam_segmentation(img_gray):
    print("Running SAM segmentation...")
    model_type = SAM_MODEL_TYPE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    points_per_side = SAM_POINTS_PER_SIDE_FAST if FAST_MODE else SAM_POINTS_PER_SIDE_FULL
    crop_n_layers = SAM_CROP_N_LAYERS_FAST if FAST_MODE else SAM_CROP_N_LAYERS_FULL
    crop_downscale = SAM_CROP_DOWNSCALE_FAST if FAST_MODE else SAM_CROP_DOWNSCALE_FULL
    pred_iou_thresh = SAM_PRED_IOU_THRESH_FAST if FAST_MODE else SAM_PRED_IOU_THRESH_FULL
    stability_score_thresh = (
        SAM_STABILITY_THRESH_FAST if FAST_MODE else SAM_STABILITY_THRESH_FULL
    )
    mode_name = "FAST" if FAST_MODE else "FULL"
    active_variants = SAM_VARIANTS_FAST if FAST_MODE else SAM_VARIANTS_FULL
    print(
        f"Using SAM model '{model_type}' on device '{device}' in {mode_name} mode "
        f"(variants={active_variants}, points_per_side={points_per_side}, crop_n_layers={crop_n_layers})"
    )

    sam = sam_model_registry[model_type](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)  # SAM est gourmand, on force le CPU pour éviter les OOM sur GPU limités

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_downscale,
        min_mask_region_area=MIN_MASK_AREA,
    )

    all_masks = []
    masks_by_variant = {}
    variants = preprocess_sem_variants(img_gray)
    for variant in variants:
        variant_rgb = cv2.cvtColor(variant.image, cv2.COLOR_GRAY2RGB)
        masks = mask_generator.generate(variant_rgb)
        print(f"  - variant {variant.name}: {len(masks)} masks")
        masks_by_variant[variant.name] = masks

        for mask in masks:
            enriched = dict(mask)
            enriched["preprocess_variant"] = variant.name
            all_masks.append(enriched)

    merged_masks = deduplicate_masks(all_masks)
    return merged_masks, masks_by_variant, variants


images = []
annotations = []
global_rejection_stats = {}

categories = [{
    "id": 1,
    "name": "fiber",
    "supercategory": "fiber",
    "keypoints": [f"p{i + 1}" for i in range(TARGET_NUM_KEYPOINTS)],
    "skeleton": [[i + 1, i + 2] for i in range(TARGET_NUM_KEYPOINTS - 1)],
}]

ann_id = 1
img_id = 1

if SAVE_DEBUG_VISUALS:
    Path(DEBUG_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

for file in os.listdir(IMAGE_DIR):
    if not file.lower().endswith((".jpg", ".png", ".tif")):
        continue

    path = os.path.join(IMAGE_DIR, file)
    img_raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        print(f"[WARN] Impossible de lire {file}")
        continue

    img = crop_percentage(img_raw, percent=15)

    print(f"Processing {file} with SAM segmentation...")
    all_masks, masks_by_variant, variants = sam_segmentation(img)
    print(f"Found {len(all_masks)} deduplicated masks for {file}.")

    image_debug_dir = Path(DEBUG_OUTPUT_DIR) / Path(file).stem
    if SAVE_DEBUG_VISUALS:
        image_debug_dir.mkdir(parents=True, exist_ok=True)

    if DEBUG_VISUALS:
        visualize_preprocessing_overview(
            img,
            variants,
            output_path=image_debug_dir / "01_preprocessing_overview.png" if SAVE_DEBUG_VISUALS else None,
            show=SHOW_DEBUG_VISUALS,
        )
        visualize_mask_summary(
            img,
            masks_by_variant,
            all_masks,
            output_path=image_debug_dir / "02_sam_mask_summary.png" if SAVE_DEBUG_VISUALS else None,
            show=SHOW_DEBUG_VISUALS,
        )

    images.append({
        "id": img_id,
        "file_name": file,
        "height": img.shape[0],
        "width": img.shape[1],
    })

    accepted_annotations_for_debug = []
    image_annotations = []
    debug_plot_count = 0
    image_rejection_stats = {}
    accepted_final_masks = []

    for mask_idx, mask_data in enumerate(all_masks):
        if mask_data["area"] < MIN_MASK_AREA:
            continue

        binary_mask = mask_data["segmentation"].astype(np.uint8) * 255
        points, visible_keypoints, keypoint_strategy, visibility, skeleton_stats = extract_40_keypoints_ordered(binary_mask)
        if points is None:
            if mask_data["area"] > 500:
                print(f"[WARN] Fibre ignorée ({file}) - strategy={keypoint_strategy}")
            continue

        fiber_width, fiber_length, orientation, fiber_curvature = compute_fiber_metrics(points, binary_mask)
        spline_mask = spline_mask_from_keypoints(binary_mask.shape, points, fiber_width)
        spline_mask_iou = compute_mask_fit_iou(binary_mask, spline_mask)
        is_valid, reject_reason = evaluate_candidate_fiber(
            skeleton_stats, fiber_width, fiber_length, spline_mask_iou
        )
        if not is_valid:
            image_rejection_stats[reject_reason] = image_rejection_stats.get(reject_reason, 0) + 1
            global_rejection_stats[reject_reason] = global_rejection_stats.get(reject_reason, 0) + 1
            continue

        final_mask = spline_mask if np.any(spline_mask) else binary_mask
        is_duplicate_final = any(
            compute_mask_fit_iou(final_mask, accepted_mask) >= FINAL_MASK_IOU_DEDUP_THRESHOLD
            for accepted_mask in accepted_final_masks
        )
        if is_duplicate_final:
            image_rejection_stats["duplicate_final_spline"] = image_rejection_stats.get(
                "duplicate_final_spline", 0
            ) + 1
            global_rejection_stats["duplicate_final_spline"] = global_rejection_stats.get(
                "duplicate_final_spline", 0
            ) + 1
            continue

        segmentation_polygons = mask_to_polygons(final_mask)
        if not segmentation_polygons:
            image_rejection_stats["empty_polygon_after_spline"] = image_rejection_stats.get(
                "empty_polygon_after_spline", 0
            ) + 1
            global_rejection_stats["empty_polygon_after_spline"] = global_rejection_stats.get(
                "empty_polygon_after_spline", 0
            ) + 1
            continue

        skeleton, _, _ = build_skeleton_graph(final_mask)

        bbox, area = mask_bbox_area(final_mask)
        if bbox is None:
            image_rejection_stats["empty_final_mask"] = image_rejection_stats.get(
                "empty_final_mask", 0
            ) + 1
            global_rejection_stats["empty_final_mask"] = global_rejection_stats.get(
                "empty_final_mask", 0
            ) + 1
            continue

        has_bead, is_blurry, is_crossing = analyze_fiber_complexity(final_mask, img)

        coco_keypoints = []
        for point, v in zip(points, visibility):
            coco_keypoints.extend([int(point[1]), int(point[0]), int(v)])

        annotation = {
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "segmentation": segmentation_polygons,
            "keypoints": coco_keypoints,
            "num_keypoints": int(visible_keypoints),
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
            "has_bead": bool(has_bead),
            "is_blurry": bool(is_blurry),
            "is_crossing": bool(is_crossing),
            "fiber_width": round(fiber_width, 3),
            "fiber_length": round(fiber_length, 3),
            "fiber_curvature": round(fiber_curvature, 6),
            "fiber_orientation": round(orientation, 2),
            "metadata": {
                "source": file,
                "order_rule": "TopToBottomLeftToRight",
                "mask_index": mask_idx,
                "preprocess_variant": mask_data.get("preprocess_variant", "raw"),
                "keypoint_strategy": keypoint_strategy,
                "visible_keypoints": int(visible_keypoints),
                "spline_mask_iou": round(spline_mask_iou, 4),
                "skeleton_stats": skeleton_stats,
                "mask_type": "spline_reconstructed",
            },
        }

        annotations.append(annotation)
        accepted_annotations_for_debug.append(annotation)
        image_annotations.append(annotation)
        accepted_final_masks.append(final_mask)

        if DEBUG_VISUALS and debug_plot_count < MAX_FIBER_DEBUG_PLOTS:
            visualize_single_fiber_debug(
                image=img,
                binary_mask=final_mask,
                skeleton=skeleton,
                keypoints=points,
                visibility=visibility,
                fiber_metrics={
                    "fiber_width": fiber_width,
                    "fiber_length": fiber_length,
                    "fiber_curvature": fiber_curvature,
                    "fiber_orientation": orientation,
                },
                complexity_flags={
                    "has_bead": bool(has_bead),
                    "is_blurry": bool(is_blurry),
                    "is_crossing": bool(is_crossing),
                },
                title_suffix=f"{file} | mask={mask_idx} | {keypoint_strategy}",
                output_path=(
                    image_debug_dir / f"fiber_{debug_plot_count + 1:02d}_mask_{mask_idx}.png"
                    if SAVE_DEBUG_VISUALS else None
                ),
                show=SHOW_DEBUG_VISUALS,
            )
            debug_plot_count += 1

        ann_id += 1

    save_debug_diagnostic(img, all_masks, accepted_annotations_for_debug, f"debug_{file}.jpg")
    print(
        f"[SUMMARY] {file}: accepted={len(image_annotations)} "
        f"rejected={sum(image_rejection_stats.values())} details={image_rejection_stats}"
    )

    if DEBUG_VISUALS:
        visualize_annotation_overview(
            img,
            image_annotations,
            output_path=image_debug_dir / "03_final_annotations.png" if SAVE_DEBUG_VISUALS else None,
            show=SHOW_DEBUG_VISUALS,
        )
    img_id += 1


coco = {
    "images": images,
    "annotations": annotations,
    "categories": categories,
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(coco, f, indent=4)

print(f"FULL PIPELINE DONE -> {OUTPUT_JSON} ({ann_id - 1} fibres)")
print(f"Global rejection stats: {global_rejection_stats}")
