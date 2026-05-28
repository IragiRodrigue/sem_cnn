import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np


TARGET_NUM_KEYPOINTS = 40
CATEGORY_ID = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic nanofiber COCO dataset for FibeR-CNN pretraining."
    )
    parser.add_argument("--output-dir", default="synthetic_dataset")
    parser.add_argument("--num-images", type=int, default=200)

    parser.add_argument("--width", type=int, default=1024)  # Au lieu de 768
    parser.add_argument("--height", type=int, default=652)  # Au lieu de 512

    parser.add_argument("--min-fibers", type=int, default=60)
    parser.add_argument("--max-fibers", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    return parser.parse_args()


def make_sem_background(height, width, rng):
    base = rng.normal(95, 18, (height, width)).astype(np.float32)
    low_freq = rng.normal(0, 1, (max(8, height // 16), max(8, width // 16))).astype(np.float32)
    low_freq = cv2.resize(low_freq, (width, height), interpolation=cv2.INTER_CUBIC)
    low_freq = cv2.GaussianBlur(low_freq, (0, 0), 18)
    base += low_freq * 22

    for _ in range(rng.integers(1, 5)):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        radius = int(rng.integers(width // 10, width // 3))
        shade = float(rng.uniform(-18, 18))
        yy, xx = np.ogrid[:height, :width]
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius * radius))
        base += blob * shade

    base = cv2.GaussianBlur(base, (0, 0), rng.uniform(0.4, 1.2))
    return np.clip(base, 0, 255).astype(np.uint8)


def random_fiber_centerline(height, width, rng):
    margin = 40
    angle = rng.uniform(0, math.pi)
    length = rng.uniform(50, 760)
    curvature = rng.uniform(-0.45, 0.45)
    n_ctrl = int(rng.integers(4, 8))

    cx = rng.uniform(margin, width - margin)
    cy = rng.uniform(margin, height - margin)
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)

    ts = np.linspace(-0.5, 0.5, n_ctrl)
    points = []
    waviness = rng.uniform(0, 0.16) * length
    for t in ts:
        along = direction * (t * length)
        bend = normal * (curvature * (t ** 2 - 0.08) * length)
        wave = normal * (math.sin(t * math.pi * rng.uniform(1.0, 3.5)) * waviness)
        jitter = rng.normal(0, length * 0.015, 2)
        point = np.array([cx, cy]) + along + bend + wave + jitter
        points.append(point)

    points = np.asarray(points, dtype=np.float32)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return interpolate_polyline(points, TARGET_NUM_KEYPOINTS)


def interpolate_polyline(points, num_points):
    points = np.asarray(points, dtype=np.float32)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = cumulative[-1]
    if total <= 1e-6:
        return np.repeat(points[:1], num_points, axis=0)

    targets = np.linspace(0, total, num_points)
    sampled = []
    for target in targets:
        idx = int(np.searchsorted(cumulative, target, side="right") - 1)
        idx = min(idx, len(points) - 2)
        local_len = max(cumulative[idx + 1] - cumulative[idx], 1e-6)
        alpha = (target - cumulative[idx]) / local_len
        sampled.append(points[idx] * (1 - alpha) + points[idx + 1] * alpha)
    return np.asarray(sampled, dtype=np.float32)


def order_keypoints_top_to_bottom_left_to_right(points):
    first = points[0]
    last = points[-1]
    correct = last[1] > first[1] or (last[1] == first[1] and last[0] > first[0])
    return points if correct else points[::-1].copy()


def draw_fiber_mask(shape, centerline, width_px, bead=False, rng=None):
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.round(centerline).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=max(1, int(round(width_px))))

    radius = max(1, int(round(width_px / 2)))
    for x, y in np.round(centerline[1:-1]).astype(np.int32):
        cv2.circle(mask, (int(x), int(y)), radius, 255, -1)

    if bead and rng is not None:
        bead_idx = int(rng.integers(5, len(centerline) - 5))
        bx, by = np.round(centerline[bead_idx]).astype(np.int32)
        bead_radius = int(round(width_px * rng.uniform(1.7, 3.2)))
        cv2.circle(mask, (int(bx), int(by)), bead_radius, 255, -1)
    return mask


def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        epsilon = max(0.8, 0.0025 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 3:
            polygons.append(approx.reshape(-1, 2).astype(float).ravel().tolist())
    return polygons


def bbox_area_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, 0.0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1], float(len(xs))


def fiber_metrics(centerline, width_px):
    diffs = np.diff(centerline, axis=0)
    length = float(np.linalg.norm(diffs, axis=1).sum())
    start, end = centerline[0], centerline[-1]
    orientation = float(math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])))

    if len(centerline) < 3:
        curvature = 0.0
    else:
        tangents = np.diff(centerline, axis=0)
        angles = np.arctan2(tangents[:, 1], tangents[:, 0])
        curvature = float(np.mean(np.abs(np.diff(np.unwrap(angles)))))

    return float(width_px), length, orientation, curvature


def coco_keypoints(centerline):
    ordered = order_keypoints_top_to_bottom_left_to_right(centerline)
    keypoints = []
    for x, y in ordered:
        keypoints.extend([int(round(x)), int(round(y)), 2])
    return keypoints, ordered


def render_image(background, fiber_records, rng):
    image = background.astype(np.float32)

    for record in fiber_records:
        mask = record["mask"]
        shade = record["shade"]
        edge = cv2.GaussianBlur(mask, (0, 0), record["edge_sigma"]).astype(np.float32) / 255.0
        image = image * (1 - edge * 0.85) + shade * edge * 0.85

        if record["shadow"]:
            kernel = np.ones((3, 3), np.uint8)
            shadow = cv2.dilate(mask, kernel, iterations=1)
            shadow = cv2.GaussianBlur(shadow, (0, 0), 2.0).astype(np.float32) / 255.0
            image -= shadow * rng.uniform(8, 20)

    image += rng.normal(0, rng.uniform(3, 8), image.shape)
    if rng.random() < 0.35:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.4, 1.0))
    return np.clip(image, 0, 255).astype(np.uint8)


def generate_sample(image_id, ann_start_id, height, width, min_fibers, max_fibers, rng):
    background = make_sem_background(height, width, rng)
    n_fibers = int(rng.integers(min_fibers, max_fibers + 1))
    union_mask = np.zeros((height, width), dtype=np.uint8)
    render_records = []
    annotations = []
    ann_id = ann_start_id

    for _ in range(n_fibers):
        centerline = random_fiber_centerline(height, width, rng)
        width_px = float(rng.uniform(7.0, 43.0))
        has_bead = bool(rng.random() < 0.12)
        mask = draw_fiber_mask((height, width), centerline, width_px, has_bead, rng)

        bbox, area = bbox_area_from_mask(mask)
        if bbox is None or area < 30:
            continue

        segmentation = mask_to_polygons(mask)
        if not segmentation:
            continue

        overlap = np.logical_and(mask > 0, union_mask > 0).sum()
        is_crossing = bool(overlap > max(8, area * 0.03))
        union_mask = np.maximum(union_mask, mask)

        keypoints, ordered_centerline = coco_keypoints(centerline)
        fiber_width, fiber_length, orientation, curvature = fiber_metrics(ordered_centerline, width_px)

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": CATEGORY_ID,
            "segmentation": segmentation,
            "keypoints": keypoints,
            "num_keypoints": TARGET_NUM_KEYPOINTS,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
            "has_bead": has_bead,
            "is_blurry": False,
            "is_crossing": is_crossing,
            "fiber_width": round(fiber_width, 3),
            "fiber_length": round(fiber_length, 3),
            "fiber_curvature": round(curvature, 6),
            "fiber_orientation": round(orientation, 2),
            "metadata": {
                "source": "synthetic",
                "order_rule": "TopToBottomLeftToRight",
                "mask_type": "synthetic_spline",
            },
        })
        ann_id += 1

        render_records.append({
            "mask": mask,
            "shade": float(rng.uniform(150, 235)),
            "edge_sigma": float(rng.uniform(0.45, 1.25)),
            "shadow": bool(rng.random() < 0.55),
        })

    image = render_image(background, render_records, rng)
    porosity = float(1.0 - (union_mask > 0).mean())
    for annotation in annotations:
        annotation["porosity"] = round(porosity, 6)
    return image, annotations, ann_id, porosity


def write_coco(output_path, images, annotations):
    data = {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": CATEGORY_ID,
            "name": "fiber",
            "supercategory": "fiber",
            "keypoints": [f"p{i + 1}" for i in range(TARGET_NUM_KEYPOINTS)],
            "skeleton": [[i + 1, i + 2] for i in range(TARGET_NUM_KEYPOINTS - 1)],
        }],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    annotation_dir = output_dir / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    all_images = []
    all_annotations = []
    ann_id = 1

    for image_id in range(1, args.num_images + 1):
        image, annotations, ann_id, porosity = generate_sample(
            image_id,
            ann_id,
            args.height,
            args.width,
            args.min_fibers,
            args.max_fibers,
            rng,
        )
        file_name = f"synthetic_{image_id:05d}.png"
        cv2.imwrite(str(image_dir / file_name), image)
        all_images.append({
            "id": image_id,
            "file_name": file_name,
            "height": args.height,
            "width": args.width,
            "porosity": round(porosity, 6),
        })
        all_annotations.extend(annotations)
        print(f"{file_name}: {len(annotations)} fibers, porosity={porosity:.3f}")

    indices = list(range(len(all_images)))
    rng.shuffle(indices)
    split = int(len(indices) * args.train_ratio)
    train_ids = {all_images[i]["id"] for i in indices[:split]}
    val_ids = {all_images[i]["id"] for i in indices[split:]}

    train_images = [img for img in all_images if img["id"] in train_ids]
    val_images = [img for img in all_images if img["id"] in val_ids]
    train_annotations = [ann for ann in all_annotations if ann["image_id"] in train_ids]
    val_annotations = [ann for ann in all_annotations if ann["image_id"] in val_ids]

    write_coco(annotation_dir / "synthetic_fiber_train.json", train_images, train_annotations)
    write_coco(annotation_dir / "synthetic_fiber_val.json", val_images, val_annotations)
    write_coco(annotation_dir / "synthetic_fiber_all.json", all_images, all_annotations)

    print(
        f"Synthetic dataset ready in {output_dir} "
        f"({len(train_images)} train images, {len(val_images)} val images, "
        f"{len(all_annotations)} annotations)."
    )


if __name__ == "__main__":
    main()
