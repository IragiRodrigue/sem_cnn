import csv
import json
import math
import os
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import ROI_HEADS_REGISTRY

from fiber_heads import FiberROIHeadsV2
from fiber_viz import analyze_fiber_complexity


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "synthetic_dataset/images"
COCO_JSON = ROOT / "synthetic_dataset/annotations/synthetic_fiber_val.json"
OUTPUT_DIR = ROOT / "output_synthetic_pretrain"
MODEL_WEIGHTS = OUTPUT_DIR / "model_final.pth"
EVAL_DIR = ROOT / "evaluation_outputs"
PREDICTION_DIR = EVAL_DIR / "predictions"
FIGURE_DIR = EVAL_DIR / "figures"
TABLE_DIR = EVAL_DIR / "tables"
ABLATION_DIR = EVAL_DIR / "ablation_inputs"
TEST_SPLIT_JSON = EVAL_DIR / "coco_fiber_test_split.json"
TEST_SPLIT_RATIO = 0.2
NUM_KEYPOINTS = 40
SCORE_THRESH_TEST = 0.5
SSR_KEYPOINT_RANGE = list(range(5, 41, 1))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a FibeR-CNN checkpoint on the synthetic fiber annotations."
    )
    parser.add_argument("--weights", default=str(MODEL_WEIGHTS), help="Path to model weights.")
    parser.add_argument("--coco-json", default=str(COCO_JSON), help="COCO annotation JSON.")
    parser.add_argument("--image-dir", default=str(IMAGE_DIR), help="Directory containing real SEM images.")
    parser.add_argument("--eval-dir", default=str(EVAL_DIR), help="Directory for evaluation outputs.")
    parser.add_argument("--score-thresh", type=float, default=SCORE_THRESH_TEST)
    return parser.parse_args()


def configure_paths(args):
    global IMAGE_DIR, COCO_JSON, EVAL_DIR, PREDICTION_DIR, FIGURE_DIR, TABLE_DIR
    global ABLATION_DIR, TEST_SPLIT_JSON

    IMAGE_DIR = Path(args.image_dir)
    COCO_JSON = Path(args.coco_json)
    EVAL_DIR = Path(args.eval_dir)
    PREDICTION_DIR = EVAL_DIR / "predictions"
    FIGURE_DIR = EVAL_DIR / "figures"
    TABLE_DIR = EVAL_DIR / "tables"
    ABLATION_DIR = EVAL_DIR / "ablation_inputs"
    TEST_SPLIT_JSON = EVAL_DIR / "coco_fiber_test_split.json"


@dataclass
class AnnotationCurve:
    image_id: int
    ann_id: int
    points: np.ndarray  # (N, 2) in (x, y)


def ensure_dirs():
    for directory in [EVAL_DIR, PREDICTION_DIR, FIGURE_DIR, TABLE_DIR, ABLATION_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def register_custom_heads():
    try:
        ROI_HEADS_REGISTRY.register(FiberROIHeadsV2)
    except AssertionError:
        pass


def build_cfg(weights_path=MODEL_WEIGHTS, score_thresh=SCORE_THRESH_TEST):
    register_custom_heads()
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NAME = "FiberROIHeadsV2"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.KEYPOINT_ON = True
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = NUM_KEYPOINTS
    cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 10.0
    cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.05] * NUM_KEYPOINTS
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.DATASETS.TEST = ("fiber_test_eval",)
    return cfg


def load_coco():
    with open(COCO_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def create_test_split():
    coco = load_coco()
    images = sorted(coco["images"], key=lambda item: (item["file_name"], item["id"]))
    split_index = max(1, int(len(images) * (1.0 - TEST_SPLIT_RATIO)))
    raw_test_images = images[split_index:]
    test_ids = {img["id"] for img in raw_test_images}

    test_annotations = [ann for ann in coco["annotations"] if ann["image_id"] in test_ids]

    test_images = []
    for img in raw_test_images:
        test_images.append({
            "id": img["id"],
            "file_name": img["file_name"],
            "height": img["height"],
            "width": img["width"],
        })

    test_coco = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": coco["categories"],
    }
    with open(TEST_SPLIT_JSON, "w", encoding="utf-8") as f:
        json.dump(test_coco, f, indent=2)
    return test_coco


def register_test_dataset():
    if "fiber_test_eval" in DatasetCatalog.list():
        DatasetCatalog.remove("fiber_test_eval")
    register_coco_instances("fiber_test_eval", {}, str(TEST_SPLIT_JSON), str(IMAGE_DIR))
    metadata = MetadataCatalog.get("fiber_test_eval")
    metadata.set(
        thing_classes=["fiber"],
        keypoint_names=[f"p{i+1}" for i in range(NUM_KEYPOINTS)],
        keypoint_flip_map=[],
    )


def parse_keypoints(keypoints_flat):
    arr = np.array(keypoints_flat, dtype=np.float32).reshape(-1, 3)
    return arr[:, :2], arr[:, 2]


def resample_curve(points_xy, num_points):
    if len(points_xy) == 0:
        return np.zeros((num_points, 2), dtype=np.float32)
    if len(points_xy) == 1:
        return np.repeat(points_xy, num_points, axis=0)

    deltas = np.diff(points_xy, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cum[-1])
    if total <= 1e-8:
        return np.repeat(points_xy[:1], num_points, axis=0)

    targets = np.linspace(0.0, total, num_points)
    sampled = []
    for t in targets:
        idx = np.searchsorted(cum, t, side="right") - 1
        idx = min(max(idx, 0), len(points_xy) - 2)
        start = points_xy[idx]
        end = points_xy[idx + 1]
        segment_len = max(cum[idx + 1] - cum[idx], 1e-8)
        alpha = (t - cum[idx]) / segment_len
        sampled.append(start * (1.0 - alpha) + end * alpha)
    return np.array(sampled, dtype=np.float32)


def compute_ssr(gt_points, approx_points):
    residuals = gt_points - approx_points
    return float(np.sum(np.square(residuals)))


def build_annotation_curves(coco):
    curves = []
    for ann in coco["annotations"]:
        pts_xy, visibility = parse_keypoints(ann["keypoints"])
        visible = pts_xy[visibility > 0]
        if len(visible) >= 2:
            curves.append(AnnotationCurve(ann["image_id"], ann["id"], visible))
    return curves


def plot_ssr_curve(annotation_curves):
    ssr_by_k = defaultdict(list)
    optimum_counts = []

    for curve in annotation_curves:
        gt_40 = resample_curve(curve.points, NUM_KEYPOINTS)
        ssr_values = []
        for k in SSR_KEYPOINT_RANGE:
            compressed = resample_curve(curve.points, k)
            approx_40 = resample_curve(compressed, NUM_KEYPOINTS)
            ssr = compute_ssr(gt_40, approx_40)
            ssr_by_k[k].append(ssr)
            ssr_values.append((k, ssr))

        baseline_ssr = max(ssr_values[0][1], 1e-8)
        threshold = baseline_ssr * 0.05
        optimum = next((k for k, ssr in ssr_values if ssr <= threshold), SSR_KEYPOINT_RANGE[-1])
        optimum_counts.append(optimum)

    mean_ssr = [float(np.mean(ssr_by_k[k])) for k in SSR_KEYPOINT_RANGE]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(SSR_KEYPOINT_RANGE, mean_ssr, color="#0C7BDC", linewidth=2.5)
    ax.set_title("SSR moyen de l'approximation des keypoints")
    ax.set_xlabel("Nombre de keypoints")
    ax.set_ylabel("Sum of squared residuals (SSR)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ssr_vs_ground_truth.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(min(optimum_counts), max(optimum_counts) + 2) - 0.5
    ax.hist(optimum_counts, bins=bins, color="#2A9D8F", edgecolor="white")
    ax.set_title("Distribution du nombre optimal de keypoints")
    ax.set_xlabel("Nombre optimal de keypoints")
    ax.set_ylabel("Nombre de fibres")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "optimum_keypoint_distribution.png", dpi=180)
    plt.close(fig)

    return mean_ssr, optimum_counts


def plot_keypoint_order_examples():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    examples = [
        np.array([[0.1, 0.1], [0.3, 0.3], [0.5, 0.55], [0.7, 0.8], [0.85, 0.92]]),
        np.array([[0.2, 0.85], [0.35, 0.65], [0.48, 0.45], [0.62, 0.25], [0.78, 0.08]]),
        np.array([[0.12, 0.35], [0.28, 0.4], [0.48, 0.52], [0.68, 0.62], [0.9, 0.75]]),
        np.array([[0.15, 0.72], [0.35, 0.61], [0.5, 0.51], [0.67, 0.41], [0.86, 0.28]]),
    ]
    titles = [
        "Fibre verticale descendante",
        "Fibre verticale montante",
        "Fibre inclinée gauche-droite",
        "Fibre inclinée droite-gauche",
    ]
    cmap = plt.get_cmap("viridis", 5)

    for ax, points, title in zip(axes.ravel(), examples, titles):
        order = np.lexsort((points[:, 0], points[:, 1]))
        ordered = points[order]
        ax.plot(ordered[:, 0], ordered[:, 1], color="#2C3E50", linewidth=1.5)
        for idx, point in enumerate(ordered):
            ax.scatter(point[0], point[1], color=cmap(idx), s=70)
            ax.text(point[0] + 0.015, point[1] + 0.015, f"p{idx+1}", fontsize=9)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    fig.suptitle('Ordre des keypoints selon la règle "top to bottom, left to right"')
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "keypoint_order_examples.png", dpi=180)
    plt.close(fig)


def save_augmentation_table():
    rows = [
        ["Étape", "Traitement", "But principal"],
        ["Entrée MEB", "Crop inférieur", "Supprimer le bandeau d'information"],
        ["Prétraitement 1", "Filtre médian + gaussien", "Réduction du bruit"],
        ["Prétraitement 2", "CLAHE", "Renforcer le contraste local"],
        ["Prétraitement 3", "Frangi", "Accentuer les structures filiformes"],
        ["Prétraitement 4", "Ouverture/Fermeture morphologique", "Nettoyer les masques"],
        ["Annotation", "SAM multi-variantes + déduplication IoU", "Augmenter le rappel"],
        ["Post-traitement", "Squelettisation + 40 keypoints", "Normaliser la géométrie"],
        ["Training mapper", "Transforms Detectron2 avec keypoint_hflip identity", "Cohérence image/masque/keypoints"],
    ]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis("off")
    table = ax.table(cellText=rows, cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#0C7BDC")
        else:
            cell.set_facecolor("#F7FBFF" if r % 2 else "#EAF3FB")
    fig.tight_layout()
    fig.savefig(TABLE_DIR / "input_augmentation_table.png", dpi=180)
    plt.close(fig)

    with open(TABLE_DIR / "input_augmentation_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def polyline_mask(points_xy, image_shape=(256, 256), thickness=16):
    canvas = np.zeros(image_shape, dtype=np.uint8)
    pts = np.round(points_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], isClosed=False, color=255, thickness=thickness)
    canvas = cv2.dilate(canvas, np.ones((5, 5), dtype=np.uint8), iterations=1)
    return canvas


def plot_misplaced_keypoint_effect():
    x = np.linspace(30, 220, 10)
    y = 128 + 40 * np.sin(np.linspace(0, math.pi, 10))
    gt_points = np.column_stack([x, y])
    bad_points = gt_points.copy()
    bad_points[5, 1] += 45

    gt_mask = polyline_mask(gt_points)
    bad_mask = polyline_mask(bad_points)
    inter = np.logical_and(gt_mask > 0, bad_mask > 0).sum()
    union = np.logical_or(gt_mask > 0, bad_mask > 0).sum()
    iou = inter / max(union, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gt_mask, cmap="gray")
    axes[0].scatter(gt_points[:, 0], gt_points[:, 1], c=np.arange(len(gt_points)), cmap="viridis", s=25)
    axes[0].set_title("Masque avec keypoints corrects")

    axes[1].imshow(bad_mask, cmap="gray")
    axes[1].scatter(bad_points[:, 0], bad_points[:, 1], c=np.arange(len(bad_points)), cmap="viridis", s=25)
    axes[1].set_title("Masque avec un keypoint déplacé")

    diff = np.zeros((256, 256, 3), dtype=np.uint8)
    diff[(gt_mask > 0) & (bad_mask == 0)] = [0, 200, 0]
    diff[(bad_mask > 0) & (gt_mask == 0)] = [220, 70, 70]
    diff[(bad_mask > 0) & (gt_mask > 0)] = [40, 80, 190]
    axes[2].imshow(diff)
    axes[2].set_title(f"Différence de masque\nIoU = {iou:.3f}")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "misplaced_keypoint_effect.png", dpi=180)
    plt.close(fig)


def parse_training_metrics():
    records = []
    with open(OUTPUT_DIR / "metrics.json", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def plot_learning_rate_schedule(records):
    lr_records = [r for r in records if "iteration" in r and "lr" in r]
    iters = [r["iteration"] for r in lr_records]
    lrs = [r["lr"] for r in lr_records]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iters, lrs, color="#E76F51", linewidth=2.2)
    ax.set_title("Learning rate schedule")
    ax.set_xlabel("Itération")
    ax.set_ylabel("Learning rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "learning_rate_schedule.png", dpi=180)
    plt.close(fig)


def prepare_ablation_templates():
    templates = {
        "baseline_ap_comparison.csv": [
            ["model", "AP", "AP50", "AP75"],
            ["Mask R-CNN baseline", "", "", ""],
            ["FibeR-CNN", "", "", ""],
        ],
        "mask_head_ablation.csv": [
            ["model", "AP", "AP50", "AP75"],
            ["Mask R-CNN baseline", "", "", ""],
            ["FibeR-CNN without extra mask head", "", "", ""],
            ["FibeR-CNN with extra mask head", "", "", ""],
        ],
        "keypoint_ordering_ablation.csv": [
            ["model", "AP", "AP50", "AP75"],
            ["Mask R-CNN baseline", "", "", ""],
            ["FibeR-CNN unordered", "", "", ""],
            ["FibeR-CNN ordered", "", "", ""],
        ],
        "input_augmentation_ablation.csv": [
            ["model", "AP", "AP50", "AP75"],
            ["Mask R-CNN baseline", "", "", ""],
            ["Mask R-CNN + augmentation", "", "", ""],
            ["FibeR-CNN + augmentation", "", "", ""],
        ],
        "error_correction_ablation.csv": [
            ["model", "AP", "AP50", "AP75"],
            ["Mask R-CNN + augmentation", "", "", ""],
            ["FibeR-CNN + error correction", "", "", ""],
        ],
        "architecture_ap_summary.csv": [
            ["architecture", "AP", "AP50", "AP75"],
            ["Mask R-CNN", "", "", ""],
            ["Keypoint R-CNN", "", "", ""],
            ["FibeR-CNN", "", "", ""],
        ],
    }

    for filename, rows in templates.items():
        path = ABLATION_DIR / filename
        if path.exists():
            continue
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)


def maybe_plot_ablation(csv_name, title, output_name):
    path = ABLATION_DIR / csv_name
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    usable_rows = [row for row in rows if row.get("AP")]
    if not usable_rows:
        return False

    label_key = next(iter(rows[0].keys()))
    labels = [row[label_key] for row in usable_rows]
    aps = [float(row["AP"]) for row in usable_rows]
    ap50 = [float(row["AP50"]) for row in usable_rows]
    ap75 = [float(row["AP75"]) for row in usable_rows]

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, aps, width=width, label="AP", color="#0C7BDC")
    ax.bar(x, ap50, width=width, label="AP50", color="#2A9D8F")
    ax.bar(x + width, ap75, width=width, label="AP75", color="#F4A261")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Average precision")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=180)
    plt.close(fig)
    return True


def mask_union_area(masks):
    if masks is None:
        return 0
    if isinstance(masks, np.ndarray):
        if masks.size == 0 or len(masks) == 0:
            return 0
        union = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            union |= mask.astype(bool)
        return int(union.sum())
    if not masks:
        return 0
    union = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        union |= mask.astype(bool)
    return int(union.sum())


def derive_predicted_properties(instances, image_gray):
    results = []
    pred_masks = instances.pred_masks.cpu().numpy() if instances.has("pred_masks") else np.zeros((0, 1, 1))
    diameters = []
    bead_flags = []

    for idx in range(len(instances)):
        mask = pred_masks[idx].astype(np.uint8) * 255
        if instances.has("pred_bead"):
            bead_score = float(instances.pred_bead[idx].item())
            has_bead = bead_score >= 0.5
        elif mask.size == 0 or np.max(mask) <= 0:
            has_bead = False
        else:
            x, y, w, h = cv2.boundingRect(mask)
            roi = image_gray[y:y+h, x:x+w]
            if w <= 0 or h <= 0 or roi.size == 0:
                has_bead = False
            else:
                has_bead, _, _ = analyze_fiber_complexity(mask, image_gray)
        bead_flags.append(bool(has_bead))

        if instances.has("pred_width"):
            diameter = float(instances.pred_width[idx].item())
        else:
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            diameter = float(2.0 * np.median(dist[mask > 0])) if np.any(mask > 0) else 0.0
        diameters.append(diameter)

        results.append({
            "instance_index": idx,
            "score": float(instances.scores[idx].item()) if instances.has("scores") else None,
            "predicted_diameter": diameter,
            "predicted_has_bead": bool(has_bead),
            "predicted_length": float(instances.pred_length[idx].item()) if instances.has("pred_length") else None,
            "predicted_orientation": float(instances.pred_orientation[idx].item()) if instances.has("pred_orientation") else None,
            "predicted_curvature": float(instances.pred_curvature[idx].item()) if instances.has("pred_curvature") else None,
            "predicted_bead_score": float(instances.pred_bead[idx].item()) if instances.has("pred_bead") else None,
            "predicted_porosity": float(instances.pred_porosity[idx].item()) if instances.has("pred_porosity") else None,
        })

    if instances.has("pred_porosity") and len(instances) > 0:
        porosity = float(instances.pred_porosity.mean().item())
    else:
        porosity = 1.0 - (mask_union_area(pred_masks) / max(image_gray.shape[0] * image_gray.shape[1], 1))
    return results, diameters, bead_flags, porosity


def evaluate_detector(cfg):
    evaluator = COCOEvaluator("fiber_test_eval", output_dir=str(EVAL_DIR / "coco_eval"), kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS)
    test_loader = build_detection_test_loader(cfg, "fiber_test_eval")
    predictor = DefaultPredictor(cfg)
    metrics = inference_on_dataset(predictor.model, test_loader, evaluator)
    with open(EVAL_DIR / "coco_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return predictor, metrics


def dataset_records_from_coco(coco):
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    records = []
    for img in coco["images"]:
        records.append({
            "image_id": img["id"],
            "file_name": str(IMAGE_DIR / img["file_name"]),
            "height": img["height"],
            "width": img["width"],
            "annotations": anns_by_image[img["id"]],
        })
    return records


def plot_property_predictions(image_level_rows):
    diameters = [row["mean_predicted_diameter"] for row in image_level_rows if row["mean_predicted_diameter"] is not None]
    porosities = [row["predicted_porosity"] for row in image_level_rows]
    bead_rates = [row["predicted_bead_rate"] for row in image_level_rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(diameters, bins=20, color="#0C7BDC", edgecolor="white")
    axes[0].set_title("Distribution des diamètres prédits")
    axes[0].set_xlabel("Diamètre")

    axes[1].hist(porosities, bins=20, color="#2A9D8F", edgecolor="white")
    axes[1].set_title("Distribution de la porosité prédite")
    axes[1].set_xlabel("Porosité")

    axes[2].hist(bead_rates, bins=20, color="#F4A261", edgecolor="white")
    axes[2].set_title("Distribution du taux de bead")
    axes[2].set_xlabel("Proportion de fibres avec bead")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "predicted_properties_summary.png", dpi=180)
    plt.close(fig)


def run_prediction_workflow(predictor, test_coco):
    records = dataset_records_from_coco(test_coco)
    image_level_rows = []
    instance_rows = []

    for record in records:
        image_gray = cv2.imread(record["file_name"], cv2.IMREAD_GRAYSCALE)
        image_bgr = cv2.imread(record["file_name"], cv2.IMREAD_COLOR)
        outputs = predictor(image_bgr)
        instances = outputs["instances"].to("cpu")

        instance_results, diameters, bead_flags, porosity = derive_predicted_properties(instances, image_gray)
        for row in instance_results:
            row["image_id"] = record["image_id"]
            row["file_name"] = os.path.basename(record["file_name"])
            instance_rows.append(row)

        image_level_rows.append({
            "image_id": record["image_id"],
            "file_name": os.path.basename(record["file_name"]),
            "num_predictions": len(instance_results),
            "mean_predicted_diameter": float(np.mean(diameters)) if diameters else None,
            "predicted_porosity": float(porosity),
            "predicted_bead_rate": float(np.mean(bead_flags)) if bead_flags else 0.0,
        })

    with open(PREDICTION_DIR / "predicted_image_properties.json", "w", encoding="utf-8") as f:
        json.dump(image_level_rows, f, indent=2)
    with open(PREDICTION_DIR / "predicted_instance_properties.json", "w", encoding="utf-8") as f:
        json.dump(instance_rows, f, indent=2)

    plot_property_predictions(image_level_rows)
    return image_level_rows, instance_rows


def save_workflow_summary(metrics, image_rows):
    summary = {
        "test_split_json": str(TEST_SPLIT_JSON),
        "num_test_images": len(image_rows),
        "mean_predicted_porosity": float(np.mean([row["predicted_porosity"] for row in image_rows])) if image_rows else None,
        "mean_predicted_bead_rate": float(np.mean([row["predicted_bead_rate"] for row in image_rows])) if image_rows else None,
        "coco_metrics": metrics,
    }
    with open(EVAL_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    args = parse_args()
    configure_paths(args)
    ensure_dirs()
    test_coco = create_test_split()
    register_test_dataset()

    annotation_curves = build_annotation_curves(test_coco)
    plot_ssr_curve(annotation_curves)
    plot_keypoint_order_examples()
    save_augmentation_table()
    plot_misplaced_keypoint_effect()

    training_records = parse_training_metrics()
    plot_learning_rate_schedule(training_records)

    prepare_ablation_templates()
    maybe_plot_ablation(
        "baseline_ap_comparison.csv",
        "AP baseline Mask R-CNN vs FibeR-CNN",
        "ap_baseline_comparison.png",
    )
    maybe_plot_ablation(
        "mask_head_ablation.csv",
        "Influence d'une tête de masque supplémentaire",
        "mask_head_ablation.png",
    )
    maybe_plot_ablation(
        "keypoint_ordering_ablation.csv",
        "Influence de l'ordre des keypoints",
        "keypoint_ordering_ablation.png",
    )
    maybe_plot_ablation(
        "input_augmentation_ablation.csv",
        "Influence de l'augmentation d'entrée",
        "input_augmentation_ablation.png",
    )
    maybe_plot_ablation(
        "error_correction_ablation.csv",
        "Influence de la correction d'erreur",
        "error_correction_ablation.png",
    )
    maybe_plot_ablation(
        "architecture_ap_summary.csv",
        "Average precisions des architectures testées",
        "architecture_ap_summary.png",
    )

    cfg = build_cfg(args.weights, args.score_thresh)
    predictor, metrics = evaluate_detector(cfg)
    image_rows, _ = run_prediction_workflow(predictor, test_coco)
    save_workflow_summary(metrics, image_rows)

    print(f"Evaluation workflow completed. Outputs saved in: {EVAL_DIR}")


if __name__ == "__main__":
    main()
