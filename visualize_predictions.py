import json
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import ROI_HEADS_REGISTRY

from fiber_heads import FiberROIHeadsV2


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "synthetic_dataset/images"
COCO_JSON = ROOT / "synthetic_dataset/annotations/synthetic_fiber_val.json"
OUTPUT_DIR = ROOT / "output_synthetic_pretrain"
MODEL_WEIGHTS = OUTPUT_DIR / "model_final.pth"
EVAL_DIR = ROOT / "evaluation_outputs"
TEST_SPLIT_JSON = EVAL_DIR / "coco_fiber_test_split.json"
VIZ_DIR = EVAL_DIR / "visualizations"
NUM_KEYPOINTS = 40
SCORE_THRESH_TEST = 0.5


def ensure_dirs():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)


def register_custom_heads():
    try:
        ROI_HEADS_REGISTRY.register(FiberROIHeadsV2)
    except AssertionError:
        pass


def build_cfg(weights_path=MODEL_WEIGHTS):
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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def load_coco_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_keypoints(keypoints_flat):
    """Convert flat keypoints [x1,y1,v1, x2,y2,v2, ...] to (N,2) and visibility"""
    arr = np.array(keypoints_flat, dtype=np.float32).reshape(-1, 3)
    return arr[:, :2], arr[:, 2]


def get_segmentation_mask(segmentation, image_shape):
    """Convert RLE or polygon to binary mask"""
    if not segmentation:
        return None
    
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    # Handle RLE format
    if isinstance(segmentation, dict) and "size" in segmentation:
        from pycocotools import mask as mask_utils
        rle = segmentation
        decoded = mask_utils.decode(rle)
        return decoded.astype(np.uint8)
    
    # Handle polygon format
    if isinstance(segmentation, list):
        for poly in segmentation:
            if len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 255)
        return mask
    
    return None


def draw_keypoints_on_image(image, keypoints_xy, visibility, color=(0, 255, 0), radius=4):
    """Draw keypoints with visibility filtering"""
    img_copy = image.copy()
    for (x, y), vis in zip(keypoints_xy, visibility):
        if vis > 0:
            cv2.circle(img_copy, (int(x), int(y)), radius, color, -1)
    return img_copy


def draw_mask_overlay(image, mask, color=(0, 255, 0), alpha=0.3):
    """Draw mask overlay on image"""
    if mask is None or mask.size == 0:
        return image
    
    img_copy = image.copy()
    mask_bool = mask.astype(bool)
    overlay = img_copy.copy()
    overlay[mask_bool] = color
    return cv2.addWeighted(img_copy, 1 - alpha, overlay, alpha, 0)


def visualize_sample(
    image_id, 
    test_coco, 
    predictor, 
    image_dir, 
    keypoint_names=None
):
    """Visualize GT and predictions for a single image"""
    
    # Find image and annotations in COCO
    img_data = next((img for img in test_coco["images"] if img["id"] == image_id), None)
    if img_data is None:
        print(f"Image {image_id} not found")
        return None
    
    image_path = image_dir / img_data["file_name"]
    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        return None
    
    # Load image
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image_bgr is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    h, w = image_bgr.shape[:2]
    
    # Get annotations
    anns = [ann for ann in test_coco["annotations"] if ann["image_id"] == image_id]
    
    # Run prediction
    outputs = predictor(image_bgr)
    instances = outputs["instances"].to("cpu")
    
    # Create side-by-side canvas: GT on left, Pred on right
    # Limit visualization to max 3 objects per side
    max_show = 3
    show_anns = anns[:max_show]
    num_preds = len(instances)
    
    # Resize to manageable size
    display_h = 512
    display_w = 512
    scale_h = display_h / h
    scale_w = display_w / w
    
    image_resized = cv2.resize(image_bgr, (display_w, display_h))
    
    canvas = np.ones((display_h + 60, display_w * 2 + 30, 3), dtype=np.uint8) * 255
    
    # Title
    cv2.putText(canvas, f"Image ID {image_id}: Ground Truth (LEFT) vs Predictions (RIGHT)", 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(canvas, f"GT Objects: {len(anns)} | Predicted Objects: {len(instances)}", 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    # --- GROUND TRUTH (LEFT) ---
    img_gt = image_resized.copy()
    
    for ann_idx, ann in enumerate(show_anns):
        # Draw mask
        if "segmentation" in ann:
            mask = get_segmentation_mask(ann["segmentation"], image_bgr.shape[:2])
            if mask is not None:
                # Resize mask to match display size
                mask_resized = cv2.resize(mask, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
                img_gt = draw_mask_overlay(img_gt, mask_resized, color=(0, 255, 0), alpha=0.3)
        
        # Draw keypoints
        if "keypoints" in ann:
            kpts_xy, visibility = parse_keypoints(ann["keypoints"])
            kpts_xy_scaled = kpts_xy * [scale_w, scale_h]
            img_gt = draw_keypoints_on_image(img_gt, kpts_xy_scaled, visibility, color=(0, 0, 255), radius=2)
        
        # Draw bbox
        if "bbox" in ann:
            x, y, bw, bh = ann["bbox"]
            x_s, y_s = int(x * scale_w), int(y * scale_h)
            w_s, h_s = int(bw * scale_w), int(bh * scale_h)
            cv2.rectangle(img_gt, (x_s, y_s), (x_s + w_s, y_s + h_s), (255, 0, 0), 2)
            cv2.putText(img_gt, f"GT#{ann_idx}", (x_s, y_s - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    canvas[60:60+display_h, 10:10+display_w] = img_gt
    
    # --- PREDICTIONS (RIGHT) ---
    img_pred = image_resized.copy()
    
    for pred_idx in range(min(max_show, num_preds)):
        # Draw predicted mask
        if instances.has("pred_masks"):
            mask = instances.pred_masks[pred_idx].cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
            img_pred = draw_mask_overlay(img_pred, mask_resized, color=(255, 0, 0), alpha=0.3)
        
        # Draw predicted keypoints
        if instances.has("pred_keypoints"):
            kpts_xy = instances.pred_keypoints[pred_idx].cpu().numpy()
            # Keypoints from Detectron2 are (N, 3) with [x, y, confidence]
            visibility = np.ones(len(kpts_xy))
            kpts_xy_scaled = kpts_xy[:, :2] * [scale_w, scale_h]
            img_pred = draw_keypoints_on_image(img_pred, kpts_xy_scaled, visibility, color=(255, 165, 0), radius=2)
        
        # Draw predicted bbox
        if instances.has("pred_boxes"):
            box = instances.pred_boxes[pred_idx].tensor[0].cpu().numpy()
            x1, y1, x2, y2 = box
            x1_s = int(x1 * scale_w)
            y1_s = int(y1 * scale_h)
            x2_s = int(x2 * scale_w)
            y2_s = int(y2 * scale_h)
            cv2.rectangle(img_pred, (x1_s, y1_s), (x2_s, y2_s), (0, 165, 255), 2)
            
            score = instances.scores[pred_idx].item() if instances.has("scores") else 0.0
            cv2.putText(img_pred, f"P#{pred_idx} ({score:.2f})", (x1_s, y1_s - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
    
    canvas[60:60+display_h, 20+display_w:20+display_w*2] = img_pred
    
    return canvas


def main():
    ensure_dirs()
    
    # Load test split
    test_coco = load_coco_json(TEST_SPLIT_JSON)
    print(f"Test split has {len(test_coco['images'])} images")
    
    # Build and load model
    cfg = build_cfg()
    predictor = DefaultPredictor(cfg)
    print(f"Model loaded from: {MODEL_WEIGHTS}")
    print(f"Running on: {cfg.MODEL.DEVICE}")
    
    # Visualize first N images
    num_samples = 5
    sample_ids = [img["id"] for img in test_coco["images"][:num_samples]]
    
    for idx, image_id in enumerate(sample_ids):
        print(f"Visualizing sample {idx+1}/{num_samples} (image_id={image_id})...")
        
        canvas = visualize_sample(
            image_id,
            test_coco,
            predictor,
            IMAGE_DIR,
        )
        
        if canvas is not None:
            output_path = VIZ_DIR / f"sample_{idx:02d}_id{image_id}.png"
            cv2.imwrite(str(output_path), canvas)
            print(f"  -> Saved: {output_path}")
        else:
            print(f"  -> Failed to visualize")
    
    print(f"\nVisualizations saved to: {VIZ_DIR}")


if __name__ == "__main__":
    main()
