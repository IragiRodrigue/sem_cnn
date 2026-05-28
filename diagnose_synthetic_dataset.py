"""
Diagnostic script to verify synthetic fiber dataset quality and format.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt


def diagnose_json(json_path):
    """Check COCO JSON file structure and content."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {json_path.name}")
    print('='*60)
    
    if not json_path.exists():
        print(f"❌ File does not exist: {json_path}")
        return False
    
    with open(json_path) as f:
        coco = json.load(f)
    
    # Check structure
    required_keys = {'images', 'annotations', 'categories'}
    missing = required_keys - set(coco.keys())
    if missing:
        print(f"❌ Missing keys: {missing}")
        return False
    print(f"✓ Valid COCO structure")
    
    # Count statistics
    num_images = len(coco['images'])
    num_annotations = len(coco['annotations'])
    num_categories = len(coco['categories'])
    
    print(f"\n📊 Statistics:")
    print(f"  - Images: {num_images}")
    print(f"  - Annotations: {num_annotations}")
    print(f"  - Categories: {num_categories}")
    print(f"  - Average annotations per image: {num_annotations/max(num_images, 1):.1f}")
    
    # Check categories
    print(f"\n📁 Categories:")
    for cat in coco['categories']:
        print(f"  - {cat['name']} (id={cat['id']})")
        if 'keypoints' in cat:
            num_kpts = len(cat['keypoints'])
            print(f"    └─ {num_kpts} keypoints")
    
    # Validate images
    print(f"\n🖼️  Image Validation:")
    invalid_images = 0
    for img in coco['images']:
        if 'id' not in img or 'file_name' not in img:
            invalid_images += 1
    
    if invalid_images == 0:
        print(f"  ✓ All {num_images} images have required fields")
    else:
        print(f"  ❌ {invalid_images}/{num_images} images are invalid")
    
    # Validate annotations
    print(f"\n📌 Annotation Validation:")
    invalid_annotations = 0
    keypoint_issues = 0
    keypoint_distribution = defaultdict(int)
    bbox_issues = 0
    area_issues = 0
    
    for ann in coco['annotations']:
        # Check required fields
        if not all(k in ann for k in ['id', 'image_id', 'category_id', 'area']):
            invalid_annotations += 1
            continue
        
        # Check keypoints
        if 'keypoints' in ann:
            num_kpts = len(ann['keypoints']) // 3  # Each keypoint is (x, y, visibility)
            keypoint_distribution[num_kpts] += 1
            if num_kpts != 40:
                keypoint_issues += 1
        
        # Check bbox
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                bbox_issues += 1
        
        # Check area
        if ann['area'] < 30:
            area_issues += 1
    
    if invalid_annotations == 0:
        print(f"  ✓ All {num_annotations} annotations have required fields")
    else:
        print(f"  ❌ {invalid_annotations} invalid annotations")
    
    print(f"\n  Keypoint Distribution:")
    for num_kpts in sorted(keypoint_distribution.keys()):
        count = keypoint_distribution[num_kpts]
        if num_kpts == 40:
            print(f"    ✓ {num_kpts} keypoints: {count} annotations")
        else:
            print(f"    ❌ {num_kpts} keypoints: {count} annotations")
    
    if keypoint_issues > 0:
        print(f"  ❌ {keypoint_issues} annotations with incorrect keypoint count")
    
    if bbox_issues > 0:
        print(f"  ⚠️  {bbox_issues} annotations with invalid bboxes")
    
    if area_issues > 0:
        print(f"  ⚠️  {area_issues} annotations with very small area")
    
    # Check area distribution
    if num_annotations > 0:
        areas = [ann['area'] for ann in coco['annotations']]
        print(f"\n  Area Statistics:")
        print(f"    - Min: {min(areas):.0f}")
        print(f"    - Mean: {np.mean(areas):.0f}")
        print(f"    - Max: {max(areas):.0f}")
    
    return True


def diagnose_images(image_dir):
    """Check if images exist and are readable."""
    print(f"\n{'='*60}")
    print(f"Checking images in: {image_dir}")
    print('='*60)
    
    if not image_dir.exists():
        print(f"❌ Directory does not exist: {image_dir}")
        return False
    
    image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return False
    
    print(f"✓ Found {len(image_files)} images")
    
    # Check a few images
    readable = 0
    dimensions = []
    
    for img_file in image_files[:min(10, len(image_files))]:
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  ❌ Cannot read: {img_file.name}")
        else:
            readable += 1
            dimensions.append(img.shape)
    
    if readable == min(10, len(image_files)):
        print(f"✓ All checked images are readable")
        if dimensions:
            h, w = dimensions[0]
            print(f"  - Dimensions: {w}×{h}")
    else:
        print(f"❌ {min(10, len(image_files)) - readable} images cannot be read")
    
    return True


def visualize_sample(json_path, image_dir, num_samples=3):
    """Visualize sample annotations."""
    print(f"\n{'='*60}")
    print(f"Visualizing samples from {json_path.name}")
    print('='*60)
    
    with open(json_path) as f:
        coco = json.load(f)
    
    # Group annotations by image
    anns_by_image = defaultdict(list)
    for ann in coco['annotations']:
        anns_by_image[ann['image_id']].append(ann)
    
    images_to_show = list(coco['images'][:num_samples])
    
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    
    for ax, img_info in zip(axes, images_to_show):
        img_path = image_dir / img_info['file_name']
        if not img_path.exists():
            ax.text(0.5, 0.5, f"Image not found", ha='center', va='center')
            ax.set_title(img_info['file_name'])
            continue
        
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            ax.text(0.5, 0.5, f"Cannot read", ha='center', va='center')
            ax.set_title(img_info['file_name'])
            continue
        
        ax.imshow(image, cmap='gray')
        
        # Draw annotations
        annotations = anns_by_image[img_info['id']]
        for ann in annotations:
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
                ax.add_patch(rect)
            
            if 'keypoints' in ann:
                keypoints = ann['keypoints']
                for i in range(0, len(keypoints), 3):
                    x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
                    if v > 0:
                        ax.plot(x, y, 'go', markersize=3)
        
        title = f"{img_info['file_name']}\n({len(annotations)} fibers)"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    output_file = Path("diagnostic_visualization.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_file}")
    plt.close()


def main():
    dataset_dir = Path("synthetic_dataset")
    annotation_dir = dataset_dir / "annotations"
    image_dir = dataset_dir / "images"
    
    print("\n" + "="*60)
    print("SYNTHETIC DATASET DIAGNOSTIC")
    print("="*60)
    
    # Check annotation files
    for json_file in ['synthetic_fiber_train.json', 'synthetic_fiber_val.json', 'synthetic_fiber_all.json']:
        json_path = annotation_dir / json_file
        diagnose_json(json_path)
    
    # Check images
    diagnose_images(image_dir)
    
    # Visualize samples
    diagnose_json(annotation_dir / "synthetic_fiber_all.json")
    visualize_sample(annotation_dir / "synthetic_fiber_all.json", image_dir, num_samples=3)
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
