import os
import json
import cv2
import numpy as np
import torch

# SAM
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.structures import BoxMode
# Morphology
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import pdist, squareform
import networkx as nx

from fiber_heads import compute_curvature, compute_orientation, width_histogram
# ==========================
# CONFIG
# ==========================
IMAGE_DIR = "images"
LABELME_DIR = "labelme_annotations"
COCO_JSON = "coco_fiber.json"
SAM_CHECKPOINT = "sam_checkpoint/sam_vit_h_4b8939.pth"

os.makedirs(LABELME_DIR, exist_ok=True)

# ==========================
#     WIDTH / LENGTH       =
# ==========================
def polygon_to_mask(img_shape, polygon):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def compute_length_graph(skeleton):
    pts = np.column_stack(np.where(skeleton))
    if len(pts) < 2:
        return 0.0

    dist = squareform(pdist(pts))
    G = nx.Graph()

    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if dist[i,j] <= np.sqrt(2):
                G.add_edge(i, j, weight=dist[i,j])

    lengths = dict(nx.all_pairs_dijkstra_path_length(G))
    return max(max(v.values()) for v in lengths.values() if v)

def compute_width_length(img_shape, polygon):
    mask = polygon_to_mask(img_shape, polygon)
    if mask.sum() == 0:
        return 0, 0

    skeleton = skeletonize(mask > 0)
    dist_map = distance_transform_edt(mask)

    pts = np.column_stack(np.where(skeleton))
    widths = [dist_map[y, x]*2 for y,x in pts]

    width = float(np.mean(widths)) if widths else 0
    length = compute_length_graph(skeleton)

    return width, length

# ==========================
# STEP 1 — SAM AUTO LABEL
# ==========================
def run_sam():
    print("Running SAM auto-label...")

    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    print("SAM loaded")
    print("SAM device:", "cuda" if torch.cuda.is_available() else "cpu")
    sam.to("cuda" if torch.cuda.is_available() else "cpu")

    mask_generator = SamAutomaticMaskGenerator(sam)

    for img_name in os.listdir(IMAGE_DIR):
        if not img_name.endswith((".png",".jpg", ".tif")):
            continue

        path = os.path.join(IMAGE_DIR, img_name)
        img = cv2.imread(path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = mask_generator.generate(rgb)
        print(f"{img_name} → {len(masks)} masks generated")

        shapes = []

        for m in masks:
            mask = m["segmentation"]

            if mask.sum() < 200:
                continue

            contours,_ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if len(cnt) < 5:
                    continue

                poly = cnt.reshape(-1,2)

                # filter elongated shapes
                x,y,w,h = cv2.boundingRect(poly)
                if max(w,h)/(min(w,h)+1e-5) < 3:
                    continue

                shapes.append({
                    "label":"fiber",
                    "points":poly.tolist(),
                    "shape_type":"polygon"
                })

        data = {
            "shapes": shapes,
            "imagePath": img_name,
            "imageHeight": img.shape[0],
            "imageWidth": img.shape[1]
        }

        out = os.path.join(LABELME_DIR, img_name.replace(".png",".json"))
        with open(out,"w") as f:
            json.dump(data,f)

        print(f"{img_name} → {len(shapes)} fibers")

# ==========================
# STEP 2 — COCO CONVERSION
# ==========================
def convert_to_coco():
    print("Converting to COCO...")

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id":1,
            "name":"fiber",
            "keypoints":["start","mid","end"],
            "skeleton":[[1,2],[2,3]]
        }]
    }

    ann_id = 1
    img_id = 1

    for file in os.listdir(LABELME_DIR):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(LABELME_DIR,file)) as f:
            data = json.load(f)

        img_path = os.path.join(IMAGE_DIR, data["imagePath"])
        img = cv2.imread(img_path)
        h,w = img.shape[:2]

        coco["images"].append({
            "id":img_id,
            "file_name":data["imagePath"],
            "height":h,
            "width":w
        })

        for shape in data["shapes"]:
            poly = shape["points"]

            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]

            bbox = [min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys)]

            mask = polygon_to_mask(img.shape, poly)
            skeleton = skeletonize(mask > 0)
            dist_map = distance_transform_edt(mask)

            # ======================
            # WIDTH + LENGTH
            # ======================
            width, length = compute_width_length(img.shape, poly)

            # ======================
            # WIDTH DISTRIBUTION
            # ======================
            sk_pts = np.column_stack(np.where(skeleton))
            widths = [dist_map[y, x]*2 for y,x in sk_pts]

            width_hist = width_histogram(widths, bins=10)

            # ======================
            # ORIENTATION
            # ======================
            orientation = compute_orientation(skeleton)

            # ======================
            # CURVATURE
            # ======================
            curvature = compute_curvature(skeleton)

            keypoints = [
                poly[0][0], poly[0][1], 2,
                poly[len(poly)//2][0], poly[len(poly)//2][1], 2,
                poly[-1][0], poly[-1][1], 2
            ]
            print(f"W: {width_hist}")
            print(f"Fiber {ann_id}: length={length:.1f}, width={width:.1f}, orientation={orientation}, curvature={curvature}")

            coco["annotations"].append({
                "id":ann_id,
                "image_id":img_id,
                "category_id":1,
                "bbox":bbox,
                "segmentation":[np.array(poly).flatten().tolist()],
                "area":float(width*length),
                "iscrowd":0,
                "keypoints":keypoints,
                "num_keypoints":3,
                "fiber_length": length,
                "fiber_width_mean": width,
                "fiber_width_hist": width_hist.tolist(),
                "fiber_orientation": orientation,
                "fiber_curvature": curvature
            })

            ann_id += 1

        img_id += 1

    with open(COCO_JSON,"w") as f:
        json.dump(coco,f,indent=4)

    print("COCO ready")

# ==========================
# STEP 3 — TRAIN DETECTRON2
# ==========================

from detectron2.data.datasets import load_coco_json

# On peut aussi utiliser load_coco_json directement dans le DatasetCatalog, mais je préfère cette approche pour injecter les métriques de fibres dans les annotations
def load_coco_with_fibers(json_file, image_root):
    """
    Charge le dataset COCO et injecte les métriques de fibres 
    ainsi que les 40 keypoints pour chaque instance.
    """
    # 1. Chargement brut du fichier JSON
    print(f"Loading COCO annotations from {json_file}...")
    print(f"JSON file size: {os.path.getsize(json_file) / 1e6:.2f} MB")
    with open(json_file, 'r') as f:
        coco_raw = json.load(f)

    # 2. Création d'un dictionnaire d'annotations indexé par image_id
    print("Indexing annotations by image_id...")
    print(f"Total annotations: {len(coco_raw['annotations'])}")
    ann_map = {}
    for ann in coco_raw["annotations"]:
        image_id = ann["image_id"]
        if image_id not in ann_map:
            ann_map[image_id] = {}
        ann_map[image_id][ann["id"]] = ann

    # 3. Préparation de la structure de données pour le modèle
    dataset_dicts = []
    
    for img_info in coco_raw["images"]:
        record = {}
        
        # Infos de base de l'image
        image_id = img_info["id"]
        record["file_name"] = os.path.join(image_root, img_info["file_name"])
        record["image_id"] = image_id
        record["height"] = img_info["height"]
        record["width"] = img_info["width"]

        objs = []
        # Récupérer toutes les annotations liées à cette image_id
        current_anns = ann_map.get(image_id, {})

        for ann_id, raw_ann in current_anns.items():
            obj = {
                "bbox": raw_ann["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": raw_ann["category_id"] - 1, # Conversion 0-indexed
                "iscrowd": 0,
            }
            
            if "segmentation" in raw_ann:
                obj["segmentation"] = raw_ann["segmentation"]

            # --- INJECTION DES KEYPOINTS (Les 40 points) ---
            if "keypoints" in raw_ann:
                obj["keypoints"] = raw_ann["keypoints"]
                obj["num_keypoints"] = raw_ann.get("num_keypoints", 40)

        
            # obj["fiber_length"] = float(raw_ann.get("fiber_length", 0.0))
            # obj["fiber_width"] = float(raw_ann.get("fiber_width", 0.0)) # Largeur moyenne
            obj["fiber_curvature"] = float(raw_ann.get("fiber_curvature", 0.0))
            obj["fiber_length"] = float(raw_ann.get("fiber_length", 0.0)) / 2000.0 
            obj["fiber_width"] = float(raw_ann.get("fiber_width", 0.0)) / 100.0
            obj["fiber_orientation"] = float(raw_ann.get("fiber_orientation", 0.0)) / 180.0
            obj["has_bead"] = float(raw_ann.get("has_bead", 0.0))
            obj["porosity"] = float(raw_ann.get("porosity", img_info.get("porosity", 0.0)))
            # Pour l'orientation (souvent un vecteur [cos, sin])
            # obj["fiber_orientation"] = raw_ann.get("fiber_orientation", [0.0, 0.0])
            objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)

    print(f"Loaded {len(dataset_dicts)} images with annotations.")
    return dataset_dicts


from fiber_mapper import FiberDatasetMapper
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader


class FiberTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=FiberDatasetMapper(cfg, True)
        )

from fiber_heads import FiberROIHeadsV2
from detectron2.modeling import ROI_HEADS_REGISTRY


def train():
    print("🚀 Démarrage de l'entraînement Detectron2 (FibeR-CNN)...")

    # 1. Nettoyage et Enregistrement du Dataset
    if "fiber_train_v2" in DatasetCatalog.list():
        DatasetCatalog.remove("fiber_train_v2")

    DatasetCatalog.register(
        "fiber_train_v2",
        lambda: load_coco_with_fibers(COCO_JSON, IMAGE_DIR)
    )

    # --- CONFIGURATION DES MÉTADONNÉES (40 Points) ---
    num_kpts = 40
    metadata = MetadataCatalog.get("fiber_train_v2")
    metadata.set(thing_classes=["fiber"])
    
    # Génération automatique des noms pour 40 points
    metadata.set(
        keypoint_names=[f"p{i+1}" for i in range(num_kpts)],
        keypoint_flip_map=[] # On ne flip pas pour garder l'ordre Haut -> Bas
    )

    # 2. Configuration du Modèle
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    )

    # --- INJECTION DES PARAMÈTRES PERSONNALISÉS ---
    # On utilise les poids que tu as définis pour équilibrer les pertes bio
    from detectron2.config import CfgNode as CN
    cfg.MODEL.FIBER = CN()

    # --- ALIGNEMENT DES HEADS ---
    # On enregistre la tête si ce n'est pas déjà fait
    if "FiberROIHeadsV2" not in ROI_HEADS_REGISTRY:
        ROI_HEADS_REGISTRY.register(FiberROIHeadsV2)
    cfg.MODEL.ROI_HEADS.NAME = "FiberROIHeadsV2"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 10.0
    
    # CRUCIAL : Switch to 40 points here to match the dataset
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = num_kpts 

    # 3. Paramètres du Dataset & Solver
    cfg.DATASETS.TRAIN = ("fiber_train_v2",)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    )
    
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 1e-5 
    cfg.SOLVER.MAX_ITER = 20000 # We can go up to 10k-20k for a complex membrane
    cfg.SOLVER.STEPS = []      # Do not reduce the LR too early

    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 4. Lancement
    trainer = FiberTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

# ==========================
# RUN ALL
# ==========================
if __name__ == "__main__":
    # convert_to_coco()
    train()
