import torch
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo


cfg = get_cfg()
cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
)

cfg.MODEL.WEIGHTS = "output/model_final.pth"  # Chemin vers le modèle entraîné
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Se
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 40
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 5.0
cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14

img_path = "test_images/test.jpg"  # Chemin vers l'image de test
predictor = DefaultPredictor(cfg)
im = cv2.imread(img_path)
outputs = predictor(im)
instances = outputs["instances"].to("cpu")
print(f"✅ Nombre de fibres détectées : {len(instances)}")

if len(instances) > 0:
    # Utilise "fiber_debug" qu'on vient de créer ci-dessus
    v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get("fiber_debug"), scale=1.2)
    
    # On dessine
    out = v.draw_instance_predictions(instances)
    
    # Affichage
    result_img = out.get_image()[:, :, ::-1]
    cv2.imshow("Detection de Fibres - CBNU", result_img)
    
    # Debug des valeurs BIO (normalisées ou non selon ton entraînement)
    if instances.has("pred_fiber_length"): # Si tu as ajouté tes têtes custom
         lengths = instances.pred_fiber_length.numpy()
         print("Longueurs prédites (normalisées) :", lengths)
    cv2.waitKey(0)
else:
    print("❌ Aucune fibre détectée. Baisse cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST")