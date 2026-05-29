from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper
from detectron2.data.transforms import AugInput
import torch
import numpy as np

class FiberDatasetMapper(DatasetMapper):

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        
        # 1. Lecture et Augmentation de l'image
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        aug_input = AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        annos = []
        # On récupère les annotations brutes avant qu'elles ne soient transformées
        raw_annotations = dataset_dict.pop("annotations")

        for obj in raw_annotations:
            # Calcul dynamique pour 40 points (ou tout autre nombre)
            keypoints_raw = obj.get("keypoints", [])
            num_keypoints = len(keypoints_raw) // 3
            
            # Règle de Flip Horizontal : 
            # Pour les fibres, on ne veut généralement pas inverser l'ordre des points 
            # (p1 reste p1), donc on garde un mapping identité.
            hflip_indices = list(range(num_keypoints))

            # Transformation standard (Box, Mask, Keypoints)
            anno = utils.transform_instance_annotations(
                obj, 
                transforms, 
                image.shape[:2],
                keypoint_hflip_indices=hflip_indices
            )
            
            # Injection des métriques scalaires
            # On normalise la longueur si nécessaire pour la perte (ex: / 300.0)
            anno["fiber_length"] = obj.get("fiber_length", 0.0)
            anno["fiber_width"] = obj.get("fiber_width", 0.0)
            anno["fiber_curvature"] = obj.get("fiber_curvature", 0.0)
            anno["fiber_orientation"] = obj.get("fiber_orientation", 0.0) # Angle scalaire
            anno["has_bead"] = float(obj.get("has_bead", 0.0))
            anno["porosity"] = float(obj.get("porosity", 0.0))
            
            annos.append(anno)

        # 2. Conversion en objet Instances (Tensor)
        # filtrage automatique des objets hors-cadre après transformation
        instances = utils.annotations_to_instances(
            annos, image.shape[:2], mask_format="polygon"
        )
        instances, kept = utils.filter_empty_instances(instances, return_mask=True)
        kept_annos = [anno for anno, keep in zip(annos, kept) if keep]

        # 3. Synchronisation des métriques personnalisées
        gt_length, gt_width, gt_orient, gt_curv, gt_bead, gt_porosity = [], [], [], [], [], []
        
        for a in kept_annos:
            # On vérifie si l'instance a survécu aux transformations (pas filtrée)
            gt_length.append(a.get("fiber_length", 0.0))
            gt_width.append(a.get("fiber_width", 0.0))
            gt_orient.append(a.get("fiber_orientation", 0.0))
            gt_curv.append(a.get("fiber_curvature", 0.0))
            gt_bead.append(a.get("has_bead", 0.0))
            gt_porosity.append(a.get("porosity", 0.0))

        # Attachement des Tensors aux instances pour la Head de FibeR-CNN
        instances.gt_length = torch.tensor(gt_length, dtype=torch.float32)
        instances.gt_width = torch.tensor(gt_width, dtype=torch.float32)
        instances.gt_orientation = torch.tensor(gt_orient, dtype=torch.float32)
        instances.gt_curvature = torch.tensor(gt_curv, dtype=torch.float32)
        instances.gt_bead = torch.tensor(gt_bead, dtype=torch.float32)
        instances.gt_porosity = torch.tensor(gt_porosity, dtype=torch.float32)

        # Format final pour le DataLoader
        dataset_dict["instances"] = instances
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        return dataset_dict
