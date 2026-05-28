import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling.roi_heads import StandardROIHeads
from skimage.morphology import skeletonize


class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FiberROIHeadsV2(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM if hasattr(cfg.MODEL.ROI_BOX_HEAD, "FC_DIM") else 1024
        self.length_head = MLPHead(dim, 1)
        self.width_head = MLPHead(dim, 1)
        self.orientation_head = MLPHead(dim, 1)
        self.curvature_head = MLPHead(dim, 1)
        self.bead_head = MLPHead(dim, 1)
        self.porosity_head = MLPHead(dim, 1)

    def _forward_box(self, features, proposals):
        features_list = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)

            pred_length = self.length_head(box_features).squeeze(-1)
            pred_width = self.width_head(box_features).squeeze(-1)
            pred_orient = self.orientation_head(box_features).squeeze(-1)
            pred_curv = self.curvature_head(box_features).squeeze(-1)
            pred_bead_logits = self.bead_head(box_features).squeeze(-1)
            pred_porosity = self.porosity_head(box_features).squeeze(-1)

            gt_length = torch.cat([p.gt_length for p in proposals], dim=0)
            gt_width = torch.cat([p.gt_width for p in proposals], dim=0)
            gt_orient = torch.cat([p.gt_orientation for p in proposals], dim=0)
            gt_curv = torch.cat([p.gt_curvature for p in proposals], dim=0)
            gt_bead = torch.cat([p.gt_bead for p in proposals], dim=0)
            gt_porosity = torch.cat([p.gt_porosity for p in proposals], dim=0)

            loss_weight = 0.1
            losses.update({
                "loss_fiber_length": F.mse_loss(pred_length, gt_length) * loss_weight,
                "loss_fiber_width": F.mse_loss(pred_width, gt_width) * loss_weight,
                "loss_fiber_orient": F.mse_loss(pred_orient, gt_orient) * loss_weight,
                "loss_fiber_curv": F.l1_loss(pred_curv, gt_curv) * loss_weight,
                "loss_fiber_bead": F.binary_cross_entropy_with_logits(
                    pred_bead_logits, gt_bead
                ) * loss_weight,
                "loss_fiber_porosity": F.mse_loss(
                    torch.sigmoid(pred_porosity), gt_porosity
                ) * loss_weight,
            })
            return losses

        pred_instances, sampled_indices = self.box_predictor.inference(predictions, proposals)

        # box_features est concatene sur toutes les propositions; on le re-split
        # par image pour appliquer les indices gardes apres NMS.
        num_props_per_image = [len(p) for p in proposals]
        box_features_per_image = box_features.split(num_props_per_image, dim=0)

        for inst, keep_inds, image_features in zip(pred_instances, sampled_indices, box_features_per_image):
            if len(inst) == 0:
                empty = torch.empty((0,), dtype=torch.float32, device=image_features.device)
                inst.pred_length = empty
                inst.pred_width = empty
                inst.pred_orientation = empty
                inst.pred_curvature = empty
                inst.pred_bead_logits = empty
                inst.pred_bead = empty
                inst.pred_porosity = empty
                continue

            selected_features = image_features[keep_inds]
            inst.pred_length = self.length_head(selected_features).squeeze(-1)
            inst.pred_width = self.width_head(selected_features).squeeze(-1)
            inst.pred_orientation = self.orientation_head(selected_features).squeeze(-1)
            inst.pred_curvature = self.curvature_head(selected_features).squeeze(-1)
            inst.pred_bead_logits = self.bead_head(selected_features).squeeze(-1)
            inst.pred_bead = torch.sigmoid(inst.pred_bead_logits)
            inst.pred_porosity = torch.sigmoid(self.porosity_head(selected_features).squeeze(-1))

        return pred_instances


def compute_orientation(skeleton):
    pts = np.column_stack(np.where(skeleton))
    p1, p2 = pts[0], pts[-1]
    vec = p2 - p1
    norm = np.linalg.norm(vec) + 1e-6
    return [float(vec[0] / norm), float(vec[1] / norm)]


def compute_curvature(skeleton):
    pts = np.column_stack(np.where(skeleton))
    if len(pts) < 3:
        return 0.0

    angles = []
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angles.append(np.arccos(np.clip(cos_angle, -1, 1)))

    return float(np.mean(angles))


def width_histogram(widths, bins=10):
    hist, _ = np.histogram(widths, bins=bins, range=(0, np.max(widths)))
    return hist / (np.sum(hist) + 1e-6)
