from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fiber_segmentation_core import (
    FiberCandidate,
    PipelineConfig,
    SamFiberSegmenter,
    assess_candidate_quality,
    build_candidate_from_mask,
    crop_percentage,
    export_coco,
    mask_to_polygons,
    spline_mask_from_keypoints,
)


@dataclass
class ProposalMask:
    proposal_id: int
    mask: np.ndarray
    source_variant: str
    score: float
    quality_ok: bool
    quality_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    image_path: Path | None = None
    working_image: np.ndarray | None = None
    proposals: dict[int, ProposalMask] = field(default_factory=dict)
    proposal_labels: np.ndarray | None = None
    instance_labels: np.ndarray | None = None
    selected_proposal_id: int | None = None
    selected_instance_id: int | None = None
    interaction_mode: str = "select"
    next_proposal_id: int = 1
    next_instance_id: int = 1


def run_app(image_path: str | None = None) -> None:
    try:
        import napari
        from skimage.draw import polygon2mask
        from skimage.measure import label
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QFileDialog,
            QFormLayout,
            QFrame,
            QHBoxLayout,
            QLabel,
            QListWidget,
            QListWidgetItem,
            QPushButton,
            QSlider,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise SystemExit(
            "napari, qtpy and their Qt bindings are required. "
            "Install dependencies from requirements-napari.txt."
        ) from exc

    config = PipelineConfig()
    state = SessionState()
    segmenter = SamFiberSegmenter(config)

    viewer = napari.Viewer(title="Fiber Segmentation Studio", ndisplay=2)
    viewer.theme = "dark"

    image_layer = viewer.add_image(np.zeros((32, 32), dtype=np.uint8), name="Original Image")
    proposal_layer = viewer.add_labels(np.zeros((32, 32), dtype=np.int32), name="SAM Proposals")
    instance_layer = viewer.add_labels(np.zeros((32, 32), dtype=np.int32), name="Instance Labels")
    keypoints_layer = viewer.add_points(np.empty((0, 2)), name="Keypoints", size=7, face_color="yellow")
    prompt_points_layer = viewer.add_points(
        np.empty((0, 2)),
        name="Prompt Points",
        size=9,
        face_color="lime",
        properties={"label": np.array([], dtype=int)},
        text={"string": "{label}", "color": "white", "size": 8},
    )
    roi_layer = viewer.add_shapes(
        name="ROI Geometry",
        edge_color="red",
        edge_width=2.0,
        face_color=[1, 0, 0, 0.12],
    )

    proposal_layer.opacity = 0.35
    instance_layer.opacity = 0.55
    instance_layer.editable = True

    if hasattr(keypoints_layer, "border_color"):
        keypoints_layer.border_color = "black"
    if hasattr(keypoints_layer, "border_width"):
        keypoints_layer.border_width = 0.3
    if hasattr(prompt_points_layer, "border_color"):
        prompt_points_layer.border_color = "black"
    if hasattr(prompt_points_layer, "border_width"):
        prompt_points_layer.border_width = 0.5

    def set_status(text: str) -> None:
        viewer.status = text
        status_label.setText(text)

    def blank_state(shape: tuple[int, int]) -> None:
        state.proposals.clear()
        state.proposal_labels = np.zeros(shape, dtype=np.int32)
        state.instance_labels = np.zeros(shape, dtype=np.int32)
        state.selected_proposal_id = None
        state.selected_instance_id = None
        state.next_proposal_id = 1
        state.next_instance_id = 1
        state.interaction_mode = "select"

        image_layer.data = np.zeros(shape, dtype=np.uint8)
        proposal_layer.data = state.proposal_labels.copy()
        instance_layer.data = state.instance_labels.copy()
        keypoints_layer.data = np.empty((0, 2))
        prompt_points_layer.data = np.empty((0, 2))
        prompt_points_layer.properties = {"label": np.array([], dtype=int)}
        prompt_points_layer.face_color = []
        roi_layer.data = []

    def clear_prompts() -> None:
        prompt_points_layer.data = np.empty((0, 2))
        prompt_points_layer.properties = {"label": np.array([], dtype=int)}
        prompt_points_layer.face_color = []
        set_status("Prompt points cleared.")

    def clear_roi() -> None:
        roi_layer.data = []
        roi_layer.mode = "pan_zoom"
        set_status("ROI geometry cleared.")

    def load_image(path: str | Path) -> None:
        path = Path(path)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            set_status(f"Unable to read image: {path}")
            return
        cropped = crop_percentage(image, config.crop_bottom_percent)
        state.image_path = path
        state.working_image = cropped
        blank_state(cropped.shape)
        image_layer.data = cropped
        image_layer.contrast_limits = (float(np.min(cropped)), float(np.max(cropped)))
        viewer.reset_view()
        refresh_layers()
        set_status(f"Loaded {path.name}.")

    def update_list() -> None:
        proposal_list.clear()
        for proposal_id in sorted(state.proposals):
            proposal = state.proposals[proposal_id]
            prefix = "*" if proposal_id == state.selected_proposal_id else " "
            tag = "ok" if proposal.quality_ok else proposal.quality_reason
            text = f"{prefix} mask {proposal_id} | {tag}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, proposal_id)
            proposal_list.addItem(item)

    def refresh_layers() -> None:
        if state.proposal_labels is None or state.instance_labels is None:
            return
        proposal_layer.data = state.proposal_labels.copy()
        instance_layer.data = state.instance_labels.copy()
        update_list()
        update_selected_keypoints()

    def labels_to_color_map(max_label: int, selected_label: int | None, selected_color: str, default_color: str) -> dict[int | None, str]:
        color_map: dict[int | None, str] = {None: "transparent", 0: "transparent"}
        for label_id in range(1, max_label + 1):
            color_map[label_id] = selected_color if label_id == selected_label else default_color
        return color_map

    def update_color_maps() -> None:
        try:
            proposal_layer.color = labels_to_color_map(
                int(state.proposal_labels.max()) if state.proposal_labels is not None else 0,
                state.selected_proposal_id,
                "white",
                "orange",
            )
        except Exception:
            pass
        try:
            instance_layer.color = labels_to_color_map(
                int(state.instance_labels.max()) if state.instance_labels is not None else 0,
                state.selected_instance_id,
                "cyan",
                "lime",
            )
        except Exception:
            pass

    def add_proposal(mask: np.ndarray, source_variant: str, score: float, metadata: dict[str, Any] | None = None) -> int | None:
        if state.working_image is None:
            return None
        mask = np.asarray(mask, dtype=np.uint8)
        if not np.any(mask):
            return None
        proposal_id = state.next_proposal_id
        state.next_proposal_id += 1
        candidate, reason = build_candidate_from_mask(
            mask,
            state.working_image,
            config,
            fiber_id=proposal_id,
            source=state.image_path.name if state.image_path else "unknown",
            mask_index=proposal_id,
            preprocess_variant=source_variant,
            extra_metadata=metadata or {},
            validate=False,
        )
        quality_ok = False
        quality_reason = "unresolved"
        if candidate is not None:
            quality_ok, quality_reason = assess_candidate_quality(candidate, config)
        proposal = ProposalMask(
            proposal_id=proposal_id,
            mask=mask.copy(),
            source_variant=source_variant,
            score=float(score),
            quality_ok=quality_ok,
            quality_reason=quality_reason,
            metadata=metadata or {},
        )
        state.proposals[proposal_id] = proposal
        if state.proposal_labels is not None:
            state.proposal_labels[mask > 0] = proposal_id
        state.selected_proposal_id = proposal_id
        refresh_layers()
        update_color_maps()
        return proposal_id

    def rebuild_proposal_labels() -> None:
        if state.working_image is None:
            return
        labels = np.zeros(state.working_image.shape, dtype=np.int32)
        for proposal_id in sorted(state.proposals):
            labels[state.proposals[proposal_id].mask > 0] = proposal_id
        state.proposal_labels = labels
        refresh_layers()
        update_color_maps()

    def prompt_points_payload() -> tuple[np.ndarray | None, np.ndarray | None]:
        points = np.asarray(prompt_points_layer.data, dtype=np.float32)
        labels = np.asarray(prompt_points_layer.properties.get("label", []), dtype=np.int32)
        if len(points) == 0 or len(labels) == 0:
            return None, None
        point_coords = np.stack([points[:, 1], points[:, 0]], axis=1)
        return point_coords, labels

    def prompt_box_payload() -> np.ndarray | None:
        if len(roi_layer.data) == 0:
            return None
        shape = np.asarray(roi_layer.data[-1], dtype=np.float32)
        ys = shape[:, 0]
        xs = shape[:, 1]
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

    def roi_mask() -> np.ndarray | None:
        if state.working_image is None or len(roi_layer.data) == 0:
            return None
        shape = np.asarray(roi_layer.data[-1], dtype=np.float32)
        return polygon2mask(state.working_image.shape, shape[:, :2]).astype(bool)

    def instance_mask(instance_id: int) -> np.ndarray:
        return (state.instance_labels == instance_id).astype(np.uint8) * 255

    def best_effort_candidate_for_instance(instance_id: int) -> FiberCandidate | None:
        if state.working_image is None:
            return None
        mask = instance_mask(instance_id)
        if not np.any(mask):
            return None
        candidate, _ = build_candidate_from_mask(
            mask,
            state.working_image,
            config,
            fiber_id=instance_id,
            source=state.image_path.name if state.image_path else "unknown",
            mask_index=instance_id,
            preprocess_variant="instance_labels",
            validate=False,
        )
        return candidate

    def update_selected_keypoints() -> None:
        if state.selected_instance_id is None:
            keypoints_layer.data = np.empty((0, 2))
            return
        candidate = best_effort_candidate_for_instance(state.selected_instance_id)
        if candidate is None:
            keypoints_layer.data = np.empty((0, 2))
            return
        keypoints_layer.data = np.asarray(candidate.keypoints_rc, dtype=np.float32)
        keypoints_layer.properties = {
            "index": np.arange(1, len(candidate.keypoints_rc) + 1, dtype=int)
        }
        keypoints_layer.text = {"string": "{index}", "color": "yellow", "size": 9}

    def select_proposal(proposal_id: int | None) -> None:
        state.selected_proposal_id = proposal_id
        if proposal_id in state.proposals:
            proposal = state.proposals[proposal_id]
            set_status(
                f"Selected proposal {proposal_id} | score={proposal.score:.3f} | quality={proposal.quality_reason}"
            )
        update_list()
        update_color_maps()

    def select_instance(instance_id: int | None) -> None:
        state.selected_instance_id = instance_id
        update_selected_keypoints()
        if instance_id is not None:
            candidate = best_effort_candidate_for_instance(instance_id)
            if candidate is not None:
                set_status(
                    f"Selected instance {instance_id} | len={candidate.fiber_length:.1f}px "
                    f"| width={candidate.fiber_width:.1f}px | spline IoU={candidate.spline_mask_iou:.2f}"
                )
        update_color_maps()

    def run_auto_sam() -> None:
        if state.working_image is None:
            set_status("Load an image first.")
            return
        set_status("Running Auto SAM...")
        masks, _, _ = segmenter.automatic_masks(state.working_image)
        state.proposals.clear()
        state.next_proposal_id = 1
        state.proposal_labels = np.zeros(state.working_image.shape, dtype=np.int32)
        for mask_data in masks:
            mask = mask_data["segmentation"].astype(np.uint8) * 255
            add_proposal(
                mask,
                source_variant=mask_data.get("preprocess_variant", "raw"),
                score=float(mask_data.get("predicted_iou", 0.0)),
                metadata={
                    "predicted_iou": float(mask_data.get("predicted_iou", 0.0)),
                    "stability_score": float(mask_data.get("stability_score", 0.0)),
                },
            )
        rebuild_proposal_labels()
        set_status(f"SAM proposals ready: {len(state.proposals)} masks. Nothing was auto-rejected.")

    def create_instance_labels_from_proposals() -> None:
        if state.working_image is None or state.proposal_labels is None:
            set_status("Run Auto SAM first.")
            return
        instance_labels = np.zeros(state.working_image.shape, dtype=np.int32)
        next_id = 1
        for proposal_id in sorted(state.proposals):
            mask = state.proposals[proposal_id].mask > 0
            if not np.any(mask):
                continue
            instance_labels[mask] = next_id
            next_id += 1
        state.instance_labels = instance_labels
        state.next_instance_id = next_id
        refresh_layers()
        update_color_maps()
        set_status(f"Created one merged Labels layer with {next_id - 1} instances.")

    def add_selected_proposal_to_instances() -> None:
        if state.selected_proposal_id is None or state.selected_proposal_id not in state.proposals:
            set_status("Select a proposal first.")
            return
        if state.instance_labels is None:
            return
        proposal = state.proposals[state.selected_proposal_id]
        state.instance_labels[proposal.mask > 0] = state.next_instance_id
        state.selected_instance_id = state.next_instance_id
        state.next_instance_id += 1
        refresh_layers()
        update_color_maps()
        set_status(f"Proposal {proposal.proposal_id} added as instance {state.selected_instance_id}.")

    def delete_selected_proposal() -> None:
        proposal_id = state.selected_proposal_id
        if proposal_id is None or proposal_id not in state.proposals:
            set_status("Select a proposal first.")
            return
        del state.proposals[proposal_id]
        state.selected_proposal_id = None
        rebuild_proposal_labels()
        set_status(f"Proposal {proposal_id} deleted.")

    def delete_selected_instance() -> None:
        instance_id = state.selected_instance_id
        if instance_id is None or state.instance_labels is None:
            set_status("Select an instance first.")
            return
        state.instance_labels[state.instance_labels == instance_id] = 0
        state.selected_instance_id = None
        refresh_layers()
        update_color_maps()
        set_status(f"Instance {instance_id} deleted.")

    def delete_proposals_in_roi() -> None:
        roi = roi_mask()
        if roi is None:
            set_status("Draw an ROI first.")
            return
        to_delete = [
            proposal_id
            for proposal_id, proposal in state.proposals.items()
            if np.any((proposal.mask > 0) & roi)
        ]
        for proposal_id in to_delete:
            del state.proposals[proposal_id]
        rebuild_proposal_labels()
        clear_roi()
        set_status(f"Deleted {len(to_delete)} proposals overlapping the ROI.")

    def erase_instance_pixels_in_roi() -> None:
        roi = roi_mask()
        if roi is None or state.instance_labels is None:
            set_status("Draw an ROI first.")
            return
        state.instance_labels[roi] = 0
        refresh_layers()
        update_color_maps()
        clear_roi()
        set_status("Erased instance pixels inside the ROI.")

    def set_prompt_mode(mode: str) -> None:
        state.interaction_mode = mode
        roi_layer.mode = "pan_zoom"
        if mode == "positive":
            set_status("Click on the image to add positive points.")
        elif mode == "negative":
            set_status("Click on the image to add negative points.")
        elif mode == "roi_rect":
            roi_layer.mode = "add_rectangle"
            set_status("Draw a rectangle ROI.")
        elif mode == "roi_poly":
            roi_layer.mode = "add_polygon"
            set_status("Draw a polygon ROI.")
        elif mode == "edit_instances":
            viewer.layers.selection.active = instance_layer
            if hasattr(instance_layer, "mode"):
                instance_layer.mode = "paint"
            set_status("Use napari paint/erase/fill tools on the Instance Labels layer.")
        elif mode == "edit_keypoints":
            viewer.layers.selection.active = keypoints_layer
            if hasattr(keypoints_layer, "mode"):
                keypoints_layer.mode = "select"
            set_status("Drag keypoints, then click Refine Spline.")
        else:
            set_status("Selection mode enabled.")

    def add_prompt_point(position_rc: tuple[float, float], label_value: int) -> None:
        current = np.asarray(prompt_points_layer.data, dtype=np.float32)
        point = np.asarray([position_rc], dtype=np.float32)
        prompt_points_layer.data = np.vstack([current, point]) if len(current) else point
        labels = np.asarray(prompt_points_layer.properties.get("label", []), dtype=int)
        labels = np.append(labels, label_value)
        prompt_points_layer.properties = {"label": labels}
        prompt_points_layer.face_color = ["lime" if val == 1 else "red" for val in labels]

    def run_prompted_sam() -> None:
        if state.working_image is None:
            set_status("Load an image first.")
            return
        point_coords, point_labels = prompt_points_payload()
        box = prompt_box_payload()
        if point_coords is None and box is None:
            set_status("Need points or a box first.")
            return
        prompted_mask, prompt_score = segmenter.prompted_mask(
            state.working_image,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        proposal_id = add_proposal(
            prompted_mask,
            source_variant="prompted",
            score=prompt_score,
            metadata={"prompt_score": prompt_score},
        )
        clear_prompts()
        set_status(f"Prompted SAM created proposal {proposal_id} with score {prompt_score:.3f}.")

    def refine_spline() -> None:
        if state.selected_instance_id is None or state.instance_labels is None:
            set_status("Select an instance first.")
            return
        instance_id = state.selected_instance_id
        edited_points = np.asarray(keypoints_layer.data, dtype=np.float32)
        if len(edited_points) < 2:
            set_status("Need at least two keypoints to refine.")
            return
        original_mask = instance_mask(instance_id)
        candidate = best_effort_candidate_for_instance(instance_id)
        width = max(candidate.fiber_width if candidate else 1.0, 1.0)
        points_rc = [(int(round(y)), int(round(x))) for y, x in edited_points]
        refined_mask = spline_mask_from_keypoints(original_mask.shape, points_rc, width, config)
        if not np.any(refined_mask):
            set_status("Refined spline produced an empty mask.")
            return
        state.instance_labels[state.instance_labels == instance_id] = 0
        state.instance_labels[refined_mask > 0] = instance_id
        refresh_layers()
        update_color_maps()
        set_status(f"Instance {instance_id} rebuilt from dragged keypoints.")

    def relabel_instances_compact() -> None:
        if state.instance_labels is None:
            return
        relabeled = label(state.instance_labels > 0).astype(np.int32)
        state.instance_labels = relabeled
        state.next_instance_id = int(relabeled.max()) + 1
        if state.selected_instance_id and state.selected_instance_id > relabeled.max():
            state.selected_instance_id = None
        refresh_layers()
        update_color_maps()
        set_status(f"Instances compacted into {int(relabeled.max())} labels.")

    def build_candidates_for_export() -> list[FiberCandidate]:
        candidates: list[FiberCandidate] = []
        if state.instance_labels is None or state.working_image is None:
            return candidates
        max_id = int(state.instance_labels.max())
        for instance_id in range(1, max_id + 1):
            mask = instance_mask(instance_id)
            if not np.any(mask):
                continue
            candidate, _ = build_candidate_from_mask(
                mask,
                state.working_image,
                config,
                fiber_id=instance_id,
                source=state.image_path.name if state.image_path else "unknown",
                mask_index=instance_id,
                preprocess_variant="instance_labels",
                validate=False,
            )
            if candidate is not None:
                candidates.append(candidate)
                continue
            polygons = mask_to_polygons(mask)
            if not polygons:
                continue
            ys, xs = np.where(mask > 0)
            if len(xs) < 2:
                continue
            candidates.append(
                FiberCandidate(
                    fiber_id=instance_id,
                    source=state.image_path.name if state.image_path else "unknown",
                    mask_index=instance_id,
                    preprocess_variant="instance_labels_raw",
                    binary_mask=mask.copy(),
                    final_mask=mask.copy(),
                    spline_mask=mask.copy(),
                    keypoints_rc=[(int(ys.min()), int(xs.min())), (int(ys.max()), int(xs.max()))],
                    visibility=[1, 1],
                    visible_keypoints=2,
                    keypoint_strategy="fallback_raw_mask",
                    fiber_width=0.0,
                    fiber_length=0.0,
                    fiber_curvature=0.0,
                    fiber_orientation=0.0,
                    spline_mask_iou=0.0,
                    skeleton_stats={"fallback": True},
                    bbox=[int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                    area=float(np.sum(mask > 0)),
                    polygons=polygons,
                )
            )
        return candidates

    def export_annotations() -> None:
        if state.working_image is None or state.image_path is None:
            set_status("Load an image first.")
            return
        candidates = build_candidates_for_export()
        if not candidates:
            set_status("No instances available for export.")
            return
        default_output = state.image_path.with_suffix(".fiber.coco.json")
        chosen, _ = QFileDialog.getSaveFileName(
            viewer.window.qt_viewer,
            "Export COCO",
            str(default_output),
            "JSON (*.json)",
        )
        if not chosen:
            return
        output_path = export_coco(state.image_path, state.working_image, candidates, chosen)
        set_status(f"COCO exported to {output_path}")

    def save_session() -> None:
        if state.working_image is None or state.image_path is None:
            set_status("Load an image before saving.")
            return
        default_output = state.image_path.with_suffix(".fiber.session.json")
        chosen, _ = QFileDialog.getSaveFileName(
            viewer.window.qt_viewer,
            "Save Session",
            str(default_output),
            "JSON (*.json)",
        )
        if not chosen:
            return
        payload = {
            "image_path": str(state.image_path),
            "proposals": [
                {
                    "proposal_id": proposal.proposal_id,
                    "mask": proposal.mask.tolist(),
                    "source_variant": proposal.source_variant,
                    "score": proposal.score,
                    "quality_ok": proposal.quality_ok,
                    "quality_reason": proposal.quality_reason,
                    "metadata": proposal.metadata,
                }
                for proposal in state.proposals.values()
            ],
            "instance_labels": state.instance_labels.tolist() if state.instance_labels is not None else None,
            "selected_proposal_id": state.selected_proposal_id,
            "selected_instance_id": state.selected_instance_id,
            "next_proposal_id": state.next_proposal_id,
            "next_instance_id": state.next_instance_id,
        }
        Path(chosen).write_text(json.dumps(payload), encoding="utf-8")
        set_status(f"Session saved to {chosen}")

    def load_session() -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            viewer.window.qt_viewer,
            "Load Session",
            str(Path.cwd()),
            "JSON (*.json)",
        )
        if not chosen:
            return
        payload = json.loads(Path(chosen).read_text(encoding="utf-8"))
        load_image(payload["image_path"])
        state.proposals = {}
        for item in payload.get("proposals", []):
            state.proposals[int(item["proposal_id"])] = ProposalMask(
                proposal_id=int(item["proposal_id"]),
                mask=np.asarray(item["mask"], dtype=np.uint8),
                source_variant=item["source_variant"],
                score=float(item["score"]),
                quality_ok=bool(item["quality_ok"]),
                quality_reason=item["quality_reason"],
                metadata=item.get("metadata", {}),
            )
        state.next_proposal_id = int(payload.get("next_proposal_id", len(state.proposals) + 1))
        state.instance_labels = np.asarray(payload.get("instance_labels"), dtype=np.int32)
        state.next_instance_id = int(payload.get("next_instance_id", int(state.instance_labels.max()) + 1 if state.instance_labels is not None else 1))
        state.selected_proposal_id = payload.get("selected_proposal_id")
        state.selected_instance_id = payload.get("selected_instance_id")
        rebuild_proposal_labels()
        refresh_layers()
        update_color_maps()
        set_status(f"Session loaded from {chosen}")

    @image_layer.mouse_drag_callbacks.append
    def handle_image_click(layer, event):
        if state.working_image is None:
            return
        yield
        if event.type != "mouse_press":
            return
        pos = layer.world_to_data(event.position)
        row = int(round(pos[0]))
        col = int(round(pos[1]))
        if row < 0 or col < 0 or row >= state.working_image.shape[0] or col >= state.working_image.shape[1]:
            return
        if state.interaction_mode == "positive":
            add_prompt_point((row, col), 1)
            set_status(f"Positive point added at ({row}, {col}).")
            return
        if state.interaction_mode == "negative":
            add_prompt_point((row, col), 0)
            set_status(f"Negative point added at ({row}, {col}).")
            return
        proposal_id = int(state.proposal_labels[row, col]) if state.proposal_labels is not None else 0
        instance_id = int(state.instance_labels[row, col]) if state.instance_labels is not None else 0
        if instance_id > 0:
            select_instance(instance_id)
        if proposal_id > 0:
            select_proposal(proposal_id)
        if proposal_id == 0 and instance_id == 0:
            select_instance(None)
            select_proposal(None)
            set_status(f"No mask at ({row}, {col}).")

    control_widget = QWidget()
    control_layout = QVBoxLayout(control_widget)
    control_layout.setContentsMargins(12, 12, 12, 12)
    control_layout.setSpacing(10)

    title_label = QLabel("Parameters")
    title_label.setStyleSheet("font-size: 18px; font-weight: 700;")
    control_layout.addWidget(title_label)

    def make_button(text: str, callback) -> None:
        button = QPushButton(text)
        button.clicked.connect(callback)
        button.setMinimumHeight(38)
        button.setStyleSheet("font-size: 14px;")
        control_layout.addWidget(button)

    make_button("Run Auto SAM", run_auto_sam)
    make_button("Create Instance Labels", create_instance_labels_from_proposals)
    make_button("Add Selected Proposal", add_selected_proposal_to_instances)
    make_button("Delete Selected Proposal", delete_selected_proposal)
    make_button("Delete Selected Instance", delete_selected_instance)
    make_button("Add Positive Point", lambda: set_prompt_mode("positive"))
    make_button("Add Negative Point", lambda: set_prompt_mode("negative"))
    make_button("Run Prompted SAM", run_prompted_sam)
    make_button("ROI Rectangle", lambda: set_prompt_mode("roi_rect"))
    make_button("ROI Polygon", lambda: set_prompt_mode("roi_poly"))
    make_button("Delete Proposals In ROI", delete_proposals_in_roi)
    make_button("Erase Instance Pixels In ROI", erase_instance_pixels_in_roi)
    make_button("Edit Instances", lambda: set_prompt_mode("edit_instances"))
    make_button("Edit Keypoints", lambda: set_prompt_mode("edit_keypoints"))
    make_button("Refine Spline", refine_spline)
    make_button("Compact Labels", relabel_instances_compact)
    make_button("Save Session", save_session)
    make_button("Load Session", load_session)

    separator = QFrame()
    separator.setFrameShape(QFrame.HLine)
    separator.setFrameShadow(QFrame.Sunken)
    control_layout.addWidget(separator)

    def add_slider_row(label_text: str, minimum: int, maximum: int, value: int, on_change) -> None:
        row = QWidget()
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(label_text))
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        slider.valueChanged.connect(on_change)
        layout.addWidget(slider)
        control_layout.addWidget(row)

    def add_spin_row(label_text: str, value: int, minimum: int, maximum: int, on_change) -> None:
        row = QWidget()
        form = QFormLayout(row)
        form.setContentsMargins(0, 0, 0, 0)
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.valueChanged.connect(on_change)
        form.addRow(QLabel(label_text), spin)
        control_layout.addWidget(row)

    add_slider_row("Minimum Mask Area", 1, 400, int(config.min_mask_area), lambda v: setattr(config, "min_mask_area", int(v)))
    add_slider_row("SAM IoU Threshold", 0, 100, int(config.pred_iou_thresh * 100), lambda v: setattr(config, "sam_pred_iou_thresh_fast" if config.fast_mode else "sam_pred_iou_thresh_full", float(v) / 100.0))
    add_slider_row("Final IoU Dedup", 0, 100, int(config.final_mask_iou_dedup_threshold * 100), lambda v: setattr(config, "final_mask_iou_dedup_threshold", float(v) / 100.0))
    add_spin_row("Minimum Fiber Length", int(config.min_fiber_length_px), 1, 1000, lambda v: setattr(config, "min_fiber_length_px", float(v)))
    add_slider_row("Spline Mask IoU", 0, 100, int(config.min_spline_mask_iou * 100), lambda v: setattr(config, "min_spline_mask_iou", float(v) / 100.0))

    export_button = QPushButton("Export COCO")
    export_button.clicked.connect(export_annotations)
    export_button.setMinimumHeight(42)
    export_button.setStyleSheet("font-size: 15px; font-weight: 600;")
    control_layout.addWidget(export_button)

    status_label = QLabel("Load an image to begin.")
    status_label.setWordWrap(True)
    status_label.setStyleSheet("font-size: 12px; color: #d7d7d7;")
    control_layout.addWidget(status_label)
    control_layout.addStretch(1)
    viewer.window.add_dock_widget(control_widget, area="left", name="Parameters")

    top_bar = QWidget()
    top_layout = QHBoxLayout(top_bar)
    top_layout.setContentsMargins(10, 8, 10, 8)
    top_layout.setSpacing(8)

    open_button = QPushButton("Open Image")
    open_button.setMinimumHeight(34)

    def open_dialog() -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            viewer.window.qt_viewer,
            "Open SEM Image",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.tif *.tiff)",
        )
        if chosen:
            load_image(chosen)

    open_button.clicked.connect(open_dialog)
    top_layout.addWidget(open_button)

    clear_prompt_button = QPushButton("Clear Prompts")
    clear_prompt_button.setMinimumHeight(34)
    clear_prompt_button.clicked.connect(clear_prompts)
    top_layout.addWidget(clear_prompt_button)

    clear_roi_button = QPushButton("Clear ROI")
    clear_roi_button.setMinimumHeight(34)
    clear_roi_button.clicked.connect(clear_roi)
    top_layout.addWidget(clear_roi_button)
    top_layout.addStretch(1)
    viewer.window.add_dock_widget(top_bar, area="top", name="Segmentation")

    right_widget = QWidget()
    right_layout = QVBoxLayout(right_widget)
    right_layout.setContentsMargins(10, 10, 10, 10)
    right_layout.setSpacing(8)
    right_layout.addWidget(QLabel("SAM Proposals"))
    proposal_list = QListWidget()
    right_layout.addWidget(proposal_list)

    def on_list_selection() -> None:
        current = proposal_list.currentItem()
        if current is None:
            return
        proposal_id = int(current.data(Qt.UserRole))
        select_proposal(proposal_id)

    proposal_list.itemSelectionChanged.connect(on_list_selection)
    viewer.window.add_dock_widget(right_widget, area="right", name="Review")

    if image_path:
        load_image(image_path)
    else:
        set_status("Open an SEM image, then start with Auto SAM.")

    update_color_maps()
    napari.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Semi-automatic fiber segmentation studio in napari.")
    parser.add_argument("image", nargs="?", default=None, help="Optional image path to open at startup.")
    args = parser.parse_args()
    run_app(args.image)


if __name__ == "__main__":
    main()
