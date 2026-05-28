from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fiber_segmentation_core import (
    FiberCandidate,
    PipelineConfig,
    SamFiberSegmenter,
    build_candidate_from_mask,
    crop_percentage,
    export_coco,
    labels_from_candidates,
    labels_from_raw_masks,
    spline_mask_from_keypoints,
)


@dataclass
class SessionState:
    image_path: Path | None = None
    original_image: np.ndarray | None = None
    working_image: np.ndarray | None = None
    raw_masks: list[dict[str, Any]] = field(default_factory=list)
    accepted: dict[int, FiberCandidate] = field(default_factory=dict)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    selected_fiber_id: int | None = None
    next_fiber_id: int = 1
    interaction_mode: str = "select"


def run_app(image_path: str | None = None) -> None:
    try:
        import napari
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QFileDialog,
            QFormLayout,
            QFrame,
            QHBoxLayout,
            QLabel,
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
    sam_masks_layer = viewer.add_labels(np.zeros((32, 32), dtype=np.int32), name="SAM Masks")
    skeletons_layer = viewer.add_shapes(
        name="Fiber Skeletons",
        shape_type="path",
        edge_width=2.0,
        edge_color="yellow",
        face_color="transparent",
    )
    final_layer = viewer.add_labels(np.zeros((32, 32), dtype=np.int32), name="Final Annotations")
    keypoints_layer = viewer.add_points(
        np.empty((0, 2)),
        name="Keypoints",
        size=6,
        face_color="yellow",
        edge_color="black",
        edge_width=0.3,
    )
    prompt_points_layer = viewer.add_points(
        np.empty((0, 2)),
        name="Prompt Points",
        size=9,
        face_color="lime",
        edge_color="black",
        edge_width=0.5,
        properties={"label": np.array([], dtype=int)},
        text={"string": "{label}", "color": "white", "size": 8},
    )
    prompt_box_layer = viewer.add_shapes(
        name="Prompt Box",
        shape_type="rectangle",
        edge_color="red",
        edge_width=2.5,
        face_color="transparent",
    )

    for layer in [sam_masks_layer, final_layer]:
        layer.opacity = 0.45

    sam_masks_layer.visible = True
    final_layer.visible = True

    def set_status(text: str) -> None:
        viewer.status = text
        status_label.setText(text)

    def reset_layers(shape: tuple[int, int]) -> None:
        image_layer.data = np.zeros(shape, dtype=np.uint8)
        sam_masks_layer.data = np.zeros(shape, dtype=np.int32)
        final_layer.data = np.zeros(shape, dtype=np.int32)
        skeletons_layer.data = []
        keypoints_layer.data = np.empty((0, 2))
        prompt_points_layer.data = np.empty((0, 2))
        prompt_points_layer.properties = {"label": np.array([], dtype=int)}
        prompt_points_layer.face_color = []
        prompt_box_layer.data = []

    def load_image(path: str | Path) -> None:
        path = Path(path)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            set_status(f"Unable to read image: {path}")
            return

        cropped = crop_percentage(image, config.crop_bottom_percent)
        state.image_path = path
        state.original_image = image
        state.working_image = cropped
        state.raw_masks.clear()
        state.accepted.clear()
        state.rejected.clear()
        state.selected_fiber_id = None
        state.next_fiber_id = 1
        state.interaction_mode = "select"

        reset_layers(cropped.shape)
        image_layer.data = cropped
        image_layer.contrast_limits = (float(np.min(cropped)), float(np.max(cropped)))
        viewer.reset_view()
        set_status(f"Loaded {path.name}. Ready for Auto SAM or prompted refinement.")

    def prompt_points_payload() -> tuple[np.ndarray | None, np.ndarray | None]:
        points = np.asarray(prompt_points_layer.data, dtype=np.float32)
        labels = np.asarray(prompt_points_layer.properties.get("label", []), dtype=np.int32)
        if len(points) == 0 or len(labels) == 0:
            return None, None
        point_coords = np.stack([points[:, 1], points[:, 0]], axis=1)
        return point_coords, labels

    def prompt_box_payload() -> np.ndarray | None:
        if len(prompt_box_layer.data) == 0:
            return None
        rect = np.asarray(prompt_box_layer.data[-1], dtype=np.float32)
        ys = rect[:, 0]
        xs = rect[:, 1]
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

    def render_selected(candidate: FiberCandidate | None) -> None:
        if candidate is None:
            skeletons_layer.data = []
            keypoints_layer.data = np.empty((0, 2))
            return

        skeleton_path = np.asarray(candidate.keypoints_rc, dtype=np.float32)
        skeletons_layer.data = [skeleton_path]
        keypoints_layer.data = skeleton_path
        keypoints_layer.properties = {
            "index": np.arange(1, len(candidate.keypoints_rc) + 1, dtype=int)
        }
        keypoints_layer.text = {"string": "{index}", "color": "yellow", "size": 9}

    def render_all() -> None:
        if state.working_image is None:
            return
        sam_masks_layer.data = labels_from_raw_masks(state.working_image.shape, state.raw_masks)
        final_candidates = list(state.accepted.values())
        final_layer.data = labels_from_candidates(state.working_image.shape, final_candidates)

        if state.selected_fiber_id in state.accepted:
            render_selected(state.accepted[state.selected_fiber_id])
            candidate = state.accepted[state.selected_fiber_id]
            set_status(
                f"Selected fiber {candidate.fiber_id} | len={candidate.fiber_length:.1f}px "
                f"| width={candidate.fiber_width:.1f}px | spline IoU={candidate.spline_mask_iou:.2f}"
            )
        else:
            render_selected(None)

    def assign_candidate(candidate: FiberCandidate) -> None:
        state.accepted[candidate.fiber_id] = candidate
        state.selected_fiber_id = candidate.fiber_id
        state.next_fiber_id = max(state.next_fiber_id, candidate.fiber_id + 1)

    def find_fiber_id_from_raw_label(raw_label: int) -> int | None:
        if raw_label <= 0:
            return None
        for fiber_id, candidate in state.accepted.items():
            if candidate.mask_index == raw_label - 1:
                return fiber_id
        return None

    def auto_sam() -> None:
        if state.working_image is None:
            set_status("Load an image first.")
            return

        set_status("Running Auto SAM...")
        masks, _, _ = segmenter.automatic_masks(state.working_image)
        state.raw_masks = masks
        state.accepted.clear()
        state.rejected.clear()
        state.selected_fiber_id = None
        state.next_fiber_id = 1

        accepted_final_masks: list[np.ndarray] = []
        for mask_index, mask_data in enumerate(masks):
            binary_mask = mask_data["segmentation"].astype(np.uint8) * 255
            candidate, reason = build_candidate_from_mask(
                binary_mask,
                state.working_image,
                config,
                fiber_id=state.next_fiber_id,
                source=state.image_path.name if state.image_path else "unknown",
                mask_index=mask_index,
                preprocess_variant=mask_data.get("preprocess_variant", "raw"),
                extra_metadata={
                    "predicted_iou": float(mask_data.get("predicted_iou", 0.0)),
                    "stability_score": float(mask_data.get("stability_score", 0.0)),
                },
            )
            if candidate is None:
                state.rejected.append(
                    {
                        "mask_index": mask_index,
                        "reason": reason,
                        "binary_mask": binary_mask.copy(),
                        "mask_data": mask_data,
                    }
                )
                continue

            duplicate = any(
                candidate.final_mask is not None
                and np.any(existing)
                and (
                    (
                        np.logical_and(candidate.final_mask > 0, existing > 0).sum()
                        / max(np.logical_or(candidate.final_mask > 0, existing > 0).sum(), 1)
                    )
                    >= config.final_mask_iou_dedup_threshold
                )
                for existing in accepted_final_masks
            )
            if duplicate:
                state.rejected.append(
                    {
                        "mask_index": mask_index,
                        "reason": "duplicate_final_spline",
                        "binary_mask": binary_mask.copy(),
                        "mask_data": mask_data,
                    }
                )
                continue

            assign_candidate(candidate)
            accepted_final_masks.append(candidate.final_mask.copy())

        render_all()
        set_status(
            f"Auto SAM done. Accepted {len(state.accepted)} fibers, "
            f"rejected {len(state.rejected)}."
        )

    def set_prompt_mode(mode: str) -> None:
        state.interaction_mode = mode
        prompt_box_layer.mode = "pan_zoom"
        if mode == "box":
            prompt_box_layer.mode = "add_rectangle"
            set_status("Draw a rectangle on the image, then press 'Box Prompt' again to run SAM.")
        elif mode == "positive":
            set_status("Click on the image to add positive prompt points.")
        elif mode == "negative":
            set_status("Click on the image to add negative prompt points.")
        else:
            set_status("Selection mode enabled.")

    def add_prompt_point(position_rc: tuple[float, float], label: int) -> None:
        current = np.asarray(prompt_points_layer.data, dtype=np.float32)
        updated = np.vstack([current, np.asarray(position_rc, dtype=np.float32)]) if len(current) else np.asarray([position_rc], dtype=np.float32)
        labels = np.asarray(prompt_points_layer.properties.get("label", []), dtype=int)
        labels = np.append(labels, label)
        prompt_points_layer.data = updated
        prompt_points_layer.properties = {"label": labels}
        colors = ["lime" if value == 1 else "red" for value in labels]
        prompt_points_layer.face_color = colors

    def run_prompted_sam() -> None:
        if state.working_image is None:
            set_status("Load an image first.")
            return

        point_coords, point_labels = prompt_points_payload()
        box = prompt_box_payload()
        if point_coords is None and box is None:
            set_status("Add prompt points or draw a box first.")
            return

        set_status("Running prompted SAM...")
        prompted_mask, prompt_score = segmenter.prompted_mask(
            state.working_image,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        candidate, reason = build_candidate_from_mask(
            prompted_mask,
            state.working_image,
            config,
            fiber_id=state.next_fiber_id,
            source=state.image_path.name if state.image_path else "unknown",
            mask_index=len(state.raw_masks),
            preprocess_variant="prompted",
            extra_metadata={"prompt_score": prompt_score},
        )
        if candidate is None:
            state.rejected.append(
                {
                    "mask_index": len(state.raw_masks),
                    "reason": reason,
                    "binary_mask": prompted_mask.copy(),
                    "mask_data": {"segmentation": prompted_mask.astype(bool)},
                }
            )
            render_all()
            set_status(f"Prompted SAM rejected: {reason}")
            return

        assign_candidate(candidate)
        state.raw_masks.append(
            {
                "segmentation": prompted_mask.astype(bool),
                "bbox": candidate.bbox,
                "area": candidate.area,
                "predicted_iou": prompt_score,
                "stability_score": prompt_score,
                "preprocess_variant": "prompted",
            }
        )
        render_all()
        set_status(f"Prompted fiber accepted with score {prompt_score:.3f}.")

    def reject_selected() -> None:
        fiber_id = state.selected_fiber_id
        if fiber_id is None or fiber_id not in state.accepted:
            set_status("Select an accepted fiber first.")
            return
        candidate = state.accepted.pop(fiber_id)
        state.rejected.append(
            {
                "mask_index": candidate.mask_index,
                "reason": "manual_reject",
                "binary_mask": candidate.final_mask.copy(),
                "mask_data": {"segmentation": candidate.final_mask.astype(bool)},
            }
        )
        state.selected_fiber_id = None
        render_all()
        set_status(f"Fiber {fiber_id} moved to rejected.")

    def refine_spline() -> None:
        fiber_id = state.selected_fiber_id
        if fiber_id is None or fiber_id not in state.accepted:
            set_status("Select an accepted fiber first.")
            return

        candidate = state.accepted[fiber_id]
        edited_points = np.asarray(keypoints_layer.data, dtype=np.float32)
        if len(edited_points) < 2:
            set_status("Need at least two keypoints to refine a spline.")
            return

        points_rc = [(int(round(row)), int(round(col))) for row, col in edited_points]
        refined_mask = spline_mask_from_keypoints(
            candidate.final_mask.shape, points_rc, candidate.fiber_width, config
        )
        rebuilt, reason = build_candidate_from_mask(
            refined_mask if np.any(refined_mask) else candidate.final_mask,
            state.working_image,
            config,
            fiber_id=fiber_id,
            source=candidate.source,
            mask_index=candidate.mask_index,
            preprocess_variant="manual_refine",
            extra_metadata={**candidate.metadata, "manual_refined": True},
        )
        if rebuilt is None:
            set_status(f"Refine failed: {reason}")
            return

        state.accepted[fiber_id] = rebuilt
        render_all()
        set_status(f"Fiber {fiber_id} refined and rebuilt.")

    def export_annotations() -> None:
        if state.working_image is None or state.image_path is None:
            set_status("Load an image first.")
            return
        if not state.accepted:
            set_status("No accepted fibers to export.")
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

        output_path = export_coco(
            state.image_path,
            state.working_image,
            list(state.accepted.values()),
            chosen,
        )
        set_status(f"COCO exported to {output_path}")

    @image_layer.mouse_drag_callbacks.append
    def handle_image_click(layer, event):
        if state.working_image is None:
            return

        yield
        if event.type != "mouse_press":
            return

        data_position = layer.world_to_data(event.position)
        row = int(round(data_position[0]))
        col = int(round(data_position[1]))
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

        selected_id = int(final_layer.data[row, col])
        if selected_id == 0:
            selected_id = find_fiber_id_from_raw_label(int(sam_masks_layer.data[row, col])) or 0

        if selected_id in state.accepted:
            state.selected_fiber_id = selected_id
            render_all()
        else:
            state.selected_fiber_id = None
            render_all()
            set_status(f"No accepted fiber at ({row}, {col}).")

    control_widget = QWidget()
    control_layout = QVBoxLayout(control_widget)
    control_layout.setContentsMargins(12, 12, 12, 12)
    control_layout.setSpacing(10)

    title_label = QLabel("Parameters")
    title_label.setStyleSheet("font-size: 18px; font-weight: 700;")
    control_layout.addWidget(title_label)

    def make_button(text: str, callback) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        button.setMinimumHeight(40)
        button.setStyleSheet("font-size: 15px;")
        control_layout.addWidget(button)
        return button

    make_button("Run Auto SAM", auto_sam)
    make_button("Add Positive Point", lambda: set_prompt_mode("positive"))
    make_button("Add Negative Point", lambda: set_prompt_mode("negative"))

    def on_box_prompt() -> None:
        if len(prompt_box_layer.data) == 0:
            set_prompt_mode("box")
            return
        run_prompted_sam()

    make_button("Box Prompt", on_box_prompt)
    make_button("Refine Spline", refine_spline)
    make_button("Reject Fiber", reject_selected)

    separator = QFrame()
    separator.setFrameShape(QFrame.HLine)
    separator.setFrameShadow(QFrame.Sunken)
    control_layout.addWidget(separator)

    def add_slider_row(label_text: str, minimum: int, maximum: int, value: int, on_change):
        row = QWidget()
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        caption = QLabel(label_text)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        slider.valueChanged.connect(on_change)
        layout.addWidget(caption)
        layout.addWidget(slider)
        control_layout.addWidget(row)
        return slider

    def add_spin_row(label_text: str, value: int, minimum: int, maximum: int, on_change):
        row = QWidget()
        form = QFormLayout(row)
        form.setContentsMargins(0, 0, 0, 0)
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.valueChanged.connect(on_change)
        form.addRow(QLabel(label_text), spin)
        control_layout.addWidget(row)
        return spin

    add_slider_row(
        "Minimum Fiber",
        1,
        200,
        int(config.min_mask_area),
        lambda value: setattr(config, "min_mask_area", int(value)),
    )
    add_slider_row(
        "Mask Threshold",
        0,
        100,
        int(config.pred_iou_thresh * 100),
        lambda value: setattr(
            config,
            "sam_pred_iou_thresh_fast" if config.fast_mode else "sam_pred_iou_thresh_full",
            float(value) / 100.0,
        ),
    )
    add_slider_row(
        "IoU Threshold",
        0,
        100,
        int(config.final_mask_iou_dedup_threshold * 100),
        lambda value: setattr(config, "final_mask_iou_dedup_threshold", float(value) / 100.0),
    )
    add_spin_row(
        "Minimum Length",
        int(config.min_fiber_length_px),
        1,
        1000,
        lambda value: setattr(config, "min_fiber_length_px", float(value)),
    )
    add_slider_row(
        "Spline IoU",
        0,
        100,
        int(config.min_spline_mask_iou * 100),
        lambda value: setattr(config, "min_spline_mask_iou", float(value) / 100.0),
    )

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

    def clear_prompts() -> None:
        prompt_points_layer.data = np.empty((0, 2))
        prompt_points_layer.properties = {"label": np.array([], dtype=int)}
        prompt_box_layer.data = []
        state.interaction_mode = "select"
        set_status("Prompt points and boxes cleared.")

    clear_prompt_button.clicked.connect(clear_prompts)
    top_layout.addWidget(clear_prompt_button)
    top_layout.addStretch(1)
    viewer.window.add_dock_widget(top_bar, area="top", name="Segmentation")

    if image_path:
        load_image(image_path)
    else:
        set_status("Open an SEM image, then start with Auto SAM or prompted editing.")

    napari.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Semi-automatic fiber segmentation studio in napari.")
    parser.add_argument("image", nargs="?", default=None, help="Optional image path to open at startup.")
    args = parser.parse_args()
    run_app(args.image)


if __name__ == "__main__":
    main()
