"""Detect prompted objects on sampled frames and retain searchable pixel boxes."""
import json
from functools import partial
from pathlib import Path
import cv2

from visual_memory.pipeline_config import DEFAULT_YOLOE_MODEL


def checklist_prompts(video_path):
    stem = Path(video_path).stem
    path = Path(__file__).resolve().parents[2] / "data" / "checklists" / f"{stem}_objects.json"
    if not path.is_file():
        raise ValueError(f"No checklist for {stem}. Supply one or more --prompt options.")
    return list(dict.fromkeys(item["object"] for item in json.loads(path.read_text())["objects"]))


def detect_objects(frame_list, frames_folder, model_name=DEFAULT_YOLOE_MODEL, confidence=0.25, *, prompts=None, device=None):
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")

    device_options = {} if device is None else {"device": device}
    uses_yoloe = Path(model_name).name.startswith("yoloe")
    if uses_yoloe and (not prompts or not all(isinstance(p, str) and p.strip() for p in prompts)):
        raise ValueError("YOLOE needs at least one nonempty object prompt.")

    # Explicit YOLO checkpoints can still reproduce the historical baseline.
    if Path(model_name).name.startswith("yolo"):
        from ultralytics import YOLO, YOLOE
        if uses_yoloe:
            model = YOLOE(model_name)
            model.set_classes(prompts, model.get_text_pe(prompts))
        else:
            model = YOLO(model_name)

        def predict(image):
            return model.predict(image, conf=confidence, verbose=False, **device_options)
    else:
        if not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
            raise ValueError("SAM 3 needs at least one nonempty object prompt.")
        if not Path(model_name).is_file():
            raise FileNotFoundError(f"Missing SAM 3 checkpoint: {model_name}. See START_HERE.md.")
        from ultralytics.models.sam import SAM3SemanticPredictor
        predictor = SAM3SemanticPredictor(overrides={
            "model": str(model_name), "conf": confidence, "imgsz": 1008,
            "save": False, "verbose": False, **device_options,
        })

        predictor.setup_model()
        # SAM 3's encoder consumes only the attention output, not its weights.
        # Skipping unused weights enables PyTorch's memory-efficient attention.
        for layer in predictor.model.transformer.encoder.layers:
            layer.self_attn.forward = partial(layer.self_attn.forward, need_weights=False)

        def predict(image):
            predictor.reset_image()
            predictor.set_image(image)
            try:
                for prompt in prompts:
                    yield predictor(source=None, text=[prompt])[0]
            finally:
                predictor.reset_image()

    detections = []

    for frame in frame_list:
        image = cv2.imread(str(Path(frames_folder) / frame["filename"]))
        if image is None:
            raise ValueError(f"Couldn't read {frame['filename']}")

        for prediction in predict(image):
            masks = getattr(prediction, "masks", None) if uses_yoloe else None
            polygons = masks.xy if masks is not None else ()
            for index, box in enumerate(prediction.boxes):
                label_number = int(box.cls.item())
                detection = {
                    "label": prediction.names[label_number],
                    "confidence": round(float(box.conf.item()), 3),
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "source_frame_index": frame["source_frame_index"],
                    "xyxy": [round(float(x), 1) for x in box.xyxy[0].tolist()],
                }
                if index < len(polygons) and len(polygons[index]) >= 3:
                    detection["polygon"] = polygons[index].round(1).tolist()
                detections.append(detection)

    return detections
