"""Step 3: ask a pretrained YOLO model which objects appear in each saved frame."""
from pathlib import Path
import cv2


def detect_objects(frame_list, frames_folder, model_name="yolo11n.pt", confidence=0.25):
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")

    # Only load YOLO when detection is actually requested.
    # The first run may need to download the pretrained model.
    from ultralytics import YOLO
    model = YOLO(model_name)
    detections = []

    for frame in frame_list:
        image = cv2.imread(str(Path(frames_folder) / frame["filename"]))
        if image is None:
            raise ValueError(f"Couldn't read {frame['filename']}")

        prediction = model.predict(image, conf=confidence, verbose=False)[0]
        for box in prediction.boxes:
            label_number = int(box.cls.item())
            detections.append({
                "label": prediction.names[label_number],
                "confidence": round(float(box.conf.item()), 3),
                "timestamp_seconds": frame["timestamp_seconds"],
                "source_frame_index": frame["source_frame_index"],
                "xyxy": [round(float(x), 1) for x in box.xyxy[0].tolist()],
            })

    return detections
