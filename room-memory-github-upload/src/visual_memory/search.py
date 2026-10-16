"""Steps 5 and 6: search the saved detections and return where an object was last seen."""
import json
from pathlib import Path

import cv2

# People type everyday words, but YOLO (COCO) uses its own label names.
# This maps common phrases to the COCO label we should search for.
# A value of None means "COCO YOLO has no class for this" so we can say so honestly.
ALIASES = {
    "water bottle": "bottle",
    "bottle": "bottle",
    "tv remote": "remote",
    "remote control": "remote",
    "remote": "remote",
    "phone": "cell phone",
    "cellphone": "cell phone",
    "mobile": "cell phone",
    "mug": "cup",
    "coffee cup": "cup",
    "bag": "backpack",
    "notebook": "book",
    "computer": "laptop",
    "glasses": None,
    "eyeglasses": None,
    "sunglasses": None,
    "keys": None,
    "key": None,
    "earbuds": None,
    "wired earbuds": None,
    "airpods": None,
    "headphones": None,
    "wallet": None,
}


def load_detections(results_folder):
    """Read detections.json written by run_baseline.py (step 4)."""
    path = Path(results_folder) / "detections.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No detections found at {path}. Run scripts/run_baseline.py without --skip-detection first."
        )
    return json.loads(path.read_text())


def resolve_query(query, known_labels=()):
    """Turn what the user typed into a YOLO label.

    Returns (label, note). label is None when YOLO can't detect that kind of object.
    """
    text = " ".join(query.lower().strip().split())
    if text in known_labels:
        return text, None
    if text in ALIASES:
        label = ALIASES[text]
        if label is None:
            return None, f'"{query}" is not one of the pretrained YOLO (COCO) classes, so it can\'t be found yet.'
        return label, None
    # Plurals like "bottles" or "books".
    if text.endswith("s") and text[:-1] in known_labels:
        return text[:-1], None
    return text, None


def find_last_seen(detections, query, min_confidence=0.25):
    """Return the most recent sighting of the object, or None.

    "Last seen" = the detection with the latest timestamp. If the object appears
    more than once in that frame, we keep the most confident box.
    """
    known_labels = {d["label"] for d in detections}
    label, note = resolve_query(query, known_labels)
    result = {"query": query, "label": label, "found": False, "note": note}
    if label is None:
        return result

    matches = [d for d in detections if d["label"] == label and d["confidence"] >= min_confidence]
    if not matches:
        result["note"] = f'No "{label}" detections at confidence >= {min_confidence}.'
        return result

    last = max(matches, key=lambda d: (d["timestamp_seconds"], d["confidence"]))
    result.update({
        "found": True,
        "last_seen": last,
        "first_seen_seconds": min(d["timestamp_seconds"] for d in matches),
        "frames_with_object": len({d["source_frame_index"] for d in matches}),
    })
    return result


def draw_last_seen(frames_folder, detection, output_path):
    """Draw the bounding box and timestamp on the frame where the object was last seen."""
    filename = f"frame_{detection['source_frame_index']:06d}.jpg"
    image = cv2.imread(str(Path(frames_folder) / filename))
    if image is None:
        raise FileNotFoundError(f"Couldn't read frame {filename} in {frames_folder}")

    x1, y1, x2, y2 = (int(round(v)) for v in detection["xyxy"])
    thickness = max(2, image.shape[1] // 200)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)

    text = f"{detection['label']} {detection['confidence']:.2f} @ {detection['timestamp_seconds']:.1f}s"
    scale = max(0.6, image.shape[1] / 900)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    top = max(0, y1 - th - base - 6)
    cv2.rectangle(image, (x1, top), (x1 + tw + 8, top + th + base + 6), (0, 255, 0), -1)
    cv2.putText(image, text, (x1 + 4, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Couldn't save {output_path}")
    return output_path
