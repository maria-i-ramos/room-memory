"""Checks for steps 5 and 6 using made-up detections (no YOLO needed)."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from visual_memory.search import find_last_seen, resolve_query, draw_last_seen, load_detections

DETECTIONS = [
    {"label": "bottle", "confidence": 0.80, "timestamp_seconds": 0.0, "source_frame_index": 0, "xyxy": [5, 5, 20, 30]},
    {"label": "bottle", "confidence": 0.40, "timestamp_seconds": 1.5, "source_frame_index": 15, "xyxy": [10, 5, 25, 30]},
    {"label": "bottle", "confidence": 0.90, "timestamp_seconds": 1.5, "source_frame_index": 15, "xyxy": [40, 5, 55, 30]},
    {"label": "bottle", "confidence": 0.10, "timestamp_seconds": 2.5, "source_frame_index": 25, "xyxy": [1, 1, 2, 2]},
    {"label": "book", "confidence": 0.70, "timestamp_seconds": 2.0, "source_frame_index": 20, "xyxy": [30, 30, 60, 50]},
]


def test_last_seen_picks_latest_confident_box():
    result = find_last_seen(DETECTIONS, "bottle")
    assert result["found"]
    # The 2.5s box is below the confidence threshold, so 1.5s is the latest real sighting.
    assert result["last_seen"]["timestamp_seconds"] == 1.5
    assert result["last_seen"]["confidence"] == 0.90  # best box in that frame
    assert result["first_seen_seconds"] == 0.0
    assert result["frames_with_object"] == 2


def test_aliases_and_plurals():
    assert resolve_query("Water Bottle", {"bottle"}) == ("bottle", None)
    assert resolve_query("books", {"book"}) == ("book", None)
    assert find_last_seen(DETECTIONS, "water bottle")["found"]


def test_unsupported_object_explains_why():
    result = find_last_seen(DETECTIONS, "keys")
    assert not result["found"]
    assert result["label"] is None
    assert "COCO" in result["note"]


def test_object_never_seen():
    result = find_last_seen(DETECTIONS, "laptop")
    assert not result["found"]


def test_draw_last_seen(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    cv2.imwrite(str(frames / "frame_000015.jpg"), np.zeros((64, 96, 3), dtype=np.uint8))
    out = draw_last_seen(frames, DETECTIONS[2], tmp_path / "out.jpg")
    image = cv2.imread(str(out))
    assert image is not None and image.shape == (64, 96, 3)
    assert image.sum() > 0  # something was drawn


def test_missing_detections_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_detections(tmp_path)


def test_find_object_script(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    cv2.imwrite(str(frames / "frame_000015.jpg"), np.zeros((64, 96, 3), dtype=np.uint8))
    (tmp_path / "detections.json").write_text(json.dumps(DETECTIONS))
    script = Path(__file__).resolve().parents[1] / "scripts" / "find_object.py"
    subprocess.run([sys.executable, str(script), str(tmp_path), "bottle"], check=True)
    assert (tmp_path / "search" / "last_seen_bottle.jpg").exists()
    saved = json.loads((tmp_path / "search" / "last_seen_bottle.json").read_text())
    assert saved["last_seen"]["source_frame_index"] == 15
