"""Quick checks we can run without downloading YOLO weights."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from visual_memory.video import inspect_video
from visual_memory.frames import extract_frames


@pytest.fixture
def example_video(tmp_path):
    # Create a tiny video so tests don't depend on someone's personal recordings.
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 64))
    assert writer.isOpened()
    for i in range(30):
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        cv2.rectangle(image, (i, 10), (i + 15, 25), (255, 255, 255), -1)
        writer.write(image)
    writer.release()
    return path


def test_video_info(example_video):
    info = inspect_video(example_video)
    assert info["fps"] == 10
    assert info["width"] == 96
    assert info["duration_seconds"] == pytest.approx(3)


def test_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_video(tmp_path / "missing.mp4")


def test_frame_extraction(example_video, tmp_path):
    frames = extract_frames(example_video, tmp_path / "frames", 2)
    assert len(frames) == 6  # 3 seconds x 2 saved frames per second
    assert [x["timestamp_seconds"] for x in frames] == [0, .5, 1, 1.5, 2, 2.5]
    assert all((tmp_path / "frames" / x["filename"]).exists() for x in frames)


def test_invalid_fps(example_video, tmp_path):
    with pytest.raises(ValueError):
        extract_frames(example_video, tmp_path / "frames", 0)


def test_full_script_without_yolo(example_video, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_baseline.py"
    output = tmp_path / "results"
    subprocess.run([sys.executable, str(script), str(example_video), "--out", str(output), "--skip-detection"], check=True)
    assert len(json.loads((output / "frames.json").read_text())) == 6
    assert not (output / "detections.json").exists()


def test_sam_checklist_boxes_are_searchable(example_video, tmp_path, monkeypatch):
    from types import SimpleNamespace
    sam = pytest.importorskip("ultralytics.models.sam")
    from visual_memory.detect import checklist_prompts, detect_objects
    from visual_memory.search import find_last_seen

    prompts = checklist_prompts("IMG_4572.mp4")
    assert prompts == ["card", "keys", "Apple Pencil"]
    frames = extract_frames(example_video, tmp_path / "frames", 1)
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.touch()

    class FakePredictor:
        def __init__(self, overrides):
            assert overrides["model"] == str(checkpoint)

        def setup_model(self):
            self.model = SimpleNamespace(transformer=SimpleNamespace(encoder=SimpleNamespace(layers=[])))

        def reset_image(self):
            self.image = None

        def set_image(self, image):
            assert self.image is None
            self.image = image

        def __call__(self, source, text):
            assert source is None and self.image is not None and text[0] in prompts and len(text) == 1
            box = SimpleNamespace(cls=np.array(0), conf=np.array(0.85),
                                  xyxy=np.array([[6, 5, 20, 18]]))
            return [SimpleNamespace(names=text, boxes=[box])]

    monkeypatch.setattr(sam, "SAM3SemanticPredictor", FakePredictor)
    detections = detect_objects(frames, tmp_path / "frames", checkpoint, prompts=prompts)
    assert {d["label"] for d in detections} == set(prompts)
    assert len(detections) == len(frames) * len(prompts)
    last = find_last_seen(detections, "apple pencil")["last_seen"]
    assert last == {"label": "Apple Pencil", "confidence": 0.85, "timestamp_seconds": 2,
                    "source_frame_index": 20, "xyxy": [6, 5, 20, 18]}


def test_yoloe_uses_text_prompts(example_video, tmp_path, monkeypatch):
    from types import SimpleNamespace
    ultralytics = pytest.importorskip("ultralytics")
    from visual_memory.detect import detect_objects

    prompts = ["eyeglasses", "desk"]
    frames = extract_frames(example_video, tmp_path / "frames", 1)

    class FakeYOLOE:
        def __init__(self, model_name):
            assert model_name == "yoloe-11l-seg.pt"

        def get_text_pe(self, names):
            assert names == prompts
            return "embeddings"

        def set_classes(self, names, embeddings):
            assert names == prompts and embeddings == "embeddings"

        def predict(self, image, conf, verbose):
            box = SimpleNamespace(cls=np.array(0), conf=np.array(.8),
                                  xyxy=np.array([[1, 2, 20, 30]]))
            return [SimpleNamespace(names=prompts, boxes=[box])]

    monkeypatch.setattr(ultralytics, "YOLOE", FakeYOLOE)
    detections = detect_objects(frames, tmp_path / "frames", "yoloe-11l-seg.pt", prompts=prompts)
    assert len(detections) == 3
    assert {item["label"] for item in detections} == {"eyeglasses"}
    with pytest.raises(ValueError, match="YOLOE needs"):
        detect_objects(frames, tmp_path / "frames", "yoloe-11l-seg.pt")
