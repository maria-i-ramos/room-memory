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
