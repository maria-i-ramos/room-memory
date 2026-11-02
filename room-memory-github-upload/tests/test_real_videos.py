"""Steps 1-6 on our real recordings in data/videos/.

These run YOLO, so they're slower (~5 s per video) and need the detection extras installed.
They're skipped automatically if YOLO or the videos aren't available.
Run only these with:  python -m pytest -q tests/test_real_videos.py
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

pytest.importorskip("ultralytics")

from visual_memory.video import inspect_video
from visual_memory.frames import extract_frames
from visual_memory.detect import detect_objects
from visual_memory.search import find_last_seen, draw_last_seen

PROJECT = Path(__file__).resolve().parents[1]
VIDEOS = PROJECT / "data" / "videos"

# What we know is in each video (from data/checklists/).
# "findable" = has a COCO label, so YOLO should find it at least once.
# "not_findable" = no COCO label, so search should say it can't look for it.
CHECKLIST = {
    "IMG_4203.mp4": {"findable": ["remote"], "not_findable": ["eyeglasses", "keys"]},
    "IMG_4206.mp4": {"findable": ["water bottle", "book"], "not_findable": ["keys"]},
    "IMG_4207.mp4": {"findable": ["water bottle"], "not_findable": ["sunglasses", "wired earbuds"]},
}


@pytest.fixture(scope="module", params=sorted(CHECKLIST))
def processed_video(request, tmp_path_factory):
    """Run steps 1-4 once per video and share the results with every test below."""
    video = VIDEOS / request.param
    if not video.exists():
        pytest.skip(f"{video.name} isn't in data/videos/")
    out = tmp_path_factory.mktemp(video.stem)
    info = inspect_video(video)
    frames = extract_frames(video, out / "frames", 2)
    detections = detect_objects(frames, out / "frames")
    (out / "detections.json").write_text(json.dumps(detections))
    return {"name": video.name, "info": info, "frames": frames, "detections": detections, "out": out}


def test_video_opens(processed_video):
    info = processed_video["info"]
    assert info["fps"] > 0 and info["width"] > 0 and info["height"] > 0
    assert 5 <= info["duration_seconds"] <= 60  # short room recordings


def test_frames_are_about_two_per_second(processed_video):
    frames = processed_video["frames"]
    expected = processed_video["info"]["duration_seconds"] * 2
    assert abs(len(frames) - expected) <= 2
    times = [f["timestamp_seconds"] for f in frames]
    assert times == sorted(times)


def test_detections_are_well_formed(processed_video):
    width, height = processed_video["info"]["width"], processed_video["info"]["height"]
    assert processed_video["detections"], "YOLO found nothing at all"
    for d in processed_video["detections"]:
        x1, y1, x2, y2 = d["xyxy"]
        assert 0 <= x1 < x2 <= width + 1 and 0 <= y1 < y2 <= height + 1
        assert 0.25 <= d["confidence"] <= 1


def test_findable_objects_are_found(processed_video):
    detections = processed_video["detections"]
    for item in CHECKLIST[processed_video["name"]]["findable"]:
        result = find_last_seen(detections, item)
        assert result["found"], f"{item} not found in {processed_video['name']}"
        last = result["last_seen"]
        # "Last seen" must really be the latest sighting of that label.
        latest = max(d["timestamp_seconds"] for d in detections if d["label"] == result["label"])
        assert last["timestamp_seconds"] == latest
        assert result["first_seen_seconds"] <= last["timestamp_seconds"]


def test_unsupported_objects_say_why(processed_video):
    for item in CHECKLIST[processed_video["name"]]["not_findable"]:
        result = find_last_seen(processed_video["detections"], item)
        assert not result["found"]
        assert "COCO" in result["note"]


def test_last_seen_image_is_drawn(processed_video):
    out = processed_video["out"]
    item = CHECKLIST[processed_video["name"]]["findable"][0]
    result = find_last_seen(processed_video["detections"], item)
    image_path = draw_last_seen(out / "frames", result["last_seen"], out / "search" / "last_seen.jpg")
    drawn = cv2.imread(str(image_path))
    original = cv2.imread(str(out / "frames" / f"frame_{result['last_seen']['source_frame_index']:06d}.jpg"))
    assert drawn.shape == original.shape
    assert (drawn != original).any()  # the box was actually drawn on


def test_find_object_script(processed_video):
    out = processed_video["out"]
    item = CHECKLIST[processed_video["name"]]["findable"][0]
    run = subprocess.run([sys.executable, str(PROJECT / "scripts" / "find_object.py"), str(out), item],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "Last seen at" in run.stdout
