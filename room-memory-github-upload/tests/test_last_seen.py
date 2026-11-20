import json

from visual_memory.last_seen import search_room


def test_last_seen_respects_clip_order_and_does_not_reuse_old_location(tmp_path):
    old = tmp_path / "z_old"
    recent = tmp_path / "a_recent"
    empty = tmp_path / "m_empty"
    for folder in (old, recent, empty):
        folder.mkdir()

    def detection(label, frame, seconds, box):
        return {"label": label, "source_frame_index": frame, "timestamp_seconds": seconds,
                "xyxy": box, "confidence": 0.9}

    glasses = [5, 5, 8, 8]
    desk = [0, 0, 20, 20]
    (old / "detections.json").write_text(json.dumps([
        detection("eyeglasses", 20, 10, glasses), detection("desk", 20, 10, desk),
        detection("eyeglasses", 21, 11, glasses), detection("desk", 21, 11, desk),
        detection("eyeglasses", 22, 12, glasses),
    ]))
    (recent / "detections.json").write_text(json.dumps([
        detection("glasses", 0, 0, glasses),
        {**detection("glasses", 1, 0.5, glasses), "confidence": 0.3},
    ]))
    (empty / "detections.json").write_text("[]")

    query = "Where are my glasses?"
    earlier = search_room([old, empty], query, ["desk"])
    assert earlier["location"] == "desk"
    assert earlier["newer_clips_without_detection"] == 1
    assert earlier["frame"].endswith("frame_000021.jpg")

    latest = search_room([old, recent, empty], query, ["desk"])
    assert latest["clip"] == "a_recent"
    assert latest["status"] == "location uncertain"
    assert latest["location"] is None
    assert latest["detection"]["source_frame_index"] == 0
    assert search_room([empty], query, ["desk"])["found"] is False
