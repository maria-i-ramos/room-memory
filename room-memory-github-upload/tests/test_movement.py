import json

from visual_memory.movement import estimate_placement, movement_history


def test_ambiguous_frame_cannot_veto_agreement_or_mask_a_clear_conflict(tmp_path):
    detections = [
        {"label": label, "xyxy": [5, 5, 8, 8] if label == "glasses" else [0, 0, 20, 20],
         "confidence": 0.9, "source_frame_index": frame, "timestamp_seconds": frame / 2}
        for frame, labels in enumerate([
            ["glasses", "bed"], ["glasses", "bed"],
            ["glasses", "glasses", "desk"], ["glasses", "desk"],
        ])
        for label in labels
    ]
    surfaces = {"bed", "desk"}
    estimate = estimate_placement(detections[:-2], tmp_path, "glasses", surfaces)
    assert estimate["location"] == "bed"
    assert estimate["supporting_frames"] == 2
    assert estimate["evidence"]["source_frame_index"] == 1

    conflict = estimate_placement(detections, tmp_path, "glasses", surfaces)
    assert "location" not in conflict

    weak_bed_evidence = [
        {**item, "confidence": 0.3} if item["source_frame_index"] in (0, 1)
        and item["label"] == "glasses" else item for item in detections[:-2]
    ]
    conflict = estimate_placement(weak_bed_evidence, tmp_path, "glasses", surfaces)
    assert conflict["status"] == "multiple matching objects"


def test_ordered_moves_retain_evidence_and_skip_uncertain_clips(tmp_path):
    folders = []
    for name, labels in [
        ("z", ["keys", "desk"]), ("y", []), ("x", ["keys", "bed", "desk"]),
        ("w", ["keys", "keys", "bed"]), ("v", ["keys", "bed"]), ("u", ["keys", "bed"]),
    ]:
        folder = tmp_path / name
        folder.mkdir()
        detections = [
            {"label": label, "xyxy": [5, 5, 8, 8] if label == "keys" else [0, 0, 20, 20],
             "confidence": 0.9, "source_frame_index": frame, "timestamp_seconds": frame / 2}
            for frame in range(2) for label in labels
        ]
        (folder / "detections.json").write_text(json.dumps(detections))
        folders.append(folder)
    result = movement_history(folders, "Keys", ["desk", "bed"])
    assert [s["status"] for s in result["sightings"]] == [
        "estimated", "object unseen", "location uncertain", "multiple matching objects",
        "estimated", "estimated",
    ]
    move, = result["possible_moves"]
    assert (move["from"], move["to"]) == ("desk", "bed")
    assert move["before"]["frame"] == str(folders[0] / "frames/frame_000001.jpg")
    assert move["after"]["frame"] == str(folders[4] / "frames/frame_000001.jpg")
