"""Estimate placement for a single distinctive object across ordered recordings."""
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from visual_memory.search import load_detections
from visual_memory.surface_references import REFERENCE_SURFACES, support_points


def covers_object(surface, obj):
    """Require most of the object's image area to overlap a larger surface box."""
    x1, y1, x2, y2 = obj["xyxy"]
    a, b, c, d = surface["xyxy"]
    area = (x2 - x1) * (y2 - y1)
    overlap = max(0, min(c, x2) - max(a, x1)) * max(0, min(d, y2) - max(b, y1))
    return area > 0 and (c - a) * (d - b) > area and overlap >= 0.8 * area


def supporting_surfaces(items, obj, surfaces, use_masks):
    candidates = [item for item in items if item["label"].casefold() in surfaces]
    if not use_masks or (not obj.get("polygon") and not any(item.get("polygon") for item in candidates)):
        # Keep older saved runs usable.
        return {item["label"].casefold() for item in candidates if covers_object(item, obj)}

    x1, y1, x2, y2 = obj["xyxy"]
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return set()
    # The surface mask may have a hole where the object occludes it. Sample
    # nearby pixels below and beside the object instead of inside its box.
    points = support_points(obj["xyxy"])
    matches = set()
    for item in candidates:
        if not item.get("polygon"):
            continue
        contour = np.asarray(item["polygon"], dtype=np.float32)
        if cv2.contourArea(contour) <= width * height:
            continue
        covered = sum(cv2.pointPolygonTest(contour, point, False) >= 0 for point in points)
        if covered / len(points) >= 0.6:
            matches.add(item["label"].casefold())
    return matches


def placement(folder, query, surfaces):
    return estimate_placement(load_detections(folder), folder, query, surfaces)


def estimate_placement(detections, folder, query, surfaces, min_confidence=0.25, *, use_masks=False, surface_references=None):
    frames = defaultdict(list)
    for item in detections:
        if item["confidence"] >= min_confidence:
            frames[item["source_frame_index"]].append(item)

    votes = defaultdict(list)
    ambiguous_support = {}
    strongest_vote = defaultdict(float)
    seen = False
    ambiguous = False
    for index, items in sorted(frames.items()):
        objects = [item for item in items if item["label"].casefold() == query.casefold()]
        if not objects:
            continue
        seen = True
        multiple = len(objects) > 1
        ambiguous |= multiple
        frame = Path(folder) / "frames" / f"frame_{index:06d}.jpg"
        for obj in objects:
            matches = supporting_surfaces(items, obj, surfaces, use_masks)
            if surface_references and matches and matches <= REFERENCE_SURFACES:
                # The surrounding color cannot validate the object's identity.
                if obj["confidence"] < max(0.5, min_confidence):
                    continue
                label = surface_references.classify(frame, obj["xyxy"])
                matches = {label} if label in surfaces else set()
            if len(matches) != 1:
                continue
            surface = matches.pop()
            if multiple:
                ambiguous_support[surface] = max(ambiguous_support.get(surface, 0), obj["confidence"])
                continue
            strongest_vote[surface] = max(strongest_vote[surface], obj["confidence"])
            votes[surface].append({
                "frame": str(frame),
                "seconds": obj["timestamp_seconds"], "object_box": obj["xyxy"],
                "source_frame_index": index,
            })
    if len(votes) == 1:
        surface, evidence = next(iter(votes.items()))
        stronger_conflict = any(other != surface and confidence > strongest_vote[surface]
                                for other, confidence in ambiguous_support.items())
        if len(evidence) >= 2 and not stronger_conflict:
            return {"status": "estimated", "location": surface, "evidence": evidence[-1],
                    "supporting_frames": len(evidence)}
    if ambiguous:
        return {"status": "multiple matching objects"}
    return {"status": "location uncertain" if seen else "object unseen"}


def movement_history(folders, query, surfaces):
    surfaces = {name.casefold() for name in surfaces}
    if query.casefold() in surfaces:
        raise ValueError("Choose different prompts for the object and its supporting surfaces.")
    history, moves = [], []
    previous = None
    for folder in folders:
        sighting = {"run": str(folder), **placement(folder, query, surfaces)}
        history.append(sighting)
        if sighting["status"] != "estimated":
            continue
        if previous and previous["location"] != sighting["location"]:
            moves.append({"from": previous["location"], "to": sighting["location"],
                          "before": previous["evidence"], "after": sighting["evidence"]})
        previous = sighting
    return {"object": query, "sightings": history, "possible_moves": moves}
