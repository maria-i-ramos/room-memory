"""Search a room's processed recordings in upload order."""
import re
from pathlib import Path

from visual_memory.movement import estimate_placement
from visual_memory.search import load_detections, resolve_query


def object_name(query):
    name = re.sub(
        r"^where (?:is|are|was|were|did i (?:last )?see) (?:my |the )?",
        "", " ".join(query.strip().rstrip("?").split()), flags=re.IGNORECASE,
    ).strip()
    if not name:
        raise ValueError("Enter an object name.")
    return name


def summarize_clip(detections, run, name, surfaces, min_confidence=0.25, *, use_masks=False, surface_references=None):
    label, _ = resolve_query(name, (item["label"] for item in detections))
    candidates = [item for item in detections
                  if item["label"] == label and item["confidence"] >= min_confidence]
    if not candidates:
        return None
    if label.casefold() in surfaces:
        raise ValueError("Choose an object label distinct from the surface labels.")

    placement = estimate_placement(detections, run, label, surfaces, min_confidence,
                                   use_masks=use_masks, surface_references=surface_references)
    if placement["status"] == "estimated":
        index = placement["evidence"]["source_frame_index"]
        candidates = [item for item in candidates if item["source_frame_index"] == index]
    detection = max(candidates, key=lambda item: (item["confidence"], item["timestamp_seconds"]))
    frame = run / "frames" / f"frame_{detection['source_frame_index']:06d}.jpg"
    return {
        "found": True,
        "label": label,
        "clip": run.name,
        "status": placement["status"],
        "location": placement.get("location"),
        "detection": detection,
        "frame": str(frame),
    }


def search_room(runs, query, surfaces, min_confidence=0.25, *, use_masks=False, surface_references=None):
    """Return the newest sighting, keeping location uncertainty in that clip."""
    name = object_name(query)
    if not 0 <= min_confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    surfaces = {surface.casefold() for surface in surfaces}

    for newer_clips, run in enumerate(reversed(runs)):
        run = Path(run).resolve()
        sighting = summarize_clip(load_detections(run), run, name, surfaces,
                                  min_confidence, use_masks=use_masks, surface_references=surface_references)
        if sighting:
            return {**sighting, "newer_clips_without_detection": newer_clips}
    return {"found": False, "label": name}
