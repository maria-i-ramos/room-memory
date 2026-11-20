"""Find the newest sighting of an object across processed room clips."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visual_memory.last_seen import search_room
from visual_memory.search import draw_last_seen
from visual_memory.surface_references import SurfaceReferences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object", help='Object name or question, e.g. "Where are my glasses?"')
    parser.add_argument("runs", nargs="+", type=Path, help="Processed clip folders in oldest-first order")
    parser.add_argument("--surface", action="append", help="Surface label, repeatable")
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--use-masks", action="store_true", help="Try experimental surface scoring from YOLOE masks")
    parser.add_argument("--surface-references", type=Path, help="Folder with cropped bed and rug examples")
    parser.add_argument("--json", action="store_true", help="Print a result suitable for an app to consume")
    args = parser.parse_args()
    try:
        references = SurfaceReferences(args.surface_references) if args.surface_references else None
        result = search_room(args.runs, args.object, args.surface or ["desk", "bed", "dresser", "rug"],
                             args.min_confidence, use_masks=args.use_masks, surface_references=references)
        if result["found"]:
            frame = Path(result["frame"])
            safe_label = re.sub(r"[^\w-]+", "_", result["label"]).strip("_") or "object"
            image = frame.parent.parent / "search" / f"room_last_seen_{safe_label}.jpg"
            result["image"] = str(draw_last_seen(frame.parent, result["detection"], image))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(result, indent=2))
    elif not result["found"]:
        print(f'No sighting of "{result["label"]}" in these clips.')
    else:
        seconds = result["detection"]["timestamp_seconds"]
        label = result["label"]
        clip = result["clip"]
        if result["location"]:
            print(f'Last seen on the {result["location"]} in {clip} at {seconds:.1f}s.')
        elif result["status"] == "multiple matching objects":
            print(f'Multiple possible matches for "{label}" in {clip}. Check this candidate at {seconds:.1f}s.')
        else:
            print(f'Last seen in {clip} at {seconds:.1f}s. The location in the room is uncertain.')
        if result["newer_clips_without_detection"]:
            print(f'No matching detection in {result["newer_clips_without_detection"]} newer clip(s).')
        print(f'Evidence image: {result["image"]}')
    return 0 if result["found"] else 1


if __name__ == "__main__":
    sys.exit(main())
