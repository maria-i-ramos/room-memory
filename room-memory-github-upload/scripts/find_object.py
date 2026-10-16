"""Steps 5 and 6: ask "where is my X?" and get back the last frame it was seen in.

Example:
    python scripts/find_object.py data/outputs/video2 "water bottle"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visual_memory.search import load_detections, find_last_seen, draw_last_seen


def main():
    parser = argparse.ArgumentParser(description="Find where an object was last seen")
    parser.add_argument("results", help="Output folder from run_baseline.py (e.g. data/outputs/video1)")
    parser.add_argument("object", help='What to look for, e.g. "bottle" or "water bottle"')
    parser.add_argument("--min-confidence", type=float, default=0.25)
    args = parser.parse_args()

    results_folder = Path(args.results)
    detections = load_detections(results_folder)
    result = find_last_seen(detections, args.object, args.min_confidence)

    if not result["found"]:
        print(f'Not found: "{args.object}". {result["note"] or ""}'.strip())
        labels = sorted({d["label"] for d in detections})
        print("Objects YOLO saw in this video:", ", ".join(labels) or "none")
        return 1

    last = result["last_seen"]
    safe_name = result["label"].replace(" ", "_")
    image_path = draw_last_seen(results_folder / "frames", last, results_folder / "search" / f"last_seen_{safe_name}.jpg")
    result["image"] = str(image_path)
    (results_folder / "search" / f"last_seen_{safe_name}.json").write_text(json.dumps(result, indent=2))

    print(f'"{args.object}" -> YOLO label "{result["label"]}"')
    print(f'  Last seen at {last["timestamp_seconds"]:.1f}s (frame {last["source_frame_index"]}), '
          f'confidence {last["confidence"]:.2f}')
    print(f'  Box (x1, y1, x2, y2): {last["xyxy"]}')
    print(f'  Seen in {result["frames_with_object"]} sampled frames, first at {result["first_seen_seconds"]:.1f}s')
    print(f"  Image: {image_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
