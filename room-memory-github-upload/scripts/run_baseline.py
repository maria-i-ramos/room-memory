"""Run steps 1, 2, and 3 on a short video of a room."""
import argparse
import json
import sys
import time
from pathlib import Path

# Let this script find our code even before installing the project as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visual_memory.video import inspect_video
from visual_memory.frames import extract_frames
from visual_memory.detect import detect_objects


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Find objects in a short room video")
    parser.add_argument("video", help="Path to your video (for example data/videos/room.mp4)")
    parser.add_argument("--out", default="data/outputs/first_run")
    parser.add_argument("--fps", type=float, default=2, help="How many frames to save per second")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--skip-detection", action="store_true")
    args = parser.parse_args()

    output_folder = Path(args.out)
    output_folder.mkdir(parents=True, exist_ok=True)

    print("Step 1: Checking the video...")
    video_info = inspect_video(args.video)
    save_json(output_folder / "video_info.json", video_info)
    print(f"  {video_info['duration_seconds']:.1f} seconds at {video_info['fps']:.1f} FPS")

    print("Step 2: Saving frames...")
    frames_folder = output_folder / "frames"
    frames = extract_frames(args.video, frames_folder, args.fps)
    save_json(output_folder / "frames.json", frames)
    print(f"  Saved {len(frames)} frames")

    if args.skip_detection:
        print("Step 3: Skipped detection (use this to test steps 1 and 2).")
        return

    print("Step 3: Looking for objects using YOLO...")
    start = time.perf_counter()
    detections = detect_objects(frames, frames_folder, model_name=args.model)
    seconds = time.perf_counter() - start
    save_json(output_folder / "detections.json", detections)

    results = {
        "frames_processed": len(frames),
        "detections": len(detections),
        "labels_seen": sorted(set(item["label"] for item in detections)),
        "processing_seconds": round(seconds, 2),
        "model": args.model,
    }
    save_json(output_folder / "summary.json", results)
    print("Done!", results)
    print(f"Results saved to: {output_folder}")


if __name__ == "__main__":
    main()
