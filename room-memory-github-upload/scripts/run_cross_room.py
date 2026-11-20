"""Replay the four cross-room recordings with the web app's YOLOE prompts."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CHECKLIST = PROJECT / "data/checklists/cross_room_recordings.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True, help="Folder holding the original MOV files")
    parser.add_argument("--out", type=Path, default=PROJECT / "data/outputs/cross-room-replay")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    recordings = json.loads(CHECKLIST.read_text())
    files = [args.videos / recording["file"] for recording in recordings]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        parser.error("Missing reference recordings: " + ", ".join(missing))

    for recording, path in zip(recordings, files):
        print(f"Processing {path.name} ({recording['room']})", flush=True)
        subprocess.run([
            sys.executable, str(PROJECT / "scripts/run_baseline.py"), str(path.resolve()),
            "--out", str((args.out / path.stem).resolve()), "--device", args.device,
            "--include-bottle-candidates",
        ], cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
