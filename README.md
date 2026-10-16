# room-memory
# Where Did My Stuff Go?

## About the Project

The idea for this project is to help people find things they've misplaced by using a video of their room. Eventually, we want someone to be able to ask something like "Where are my AirPods?" and have the system show where they were last seen.

## Current Status

The baseline pipeline works end to end:

| Step | What it does | Status |
|---|---|---|
| 1. Input video | Open a 10–30 second room recording with OpenCV | ✅ Done |
| 2. Extract frames | Save about 2 frames per second, with timestamps | ✅ Done |
| 3. Detect objects | Run a pretrained YOLO11 model on each frame | ✅ Done |
| 4. Save detections | Store label, confidence, timestamp, and box in `detections.json` | ✅ Done |
| 5. Search object | User types an object, e.g. "water bottle" | ✅ Done |
| 6. Return last-seen frame | Show the frame with the bounding box and timestamp drawn on | ✅ Done |

Everything runs from the command line for now. A UI (app or website) is the next big step after improving detection. 

## Quick Start

All the code lives in `room-memory-github-upload/`. See [its README](room-memory-github-upload/README.md) for the full guide.

```bash
cd room-memory-github-upload
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[detection,test]'

# Steps 1–4: process a video
python scripts/run_baseline.py data/videos/IMG_4206.mp4 --out data/outputs/video2

# Steps 5–6: find an object
python scripts/find_object.py data/outputs/video2 "water bottle"
open data/outputs/video2/search/last_seen_bottle.jpg

# Tests
python -m pytest -v
```

Only the `python3 -m venv` and `pip install` lines are one-time setup. Next time, just `cd` into the folder and run `source .venv/bin/activate`.
