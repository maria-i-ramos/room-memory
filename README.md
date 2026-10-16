# room-memory
# Where Did My Stuff Go?

## About the Project

The idea for this project is to help people find things they've misplaced by using a video of their room. Eventually, we want someone to be able to ask something like "Where are my AirPods?" and have the system show where they were last seen.

For now, we're just focusing on getting the first three steps working:

1. **Load a video** using OpenCV.
2. **Extract frames** from the video (2 frames per second).
3. **Detect objects** in the frames using a pretrained YOLO11 model.

This is our initial baseline before getting into tracking or figuring out exactly where objects are located.

## Project Files

- `src/visual_memory/` — Code for reading videos, getting frames, and detecting objects.
- `scripts/run_baseline.py` — Runs everything together.
- `tests/` — Some basic tests to make sure the code works.
- `data/videos/` — Where we put our test videos.
- `data/checklists/` — Lists of objects we know are in each video.
- `data/outputs/` — Where the results get saved.

## Getting Started

Open the project in VS Code and open a terminal.

**1. Set up Python (Mac)**

Make sure Python 3.10 or newer is installed.

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[detection,test]'
```

We only need to install everything once. When opening the project again, just run `source .venv/bin/activate`.

**2. Run the tests**

```bash
python -m pytest -q
```

**3. Run the videos**

We have three short recordings with different objects in them. Put them inside `data/videos/` using these filenames:

- `IMG_4203.mp4` — Remote, glasses, keys
- `IMG_4206.mp4` — Water bottle, book, keys
- `IMG_4207.mp4` — Water bottle, glasses, earbuds

Run each video seperately:

```bash
python scripts/run_baseline.py data/videos/IMG_4203.mp4 --out data/outputs/video1
python scripts/run_baseline.py data/videos/IMG_4206.mp4 --out data/outputs/video2
python scripts/run_baseline.py data/videos/IMG_4207.mp4 --out data/outputs/video3
```

The first run might take a little longer since YOLO has to download the model.

If we just want to check that the video and frame extraction works, we can add `--skip-detection` to the command.

## Results

After running the code, the results should show up in `data/outputs/`.

- `frames/` — Images taken from the video.
- `frames.json` — The frames and their timestamps.
- `detections.json` — Objects YOLO found, including confidence scores and bounding boxes.
- `summary.json` — A summary of the detections and how long it took.

For our first experiment, we're using three real videos and comparing YOLO's detections to the checklists we made. This should help us figure out which objects it can recognize and which ones it struggles with.

We also want to see if extracting more frames makes much of a difference. For example, we can compare 1 FPS and 5 FPS:

```bash
python scripts/run_baseline.py data/videos/IMG_4203.mp4 --fps 1 --out data/outputs/one_fps
python scripts/run_baseline.py data/videos/IMG_4203.mp4 --fps 5 --out data/outputs/five_fps
```

Right now, we're mostly looking at whether each object gets detected at least once. This isn't a full accuracy evaluation since we aren't manually drawing bounding boxes on every frame.

There are also some limitations. YOLO doesn't have categories for everything, especially smaller personal items like keys or earbuds. The current code also can't tell two objects of the same type apart or keep track of an object as the camera moves.

## Next Steps

Once we have some initial results, we can see what needs to be improved. I'm especially interested in how well it detects smaller objects since those are probaly the things someone would actually lose.

After that, we can work on object tracking and eventually letting users search for a specific item and see where it was last spotted.
