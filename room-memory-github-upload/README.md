# Where Did My Stuff Go?

## What we're building

The idea is pretty simple: take a short video of a room and later use it to help find misplaced objects. Eventually, I want someone to be able to ask "Where are my AirPods?" and see where they were last visible.

For now, I'm starting with just **three steps**. I want to see whether I can get object detection working on a real video before adding tracking or anything more complicated.

1. **Open a video** and check that Python can read it.
2. **Save a few frames** from the video (2 per second for now).
3. **Run YOLO** on those frames to identify objects it recognizes.

This is an early baseline. It doesn't track objects yet or understand locations like "next to my laptop." It also won't necessarily recognize AirPods: the pretrained YOLO model has a limited list of object categories.

## Files you'll actually work with

- `src/visual_memory/video.py` — step 1, opening a video
- `src/visual_memory/frames.py` — step 2, saving frames with timestamps
- `src/visual_memory/detect.py` — step 3, YOLO detections
- `scripts/run_baseline.py` — runs everything in order
- `tests/test_pipeline.py` — checks that the first two steps run correctly
- `data/videos/` — put your own room recording here
- `data/outputs/` — see the saved frames and results here

## Getting started

Use **Python 3.10 or newer**. Open the project folder in your terminal, then run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[detection,test]'
```

On Windows, activate the environment with `.venv\\Scripts\\activate` instead.

Record about 10–30 seconds of your room on your phone and copy the file into `data/videos/` as `room.mp4`. Then run:

```bash
python scripts/run_baseline.py data/videos/room.mp4
```

The first YOLO run might download the model weights, so it can take a bit longer. If you don't want to download YOLO yet, try steps 1 and 2 alone:

```bash
python scripts/run_baseline.py data/videos/room.mp4 --skip-detection
```

To run the automated tests:

```bash
python -m pytest -q
```

## Where do the results go?

After running the script, check `data/outputs/first_run/`:

- `frames/` — screenshots extracted from the video
- `frames.json` — each saved frame and its timestamp
- `detections.json` — detected objects and their boxes (only after YOLO runs)
- `summary.json` — number of frames, detections, labels, and runtime

**Important:** A detection count is not accuracy! One bottle appearing in five frames might produce five detections.

## Simple baseline experiment

My first test will be on a real recording of my room. I'll write down a short checklist of objects that are visible (for example, a bottle, laptop, and backpack), run the script, then compare what YOLO finds with my checklist. I also want to see which objects it misses.

I can try different frame rates to see whether sampling more frequently is worth the extra processing time:

```bash
python scripts/run_baseline.py data/videos/room.mp4 --fps 1 --out data/outputs/one_fps
python scripts/run_baseline.py data/videos/room.mp4 --fps 2 --out data/outputs/two_fps
python scripts/run_baseline.py data/videos/room.mp4 --fps 5 --out data/outputs/five_fps
```

I could also try a larger YOLO model using `--model yolo11s.pt`.

A checklist gives me a rough count of **objects spotted at least once**, after I manually check the output. It does **not** give me proper bounding-box precision or recall. For that, I'd need to draw correct boxes on some of the frames and compare them with the model's boxes.

| Video | Model | Saved FPS | Objects on checklist | Found at least once | Time (seconds) |
|---|---|---|---|---|---|
| room1 | YOLO11n | 1 | Not tested yet | Not tested yet | Not tested yet |
| room1 | YOLO11n | 2 | Not tested yet | Not tested yet | Not tested yet |
| room1 | YOLO11n | 5 | Not tested yet | Not tested yet | Not tested yet |

This gives us a starting point for deciding whether we need tracking, a different model, or better handling of small objects.

## A couple of limitations

- The timestamps are estimated using frame number and FPS. Phone videos can have variable frame rates, so these aren't guaranteed exact timestamps.
- Pretrained YOLO doesn't have labels for every object people lose.
- If the same type of object appears more than once, this version can't tell them apart.
- Don't upload personal room videos to GitHub. The `.gitignore` is set up to keep recordings and results out of commits.

## Pushing to GitHub

The ZIP includes Git history. After unzipping, make a new empty repo on GitHub and run:

```bash
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```
