# Where Did My Stuff Go? — Project Guide

Take a short video of a room and use it later to find misplaced objects. Ask "where is my water bottle?" and get back the frame where it was last seen, with a box drawn around it.

This folder has the code, tests, sample videos, and checklists. All the commands below are run from **this folder** (`room-memory-github-upload/`).

## How it works

```
Input video (10–30 s room recording)
   ↓  Step 1: video.py       open the video, read FPS / size / duration
Extract frames
   ↓  Step 2: frames.py      save about 2 frames per second as JPEGs with timestamps
Detect objects
   ↓  Step 3: detect.py      run pretrained YOLO11 on each frame
Save detections
   ↓  Step 4: run_baseline.py   write label, confidence, timestamp, box → detections.json
Search object
   ↓  Step 5: search.py      map "water bottle" → YOLO label "bottle"; find matches
Return last-seen frame
      Step 6: search.py      pick the latest confident sighting; draw box + timestamp
```

**Step 1: open the video** (`video.py`). Checks that the file exists, opens it with OpenCV, reads the FPS and frame count, and decodes the first frame to make sure the video isn't empty. It takes the width and height from that frame, which handles rotated phone videos better.

**Step 2: extract frames** (`frames.py`). Reads every frame and works out its time as frame number ÷ FPS. It saves a frame each time that time reaches the next 0.5 s mark (at 2 FPS). Each saved frame gets a JPEG and an entry in `frames.json` with its filename, timestamp, and frame number.

**Step 3: detect objects** (`detect.py`). Loads YOLO (`yolo11n.pt` by default; it downloads on the first run) and runs it on every saved frame, keeping boxes with confidence ≥ 0.25.

**Step 4: save detections** (`run_baseline.py`). Writes every box to `detections.json`, plus a short `summary.json`.

**Step 5: search** (`search.py`). Turns everyday words into YOLO label names ("water bottle" → `bottle`, "phone" → `cell phone`). If an object has no YOLO category (keys, glasses, earbuds), it says so instead of silently returning nothing.

**Step 6: last seen** (`search.py`). Among detections with confidence ≥ 0.25, picks the one with the latest timestamp. If that frame has two boxes for the object, it uses the more confident one. It draws the box and timestamp on that frame.

## Files

```
room-memory-github-upload/
├── README.md                this guide
├── START_HERE.md            original setup notes
├── pyproject.toml           project name + libraries to install
├── .gitignore               keeps outputs, .venv, model weights out of Git
├── src/visual_memory/
│   ├── video.py             step 1
│   ├── frames.py            step 2
│   ├── detect.py            step 3
│   └── search.py            steps 5–6
├── scripts/
│   ├── run_baseline.py      runs steps 1–4 on one video
│   └── find_object.py       runs steps 5–6 on a results folder
├── tests/
│   ├── conftest.py          lets tests import the code in src/
│   ├── test_pipeline.py     steps 1–2 on a tiny generated video (fast)
│   ├── test_search.py       steps 5–6 on made-up detections (fast)
│   └── test_real_videos.py  steps 1–6 on our 3 real videos (~20 s, needs YOLO)
└── data/
    ├── videos/              IMG_4203.mp4, IMG_4206.mp4, IMG_4207.mp4
    ├── checklists/          what's really in each video + whether YOLO found it
    └── outputs/             results (not committed to Git)
```

## Setup (first time only)

Use **Python 3.10 or newer**.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[detection,test]'
```

- `python3 -m venv .venv` makes a private Python just for this project, inside the `.venv` folder.
- `source .venv/bin/activate` switches the current Terminal window to that Python. Your prompt will show `(.venv)`.
- `pip install -e '.[detection,test]'` installs OpenCV and NumPy, plus `ultralytics` (YOLO, which brings in PyTorch) and `pytest`.
  - `-e` means edits to `src/` take effect without reinstalling.
  - The quotes stop the Mac shell from treating the brackets as a special pattern.

On Windows, activate with `.venv\Scripts\activate` instead.

**Every time after that:**

```bash
cd room-memory-github-upload
source .venv/bin/activate
```

## Running it

**Steps 1–4: process a video**

```bash
python scripts/run_baseline.py data/videos/IMG_4203.mp4 --out data/outputs/video1
python scripts/run_baseline.py data/videos/IMG_4206.mp4 --out data/outputs/video2
python scripts/run_baseline.py data/videos/IMG_4207.mp4 --out data/outputs/video3
```

Options:

| Option | What it does |
|---|---|
| `--fps 1` | Change how many frames are saved per second (default 2) |
| `--model yolo11s.pt` | Use a different YOLO model (default `yolo11n.pt`) |
| `--skip-detection` | Run only steps 1–2 (no YOLO download needed) |

**Steps 5–6: find an object**

```bash
python scripts/find_object.py data/outputs/video2 "water bottle"
```

```
"water bottle" -> YOLO label "bottle"
  Last seen at 4.7s (frame 14), confidence 0.78
  Box (x1, y1, x2, y2): [0.6, 96.1, 196.6, 701.3]
  Seen in 4 sampled frames, first at 3.0s
  Image: data/outputs/video2/search/last_seen_bottle.jpg
```

Use `--min-confidence 0.4` to ignore weak detections.

## Where the results go

Each run creates a folder like `data/outputs/video2/`:

| File | What's in it |
|---|---|
| `video_info.json` | FPS, width, height, duration |
| `frames/` | the saved JPEG frames |
| `frames.json` | each frame's filename and timestamp |
| `detections.json` | **every object found**: label, confidence, timestamp, frame number, box |
| `summary.json` | number of frames and detections, object types seen, run time (no confidences) |
| `search/last_seen_<label>.jpg` | the last-seen frame with the box drawn on |
| `search/last_seen_<label>.json` | the search result as data |

To open them on a Mac: `open data/outputs/video2/detections.json` or `open data/outputs/video2/frames`.

To get a quick table of objects and confidence scores for one video:

```bash
python -c "
import json, sys, collections
d = json.load(open(sys.argv[1]))
by = collections.defaultdict(list)
for x in d: by[x['label']].append(x['confidence'])
print(f\"{'label':<12}{'boxes':>6}{'best':>7}{'avg':>7}\")
for k, c in sorted(by.items(), key=lambda kv: -max(kv[1])):
    print(f'{k:<12}{len(c):>6}{max(c):>7.2f}{sum(c)/len(c):>7.2f}')
" data/outputs/video1/detections.json
```

## Tests

```bash
python -m pytest -v                            # all 33 tests, one line each
python -m pytest -v tests/test_real_videos.py  # just the real-video tests
python -m pytest -v | tee test_results.txt     # also save the output to a file
```

Test results print in the Terminal; they aren't saved anywhere unless you use `tee` (or `pytest-html` for a browsable report).

| Test file | Checks | Needs YOLO? |
|---|---|---|
| `test_pipeline.py` | Step 1 reads the video correctly; step 2 saves the right frames and timestamps | No |
| `test_search.py` | Aliases, plurals, latest sighting, confidence cutoff, unsupported objects, drawing the box | No |
| `test_real_videos.py` | On each real video: it opens, ~2 frames/s are saved, boxes are inside the image, checklist objects are found, unsupported ones explain why, the last-seen image is drawn, `find_object.py` works | Yes (skips itself if YOLO or the videos are missing) |

## Results

All runs use a minimum confidence of 0.25. In the "Really is" columns below, we checked the main labels by looking at the frames.

### Iteration 1: YOLO11n, 2 FPS

| Video | Checklist object | COCO class? | Found? | Notes |
|---|---|---|---|---|
| IMG_4203 | remote | yes | ✅ | 10/32 frames, best 0.97 |
| | eyeglasses | no | ❌ | Boxed but labeled "mouse" (0.69) |
| | keys | no | ❌ | Boxed but labeled "scissors" (0.94) / "airplane" (0.75) |
| IMG_4206 | water bottle | yes | ✅ | 4/31 frames, best 0.82 |
| | book | yes | ✅ | 4/31 frames, best only 0.44 |
| | keys | no | ❌ | Not boxed |
| IMG_4207 | water bottle | yes | ✅ | 8/31 frames, best 0.83; also labeled "vase" (0.84) |
| | sunglasses | no | ❌ | Not boxed |
| | wired earbuds | no | ❌ | Not boxed |

**Score: 4/4 supported objects found, 0/5 unsupported.**

All objects YOLO11n reported (95 boxes):

| Video | YOLO label | Boxes | Best | Avg | Really is |
|---|---|---|---|---|---|
| IMG_4203 | remote | 10 | 0.97 | 0.83 | ✅ remote |
| | scissors | 15 | 0.94 | 0.79 | ❌ keys |
| | airplane | 2 | 0.75 | 0.58 | ❌ keys |
| | mouse | 6 | 0.69 | 0.49 | ❌ eyeglasses |
| | person | 28 | 0.69 | 0.46 | feet/hands at the frame edge |
| | keyboard | 1 | 0.45 | 0.45 | unclear |
| IMG_4206 | bottle | 4 | 0.82 | 0.72 | ✅ water bottle |
| | book | 4 | 0.44 | 0.33 | ✅ book |
| | person | 2 | 0.38 | 0.34 | feet/hands |
| IMG_4207 | vase | 4 | 0.84 | 0.49 | ❌ water bottle |
| | bottle | 8 | 0.83 | 0.58 | ✅ water bottle |
| | tv | 5 | 0.56 | 0.46 | ✅ TV in background |
| | microwave | 3 | 0.34 | 0.31 | false alarm |
| | spoon / cup / sink | 1 each | ≤ 0.40 | | false alarms |

### Iteration 2: frame rate (YOLO11n)

The source videos are 3 FPS, so "3 FPS" means every frame. Each cell shows frames with the object / frames processed.

| Object | 1 FPS | 2 FPS | 3 FPS |
|---|---|---|---|
| remote (4203) | 6/16 | 10/32 | 15/48 |
| bottle (4206) | 2/16 | 4/31 | 6/46 |
| book (4206) | 1/16 | 4/31 | 9/46 |
| bottle (4207) | 5/16 | 8/31 | 13/47 |
| Time per video | ~3.5 s | 4.4–6.0 s | 5.1–5.4 s |

Times include about 2–3 s to load the model. Every supported object was found at every rate, but at 1 FPS the book appeared in only one frame, so we use **2 FPS**.

### Iteration 3: YOLO11s vs YOLO11n, 2 FPS

```bash
python scripts/run_baseline.py data/videos/IMG_4203.mp4 --model yolo11s.pt --out data/outputs/video1_s
```

| Video | Object | YOLO11n | YOLO11s |
|---|---|---|---|
| IMG_4203 | remote | ✅ remote (0.97) | ✅ remote (0.94) |
| | eyeglasses | "mouse" ×6 (0.69) | "mouse" ×17 (0.86) |
| | keys | "scissors" ×15 (0.94) | "scissors" ×12 (0.93) |
| IMG_4206 | water bottle | ✅ bottle (0.82) | "cup" ×9 (0.86); "bottle" ×1 (0.35) |
| | book | ✅ book (0.44) | ❌ labeled "tv" (0.53) |
| | keys | not boxed | not boxed |
| IMG_4207 | water bottle | ✅ bottle (0.83) + "vase" | ✅ bottle (0.90), no "vase" |
| | sunglasses | not boxed | "cat" (0.66) / "spoon" (0.62) / "scissors" (0.48) |
| | wired earbuds | not boxed | "mouse" (0.52) |

| | YOLO11n | YOLO11s |
|---|---|---|
| Supported objects found under the right name | **4/4** | 2/4 |
| Checklist objects boxed at all, under any name | 6/9 | **8/9** |
| "Person" noise | 30 boxes | 4 boxes |
| Time per video | 4.4–6.0 s | 4.0–7.0 s |

YOLO11s **finds** more objects (sunglasses and earbuds are boxed for the first time) but **names** them worse. For search, we're keeping YOLO11n for now.

## Problems with the YOLO model

1. **Only 80 object types.** Pretrained YOLO uses the COCO categories, which don't include keys, glasses, sunglasses, or earbuds.
2. **Confident wrong labels.** There's no "unknown" option, so unknown objects get the nearest category, often with high confidence (keys → "scissors" at 0.94). A confidence cutoff can't remove these mistakes.
3. **Unstable labels.** The same water bottle came back as "bottle", "vase", or "cup".
4. **Missed and flickering detections.** Each frame is checked separately, so objects appear in some frames and not others.
5. **Small and thin objects.** YOLO11n completely missed the sunglasses, earbuds, and the keys in IMG_4206.
6. **Objects cut off at the edge.** Many boxes touch the left edge of the image, so "last seen" is often a half-visible view.
7. **Noise.** Feet and hands get boxed as "person", plus a few low-confidence false alarms.
8. **Unusual camera angle.** COCO photos are mostly from eye level; our clips look down at a dim tabletop from a moving phone. This likely lowers confidence, though we haven't tested it.

## Next steps

| Priority | Change | Fixes |
|---|---|---|
| 1 | Try **YOLO-World** (open vocabulary) with our own names: "keys", "sunglasses", "earbuds", "water bottle" | Problems 1–3 |
| 2 | Search clean-up: ignore "person", treat bottle/cup/vase as one, cutoff of about 0.4 | Problems 3 and 7 |
| 3 | Prefer last-seen boxes that don't touch the edge of the image | Problem 6 |
| 4 | Record more test videos with more objects, rooms, and lighting | Three clips are too few to trust the numbers |
| 5 | Fine-tune YOLO on our own items | Problems 1–5, if YOLO-World isn't enough |
| 6 | Track objects across frames | Problem 4; telling duplicates apart |

**Toward an app or website**

1. One shared database (SQLite) of detections across all videos, with when each video was recorded, so "last seen" can mean a real date and time.
2. A Streamlit prototype: upload a video, type an object, see the last-seen frame.
3. A FastAPI backend (upload and search endpoints) with a website or phone app.

## Limitations

- Timestamps are estimated from frame number ÷ FPS. Phone videos can have variable frame rates, so they're approximate.
- "Last seen" currently means within one video, not across recordings.
- Two objects of the same type can't be told apart.
- These results come from three 16-second clips. They're a first check, not a measure of accuracy. For true precision and recall, we'd need to draw correct boxes on some frames and compare them with the model's boxes.

## Git

- `data/outputs/`, `.venv/`, `*.pt`, and cache folders are ignored by `.gitignore`.
- Before committing: run `git status`, `git add` specific files (not `git add .`), and check `git diff --cached --stat`.
- Don't commit new personal room recordings without agreeing as a team.
