# Room Memory — start here

This project includes the beginner-friendly Python code, tests, and object checklists. **Videos are not included in the ZIP.** You will add your own recordings to `data/videos/` before running the baseline.

## 1. Add your three videos

Copy the original recordings into `room-memory/data/videos/` using these exact filenames (keep the `.MOV` extension):

- `IMG_4203.MOV` — remote, glasses, keys
- `IMG_4206.MOV` — water bottle, book, keys
- `IMG_4207.MOV` — water bottle, glasses, earbuds

Do not put videos in `data/checklists/` (that folder contains the annotation files).

## 2. Install the project
Open a terminal inside `room-memory` and enter:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[detection,test]'
```

If you're using Windows, activate with `.venv\Scripts\activate` instead.

## 3. Check that the code works

```bash
python -m pytest -q
```

## 4. Run each real video

```bash
python scripts/run_baseline.py data/videos/IMG_4203.MOV --out data/outputs/video1
python scripts/run_baseline.py data/videos/IMG_4206.MOV --out data/outputs/video2
python scripts/run_baseline.py data/videos/IMG_4207.MOV --out data/outputs/video3
```

YOLO model weights may download on the first run, so internet access may be required once.

## 5. Read the results

For each run, look inside `data/outputs/video1` etc. at `summary.json`, `detections.json`, and the extracted `frames/`.

Compare the detected labels with the independent checklists in `data/checklists/`:

- IMG_4203.MOV — remote, eyeglasses, keys
- IMG_4206.MOV — water bottle, book, keys
- IMG_4207.MOV — water bottle, sunglasses, wired earbuds

**Important:** `summary.json` reports predictions, not true accuracy. Standard COCO YOLO11 has a class for remote, bottle and book, but no dedicated class for eyeglasses, keys, sunglasses, or earbuds. Manually check the frames and the prediction boxes before marking checklist items detected.

## 6. GitHub

This folder contains an initialized `.git` directory from the original student repo. Videos are intentionally ignored by `.gitignore`, so **they stay on your computer and aren't pushed to GitHub**. This is important for privacy and GitHub size limits. Create an empty GitHub repository, then run:

```bash
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

See `README.md` for what each source file does and how to compare FPS settings.
