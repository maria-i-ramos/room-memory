# Where Did My Stuff Go?

Room Memory is a local web app for finding where you last saw a belonging. Upload a short room video, then ask a question like **“Where are my glasses?”** The app searches saved sightings and shows the recording, an estimated surface when available, and a highlighted frame.

New uploads are sampled at two frames per second and processed in the background with **YOLOE**. The app stores sightings in **SQLite**. With optional **CLIP** matching, you can name a specific belonging and save reference photos. An uncertain match appears for review without replacing the last reliable location, and you can dismiss it with **Not this item**.

## Quick start

On a Mac, install **Python 3.10 or newer** and **Git**. From the repository root:

```bash
cd room-memory-github-upload
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[detection,web]'
.venv/bin/python scripts/room_app.py
```

Open [Room Memory](http://127.0.0.1:8000). The install command brings in the Python dependencies, and the first upload can download the YOLOE checkpoint and its text encoder. The default pipeline runs on CPU. SQLite comes with Python, and SAM3 weights aren't required.

- [Enable CLIP matching for named belongings](room-memory-github-upload/START_HERE.md#remember-a-specific-belonging)
- [Set up and run the web app](room-memory-github-upload/START_HERE.md#run-room-memory-in-a-browser)
- [Replay the six original movement recordings](room-memory-github-upload/START_HERE.md#replay-the-six-original-movement-recordings)
- [Replay the four cross-room recordings](room-memory-github-upload/START_HERE.md#replay-the-cross-room-recordings)
- [Results so far](room-memory-github-upload/README.md#results-so-far)
- [Cross-room and red-bottle-only results](room-memory-github-upload/CROSS_ROOM_RESULTS.md)
- [Original YOLO baselines and later model comparisons](room-memory-github-upload/README.md)

The app runs locally for one user. It can search clips recorded in different rooms, although room names are only in the evaluation checklist today. The repository includes the six original movement recordings, IMG_2133–IMG_2138. New uploads, generated evidence, model weights, and the SQLite database stay local.
