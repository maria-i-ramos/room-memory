# Room Memory setup and experiments

## Run Room Memory in a browser

On a Mac, install Python 3.10 or newer and Git. From the repository root, run `cd room-memory-github-upload`. Create a virtual environment with `python3 -m venv .venv` if needed, then run:

```bash
.venv/bin/python -m pip install -e '.[detection,web]'
.venv/bin/python scripts/room_app.py
```

Open [Room Memory](http://127.0.0.1:8000). Choose **Add a recording**, select a short room video, and enter when you recorded it. Once its status is **ready**, search for a belonging. New uploads use YOLOE on CPU by default. The first run may download the YOLOE checkpoint and text encoder. Keep the server running while processing. The app logs processing failures under `data/room-app/runs/`.

The local database, uploads, and evidence are stored under `data/room-app/`. Use `--data-dir` for a separate test database. Use `--import-runs` with saved output folders in oldest-first order to index them without running the detector. Imports have no recording date and are ordered by the supplied list. Dated uploads are ordered by recording time and take precedence over imported history. For identity matching, see [Remember a specific belonging](#remember-a-specific-belonging).

## Replay the six original movement recordings

The original movement set is `IMG_2133.MOV` through `IMG_2138.MOV`, recorded in the same room. It tests changing placements for glasses, keys, the remote, camera, water bottle, and card holder, with a stationary medicine bottle as a control. The [placement checklist](data/checklists/movement_review.md) gives the expected surfaces and frame evidence.

The six MOVs are included in `data/videos/`. From the project folder, run the current YOLOE pipeline in recording order:

```bash
for id in 2133 2134 2135 2136 2137 2138; do
  .venv/bin/python scripts/run_baseline.py "data/videos/IMG_${id}.MOV" \
    --device cpu --include-bottle-candidates \
    --out "data/outputs/movement-current-yoloe/IMG_${id}" || break
done
```

To inspect those results in a separate app database, start the app with:

```bash
.venv/bin/python scripts/room_app.py --port 8001 \
  --data-dir data/room-app/movement-replay \
  --import-runs data/outputs/movement-current-yoloe/IMG_{2133,2134,2135,2136,2137,2138}
```

Open [the movement replay app](http://127.0.0.1:8001). Search returns the available evidence and estimated surface. Compare these with the checklist, including missed or uncertain sightings. Generated outputs stay local. The older `IMG_4569.mp4`–`IMG_4574.mp4` checklist clips are a separate historical set covered below.

## Replay the cross-room recordings

[The recording checklist](data/checklists/cross_room_recordings.json) lists the expected placements and the original bottle's absence from the last clip.

| Recording | Room | What it tests |
|---|---|---|
| IMG_2367.MOV | Original room | Glasses, keys, and the original bottle in the familiar room |
| IMG_2368.MOV | Second room | The same items in a new room |
| IMG_2369.MOV | Second room | The original bottle and a similar red bottle both appear |
| IMG_2370.MOV | Original room | Only the red bottle is present, so it should never replace the original's sighting |

Keep the MOVs in a local folder. If they are in Downloads, run:

```bash
.venv/bin/python scripts/run_cross_room.py --videos ~/Downloads
.venv/bin/python scripts/room_app.py --port 8001 \
  --data-dir data/room-app/cross-room-replay \
  --import-runs data/outputs/cross-room-replay/IMG_{2367,2368,2369,2370}
```

Open [the replay app](http://127.0.0.1:8001) to inspect the label-based sightings. The replay script uses the same YOLOE defaults as uploads, including the extra generic bottle prompt used when a bottle is enrolled. The room names are evaluation notes, not app fields. Importing does not recreate the earlier enrollment or re-ID scores automatically. See [the cross-room results](CROSS_ROOM_RESULTS.md) for the original identity experiment and [the re-ID report](REID_RESULTS.md) for its setup.

## Historical SAM 3 setup

### 1. Put the clips and checklists together

The historical checklist set is `IMG_4569.mp4` through `IMG_4574.mp4`. These older clips and their `data/checklists/IMG_..._objects.json` files are already tracked. New clips, model weights, and new outputs are ignored.

SAM 3 uses each checklist's `object` values as text prompts, including names such as “Apple Pencil” and “keys”. A new clip can use the same checklist format or explicit prompts, e.g. `--prompt keys --prompt "Apple Pencil"`. Use the same prompt list across clips if you want to test both presence and absence.

To rerun a historical clip with the current YOLOE detector instead, omit `--model sam3.pt`. Pass explicit `--prompt` options for items outside the app's default list. For example, compare the three prompts in the IMG_4572 checklist:

```bash
.venv/bin/python scripts/run_baseline.py data/videos/IMG_4572.mp4 \
  --prompt card --prompt keys --prompt "Apple Pencil" \
  --out data/outputs/checklist-yoloe/IMG_4572
```

### 2. Set up the environment and checkpoint

From this project folder, create the environment with `python3 -m venv .venv` if needed, then run:

```bash
source .venv/bin/activate
python -m pip install -e '.[detection,test]'
```

Sign in and request access on [Meta's SAM 3 model page](https://huggingface.co/facebook/sam3). Once approved, download `sam3.pt` from the Files tab and place it in this folder alongside `pyproject.toml`. To keep it elsewhere, pass its path with `--model`.

If the checkpoint is in Downloads, pass `--model ~/Downloads/sam3.pt`. On a compatible Apple Silicon Mac, `--device mps` enables GPU acceleration.

### 3. Run the six clips

```bash
for id in 4569 4570 4571 4572 4573 4574; do
  python scripts/run_baseline.py "data/videos/IMG_${id}.mp4" \
    --model sam3.pt --out "data/outputs/sam3/IMG_${id}" || break
done
```

The default sampling rate is two frames per second. Each output folder contains the sampled JPEGs, frame timestamps, detection boxes and scores, and a summary listing the model and prompts. Use a fresh output folder for each experiment. To reproduce the earlier baseline, pass `--model yolo11n.pt` and use a separate output folder.

## Ask where an object was last seen

From this project folder with the virtual environment active, try the existing six movement runs:

```bash
python scripts/where_is.py "Where are my glasses?" \
  data/outputs/movement-sam3/IMG_213{3,4,5,6,7,8}
```

Pass processed clip folders from oldest to newest for the same room. The command searches the saved results and writes a highlighted JPEG in the matching clip's `search/` folder. Use `--json` to get the clip name, timestamp, location status, and image path for a future web interface. The default surfaces are desk, bed, dresser, and rug. To use different surfaces, pass each label with `--surface`.

To add a recording with the current default YOLOE model, process it once and append its output folder to the search command:

```bash
python scripts/run_baseline.py "/path/to/new-room-clip.MOV" \
  --out data/outputs/new-room-clip
```

Queries use the newest clip with a matching detection. When the object is detected but the surface is uncertain, the command shows that clip and its evidence rather than reporting a location from an older clip. Frames with multiple matching boxes are skipped when estimating the surface. At least two clear frames must agree, with no clear votes for a different surface. A skipped frame can still block the estimate if it contains a higher-confidence candidate on a different surface. If there is still no estimate and some frames had multiple matches, the command flags the candidate for review. A label like “eyeglasses” cannot distinguish two pairs across recordings. For the prototype, use distinctive prompts consistently when recording more than one similar item, such as “red glasses” and “black glasses.”

To evaluate the search, run the query with successively longer prefixes of the same ordered list. Compare each response's object, surface, and clip against the [placement notes](data/checklists/movement_review.md). Querying the saved clips runs no model.

## Check where an object moved

```bash
python scripts/find_object.py data/outputs/sam3/IMG_4571 keys
python scripts/find_object.py data/outputs/sam3/IMG_4572 keys
```

Open each clip's `search/last_seen_keys.jpg` and compare the box and its surroundings with your placement notes. Timestamps start at zero for each recording. Record the expected location, whether the box covers the right object, and any missed or incorrect sightings. A checklist entry supplies a prompt and does not guarantee a match.

## Compare placements in IMG_2133 through IMG_2138

These six MOV files belong in `data/videos/`. Their embedded creation times put them in ascending filename order. The eyeglasses are a first test object. The confirmed placement sequence is desk → bed → white dresser → desk. See the [movement answer key](data/checklists/movement_review.md) for frames and timestamps. Use the same prompts in every clip so the detector can search all candidate surfaces independently.

After placing `sam3.pt` alongside `pyproject.toml`, run:

```bash
for id in 2133 2134 2135 2136 2137 2138; do
  python scripts/run_baseline.py "data/videos/IMG_${id}.MOV" \
    --model sam3.pt --out "data/outputs/movement-sam3/IMG_${id}" \
    --prompt eyeglasses --prompt "water bottle" --prompt "remote control" --prompt "compact camera" --prompt "card holder" --prompt keys --prompt "medicine bottle" \
    --prompt desk --prompt bed --prompt dresser --prompt rug || break
done

python scripts/find_moves.py eyeglasses data/outputs/movement-sam3/IMG_213{3,4,5,6,7,8} \
  --surface desk --surface bed --surface dresser --surface rug
```

The report includes a sighting status for every clip and a before/after frame path for each suggested move. It estimates a surface when its box covers at least 80% of the object box in two or more sampled frames, with no competing single-surface votes. Camera perspective can create overlap. Review both frames to verify physical support and identity. Frames with multiple matching objects do not vote on the placement. An unseen object leaves its last known placement intact. Identical-looking objects appearing in separate clips still need human confirmation.

## Try YOLOE on the same recordings

YOLOE accepts the same object prompts. This experiment uses YOLOE-11 large, a confidence threshold of 0.25 and two sampled frames per second. The first run downloads the checkpoint and its text encoder.

```bash
for id in 2133 2134 2135 2136 2137 2138; do
  python scripts/run_baseline.py "data/videos/IMG_${id}.MOV" \
    --model yoloe-11l-seg.pt --out "data/outputs/movement-yoloe/IMG_${id}" \
    --prompt eyeglasses --prompt "water bottle" --prompt "remote control" \
    --prompt "compact camera" --prompt "card holder" --prompt keys \
    --prompt "medicine bottle" --prompt desk --prompt bed --prompt dresser --prompt rug || break
done

python scripts/find_moves.py eyeglasses data/outputs/movement-yoloe/IMG_213{3,4,5,6,7,8} \
  --surface desk --surface bed --surface dresser --surface rug
```

See [YOLOE_RESULTS.md](YOLOE_RESULTS.md) for the two prompt experiments and reviewed results.

## Try FastSAM on the same sampled frames

FastSAM segments candidate regions and CLIP selects the closest region for each text prompt. A selected region can be a false match when the object is absent. The experiment saves `selections.json`, including CLIP similarity and segmentation confidence, for visual review.

```bash
python scripts/run_fastsam.py data/outputs/movement-yolo/IMG_213{3,4,5,6,7,8} \
  --model FastSAM-x.pt --out data/outputs/movement-fastsam \
  --prompt glasses --prompt bottle --prompt remote --prompt camera \
  --prompt wallet --prompt keys --prompt "pill bottle" \
  --prompt desk --prompt bed --prompt dresser --prompt rug
```

The input folders must contain `frames.json` and the associated sampled JPEGs from the earlier pipeline run. This experiment runs on CPU at 1024-pixel inference size. The first run downloads FastSAM-x and CLIP ViT-B/32 weights.

See [FASTSAM_RESULTS.md](FASTSAM_RESULTS.md) for reviewed results and comparison with YOLOE.

## SAM 3 on the movement recordings with Apple GPU acceleration

```bash
for id in 2133 2134 2135 2136 2137 2138; do
  python scripts/run_baseline.py "data/videos/IMG_${id}.MOV" \
    --model ~/Downloads/sam3.pt --device mps \
    --out "data/outputs/movement-sam3/IMG_${id}" \
    --prompt eyeglasses --prompt "water bottle" --prompt "remote control" \
    --prompt "compact camera" --prompt "card holder" --prompt keys \
    --prompt "medicine bottle" --prompt desk --prompt bed --prompt dresser --prompt rug || break
done
```

### Try automatic bed-versus-rug correction

Prepare two tight crops of plain surface from one room. Save them as `bed.jpg` and `rug.jpg` in the same folder. Avoid including objects, pillows, or adjacent flooring. You can add multiple examples, such as `bed-daylight.jpg` and `rug-evening.jpg`. The local demo crops are in `data/room-app/surface-references/`.

```bash
.venv/bin/python scripts/where_is.py remote \
  data/outputs/movement-yoloe-masks/IMG_2137 \
  --surface-references data/room-app/surface-references
```

The optional check compares the pixels next to and below a confident object detection with the reference colors. It only checks frames whose geometric candidates are bed or rug. At least two frames must still agree. It leaves uncertain colors and competing votes unresolved. The method can fail if the surfaces look alike, the lighting changes, or no plausible surface was detected.

Pass the same `--surface-references` flag to `scripts/room_app.py` to use it for imports and future uploads. This does not rewrite earlier sightings or confirmations. Use a fresh `--data-dir` to compare an import against the existing database. Imported clips still use the saved detections.

### Remember a specific belonging

If the CLIP checkpoint is missing, download it once from the project folder:

```bash
.venv/bin/python -c 'import clip; clip.load("ViT-B/32", device="cpu", download_root="../weights/clip")'
```

Stop the running local server with Control-C and start it with those local weights:

```bash
.venv/bin/python scripts/room_app.py --reid-weights ../weights/clip/ViT-B-32.pt
```

Use the same `--data-dir` and `--surface-references` values you used before, if any. Opening the same database preserves existing recordings and confirmations. The optional weights flag enables enrollment and matching. If the checkpoint is unavailable, the original command still runs the label-based app.

1. Search for an existing category such as **water bottle**. Check that the highlighted object is the right one, then choose **Remember this item** and give it a distinctive name such as **Brown bottle**.
2. Search by that name, or click it under **Remembered items**. To add another angle, open **Save another reference photo** and choose a tight JPG or PNG crop. You can also save a verified detection as another example.
3. Upload another clip. YOLOE proposes detections, then CLIP compares crops with saved references in the relevant categories. The water and medicine bottles compete for the same bottle candidates. The app requires matching detections in at least two sampled frames to accept an automatic identity match.
4. Review a **Possible newer sighting** if it appears. It will not replace the last reliable location. If the highlighted object is correct, use **Confirm item and save photo**, then confirm its surface separately if needed. If it shows a different item, choose **Not this item**. The app hides that suggestion and can show the next earlier candidate.

Enrollment creates a separate history for the named belonging. It preserves the selected reference sighting, and matching runs on new uploads and newly imported runs. Earlier clips are not silently reassigned. Use names that differ from generic categories if you want to keep both easy to search. The model stays local, and the similarity rules are provisional. Near-identical belongings and cases where YOLOE misses the object can still require review. See [REID_RESULTS.md](REID_RESULTS.md) for the experiment behind this feature.
