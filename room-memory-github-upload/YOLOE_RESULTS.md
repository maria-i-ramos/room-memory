# YOLOE results on the six movement videos

Ran YOLOE-11 large (`yoloe-11l-seg.pt`) locally on CPU on November 16, 2026. Each experiment processed 99 sampled frames from IMG_2133 through IMG_2138, at two frames per second and a confidence threshold of 0.25. Detection used the default 640-pixel inference size. Reported processing time includes loading the model and encoding prompts for each clip, and excludes video decoding, frame extraction and initial downloads.

| Experiment | Object prompts | Processing time |
|---|---|---|
| Original names | eyeglasses, water bottle, remote control, compact camera, card holder, keys, medicine bottle | 43.53 seconds |
| Simpler names | glasses, bottle, remote, camera, wallet, key, pill bottle | 48.44 seconds |

Both experiments also prompted desk, bed, dresser and rug. Outputs are in `data/outputs/movement-yoloe/` and `data/outputs/movement-yoloe-simple/`. Each clip has detections, a summary, sampled frames, and annotated review images. The folders also contain per-object contact sheets and `movement_histories.json`.

## Findings from visual review

Each contact sheet shows the highest-confidence prediction per object per clip. These are spot checks, with no exhaustive box annotations or precision/recall score.

| Item | Original prompts | Simpler prompts |
|---|---|---|
| Glasses | Detected only in IMG_2138 | Correct reviewed boxes in 2133, 2136, 2137, 2138. Missed both bed clips. |
| Water bottle | Zero detections under its requested name. Some water-bottle boxes were labeled medicine bottle. | Generic bottle finds bottles, but combines the water bottle and medicine bottle into one category. |
| Remote | Correct reviewed boxes in 2133–2137. Missed dresser in 2138. | Correct reviewed boxes in all six clips, including the dresser. |
| Camera | Correct reviewed boxes on desk in 2133–2135. Missed bed. The 2138 prediction covers part of the lanyard. | Correct reviewed boxes on desk in 2133–2135 and dresser in 2138. Missed bed in 2136–2137. |
| Card holder | Correct reviewed floor and bed boxes in 2137–2138. Some earlier predictions cover a green box on the monitor stand. | Wallet finds the card holder on the desk in 2133–2135, but the strongest later predictions include unrelated objects. |
| Keys | One low-confidence detection in 2138, around the keys and lanyard | Zero detections under key |
| Medicine bottle | Strongest predictions include the water bottle and a floor object | Pill bottle correctly covers the medicine bottle in all six reviewed examples |

## Movement results

The current box-overlap location estimator recovered desk → bed for the remote. It incorrectly assigns the remote on the rug in IMG_2137 to bed because a detected bed box overlaps that region. It leaves the remote's dresser location uncertain.

The simpler wallet run also reports desk → bed, but its reviewed examples include false positives and the result misses the actual intermediate floor placement. The medicine bottle produces zero move events, with desk estimates in four clips and uncertain placement in two.

YOLOE improved detection of several custom items compared with the earlier COCO baseline, whose vocabulary lacks glasses and keys. Full placement histories remain incomplete. Matching the same physical object across clips and estimating the supporting surface still require more work.

## Review images

- [Glasses with simpler prompts](data/outputs/movement-yoloe-simple/glasses_review.jpg)
- [Remote with simpler prompts](data/outputs/movement-yoloe-simple/remote_review.jpg)
- [Medicine bottle with simpler prompts](data/outputs/movement-yoloe-simple/pill_bottle_review.jpg)
- [Camera with simpler prompts](data/outputs/movement-yoloe-simple/camera_review.jpg)
- [Card holder with original prompts](data/outputs/movement-yoloe/card_holder_review.jpg)

## Reproduce the simpler prompt experiment

From this project folder with the virtual environment active:

```bash
for id in 2133 2134 2135 2136 2137 2138; do
  python scripts/run_baseline.py "data/videos/IMG_${id}.MOV" \
    --model yoloe-11l-seg.pt --out "data/outputs/movement-yoloe-simple/IMG_${id}" \
    --prompt glasses --prompt bottle --prompt remote --prompt camera \
    --prompt wallet --prompt key --prompt "pill bottle" \
    --prompt desk --prompt bed --prompt dresser --prompt rug || break
done
python scripts/find_moves.py remote data/outputs/movement-yoloe-simple/IMG_213{3,4,5,6,7,8} \
  --surface desk --surface bed --surface dresser --surface rug
```

The first experiment's commands are in [START_HERE.md](START_HERE.md). The model weights and text encoder are cached locally and ignored by Git. The installed PyTorch emits a Python 3.14 TorchScript compatibility warning while loading the text encoder, but both experiments completed successfully.

## Surface masks and the remote-on-rug case

YOLOE runs now save a `polygon` for each available segmentation mask. The optional `--use-masks` search flag tests support using points next to and below the detected item. Box overlap stays the default. On the 29 annotated placements, the simple-prompt run returned the right clip and surface in 10 cases with boxes and 9 with masks. Incorrect surface answers fell from 3 to 2 because one wrong box result became uncertain. One correct box result also became uncertain.

Masks did not fix IMG_2137. YOLOE traced the visible rug around the remote and labeled that region `bed`. Trying alternative surface prompts found some floor evidence but also missed real bed sightings or continued to mislabel the rug. The browser app lets the user correct the surface for a particular sighting after reviewing its evidence. This updates the local SQLite database and does not change the detector or other sightings.

To try that correction in isolation from your existing app data:

```bash
.venv/bin/python scripts/room_app.py --data-dir data/room-app/rug-check \
  --import-runs data/outputs/movement-yoloe-masks/IMG_2137
```

Search for “remote”. Under the evidence image choose **Confirm or correct the location**, select **Floor (on the rug)**, and save. The saved answer is marked **Confirmed by you**.
