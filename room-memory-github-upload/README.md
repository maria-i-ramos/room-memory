# Where Did My Stuff Go? — Project Guide

[Run the local web app](START_HERE.md#run-room-memory-in-a-browser) to upload a room recording and search across saved sightings. New uploads run YOLOE in the background and save evidence and estimated surfaces in SQLite. Optional CLIP matching lets the user search for a specific remembered item. SAM3 and FastSAM remain available for comparisons.

The evaluation has two sets. The [six original movement recordings](START_HERE.md#replay-the-six-original-movement-recordings) (IMG_2133–IMG_2138) test last-seen locations in one room. The [cross-room test](CROSS_ROOM_RESULTS.md) (IMG_2367–IMG_2370) adds a second room and a red bottle that could be mistaken for the enrolled bottle. Both sets can be replayed with the web app's YOLOE prompts. See the [YOLOE](YOLOE_RESULTS.md), [FastSAM](FASTSAM_RESULTS.md), [SAM3](SAM3_RESULTS.md), and [re-identification](REID_RESULTS.md) reports for the follow-up experiments.

Run `python -m pytest -q` from this folder to check frame extraction, detection integration, last-seen search, and the historical YOLO baseline.

## Results so far

| Experiment | Measured result | What it showed |
|---|---|---|
| [Original YOLO baselines](#historical-yolo-results) | YOLO11n found 4 of 4 supported checklist items and correctly named 0 of 5 outside its vocabulary. YOLO11s boxed 8 of 9 items versus 6 of 9 with YOLO11n, but correctly named fewer supported items. | The app needed a detector that could use prompts for everyday belongings. |
| [Prompted detection and last-seen search](YOLOE_RESULTS.md) | The simpler-prompt YOLOE run processed 99 frames in 48.44 seconds and returned the correct clip and surface for 10 of 29 annotated placements. | YOLOE is the current upload detector. Its location estimates still need checks for missed or mislabeled surfaces. |
| [SAM3 comparison](SAM3_RESULTS.md) | SAM3 took 26 minutes 40 seconds on the same sampled frames in a different hardware and inference configuration. The revised location rule returned the correct clip and surface for 13 of 29 placements, with zero wrong surface answers. | SAM3 found some small objects that YOLOE missed. The timings describe these configurations and do not isolate model speed. |
| [FastSAM comparison](FASTSAM_RESULTS.md) | FastSAM and CLIP processed the 99 sampled frames in 244.32 seconds. | It found some objects missed by YOLOE, but the best segment for a prompt could be unrelated when an item was absent. |
| [Object re-identification](REID_RESULTS.md) | Across 13 reviewed bottle detections, YOLOE labels gave 8 correct and 5 wrong identities. CLIP with multiple reference views gave 10 correct, 0 wrong, and 3 withheld. | Reference photos helped distinguish the bottles. This is a per-crop result, and the prototype still missed the latest clip for the original bottle. |
| [Cross-room and red-bottle-only replay](CROSS_ROOM_RESULTS.md) | The red-only clip did not replace the original bottle's saved location. It did generate a false review suggestion. | The app now has a **Not this item** button to dismiss that suggestion without changing the last reliable location. |

These are small experiments with different scoring units. The [placement notes](data/checklists/movement_review.md) and [cross-room checklist](data/checklists/cross_room_recordings.json) record the expected results. This summary and the linked reports are intended to live in Git. Generated frames, detection files, and local app history stay on the machine that ran the experiments.

## Historical YOLO results


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

## Limitations of the original YOLO baseline

- Timestamps are estimated from frame number ÷ FPS. Phone videos can have variable frame rates, so they're approximate.
- "Last seen" currently means within one video, not across recordings.
- Two objects of the same type can't be told apart.
- These results come from three 16-second clips. They're a first check, not a measure of accuracy. For true precision and recall, we'd need to draw correct boxes on some frames and compare them with the model's boxes.

## Git

- `data/outputs/`, `.venv/`, `*.pt`, and cache folders are ignored by `.gitignore`.
- Before committing: run `git status`, `git add` specific files (not `git add .`), and check `git diff --cached --stat`.
- Don't commit new personal room recordings without agreeing as a team.
