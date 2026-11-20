# First cross-room test

Use the [cross-room replay command](START_HERE.md#replay-the-cross-room-recordings) to rerun the detector on all four clips. The [recording checklist](data/checklists/cross_room_recordings.json) records their order and expected placements.

Processed IMG_2367, IMG_2368, and IMG_2369 in filename order with the web app's YOLOE prompts for an enrolled bottle. IMG_2367 is the room used for enrollment. The other two clips are from a second room. IMG_2369 adds a red bottle. YOLOE sampled 40 frames total at 2 FPS. Detection took 19.94 seconds on CPU, excluding frame extraction and appearance matching. The original bottle used two references from earlier recordings. Glasses and keys used one reference each from IMG_2138. The red bottle was never enrolled. The three test clips did not contribute reference photos.

| Clip | Original bottle | Glasses | Keys |
|---|---|---|---|
| IMG_2367, original room | Correct review candidate on the desk. | Visible on the dresser. The only glasses prediction was the key lanyard. | Visible on the bed. No detection with the standard prompt. |
| IMG_2368, second room | Correct review candidate on the carpeted floor. The label-only surface estimator incorrectly said desk. | Correct review candidate on the white dresser. | On the desk. No detection with the standard prompt. |
| IMG_2369, second room | Correct review candidate on the carpeted floor. YOLOE also detected the red bottle on the white dresser. | Correct review candidate on the white dresser. | On the desk. No detection with the standard prompt. |

No new named sighting was automatically accepted. The app retained the earlier reliable desk sightings for the original bottle and glasses. It did not replace the original bottle with the red bottle. The strongest original candidate in the last clip scored 0.784 and the strongest red candidate 0.759 against the original's references. A weaker crop of the original scored 0.690. These are cosine similarities, not probabilities. The current single-identity update requires scores of at least 0.88 in two frames. Lowering that cutoff enough to accept both original candidates in this clip would also accept at least two red candidates. This is a coexistence test. A newer clip with only the red bottle is still needed to test whether it can produce a misleading review candidate when the original is absent.

In a separate prompt probe on the same frames, `house keys` found the lanyard in all three clips. The second room still produced only one key detection per clip. The `floor` prompt found a floor box that supported the original bottle in both second-room clips. The probe used a reduced prompt list. The next test is to add these labels to the full prompt set and evaluate them alongside desk and the other surfaces before changing the app defaults.

Local artifacts are in `data/outputs/cross-room-yoloe/`. The test database is `data/room-app/cross-room-test/`. To browse the saved test separately:

```bash
.venv/bin/python scripts/room_app.py --port 8001 \
  --data-dir data/room-app/cross-room-test \
  --reid-weights ../weights/clip/ViT-B-32.pt
```

Search for **Original bottle**, **My glasses**, or **Lanyard keys**. The original web app database was not modified. Room labels in this report come from the supplied clip notes. The app still does not store them.

## Red-bottle-only check in the original room

IMG_2370 puts the red bottle in the original room without the enrolled bottle. YOLOE sampled 21 frames and ran in 7.70 seconds on CPU. With the original bottle's references unchanged, no detection reached the 0.88 single-identity automatic update threshold. The app retained the previous reliable desk sighting from IMG_2133. It still showed the red bottle at 0.5s as a possible newer sighting because its 0.757 similarity passed the 0.75 review threshold. The number is cosine similarity, not a probability. This passes the automatic update check and reveals a false review suggestion.

In a separate replay, adding a red bottle crop from IMG_2369 as a competing identity matched the red bottle in four frames in IMG_2370. It did not accept an original bottle match. A low-confidence YOLOE box around a small item on the dresser at 6.5s still produced a false review candidate for the original. The competing reference was used only for this experiment, so the original isolated test database still shows the baseline behavior. See the local output files `red-only-review.json` and `red-competitor-review.json` under `data/outputs/cross-room-yoloe/`.
