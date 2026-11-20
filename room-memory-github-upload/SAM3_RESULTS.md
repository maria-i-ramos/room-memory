# SAM 3 results on the six movement videos

Tested the downloaded `sam3.pt` locally on November 16, 2026, through Ultralytics 8.4.178 with PyTorch 2.14.1 and Python 3.14.7. Inference used the Apple M3 GPU through MPS.

The experiment samples IMG_2133 through IMG_2138 at two frames per second, uses 1008-pixel model input and a confidence threshold of 0.25. Prompts are eyeglasses, water bottle, remote control, compact camera, card holder, keys, medicine bottle, desk, bed, dresser and rug. Each frame is encoded once and queried with one prompt at a time. This is image-based prompted detection; this run does not exercise SAM 3's video tracking model or establish persistent physical-object identities.

## Runtime implementation

The adapter caches image features with `set_image`, then calls the predictor with `source=None` for each prompt. Passing the image again would invalidate that cache. It also disables unused encoder self-attention weights, which enables PyTorch's efficient attention path. On one cached camera example, this reduced a prompt call from 2.44 to 0.60 seconds; the returned box and confidence matched within `rtol=1e-3, atol=1e-2`. That numerical spot check is not a full equivalence test. The optimization accesses Ultralytics model internals and should be rechecked when upgrading the library.

The adapter retains searchable boxes, labels, confidence scores and frame timestamps. It does not currently save SAM masks or use them to determine supporting surfaces.

## Review method

Review sheets show the highest-confidence prediction for each object in each clip. Correct reviewed boxes do not imply that all predictions were correct: lower-confidence false matches can still affect the movement estimator. These are visual spot checks, not an exhaustive precision/recall evaluation. Location estimates use the existing box-overlap heuristic, which can confuse overlapping bed, rug and furniture regions.

## Results

All six clips completed. The detection step took **26 minutes 40 seconds** for 99 frames, including model loading for each clip and excluding frame extraction. This is a measurement of the current M3 configuration with 11 prompts per frame. YOLOE took about 44–48 seconds on the same sampled frames with a smaller input and CPU inference. Those times describe the two configurations and do not isolate model speed.

| Object | Visual review | Automatic movement result |
|---|---|---|
| Glasses | Correct strongest boxes in all six clips, including both bed clips that YOLOE missed. | No moves. It could only assign desk confidently in the first and last clips. |
| Remote | Correct strongest boxes in all six clips, including the final dresser placement. | Desk → bed → rug. The dresser placement remained uncertain. |
| Camera | Correct strongest boxes in all six clips, including both bed clips that YOLOE missed. | Desk → bed → dresser, matching the confirmed sequence. |
| Water bottle | Correct strongest boxes on desk, dresser and bed. No detection in 2137. | No moves. Some clips had multiple candidates or uncertain locations. |
| Card holder | Correct strongest boxes on the desk and rug. The strongest prediction in the final clip was the remote. | Desk → rug. The last bed placement was unresolved. |
| Keys | Strongest boxes covered keys or part of the key set in 2135–2138. The model also mislabeled a remote button in 2133. | No moves. Multiple candidates made the relevant clips ambiguous. |
| Medicine bottle | Correct strongest boxes in three clips, two misses and one strongest false match. YOLOE with the “pill bottle” prompt had correct strongest boxes in all six. | No moves, with no reliable desk assignment. |

The movement report links sightings by label. The correct camera sequence shows that detections and surface estimates can recover a known sequence on these clips. Establishing the same physical object's identity across independent recordings requires a separate matching step.

## Outputs and reproduction

Per-clip frames, detections and summaries are saved under `data/outputs/movement-sam3/`. Review sheets and `movement_histories.json` are in that same folder. The output directory also retains `review_run.py` to regenerate the local review artifacts.

The command for all six videos is in [START_HERE.md](START_HERE.md#sam-3-on-the-movement-recordings-with-apple-gpu-acceleration). Weights remain in Downloads. Videos, generated outputs and weights are excluded from Git. Code changes are local on branch `roshni-pipeline`.
