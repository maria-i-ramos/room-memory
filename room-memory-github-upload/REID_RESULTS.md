# Cross-upload object re-identification

## Research and what transfers to Room Memory

- [ByteTrack (ECCV 2022)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf) associates detections across neighboring frames, including low-confidence detections that can help preserve a track through occlusion. A track can help choose representative crops within a recording. Room Memory's primary task is matching identity after an arbitrary gap between uploads, so continuous tracking is optional.
- [Egocentric Streaming Object Memory (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/papers/Manigrasso_Online_Episodic_Memory_Visual_Query_Localization_with_Egocentric_Streaming_Object_WACV_2026_paper.pdf) separates object discovery, visual matching, and temporal memory. It compares a visual query against stored object representations and returns the latest matching evidence. The paper evaluates DINOv2 cosine similarity and a Siam-RCNN head for visual matching. Room Memory can adopt the staged design without storing response tracks.
- [Object ReID in an office environment (2026)](https://openjournals.uwaterloo.ca/vsl/article/view/7160) compares frozen encoders including CLIP and DINOv2 on crops of everyday objects and evaluates cosine retrieval. These are candidates for a matched experiment on the app's own clips.
- [Embodied VideoAgent (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Fan_Embodied_VideoAgent_Persistent_Memory_from_Egocentric_Videos_and_Embodied_Sensors_ICCV_2025_paper.pdf) associates persistent object IDs with thumbnails, appearance, context, and spatial relationships. It combines 2D appearance with depth and camera poses. For phone uploads, store the room and supporting surface as separate evidence and avoid relying on an object's previous position when deciding its identity.
- [MyVLM (ECCV 2024)](https://snap-research.github.io/MyVLM/) uses external concept heads to help a VLM identify specific people or objects. The app can first test an image-only recognizer and keep its evidence structured. A VLM can be evaluated later on ambiguous candidate pairs or surface descriptions. [Beyond3D](https://arxiv.org/abs/2609.34630) reports that even general and spatially specialized VLMs struggled with last-visible timing in a benchmark involving relocated objects.

## Local prototype

[scripts/probe_reid.py](scripts/probe_reid.py) reads saved YOLOE detections and crops candidate regions from existing JPEG frames. It embeds crops and user-provided reference photos using [CLIP ViT-B/32](https://arxiv.org/abs/2103.00020) with the cached local checkpoint. For each crop it compares the normalized embedding with every reference, keeps the best score per identity, and rejects candidates below the similarity threshold or too close to the runner-up. The latest candidate for an item requires accepted detections in at least two distinct frames in the same clip. This path stays separate from the web app's automatic index.

The experiment uses water and medicine bottles because YOLOE often labels both `bottle`. The initial reference for each item is a crop from IMG_2133. An additional side-view water bottle crop from IMG_2136 was used in the second run. The comparison below uses exactly the same 13 candidate boxes from IMG_2137 and IMG_2138. The crops were visually reviewed, including a distractor. The simple baseline maps `bottle` to the water bottle and `pill bottle` to the medicine bottle.

| Method | Correct matches | Wrong matches | Abstentions |
|---|---:|---:|---:|
| YOLOE label mapping | 8 | 5 | 0 |
| CLIP, one photo per item | 7 | 1 | 5 |
| CLIP, added a sideways water bottle photo | 10 | 0 | 3 |

With multiple views, one abstention correctly rejected a distractor and two withheld a real item. The default score threshold was 0.75 and the runner-up margin was 0.08 for both reference sets. These are trial thresholds and the scores are not probabilities. The result uses clips from one room and the candidate boxes that YOLOE already found. The multi-view run ranked 13 crops in about 2.7 seconds on CPU, including model load. It returned the medicine bottle in IMG_2138. It returned the water bottle in IMG_2137 even though it was also visible in IMG_2138, because only one later frame passed both thresholds. Do not infer last-seen accuracy from per-crop matching.

To reproduce locally using the demo crops:

```bash
.venv/bin/python scripts/probe_reid.py data/outputs/movement-yoloe-masks/IMG_213{7,8} \
  --references data/room-app/reid-references-multiview \
  --weights ../weights/clip/ViT-B-32.pt \
  --label bottle --label "pill bottle" \
  --out data/outputs/reid-bottles-multiview
```

The reference directory has one subfolder per enrolled item. Each contains one or more close-up images. These demo photos and the generated report are local and ignored by Git. If setting up on another machine, create reference crops from verified sightings and keep the enrollment images out of the evaluation clips.

## Criteria for putting identity matching into the app

Give each enrolled belonging a stable database ID with a display name and multiple user-confirmed views. At upload time, YOLOE proposes crops. The identity matcher can accept, reject, or leave them for review. Store the object ID, recording, crop, and location separately so the same item can be matched after moving rooms without matching on its surroundings. Keep a candidate's detector confidence distinct from its identity similarity and avoid updating its reference photos from unconfirmed predictions.

Record two visually similar items and a newer clip where only the wrong one is present. Change viewpoint and room, and hold out the test clips from enrollment. Compare label-only, CLIP, and DINOv2 on the same candidates. Track wrong identity updates, rejected true sightings, and end-to-end latest reliable sighting accuracy. Use these results to choose acceptance rules. When evidence is insufficient, the app can show the earlier sighting with its date and expose the newer candidate for review.
