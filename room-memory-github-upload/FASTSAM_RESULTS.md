# FastSAM results on the six movement videos

Ran FastSAM-x with CLIP ViT-B/32 locally on CPU on November 16, 2026, using the 99 JPEG frames already sampled from IMG_2133 through IMG_2138. Settings were 1024-pixel input, segmentation confidence 0.25, IoU 0.9 and full-resolution masks.

Prompts were glasses, bottle, remote, camera, wallet, keys, pill bottle, desk, bed, dresser and rug. The script segments each frame once, filters masks of at most 100 pixels, and uses the installed FastSAM CLIP similarity implementation to select the best region for each prompt. It preserves the prompt-to-region mapping when several prompts select the same mask.

Processing took **244.32 seconds**, including first-use CLIP loading, segmentation, crop preparation and text selection. FastSAM model initialization, downloads and prior frame extraction are excluded. The prior YOLOE experiments took 43.53 and 48.44 seconds at 640-pixel input. These runs use different input sizes and model loading patterns, so the timings describe these experiments.

## Visual review

The review images show the highest-CLIP-similarity selection per prompt per clip. Scores are similarities, not calibrated object-presence probabilities. Every prompt chooses a candidate whenever eligible segments exist, including in frames where the requested item is absent. These are qualitative spot checks rather than precision/recall measurements.

| Object | What the strongest selections showed |
|---|---|
| Glasses | Correct on the desk in 2133, bed in 2134 and dresser in 2136–2137. The 2135 selection is unrelated. The 2138 selection covers the desk rather than isolating the glasses. |
| Water bottle | Generic bottle selected lamps, furniture parts and other containers in the strongest examples. |
| Remote | Correct on desk in 2133, bed in 2134 and 2136, and floor in 2137. Strongest selections in 2135 and 2138 are unrelated. |
| Camera | Correct desk selections in 2133–2135, bed selection in 2137 and dresser selection in 2138. The strongest 2136 selection is a framed photo. |
| Card holder | Wallet correctly selected it on the desk in 2133 and 2135, and floor in 2137. Other strongest selections include unrelated objects and a patch of bedding. |
| Keys | Selected visible keys on bed in 2135 and dresser in 2136–2137, sometimes only part of the key set. In 2133, 2134 and 2138 the strongest selections are unrelated. |
| Medicine bottle | Pill bottle correctly selected the medicine bottle on the desk in all six strongest examples. |

FastSAM produced additional correct selections for small objects that YOLOE missed, including glasses and camera on the bed and portions of the keys. Its text selection also frequently chose unrelated regions. These selections alone cannot reliably establish the complete movement sequences.

## Saved outputs

The experiment writes `data/outputs/movement-fastsam/IMG_.../selections.json` and `summary.json`. Each selection includes the original frame path, timestamp, prompt, box, CLIP similarity and segmentation confidence. The raw selections retain uncertainty about object presence and were not used to produce automatic movement claims.

- [Glasses review](data/outputs/movement-fastsam/glasses_review.jpg)
- [Water bottle review](data/outputs/movement-fastsam/bottle_review.jpg)
- [Remote review](data/outputs/movement-fastsam/remote_review.jpg)
- [Camera review](data/outputs/movement-fastsam/camera_review.jpg)
- [Card holder review](data/outputs/movement-fastsam/wallet_review.jpg)
- [Keys review](data/outputs/movement-fastsam/keys_review.jpg)
- [Medicine bottle review](data/outputs/movement-fastsam/pill_bottle_review.jpg)

Run commands are in [START_HERE.md](START_HERE.md). The reusable experiment entry point is [scripts/run_fastsam.py](scripts/run_fastsam.py). It uses the installed Ultralytics FastSAM CLIP helper, whose private API may change with package upgrades. All six clips completed with the current installation. The installed PyTorch issued a Python 3.14 TorchScript compatibility warning while loading CLIP.
