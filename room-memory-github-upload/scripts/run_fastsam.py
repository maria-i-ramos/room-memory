"""Evaluate FastSAM text-selected regions on existing sampled video frames."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import FastSAM


def select_regions(model, result, prompts):
    """Use the installed FastSAM CLIP selector, retaining prompt-to-region mapping.

    As in FastSAMPredictor.prompt, ignore masks with at most 100 pixels and
    select the highest CLIP similarity per prompt. A selection does not establish
    that the requested object is present.
    """
    if result.masks is None:
        return []
    crops, indices = [], []
    height, width = result.orig_img.shape[:2]
    masks = result.masks.data
    for index, box in enumerate(result.boxes.xyxy.tolist()):
        if masks[index].sum().item() <= 100:
            continue
        x1, y1, x2, y2 = (int(value) for value in box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = result.orig_img[y1:y2, x1:x2]
        crop = crop * masks[index, y1:y2, x1:x2, None].cpu().numpy()
        crops.append(Image.fromarray(crop.astype(np.uint8)[:, :, ::-1]))
        indices.append(index)
    if not crops:
        return []
    with torch.inference_mode():
        similarities = model.predictor._clip_inference(crops, prompts)
    selected = []
    for prompt, scores in zip(prompts, similarities):
        best = int(scores.argmax().item())
        index = indices[best]
        selected.append({
            "prompt": prompt,
            "segment_index": index,
            "clip_similarity": round(float(scores[best].item()), 4),
            "segmentation_confidence": round(float(result.boxes.conf[index].item()), 4),
            "xyxy": result.boxes.xyxy[index].tolist(),
        })
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Existing output folders with frames.json")
    parser.add_argument("--out", type=Path, default=Path("data/outputs/movement-fastsam"))
    parser.add_argument("--model", default="FastSAM-x.pt")
    parser.add_argument("--prompt", action="append", required=True)
    args = parser.parse_args()
    names = [folder.name for folder in args.runs]
    if len(set(names)) != len(names):
        parser.error("Each input folder must have a unique name.")
    model = FastSAM(args.model)
    for folder in args.runs:
        output = args.out / folder.name
        output.mkdir(parents=True, exist_ok=True)
        frames = json.loads((folder / "frames.json").read_text())
        selections = []
        started = time.perf_counter()
        for frame in frames:
            source = folder / "frames" / frame["filename"]
            result = model.predict(str(source), device="cpu", retina_masks=True,
                                   imgsz=1024, conf=.25, iou=.9, verbose=False)[0]
            for selection in select_regions(model, result, args.prompt):
                selection.update({
                    "source_frame_index": frame["source_frame_index"],
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "frame": str(source.resolve()),
                })
                selections.append(selection)
        summary = {
            "model": args.model, "device": "cpu", "imgsz": 1024, "confidence": .25,
            "iou": .9, "prompts": args.prompt, "frames_processed": len(frames),
            "processing_seconds": round(time.perf_counter() - started, 2),
            "selections": len(selections),
            "selection_rule": "highest CLIP similarity per prompt per frame; presence unverified",
        }
        (output / "selections.json").write_text(json.dumps(selections, indent=2))
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(folder.name, summary, flush=True)


if __name__ == "__main__":
    main()
