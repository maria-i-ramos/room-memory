"""Rank detections from saved videos against photos of named belongings."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visual_memory.object_reid import ClipEncoder, crop_detection
from visual_memory.search import draw_last_seen, load_detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Saved clip folders, oldest first")
    parser.add_argument("--references", type=Path, required=True, help="One image subfolder per known item")
    parser.add_argument("--weights", type=Path, required=True, help="Local CLIP ViT-B/32 checkpoint")
    parser.add_argument("--label", action="append", required=True, help="Detection label to consider, repeatable")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-similarity", type=float, default=0.75)
    parser.add_argument("--min-margin", type=float, default=0.08)
    args = parser.parse_args()

    if not args.weights.is_file():
        parser.error(f"Missing local CLIP weights: {args.weights}")
    if not args.references.is_dir():
        parser.error(f"Missing references folder: {args.references}")
    if not -1 <= args.min_similarity <= 1 or not 0 <= args.min_margin <= 2:
        parser.error("Similarity must be in [-1, 1] and margin in [0, 2].")
    if len({run.name for run in args.runs}) != len(args.runs):
        parser.error("Clip folders must have distinct names.")

    started = time.perf_counter()
    embeddings = ClipEncoder(args.weights).encode

    photos, owners = [], []
    for folder in sorted(args.references.iterdir()):
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                with Image.open(path) as image:
                    photos.append(image.convert("RGB"))
                owners.append(folder.name)
    identities = sorted(set(owners))
    if len(identities) < 2:
        parser.error("Supply reference photos for at least two identities to measure the winning margin.")
    reference_vectors = embeddings(photos)
    owner_indexes = {name: [i for i, owner in enumerate(owners) if owner == name] for name in identities}

    results, latest = [], {}
    labels = {label.casefold() for label in args.label}
    args.out.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        images, candidates = [], []
        for detection in load_detections(run):
            if detection["label"].casefold() not in labels:
                continue
            frame = run / "frames" / f"frame_{detection['source_frame_index']:06d}.jpg"
            image = crop_detection(frame, detection["xyxy"])
            if image is not None:
                images.append(image)
                candidates.append(detection)
        matches = {identity: [] for identity in identities}
        for start in range(0, len(images), 16):
            similarities = embeddings(images[start:start + 16]) @ reference_vectors.T
            for detection, scores in zip(candidates[start:start + 16], similarities):
                ranked = sorted(((float(np.max(scores[indexes])), item) for item, indexes in owner_indexes.items()),
                                reverse=True)
                score, identity = ranked[0]
                margin = score - ranked[1][0]
                accepted = score >= args.min_similarity and margin >= args.min_margin
                record = {"clip": run.name, "detection": detection, "candidate": identity,
                          "similarity": round(score, 4), "margin": round(margin, 4), "accepted": accepted}
                results.append(record)
                if accepted:
                    matches[identity].append(record)

        for identity, matching in matches.items():
            frames = {match["detection"]["source_frame_index"] for match in matching}
            if len(frames) >= 2:
                best = max(matching, key=lambda match: match["similarity"])
                filename = f"identity_{identities.index(identity)}.jpg"
                image = draw_last_seen(run / "frames", best["detection"], args.out / filename)
                latest[identity] = {**best, "supporting_frames": len(frames), "image": str(image)}

    report = {
        "min_similarity": args.min_similarity, "min_margin": args.min_margin,
        "seconds": round(time.perf_counter() - started, 2),
        "latest_candidates": latest, "detections": results,
    }
    (args.out / "reid.json").write_text(json.dumps(report, indent=2))
    print(f"Ranked {len(results)} detections in {report['seconds']:.1f}s.")
    for identity in identities:
        result = latest.get(identity)
        if result:
            print(f"{identity}: {result['clip']} ({result['supporting_frames']} matching frames).")
        else:
            print(f"{identity}: no clip with two matching frames.")
    print(f"Report: {args.out / 'reid.json'}")


if __name__ == "__main__":
    main()
