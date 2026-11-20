"""Local appearance matching for enrolled belongings."""
import math
from collections import defaultdict

import numpy as np
from PIL import Image


# The bottle categories compete for the same YOLOE candidates.
BOTTLE_LABELS = {"bottle", "water bottle", "medicine bottle", "pill bottle"}


def crop_detection(frame, box):
    with Image.open(frame) as image:
        x1, y1, x2, y2 = box
        bounds = (max(0, math.floor(x1)), max(0, math.floor(y1)),
                  min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2)))
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            return None
        return image.convert("RGB").crop(bounds)


class ClipEncoder:
    def __init__(self, weights):
        self.weights = str(weights)
        self.model = None

    def encode(self, images):
        import torch

        if self.model is None:
            import clip

            self.model, self.preprocess = clip.load(self.weights, device="cpu", jit=False)
        with torch.inference_mode():
            inputs = torch.stack([self.preprocess(image) for image in images])
            features = self.model.encode_image(inputs).float()
            return torch.nn.functional.normalize(features, dim=1).cpu().numpy()


class IdentityMatcher:
    def __init__(self, encoder):
        self.encoder = encoder
        self.references = {}

    def match(self, run, detections, belongings):
        """Return known identities with automatic or review candidates for one clip."""
        by_label = defaultdict(set)
        owners, vectors = [], []
        for key, belonging in belongings.items():
            for label in belonging["labels"]:
                by_label[label].add(key)
            for path in belonging["references"]:
                if path not in self.references:
                    with Image.open(path) as image:
                        self.references[path] = self.encoder.encode([image.convert("RGB")])[0]
                owners.append(key)
                vectors.append(self.references[path])
        if not vectors:
            return {}

        references = np.asarray(vectors)
        images, candidates = [], []
        for detection in detections:
            if detection["label"].casefold() not in by_label or detection["confidence"] < 0.25:
                continue
            frame = run / "frames" / f"frame_{detection['source_frame_index']:06d}.jpg"
            image = crop_detection(frame, detection["xyxy"])
            if image is not None:
                images.append(image)
                candidates.append(detection)

        accepted, review = defaultdict(list), {}
        for start in range(0, len(images), 16):
            scores = self.encoder.encode(images[start:start + 16]) @ references.T
            for detection, similarities in zip(candidates[start:start + 16], scores):
                eligible = by_label[detection["label"].casefold()]
                ranked = sorted(
                    ((max(float(s) for s, owner in zip(similarities, owners) if owner == key), key)
                     for key in eligible), reverse=True,
                )
                score, key = ranked[0]
                runner_up = ranked[1][0] if len(ranked) > 1 else None
                if score < 0.75:
                    continue
                candidate = {**detection, "label": key, "identity_similarity": score}
                if key not in review or score > review[key]["identity_similarity"]:
                    review[key] = candidate
                # A single available identity has no runner-up to reject lookalikes.
                if (runner_up is None and score >= 0.88) or (
                    runner_up is not None and score - runner_up >= 0.08
                ):
                    accepted[key].append(candidate)

        results = {}
        for key, candidate in review.items():
            matches = accepted[key]
            if len({d["source_frame_index"] for d in matches}) >= 2:
                results[key] = ("matched", matches)
            else:
                results[key] = ("review", [candidate])
        return results
