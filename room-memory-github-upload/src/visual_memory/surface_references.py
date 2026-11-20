"""Compare the visible pixels around an object with room-specific examples."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

REFERENCE_SURFACES = {"bed", "rug"}


def support_points(box):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return []
    return [(x1 + width * x, y2 + height * y)
            for x in (0.15, 0.35, 0.5, 0.65, 0.85) for y in (0.03, 0.1, 0.2)] + [
                (x1 + width * x, y1 + height * y)
                for x in (-0.12, 1.12) for y in (0.55, 0.75, 0.95)
            ]


def lab_colors(bgr):
    return cv2.cvtColor(bgr.reshape(-1, 1, 3).astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[:, 0, :]


class SurfaceReferences:
    def __init__(self, folder):
        self.colors = {}
        for label in sorted(REFERENCE_SURFACES):
            files = sorted(path for path in Path(folder).glob(f"{label}*")
                           if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if not files:
                raise ValueError(f"Add at least one {label} reference image in {folder}.")
            self.colors[label] = []
            for file in files:
                image = cv2.imread(str(file))
                if image is None:
                    raise ValueError(f"Could not read surface reference {file}.")
                patch = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
                self.colors[label].append(np.median(lab_colors(patch), axis=0))
        self.read_frame = lru_cache(maxsize=4)(cv2.imread)

    def classify(self, frame, box):
        image = self.read_frame(str(frame))
        if image is None:
            return None
        height, width = image.shape[:2]
        points = [(round(x), round(y)) for x, y in support_points(box)]
        pixels = [image[y, x] for x, y in points if 0 <= x < width and 0 <= y < height]
        if len(pixels) < 8:
            return None
        colors = lab_colors(np.asarray(pixels))
        distances = [
            np.min(np.linalg.norm(colors[:, None, :] - examples, axis=2), axis=1)
            for examples in self.colors.values()
        ]
        first, second = distances
        # Most sampled pixels must look like one reference and be clearly
        # farther from the other. Distances are CIELAB color differences.
        for label, near, far in zip(self.colors, distances, (second, first)):
            if np.mean((near <= 24) & (far - near >= 10)) >= 0.65:
                return label
        return None
