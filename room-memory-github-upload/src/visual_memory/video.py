"""Step 1: check that a video can be opened before we process it."""
from pathlib import Path
import cv2


def inspect_video(video_path):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"I couldn't find the video: {video_path}")

    video = cv2.VideoCapture(str(video_path))
    try:
        if not video.isOpened():
            raise ValueError(f"I couldn't open this video: {video_path}")

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        success, first_frame = video.read()
        if not success or first_frame is None or fps <= 0:
            raise ValueError("The video is empty or its frame rate is missing.")

        # Getting the size from a decoded frame also works better with rotated phone clips.
        height, width = first_frame.shape[:2]
        return {
            "path": str(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration_seconds": frame_count / fps,
        }
    finally:
        video.release()
