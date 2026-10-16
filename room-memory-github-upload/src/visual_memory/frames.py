"""Step 2: take a few frames each second instead of processing the whole video."""
from pathlib import Path
import cv2


def extract_frames(video_path, output_folder, frames_per_second=2):
    if frames_per_second <= 0:
        raise ValueError("frames_per_second must be greater than zero")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Couldn't open video: {video_path}")

    saved_frames = []
    try:
        original_fps = video.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            raise ValueError("Couldn't read the video's FPS")

        frame_number = 0
        next_save_time = 0.0
        while True:
            success, image = video.read()
            if not success:
                break

            # Example: a 10 FPS video sampled at 2 FPS keeps frames 0, 5, 10, ...
            seconds = frame_number / original_fps
            if seconds + 1e-8 >= next_save_time:
                filename = f"frame_{frame_number:06d}.jpg"
                if not cv2.imwrite(str(output_folder / filename), image):
                    raise OSError(f"Couldn't save {filename}")
                saved_frames.append({
                    "filename": filename,
                    "timestamp_seconds": round(seconds, 3),
                    "source_frame_index": frame_number,
                })
                next_save_time += 1 / frames_per_second
                while next_save_time <= seconds + 1e-8:
                    next_save_time += 1 / frames_per_second

            frame_number += 1
    finally:
        video.release()

    return saved_frames
