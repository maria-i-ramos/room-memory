
import os
import json
import time
import argparse
import random
from pathlib import Path

from google import genai


# Project folders
VIDEO_FOLDER = Path("data/videos")
OUTPUT_FOLDER = Path("data/outputs/gemini")

# Change models using GEMINI_MODEL in the terminal
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Keep different model results separate
MODEL_OUTPUT_FOLDER = OUTPUT_FOLDER / MODEL

# Retry settings
MAX_RETRIES = 3
RETRY_ERRORS = [429, 500, 502, 503, 504]


def upload_video(client, video_path):
    print(f"Uploading {video_path.name}...", flush=True)

    video = client.files.upload(file=str(video_path))

    # Wait for Gemini to finish processing the upload
    while not video.state or video.state.name == "PROCESSING":
        print("Waiting for Gemini to process the video...", flush=True)
        time.sleep(5)
        video = client.files.get(name=video.name)

    if video.state.name != "ACTIVE":
        raise RuntimeError(f"Could not process {video_path.name}")

    return video


def ask_gemini(client, videos, prompt, output_file):
    start_time = time.time()

    # Upload the videos once
    uploaded_videos = []

    for video_path in videos:
        uploaded = upload_video(client, video_path)
        uploaded_videos.append(uploaded)

    response = None

    # Retry temporary errors without uploading again
    for attempt in range(MAX_RETRIES):
        try:
            print(
                f"Asking Gemini ({MODEL}) - "
                f"attempt {attempt + 1}/{MAX_RETRIES}",
                flush=True
            )

            response = client.models.generate_content(
                model=MODEL,
                contents=uploaded_videos + [prompt],
                config={
                    "response_mime_type": "application/json"
                }
            )

            break

        except Exception as error:
            error_code = getattr(error, "code", None)

            if error_code not in RETRY_ERRORS:
                raise

            if attempt == MAX_RETRIES - 1:
                print("All retry attempts failed.", flush=True)
                raise

            # Wait longer after each failed attempt
            wait_time = 15 * (2 ** attempt) + random.uniform(0, 3)

            print(f"Temporary Gemini error: {error_code}")
            print(f"Retrying in {wait_time:.0f} seconds...", flush=True)

            time.sleep(wait_time)

    raw_response = response.text or ""

    try:
        predictions = json.loads(raw_response)
    except json.JSONDecodeError:
        predictions = None
        print("Gemini did not return valid JSON.")

    runtime = round(time.time() - start_time, 2)

    results = {
        "videos": [video.name for video in videos],
        "model": MODEL,
        "prompt": prompt,
        "predictions": predictions,
        "raw_response": raw_response,
        "runtime_seconds": runtime
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as file:
        json.dump(results, file, indent=2)

    print(f"Saved results to {output_file}")
    print(f"Total time: {runtime} seconds")

    return results


def identify_objects(client, limit=None):
    videos = sorted(
        video for video in VIDEO_FOLDER.iterdir()
        if video.suffix.lower() in [".mp4", ".mov"]
    )

    prompt = """
    We are testing how well Gemini identifies objects
    and remembers their locations in room videos.

    Watch the entire video carefully.

    Identify the everyday objects you can see, especially
    items someone might misplace, such as keys, glasses,
    wallets, water bottles, chargers, phones, pencils,
    laptops, and similar items.

    Include furniture when it helps explain an
    object's location.

    For each object, return:
    - name: the name of the object
    - found: true
    - surface: what the object is sitting on
      (desk, floor, table, chair, bed, etc.)
    - nearby_objects: other objects close to it
    - location: a specific description of its location
    - first_seen_seconds: when the object first appears
    - last_seen_seconds: when the object was last visible
    - confidence: low, medium, or high

    If an object changes locations during the video,
    report its last visible location.

    If an object disappears behind something, do not
    assume it has moved unless the video shows evidence.

    Only report objects and spatial relationships that
    are visually supported.

    If the supporting surface is unclear, return null.

    Do not invent objects or timestamps.

    Return valid JSON in this format:

    {
        "objects": [
            {
                "name": "water bottle",
                "found": true,
                "surface": "desk",
                "nearby_objects": ["laptop"],
                "location": "on the desk beside the laptop",
                "first_seen_seconds": 1,
                "last_seen_seconds": 8,
                "confidence": "high"
            }
        ]
    }

    The example only shows the JSON structure.
    Do not assume that example objects are present.
    """

    completed = 0
    attempted = 0

    for video in videos:
        output = (
            MODEL_OUTPUT_FOLDER
            / "identification"
            / f"{video.stem}.json"
        )

        # Skip results already completed with this model
        if output.exists():
            print(f"Already tested {video.name}, skipping.")
            continue

        # Limit the number of new video attempts
        if limit is not None and attempted >= limit:
            break

        attempted += 1

        try:
            result = ask_gemini(
                client,
                [video],
                prompt,
                output
            )

            if result["predictions"] is not None:
                completed += 1

        except Exception as error:
            print(f"Error with {video.name}: {error}")

    print()
    print("Identification experiments finished.")
    print(f"Attempted: {attempted}")
    print(f"Completed with valid JSON: {completed}")
    print(f"Model: {MODEL}")


def compare_movement(client, first_name, second_name, test_name):
    first_video = VIDEO_FOLDER / first_name
    second_video = VIDEO_FOLDER / second_name

    if not first_video.exists():
        print(f"Missing video: {first_video}")
        return

    if not second_video.exists():
        print(f"Missing video: {second_video}")
        return

    output = (
        MODEL_OUTPUT_FOLDER
        / "movement"
        / f"{test_name}.json"
    )

    if output.exists():
        print(f"Already tested {test_name}, skipping.")
        return

    prompt = f"""
    We are testing whether Gemini can recognize changes
    in object locations across two room recordings.

    You are watching TWO videos in chronological order.

    Video 1: {first_name}
    Video 2: {second_name}

    Identify the water bottle and wallet in both videos,
    if they are visible.

    For each object, return:
    - object_name
    - found_in_first_video: true or false
    - first_video_surface: desk, floor, table, etc.
    - first_video_location: detailed location in video 1
    - found_in_second_video: true or false
    - second_video_surface: desk, floor, table, etc.
    - second_video_location: detailed location in video 2
    - moved: true, false, or null if unclear
    - confidence: low, medium, or high

    Also return:
    - objects_swapped_locations: true, false, or null
    - explanation: describe any location changes

    Compare the last visible location of each object
    in the first video with its last visible location
    in the second video.

    Do not assume objects moved just because there
    are two recordings.

    If an object is missing from either video, return
    null for any location you cannot establish.

    If the supporting surface is unclear, use null.

    Return valid JSON in this format:

    {{
        "objects": [
            {{
                "object_name": "water bottle",
                "found_in_first_video": true,
                "first_video_surface": null,
                "first_video_location": null,
                "found_in_second_video": true,
                "second_video_surface": null,
                "second_video_location": null,
                "moved": null,
                "confidence": "low"
            }}
        ],
        "objects_swapped_locations": null,
        "explanation": ""
    }}

    The example only shows the JSON structure.
    Make all predictions from the actual videos.
    """

    try:
        ask_gemini(
            client,
            [first_video, second_video],
            prompt,
            output
        )

    except Exception as error:
        print(f"Error with {test_name}: {error}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "experiment",
        choices=["identify", "movement", "all"],
        help="Choose which Gemini experiment to run"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of new identification attempts"
    )

    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is missing")

    if not VIDEO_FOLDER.exists():
        raise FileNotFoundError("data/videos folder is missing")

    client = genai.Client()

    print(f"Using Gemini model: {MODEL}")
    print(f"Saving results to: {MODEL_OUTPUT_FOLDER}")
    print()

    # Test identification and location recognition
    if args.experiment in ["identify", "all"]:
        identify_objects(client, args.limit)

    # Test changes in object locations across recordings
    if args.experiment in ["movement", "all"]:

        # Water bottle changes from desk to floor
        compare_movement(
            client,
            "IMG_4889_from_MP4.mp4",
            "IMG_4890-2.mp4",
            "bottle_desk_to_floor"
        )

        # Water bottle and wallet swap locations
        compare_movement(
            client,
            "IMG_4866 2.mp4",
            "IMG_4887.mp4",
            "bottle_wallet_swap"
        )


if __name__ == "__main__":
    main()