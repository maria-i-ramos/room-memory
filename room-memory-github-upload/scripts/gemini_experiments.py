
import os
import json
import time
import argparse
from pathlib import Path

from google import genai


# Where our videos and results are saved
VIDEO_FOLDER = Path("data/videos")
OUTPUT_FOLDER = Path("data/outputs/gemini")
MODEL = "gemini-3.8-flash"


def upload_video(client, video_path):
    print(f"Uploading {video_path.name}...")

    video = client.files.upload(file=str(video_path))

    # Gemini needs to process the video before we can use it
    while not video.state or video.state.name == "PROCESSING":
        print("Waiting for Gemini to process the video...")
        time.sleep(5)
        video = client.files.get(name=video.name)

    if video.state.name != "ACTIVE":
        raise RuntimeError(f"Couldn't process {video_path.name}")

    return video


def ask_gemini(client, videos, prompt, output_file):
    start_time = time.time()

    uploaded_videos = []

    for video_path in videos:
        uploaded = upload_video(client, video_path)
        uploaded_videos.append(uploaded)

    print("Asking Gemini about the video...")

    response = client.models.generate_content(
        model=MODEL,
        contents=uploaded_videos + [prompt],
        config={"response_mime_type": "application/json"}
    )

    raw_response = response.text or ""

    try:
        predictions = json.loads(raw_response)
    except json.JSONDecodeError:
        predictions = None
        print("Gemini didn't return valid JSON.")

    runtime = round(time.time() - start_time, 2)

    # Save the response so we can evaluate it later
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


def identify_objects(client, limit=None):

    # Find all the videos in our folder
    videos = sorted(
        video for video in VIDEO_FOLDER.iterdir()
        if video.suffix.lower() in [".mp4", ".mov"]
    )

    prompt = """
    We're testing how well Gemini can identify objects
    and describe where they are in a room.

    Watch the entire video and identify the objects you see.

    Focus on everyday items like water bottles, wallets,
    keys, glasses, chargers, phones, pencils, and laptops.
    Also include furniture that helps describe their locations.

    For each object, return:
    - name: what the object is
    - found: true
    - surface: what it is sitting on (desk, floor, bed, etc.)
    - nearby_objects: other objects close to it
    - location: a detailed description of where it is
    - first_seen_seconds: when it first appears
    - last_seen_seconds: when it was last visible
    - confidence: low, medium, or high

    If an object appears in multiple locations, use its
    last visible location.

    Only report objects and relationships you can actually see.
    If the surface isn't clear, use null instead of guessing.

    Don't make up objects or timestamps.

    Return valid JSON with an "objects" list.
    """

    completed = 0
    attempted = 0

    for video in videos:

        output = (
            OUTPUT_FOLDER
            / "identification"
            / f"{video.stem}.json"
        )

        # Skip videos we've already tested
        if output.exists():
            print(f"Already tested {video.name}, skipping.")
            continue

        if limit is not None and attempted >= limit:
            break

        attempted += 1
        
        try:
            ask_gemini(client, [video], prompt, output)
            completed += 1

        except Exception as error:
            print(f"Error with {video.name}: {error}")

        if limit is not None and completed >= limit:
            break

    print(f"Completed {completed} new identification tests.")


def compare_movement(client, first_name, second_name, test_name):

    first_video = VIDEO_FOLDER / first_name
    second_video = VIDEO_FOLDER / second_name

    if not first_video.exists() or not second_video.exists():
        print(f"Missing a video for {test_name}. Skipping.")
        return

    output = (
        OUTPUT_FOLDER
        / "movement"
        / f"{test_name}.json"
    )

    if output.exists():
        print(f"Already tested {test_name}, skipping.")
        return

    prompt = f"""
    We're testing if Gemini can recognize when objects
    have changed locations between two room recordings.

    You are watching TWO videos in chronological order.

    Video 1: {first_name}
    Video 2: {second_name}

    Look for the water bottle and wallet in both videos.

    For EACH object, return:
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
    - explanation: explain any location changes you noticed

    Use only what you can actually see in both recordings.

    Don't assume objects moved just because there are
    two different videos.

    If you can't clearly identify a surface or location,
    return null rather than guessing.

    Return valid JSON.
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
        help="Only run this many new identification videos"
    )

    args = parser.parse_args()

    # Make sure our API key and videos are available
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is missing")

    if not VIDEO_FOLDER.exists():
        raise FileNotFoundError("data/videos folder is missing")

    client = genai.Client()

    # Test object identification and locations
    if args.experiment in ["identify", "all"]:
        identify_objects(client, args.limit)

    # Test whether Gemini recognizes location changes
    if args.experiment in ["movement", "all"]:

        # Water bottle moves from desk to floor
        compare_movement(
            client,
            "IMG_4889_from_MP4.mp4",
            "IMG_4890-2.mp4",
            "bottle_desk_to_floor"
        )

        # Water bottle and wallet swap locations
        compare_movement(
            client,
            "IMG_4886 2.mp4",
            "IMG_4887.mp4",
            "bottle_wallet_swap"
        )


if __name__ == "__main__":
    main()
