
import os
import json
import time
import argparse
from pathlib import Path

from google import genai


# These are the objects we're looking for in each video.
# We can add more videos to this list later.
video_objects = {
    "IMG_4574": ["glasses", "card", "keys", "charger", "Apple Pencil", "mouse"],
    "IMG_4573": ["laptop", "glasses", "card", "phone", "keys", "pencil"],
    "IMG_4572": ["card", "keys", "Apple Pencil"],
    "IMG_4571": ["glasses", "keys", "Apple Pencil", "mouse", "keyboard"],
    "IMG_4570": ["laptop", "keys", "Apple Pencil", "mouse", "keyboard"],
    "IMG_4569": ["laptop", "keyboard", "Apple Pencil", "iPad", "mouse"]
}


def run_gemini(video_path):

    video_path = Path(video_path)
    video_name = video_path.stem

    # Check everything we need before running
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("Gemini API key is missing")

    if not video_path.exists():
        raise FileNotFoundError(f"Couldn't find video: {video_path}")

    if video_name not in video_objects:
        raise ValueError(f"No checklist found for {video_name}")

    objects = video_objects[video_name]

    # Add a few objects that arent in our checklist.
    # This helps us see if Gemini makes up objects.
    absent_objects = ["banana", "scissors", "wallet"]
    search_objects = objects + [
        item for item in absent_objects if item not in objects
    ]

    client = genai.Client()
    start_time = time.time()

    print(f"Uploading {video_name} to Gemini...")

    video = client.files.upload(file=str(video_path))

    # Wait for Gemini to finish processing the video
    while video.state and video.state.name == "PROCESSING":
        print("Gemini is processing the video...")
        time.sleep(5)
        video = client.files.get(name=video.name)

    if not video.state or video.state.name != "ACTIVE":
        raise RuntimeError("Gemini couldn't process the video")

    # We ask the same questions for every video to make
    # our results easier to compare later.
    prompt = f"""
    We're testing how well a video model can find misplaced objects.

    Watch the entire video and look for these objects:
    {", ".join(search_objects)}

    For EACH object, tell us:
    - name: the object we asked about
    - found: true or false
    - first_seen_seconds: when it first appeared
    - last_seen_seconds: when it was last visible
    - location: where it was last seen
    - confidence: low, medium, or high

    Some objects might not appear in the video.
    Only say an object was found if you can see it.
    Don't assume an object is there just because we asked about it.

    If you don't see an object, use false and null timestamps.
    Don't guess missing information.

    Return only valid JSON in this format:

    {{
        "objects": [
            {{
                "name": "keys",
                "found": true,
                "first_seen_seconds": 1.0,
                "last_seen_seconds": 5.5,
                "location": "on the desk",
                "confidence": "high"
            }}
        ]
    }}

    Include every requested object in your answer.
    """

    print("Asking Gemini to find our objects...")

    model = "gemini-3.8-flash"

    response = client.models.generate_content(
        model=model,
        contents=[video, prompt],
        config={"response_mime_type": "application/json"}
    )

    # Try to turn Gemini's response into a Python dictionary
    raw_response = response.text or ""

    try:
        predictions = json.loads(raw_response)
    except json.JSONDecodeError:
        predictions = None
        print("Gemini didn't return valid JSON, saving its response anyway")

    runtime = round(time.time() - start_time, 2)

    # Save everything for when we compare models later
    results = {
        "video": video_path.name,
        "model": model,
        "target_objects": objects,
        "queried_objects": search_objects,
        "predictions": predictions,
        "raw_response": raw_response,
        "runtime_seconds": runtime
    }

    output_folder = Path("data/outputs") / video_name
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / "gemini.json"

    with open(output_file, "w") as file:
        json.dump(results, file, indent=2)

    print(f"Finished! Results saved to {output_file}")
    print(f"Total time: {runtime} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to the video we want to test")
    args = parser.parse_args()

    run_gemini(args.video)