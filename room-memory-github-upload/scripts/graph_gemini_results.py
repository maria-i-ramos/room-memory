
import json
import csv
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# This is where our Gemini results are saved
gemini_folder = Path("data/outputs/gemini/gemini-3.6-flash")
results_folder = Path("results")
graphs_folder = results_folder / "graphs"

results_folder.mkdir(exist_ok=True)
graphs_folder.mkdir(exist_ok=True)


# Save a list of dictionaries as a CSV file
def save_csv(filename, rows, columns):
    with open(results_folder / filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# Read Gemini's object identification results
predictions = []
summaries = []

identification_folder = gemini_folder / "identification"

for json_file in sorted(identification_folder.glob("*.json")):

    with open(json_file) as file:
        result = json.load(file)

    video_name = result["videos"][0]
    objects = (result.get("predictions") or {}).get("objects", [])

    # Count objects Gemini said it saw
    found_count = 0

    for obj in objects:
        if obj.get("found", True):
            found_count += 1

        predictions.append({
            "video": video_name,
            "object": obj.get("name"),
            "found": obj.get("found", True),
            "surface": obj.get("surface"),
            "location": obj.get("location"),
            "nearby_objects": ", ".join(obj.get("nearby_objects") or []),
            "first_seen_seconds": obj.get("first_seen_seconds"),
            "last_seen_seconds": obj.get("last_seen_seconds"),
            "confidence": obj.get("confidence")
        })

    summaries.append({
        "video": video_name,
        "objects_reported": found_count,
        "runtime_seconds": result.get("runtime_seconds", 0)
    })


# Save results that we can open in Excel or Google Sheets
save_csv(
    "gemini_predictions.csv",
    predictions,
    [
        "video", "object", "found", "surface", "location",
        "nearby_objects", "first_seen_seconds",
        "last_seen_seconds", "confidence"
    ]
)

save_csv(
    "gemini_summary.csv",
    summaries,
    ["video", "objects_reported", "runtime_seconds"]
)


# Graph 1: Objects identified in each video
if summaries:
    names = [row["video"] for row in summaries]
    counts = [row["objects_reported"] for row in summaries]
    times = [row["runtime_seconds"] for row in summaries]

    plt.figure(figsize=(max(10, len(names) * 0.6), 5))
    plt.bar(names, counts)
    plt.title("Objects Gemini Reported by Video")
    plt.ylabel("Number of Objects")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(graphs_folder / "objects_by_video.png")
    plt.close()

    # Graph 2: Processing time for each video
    plt.figure(figsize=(max(10, len(names) * 0.6), 5))
    plt.bar(names, times)
    plt.title("Gemini Processing Time by Video")
    plt.ylabel("Seconds")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(graphs_folder / "processing_time.png")
    plt.close()


# Graph 3: What surfaces Gemini thinks objects are on
surface_counts = Counter()

for row in predictions:
    if row["found"] and row["surface"]:
        surface_counts[str(row["surface"]).lower()] += 1

if surface_counts:
    surfaces = list(surface_counts.keys())
    counts = list(surface_counts.values())

    plt.figure(figsize=(9, 5))
    plt.bar(surfaces, counts)
    plt.title("Surfaces Gemini Reported")
    plt.ylabel("Number of Objects")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(graphs_folder / "surfaces_reported.png")
    plt.close()


# Summarize our movement experiments
movement_rows = []
movement_folder = gemini_folder / "movement"

for json_file in sorted(movement_folder.glob("*.json")):
    with open(json_file) as file:
        result = json.load(file)

    answer = result.get("predictions") or {}

    movement_rows.append({
        "experiment": json_file.stem,
        "videos": ", ".join(result.get("videos", [])),
        "objects_swapped_locations": answer.get(
            "objects_swapped_locations"
        ),
        "explanation": answer.get("explanation"),
        "runtime_seconds": result.get("runtime_seconds", 0)
    })

save_csv(
    "gemini_movement.csv",
    movement_rows,
    [
        "experiment", "videos", "objects_swapped_locations",
        "explanation", "runtime_seconds"
    ]
)


print("Finished! Saved Gemini results in the results folder.")
print(f"Identification videos: {len(summaries)}")
print(f"Movement experiments: {len(movement_rows)}")
print(f"Graphs saved to: {graphs_folder}")
