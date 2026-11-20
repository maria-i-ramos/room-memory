
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Project folders
MODEL = "gemini-3.6-flash"

PREDICTIONS = Path("data/outputs/gemini") / MODEL / "identification"
CHECKLISTS = Path("data/checklists")
RESULTS = Path("results/gemini_evaluation")
GRAPHS = RESULTS / "graphs"
FAILURE_LABELS = RESULTS / "failure_case_labels.csv"

RESULTS.mkdir(parents=True, exist_ok=True)
GRAPHS.mkdir(parents=True, exist_ok=True)


# Treat common names for the same object as equivalent.
# Add more aliases after reviewing your predictions.
ALIASES = {
    "cell phone": "phone",
    "mobile phone": "phone",
    "smartphone": "phone",
    "iphone": "phone",
    "spectacles": "glasses",
    "eyeglasses": "glasses",
    "sunglasses": "glasses",
    "water bottles": "water bottle",
    "bottle of water": "water bottle",
    "bottled water": "water bottle",
    "computer mouse": "mouse",
    "wireless mouse": "mouse",
    "apple pencil": "apple pencil",
    "stylus": "apple pencil",
    "key": "keys",
    "keychain": "keys",
    "book": "book",
    "books": "book",
    "dining table": "table",
}


def normalize(name):
    if name is None:
        return ""

    name = str(name).lower().strip()
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    return ALIASES.get(name, name)


def video_id(filename):
    # Match videos to checklists despite small filename changes.
    name = Path(filename).stem.lower()
    name = re.sub(r"_objects$", "", name)
    name = re.sub(r"_from_mp4$", "", name)
    name = re.sub(r"\s*\(\d+\)$", "", name)
    name = re.sub(r"[-_ ]2$", "", name)
    return name


def save_csv(filename, rows, columns):
    with open(RESULTS / filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_objects(data):
    if isinstance(data, dict):
        objects = data.get("objects", [])
    elif isinstance(data, list):
        objects = data
    else:
        return []

    return objects if isinstance(objects, list) else []


def object_name(obj):
    if isinstance(obj, str):
        return normalize(obj)

    return normalize(
        obj.get("object") or obj.get("name")
    )


def get_number(value):
    if isinstance(value, bool):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def make_placeholder(filename, title, message):
    # Never invent accuracy values when labels are missing.
    plt.figure(figsize=(10, 5))
    plt.text(
        0.5, 0.5, message,
        ha="center", va="center",
        fontsize=12, wrap=True
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(GRAPHS / filename, dpi=180)
    plt.close()


# Load our ground-truth checklists

# Load our ground-truth checklists
checklists = {}

for path in CHECKLISTS.glob("*.json"):
    with open(path) as file:
        data = json.load(file)

    # Some checklists are dictionaries, others are lists
    if isinstance(data, dict):
        video_name = data.get("video", path.name)
    elif isinstance(data, list):
        video_name = path.name
    else:
        print(f"Skipping unsupported checklist: {path.name}")
        continue

    key = video_id(video_name)
    checklists[key] = data


# Load successful Gemini predictions
predictions_by_video = {}

for path in PREDICTIONS.glob("*.json"):
    with open(path) as file:
        result = json.load(file)

    predictions = result.get("predictions")

    if not isinstance(predictions, dict):
        print(f"Skipping invalid prediction: {path.name}")
        continue

    videos = result.get("videos") or [path.name]
    key = video_id(videos[0])

    predictions_by_video[key] = {
        "filename": videos[0],
        "objects": read_objects(predictions)
    }


# Compare identification results to ground truth
identification_rows = []
surface_rows = []
timestamp_rows = []
detailed_rows = []

for key, result in sorted(predictions_by_video.items()):

    if key not in checklists:
        print(f"No checklist found for {result['filename']}")
        continue

    truth_objects = read_objects(checklists[key])
    predicted_objects = result["objects"]

    truth = {}
    predicted = {}

    for obj in truth_objects:
        name = object_name(obj)

        if not name:
            continue

        if isinstance(obj, dict) and obj.get("present", True) is False:
            continue

        truth[name] = obj

    for obj in predicted_objects:
        name = object_name(obj)

        if not name:
            continue

        if isinstance(obj, dict) and obj.get("found", True) is False:
            continue

        predicted[name] = obj

    truth_names = set(truth)
    predicted_names = set(predicted)

    correct = truth_names & predicted_names
    missed = truth_names - predicted_names
    extra = predicted_names - truth_names

    tp = len(correct)
    fp = len(extra)
    fn = len(missed)

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None

    identification_rows.append({
        "video": result["filename"],
        "correct": tp,
        "missed": fn,
        "extra": fp,
        "precision": precision,
        "recall": recall
    })

    for name in sorted(truth_names | predicted_names):
        status = (
            "correct" if name in correct
            else "missed" if name in missed
            else "extra"
        )

        detailed_rows.append({
            "video": result["filename"],
            "object": name,
            "status": status
        })

    # Surface accuracy can only be calculated when
    # the ground-truth checklist has a surface label.
    for name in correct:
        gt = truth[name]
        pred = predicted[name]

        if not isinstance(gt, dict) or not isinstance(pred, dict):
            continue

        true_surface = gt.get("surface") or gt.get("supporting_surface")
        predicted_surface = pred.get("surface")

        if true_surface is not None:
            surface_rows.append({
                "video": result["filename"],
                "object": name,
                "ground_truth": true_surface,
                "predicted": predicted_surface,
                "correct": (
                    normalize(true_surface) == normalize(predicted_surface)
                )
            })

        # Only use independently annotated last-seen times.
        true_time = get_number(gt.get("last_seen_seconds"))
        predicted_time = get_number(pred.get("last_seen_seconds"))

        if true_time is not None:
            error = (
                abs(predicted_time - true_time)
                if predicted_time is not None else None
            )

            timestamp_rows.append({
                "video": result["filename"],
                "object": name,
                "ground_truth_seconds": true_time,
                "predicted_seconds": predicted_time,
                "absolute_error_seconds": error
            })


# Save the detailed evaluation tables
save_csv(
    "identification_accuracy.csv",
    identification_rows,
    ["video", "correct", "missed", "extra", "precision", "recall"]
)

save_csv(
    "object_comparisons.csv",
    detailed_rows,
    ["video", "object", "status"]
)

save_csv(
    "surface_accuracy.csv",
    surface_rows,
    ["video", "object", "ground_truth", "predicted", "correct"]
)

save_csv(
    "timestamp_errors.csv",
    timestamp_rows,
    [
        "video", "object", "ground_truth_seconds",
        "predicted_seconds", "absolute_error_seconds"
    ]
)


# GRAPH 1: Identification precision and recall
if identification_rows:
    names = [row["video"] for row in identification_rows]
    precision = [
        100 * (row["precision"] or 0)
        for row in identification_rows
    ]
    recall = [
        100 * (row["recall"] or 0)
        for row in identification_rows
    ]

    positions = list(range(len(names)))

    plt.figure(figsize=(max(12, len(names) * 0.6), 6))
    plt.bar(
        [x - 0.2 for x in positions],
        precision,
        width=0.4,
        label="Precision"
    )
    plt.bar(
        [x + 0.2 for x in positions],
        recall,
        width=0.4,
        label="Recall"
    )

    plt.xticks(positions, names, rotation=75, ha="right")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 105)
    plt.title("Gemini Object Identification: Precision and Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS / "identification_accuracy.png", dpi=180)
    plt.close()

    # Additional graph: correct, missed, and extra objects
    correct_counts = [row["correct"] for row in identification_rows]
    missed_counts = [row["missed"] for row in identification_rows]
    extra_counts = [row["extra"] for row in identification_rows]

    plt.figure(figsize=(max(12, len(names) * 0.6), 6))
    plt.bar(positions, correct_counts, label="Correct")
    plt.bar(
        positions, missed_counts,
        bottom=correct_counts,
        label="Missed"
    )
    plt.bar(
        positions, extra_counts,
        bottom=[
            a + b for a, b in zip(correct_counts, missed_counts)
        ],
        label="Extra"
    )

    plt.xticks(positions, names, rotation=75, ha="right")
    plt.ylabel("Number of Objects")
    plt.title("Gemini: Correct, Missed, and Extra Objects")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS / "identification_errors.png", dpi=180)
    plt.close()


# GRAPH 2: Supporting surface accuracy
if surface_rows:
    correct_count = sum(row["correct"] for row in surface_rows)
    incorrect_count = len(surface_rows) - correct_count

    plt.figure(figsize=(7, 5))
    plt.bar(
        ["Correct", "Incorrect"],
        [correct_count, incorrect_count]
    )
    plt.ylabel("Number of Objects")
    plt.title("Gemini Supporting Surface Accuracy")
    plt.tight_layout()
    plt.savefig(GRAPHS / "surface_accuracy.png", dpi=180)
    plt.close()
else:
    make_placeholder(
        "surface_accuracy.png",
        "Gemini Supporting Surface Accuracy",
        "Not evaluated yet:\n"
        "Ground-truth surface labels are needed."
    )


# GRAPH 3: Last-seen timestamp error
valid_times = [
    row for row in timestamp_rows
    if row["absolute_error_seconds"] is not None
]

if valid_times:
    labels = [
        f"{row['video']} - {row['object']}"
        for row in valid_times
    ]
    errors = [
        row["absolute_error_seconds"]
        for row in valid_times
    ]

    plt.figure(figsize=(max(10, len(labels) * 0.35), 5))
    plt.bar(range(len(labels)), errors)
    plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
    plt.ylabel("Absolute Error (seconds)")
    plt.title("Gemini Last-Seen Timestamp Error")
    plt.tight_layout()
    plt.savefig(GRAPHS / "timestamp_error.png", dpi=180)
    plt.close()
else:
    make_placeholder(
        "timestamp_error.png",
        "Gemini Last-Seen Timestamp Error",
        "Not evaluated yet:\n"
        "Verified last-seen timestamps are needed."
    )


# GRAPH 4: Accuracy grouped by failure case
# We'll add video-specific labels in a separate CSV.
failure_types = [
    "small objects in clutter",
    "objects visible for less than a second",
    "objects that disappear behind something",
    "two similar-looking objects",
    "incorrect last-seen timestamps",
    "objects that are not captured",
    "longer videos with lots of irrelevant footage"
]

if not FAILURE_LABELS.exists():
    labels = []

    for row in identification_rows:
        labels.append({
            "video": row["video"],
            "failure_case": ""
        })

    with open(FAILURE_LABELS, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["video", "failure_case"]
        )
        writer.writeheader()
        writer.writerows(labels)

failure_scores = {}

with open(FAILURE_LABELS, newline="") as file:
    for row in csv.DictReader(file):
        video = row["video"]
        category = row["failure_case"].strip().lower()

        if not category:
            continue

        matching = next(
            (
                item for item in identification_rows
                if item["video"] == video
            ),
            None
        )

        if matching is None:
            continue

        failure_scores.setdefault(category, []).append(matching)

if failure_scores:
    categories = []
    recalls = []

    for category, rows in sorted(failure_scores.items()):
        total_correct = sum(row["correct"] for row in rows)
        total_truth = sum(
            row["correct"] + row["missed"]
            for row in rows
        )

        if total_truth == 0:
            continue

        categories.append(category)
        recalls.append(100 * total_correct / total_truth)

    if categories:
        plt.figure(figsize=(11, 6))
        plt.barh(categories, recalls)
        plt.xlabel("Object Recall (%)")
        plt.xlim(0, 105)
        plt.title("Gemini Object Recall by Failure Case")
        plt.tight_layout()
        plt.savefig(GRAPHS / "failure_case_accuracy.png", dpi=180)
        plt.close()
    else:
        make_placeholder(
            "failure_case_accuracy.png",
            "Accuracy by Failure Case",
            "No evaluable failure-case labels."
        )
else:
    make_placeholder(
        "failure_case_accuracy.png",
        "Accuracy by Failure Case",
        "Not evaluated yet:\n"
        "Assign failure-case labels in failure_case_labels.csv."
    )


# Print a summary
print()
print("Gemini evaluation complete!")
print(f"Model: {MODEL}")
print(f"Videos evaluated: {len(identification_rows)}")
print(f"Surface annotations: {len(surface_rows)}")
print(f"Timestamp annotations: {len(timestamp_rows)}")
print(f"Results saved to: {RESULTS}")
print(f"Graphs saved to: {GRAPHS}")

if identification_rows:
    tp = sum(row["correct"] for row in identification_rows)
    fp = sum(row["extra"] for row in identification_rows)
    fn = sum(row["missed"] for row in identification_rows)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    print(f"Overall object precision: {precision:.1%}")
    print(f"Overall object recall: {recall:.1%}")
