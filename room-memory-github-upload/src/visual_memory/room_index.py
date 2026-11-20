"""SQLite index and a serial queue for the local room app."""
import json
import sqlite3
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from visual_memory.last_seen import summarize_clip
from visual_memory.object_reid import BOTTLE_LABELS, crop_detection
from visual_memory.pipeline_config import ITEM_PROMPTS, SURFACES
from visual_memory.search import ALIASES, draw_last_seen, load_detections, resolve_query

@contextmanager
def connect(root):
    db = sqlite3.connect(root / "room.sqlite3")
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA foreign_keys = ON")
        with db:
            yield db
    finally:
        db.close()


def initialize(root):
    root.mkdir(parents=True, exist_ok=True)
    with connect(root) as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS clips (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                recorded_at REAL,
                source TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY,
                clip_id INTEGER NOT NULL REFERENCES clips(id),
                item TEXT NOT NULL,
                location TEXT,
                status TEXT NOT NULL,
                seconds REAL NOT NULL,
                confidence REAL NOT NULL,
                image TEXT NOT NULL,
                UNIQUE (clip_id, item)
            );
            CREATE INDEX IF NOT EXISTS sightings_item ON sightings(item);
            CREATE TABLE IF NOT EXISTS belongings (
                key TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                normalized_name TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS belonging_references (
                belonging_key TEXT NOT NULL REFERENCES belongings(key),
                crop TEXT NOT NULL,
                PRIMARY KEY (belonging_key, crop)
            );
        """)
        columns = {row["name"] for row in db.execute("PRAGMA table_info(sightings)")}
        for name, kind in (("crop", "TEXT"), ("identity_status", "TEXT"), ("identity_similarity", "REAL")):
            if name not in columns:
                db.execute(f"ALTER TABLE sightings ADD COLUMN {name} {kind}")


def enrolled_belongings(root):
    with connect(root) as db:
        rows = db.execute("""
            SELECT belongings.*, belonging_references.crop FROM belongings
            JOIN belonging_references ON belonging_references.belonging_key = belongings.key
        """).fetchall()
    result = {}
    for row in rows:
        category = row["category"]
        labels = {category, ITEM_PROMPTS[category]}
        canonical = ALIASES.get(category, category)
        labels.update(alias for alias, target in ALIASES.items() if target == canonical)
        if category in BOTTLE_LABELS:
            labels.update(BOTTLE_LABELS)
        belonging = result.setdefault(
            row["key"], {"name": row["name"], "labels": labels, "references": []},
        )
        belonging["references"].append(root / row["crop"])
    return result


def index_run(root, run, clip_id=None, *, surface_references=None, identity_matcher=None):
    """Write evidence first, then publish the clip and its sightings together."""
    run = Path(run).resolve()
    detections = load_detections(run)
    source = str(run)
    if clip_id is None:
        with connect(root) as db:
            if db.execute("SELECT 1 FROM clips WHERE source = ?", (source,)).fetchone():
                return
        info = run / "video_info.json"
        name = Path(json.loads(info.read_text())["path"]).name if info.is_file() else run.name

    entries = []
    evidence_folder = root / "evidence" / sha256(source.encode()).hexdigest()

    def add_sighting(item, result, identity_status=None, display_name=None):
        detection = result["detection"]
        stem = item.replace(" ", "_")
        annotation = {**detection, "label": display_name} if display_name else detection
        image = draw_last_seen(run / "frames", annotation, evidence_folder / f"{stem}.jpg")
        crop = crop_detection(result["frame"], detection["xyxy"])
        crop_path = evidence_folder / f"{stem}_crop.jpg"
        if crop is not None:
            crop.save(crop_path)
        entries.append((item, result["location"], result["status"],
                        detection["timestamp_seconds"], detection["confidence"],
                        str(image.relative_to(root)),
                        str(crop_path.relative_to(root)) if crop is not None else None,
                        identity_status, detection.get("identity_similarity")))

    for item in ITEM_PROMPTS:
        result = summarize_clip(detections, run, item, SURFACES, surface_references=surface_references)
        if result:
            add_sighting(item, result)

    if identity_matcher is not None:
        enrolled = enrolled_belongings(root)
        surfaces = [d for d in detections if d["label"].casefold() in SURFACES]
        for key, (status, matches) in identity_matcher.match(run, detections, enrolled).items():
            result = summarize_clip(surfaces + matches, run, key, SURFACES, surface_references=surface_references)
            if result:
                add_sighting(key, result, status, enrolled[key]["name"])

    with connect(root) as db:
        if clip_id is None:
            clip_id = db.execute(
                "INSERT INTO clips(name, source, status) VALUES (?, ?, 'ready')", (name, source)
            ).lastrowid
        db.executemany("""
            INSERT INTO sightings(clip_id, item, location, status, seconds, confidence, image,
                                  crop, identity_status, identity_similarity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(clip_id, *entry) for entry in entries])
        db.execute("UPDATE clips SET status = 'ready', error = NULL WHERE id = ?", (clip_id,))


def find_sightings(root, item):
    with connect(root) as db:
        rows = db.execute("""
            SELECT sightings.*, clips.name, clips.recorded_at,
                   COALESCE(belongings.name, sightings.item) AS display_name
            FROM sightings JOIN clips ON clips.id = sightings.clip_id
            LEFT JOIN belongings ON belongings.key = sightings.item
            WHERE item = ? AND clips.status = 'ready' AND sightings.identity_status IS NOT 'dismissed'
            ORDER BY clips.recorded_at DESC, clips.id DESC
        """, (item,)).fetchall()
    latest = rows[0] if rows else None
    estimated = next((row for row in rows if row["status"] in {"estimated", "confirmed"}
                      and row["identity_status"] != "review"), None)
    return latest, estimated


def dismiss_candidate(root, sighting_id):
    """Hide one unverified identity match while retaining its source evidence."""
    with connect(root) as db:
        sighting = db.execute("""
            UPDATE sightings SET identity_status = 'dismissed'
            WHERE id = ? AND identity_status IN ('review', 'dismissed')
              AND item IN (SELECT key FROM belongings)
            RETURNING item
        """, (sighting_id,)).fetchone()
        if sighting is None:
            return None
        return db.execute("SELECT name FROM belongings WHERE key = ?", (sighting["item"],)).fetchone()["name"]


def resolve_belonging(root, name):
    with connect(root) as db:
        return db.execute("SELECT key, name FROM belongings WHERE normalized_name = ?",
                          (name.casefold(),)).fetchone()


def belonging_names(root):
    with connect(root) as db:
        return [row["name"] for row in db.execute("SELECT name FROM belongings ORDER BY normalized_name")]


def add_reference_photo(root, name, photo):
    with connect(root) as db:
        belonging = db.execute("SELECT key, name FROM belongings WHERE normalized_name = ?",
                               (name.casefold(),)).fetchone()
        if belonging is None:
            return None
        path = root / "evidence" / f"reference-{uuid4().hex}.jpg"
        path.parent.mkdir(exist_ok=True)
        photo.save(path)
        db.execute("INSERT INTO belonging_references VALUES (?, ?)",
                   (belonging["key"], str(path.relative_to(root))))
    return belonging["name"]


def remember_item(root, sighting_id, name):
    name = " ".join(name.split())
    if not name or len(name) > 80:
        raise ValueError("Enter a name from 1 to 80 characters.")
    with connect(root) as db:
        source = db.execute("""
            SELECT sightings.*, clips.source, belongings.category, belongings.name AS belonging_name
            FROM sightings JOIN clips ON clips.id = sightings.clip_id
            LEFT JOIN belongings ON belongings.key = sightings.item
            WHERE sightings.id = ? AND clips.status = 'ready'
              AND sightings.identity_status IS NOT 'dismissed'
        """, (sighting_id,)).fetchone()
        if source is None:
            return None
        category = source["category"] or source["item"]
        if source["belonging_name"] and name.casefold() != source["belonging_name"].casefold():
            raise ValueError("Save this reference under the named belonging shown.")
        existing = db.execute("SELECT * FROM belongings WHERE normalized_name = ?",
                              (name.casefold(),)).fetchone()
        if existing and existing["category"] != category:
            raise ValueError("That name already belongs to a different item category.")
        if existing:
            key, name = existing["key"], existing["name"]
        else:
            key = f"belonging-{uuid4().hex}"
            db.execute("INSERT INTO belongings VALUES (?, ?, ?, ?)", (key, name, name.casefold(), category))

        crop = source["crop"]
        if not crop:
            # Older databases recorded the timestamp but did not keep the crop.
            run = Path(source["source"])
            if not run.is_dir():
                run = root / "runs" / str(source["clip_id"])
            try:
                detections = load_detections(run)
                label, _ = resolve_query(category, (d["label"] for d in detections))
                detection = max(
                    (d for d in detections if d["label"] == label and d["timestamp_seconds"] == source["seconds"]),
                    key=lambda d: d["confidence"],
                )
                frame = run / "frames" / f"frame_{detection['source_frame_index']:06d}.jpg"
                photo = crop_detection(frame, detection["xyxy"])
                if photo is None:
                    raise ValueError("Empty crop.")
            except (OSError, ValueError) as exc:
                raise ValueError("The original detection crop is unavailable. Try a newer upload.") from exc
            path = root / "evidence" / f"reference-{sighting_id}.jpg"
            path.parent.mkdir(exist_ok=True)
            photo.save(path)
            crop = str(path.relative_to(root))
            db.execute("UPDATE sightings SET crop = ? WHERE id = ?", (crop, sighting_id))
        if not (root / crop).is_file():
            raise ValueError("The saved detection crop is missing. Try a newer upload.")

        db.execute("INSERT OR IGNORE INTO belonging_references VALUES (?, ?)", (key, crop))
        db.execute("""
            INSERT INTO sightings(clip_id, item, location, status, seconds, confidence, image, crop, identity_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'confirmed')
            ON CONFLICT(clip_id, item) DO UPDATE SET
                location = excluded.location, status = excluded.status, seconds = excluded.seconds,
                confidence = excluded.confidence, image = excluded.image, crop = excluded.crop,
                identity_status = 'confirmed', identity_similarity = NULL
        """, (source["clip_id"], key, source["location"], source["status"], source["seconds"],
              source["confidence"], source["image"], crop))
    return name


def confirm_surface(root, sighting_id, surface):
    if surface not in SURFACES:
        raise ValueError("Choose a known surface.")
    with connect(root) as db:
        sighting = db.execute("""
            UPDATE sightings SET location = ?, status = 'confirmed'
            WHERE id = ? AND identity_status IS NOT 'dismissed' RETURNING item
        """, (surface, sighting_id)).fetchone()
        if sighting is None:
            return None
        item = sighting["item"]
        belonging = db.execute("SELECT name FROM belongings WHERE key = ?", (item,)).fetchone()
    return belonging["name"] if belonging else item
