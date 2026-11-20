"""Launch the local room memory app."""
import argparse
import fcntl
import hmac
import io
import secrets
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_file, session, url_for
from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from visual_memory.last_seen import object_name
from visual_memory.object_reid import ClipEncoder, IdentityMatcher
from visual_memory.pipeline_config import DEFAULT_YOLOE_MODEL, ITEM_PROMPTS, SURFACES
from visual_memory.room_index import (
    add_reference_photo, belonging_names, confirm_surface, connect, dismiss_candidate,
    enrolled_belongings, find_sightings,
    index_run, initialize, remember_item, resolve_belonging,
)
from visual_memory.search import resolve_query
from visual_memory.surface_references import SurfaceReferences


def process_queue(root, device, stop, surface_references=None, identity_matcher=None):
    while not stop.is_set():
        with connect(root) as db:
            clip = db.execute("""
                UPDATE clips SET status = 'processing'
                WHERE id = (SELECT id FROM clips WHERE status = 'queued' ORDER BY id LIMIT 1)
                RETURNING id, source
            """).fetchone()
        if clip is None:
            stop.wait(1)
            continue

        run = root / "runs" / str(clip["id"])
        try:
            run.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, str(PROJECT / "scripts/run_baseline.py"), clip["source"],
                "--out", str(run), "--model", str(PROJECT / DEFAULT_YOLOE_MODEL), "--device", device,
            ]
            if identity_matcher and any("bottle" in item["labels"] for item in enrolled_belongings(root).values()):
                command.append("--include-bottle-candidates")
            with (run / "processing.log").open("w") as log:
                subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=1800)
            index_run(root, run, clip["id"], surface_references=surface_references,
                      identity_matcher=identity_matcher)
        except Exception:
            # Keep the queue moving even when a particular recording fails.
            import traceback
            traceback.print_exc()
            message = f"Processing failed. See runs/{clip['id']}/processing.log."
            with connect(root) as db:
                db.execute("UPDATE clips SET status = 'failed', error = ? WHERE id = ?", (message, clip["id"]))


def create_app(root, reid_enabled=False):
    app = Flask(__name__, template_folder="room_templates")
    app.config.update(
        SECRET_KEY=secrets.token_hex(32), TRUSTED_HOSTS=["127.0.0.1", "localhost"],
        MAX_CONTENT_LENGTH=251 * 1024 * 1024, SESSION_COOKIE_SAMESITE="Strict",
    )

    @app.template_filter("recorded")
    def recorded(timestamp):
        return datetime.fromtimestamp(timestamp).strftime("%b %d, %Y at %I:%M %p") if timestamp is not None else ""

    @app.get("/")
    def home():
        if "upload_token" not in session:
            session["upload_token"] = secrets.token_urlsafe(32)
        query = request.args.get("q", "").strip()
        latest = located = None
        label = ""
        if query:
            try:
                name = object_name(query)
                belonging = resolve_belonging(root, name)
                if belonging and request.args.get("category") != "1":
                    label = belonging["name"]
                    latest, located = find_sightings(root, belonging["key"])
                else:
                    label, _ = resolve_query(name, ITEM_PROMPTS)
                    latest, located = find_sightings(root, label)
            except ValueError as exc:
                flash(str(exc))
        return render_template(
            "room.html", query=query, label=label, latest=latest, located=located,
            items=ITEM_PROMPTS, surfaces=sorted(SURFACES), upload_token=session["upload_token"],
            reid_enabled=reid_enabled, remembered=belonging_names(root),
            now=datetime.now().strftime("%Y-%m-%dT%H:%M"),
        )

    def check_form_token():
        token = session.get("upload_token")
        if not token or not hmac.compare_digest(token, request.form.get("upload_token", "")):
            abort(400, "Refresh the local app before submitting this form.")

    @app.post("/upload")
    def upload():
        check_form_token()
        video = request.files.get("video")
        name = Path(video.filename or "").name if video else ""
        if Path(name).suffix.lower() not in {".mov", ".mp4", ".m4v"}:
            flash("Choose a MOV, MP4, or M4V recording.")
            return redirect(url_for("home"), code=303)
        try:
            timestamp = datetime.strptime(request.form.get("recorded_at", ""), "%Y-%m-%dT%H:%M").timestamp()
        except (ValueError, OverflowError, OSError):
            flash("Enter the recording date and time.")
            return redirect(url_for("home"), code=303)

        folder = root / "uploads"
        folder.mkdir(exist_ok=True)
        path = folder / f"{uuid4().hex}{Path(name).suffix.lower()}"
        try:
            video.save(path)
            if not 0 < path.stat().st_size <= 250 * 1024 * 1024:
                path.unlink()
                flash("Choose a nonempty recording up to 250 MB.")
                return redirect(url_for("home"), code=303)
            with connect(root) as db:
                db.execute("""
                    INSERT INTO clips(name, recorded_at, source, status) VALUES (?, ?, ?, 'queued')
                """, (name, timestamp, str(path)))
        except (OSError, sqlite3.Error):
            path.unlink(missing_ok=True)
            raise
        flash("Recording uploaded. Search is available while it processes.")
        return redirect(url_for("home"), code=303)

    @app.errorhandler(413)
    def too_large(_error):
        flash("That upload is too large. Choose a recording up to 250 MB.")
        return redirect(url_for("home"), code=303)

    @app.get("/clips")
    def clips():
        with connect(root) as db:
            rows = db.execute("""
                SELECT name, recorded_at, status, error FROM clips
                ORDER BY id DESC LIMIT 30
            """).fetchall()
        return jsonify([{**dict(row), "recorded": recorded(row["recorded_at"])} for row in rows])

    @app.get("/evidence/<int:sighting_id>")
    def evidence(sighting_id):
        with connect(root) as db:
            row = db.execute("SELECT image FROM sightings WHERE id = ?", (sighting_id,)).fetchone()
        if row is None:
            abort(404)
        image = root / row["image"]
        if not image.is_file():
            abort(404)
        return send_file(image, mimetype="image/jpeg", max_age=0)

    @app.post("/sightings/<int:sighting_id>/surface")
    def set_surface(sighting_id):
        check_form_token()
        try:
            item = confirm_surface(root, sighting_id, request.form.get("surface"))
        except ValueError as exc:
            abort(400, str(exc))
        if item is None:
            abort(404)
        flash("Location confirmed for this recording.")
        return redirect(url_for("home", q=item), code=303)

    @app.post("/sightings/<int:sighting_id>/dismiss")
    def dismiss(sighting_id):
        check_form_token()
        name = dismiss_candidate(root, sighting_id)
        if name is None:
            abort(404)
        flash("Suggestion dismissed.")
        return redirect(url_for("home", q=name), code=303)

    @app.post("/sightings/<int:sighting_id>/remember")
    def remember(sighting_id):
        if not reid_enabled:
            abort(404)
        check_form_token()
        try:
            name = remember_item(root, sighting_id, request.form.get("name", ""))
        except ValueError as exc:
            flash(str(exc))
            return redirect(url_for("home", q=request.form.get("query", "")), code=303)
        if name is None:
            abort(404)
        flash(f"Saved a reference photo for {name}.")
        return redirect(url_for("home", q=name), code=303)

    @app.post("/reference-photo")
    def reference_photo():
        if not reid_enabled:
            abort(404)
        check_form_token()
        name = request.form.get("name", "")
        upload = request.files.get("photo")
        try:
            if upload is None:
                raise ValueError()
            limit = 10 * 1024 * 1024
            data = upload.stream.read(limit + 1)
            if len(data) > limit:
                raise ValueError()
            with Image.open(io.BytesIO(data), formats=["JPEG", "PNG"]) as image:
                if image.width * image.height > 32_000_000:
                    raise ValueError()
                photo = ImageOps.exif_transpose(image).convert("RGB")
                photo.thumbnail((1024, 1024))
        except (OSError, ValueError, Image.DecompressionBombError):
            flash("Choose a JPG or PNG photo up to 10 MB and 32 megapixels.")
            return redirect(url_for("home", q=name), code=303)
        label = add_reference_photo(root, name, photo)
        if label is None:
            abort(404)
        flash(f"Saved another reference photo for {label}.")
        return redirect(url_for("home", q=label), code=303)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data/room-app")
    parser.add_argument("--import-runs", nargs="+", type=Path, default=[], help="Saved runs, oldest first")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--surface-references", type=Path, help="Folder with cropped bed and rug examples")
    parser.add_argument("--reid-weights", type=Path, help="Local CLIP ViT-B/32 weights for remembering named items")
    args = parser.parse_args()
    matcher = None
    if args.reid_weights:
        if not args.reid_weights.is_file():
            parser.error(f"Missing CLIP weights: {args.reid_weights}")
        matcher = IdentityMatcher(ClipEncoder(args.reid_weights.resolve()))
    try:
        references = SurfaceReferences(args.surface_references) if args.surface_references else None
    except ValueError as exc:
        parser.error(str(exc))
    root = args.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "app.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("The app is already running with this data folder.")
        initialize(root)
        for run in args.import_runs:
            index_run(root, run, surface_references=references, identity_matcher=matcher)
        with connect(root) as db:
            db.execute("UPDATE clips SET status = 'queued' WHERE status = 'processing'")
        stop = threading.Event()
        worker = threading.Thread(target=process_queue, args=(root, args.device, stop, references, matcher), daemon=True)
        worker.start()
        try:
            create_app(root, reid_enabled=matcher is not None).run(host="127.0.0.1", port=args.port, use_reloader=False)
        finally:
            stop.set()
            worker.join(timeout=1)


if __name__ == "__main__":
    main()
