"""Local demo app: pick a clip and a viewer, see which ad that viewer gets and why, watch it, verify any export.

  .venv/bin/uvicorn app.main:app --port 8000      # then open http://127.0.0.1:8000
Analysis (SAM 3 + tracking) is done ahead of time by scripts/analyse.py and scripts/track_eval.py and is shown as
cached. Rendering a (clip, brand) pair runs for real the first time and is reused afterwards.
"""
import functools
import hashlib
import hmac
import secrets
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zlib
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import auction, creator, discover, rank, watermark  # noqa: E402
from scripts.render import render_campaign  # noqa: E402

app = FastAPI(title="AdPlace")
for d in ("outputs", "assets/clips", "runs"):   # git-ignored: a fresh clone has none of them
    (ROOT / d).mkdir(parents=True, exist_ok=True)
PLAYERS = Path(__file__).parent / "static" / "player"


@app.middleware("http")
async def removed_videos_stay_gone(request, call_next):
    """A video its creator removed is gone for viewers: its player files (the video, the artwork) are refused too."""
    parts = request.url.path.split("/")
    gone = JSONResponse({"detail": "This video was removed by its creator."}, status_code=404)
    if len(parts) > 4 and parts[1:3] == ["static", "player"] and (PLAYERS / parts[3] / "removed").exists():
        return gone
    if len(parts) > 3 and parts[1] == "runs":
        # a run's folder holds the creator's source video and logs: only its poster and thumbnails are public,
        # and none of it once the creator removed the video
        if not (parts[3] == "poster.jpg" or (parts[3] == "thumbs" and len(parts) == 5 and parts[4].endswith(".jpg"))):
            return JSONResponse({"detail": "Not public."}, status_code=404)
        if (ROOT / "runs" / parts[2] / "removed").exists():
            return gone
    if len(parts) > 2 and parts[1] == "clips" and parts[2] == "uploads":   # creators' source videos stay private
        return JSONResponse({"detail": "Not public."}, status_code=404)
    return await call_next(request)


app.mount("/outputs", StaticFiles(directory=ROOT / "outputs"), name="outputs")
app.mount("/clips", StaticFiles(directory=ROOT / "assets" / "clips"), name="clips")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
app.mount("/runs", StaticFiles(directory=ROOT / "runs"), name="runs")


def page(name):
    """An app page, revalidated on every visit so a browser never runs a stale copy after an update."""
    return FileResponse(Path(__file__).parent / "static" / name, headers={"Cache-Control": "no-cache"})


def load(name):
    if name == "clips.json":  # hand-made clips plus uploaded ones
        from pipeline.clipcfg import all_clips
        return all_clips()
    return json.loads((ROOT / "configs" / name).read_text())


def clip_entry(clip_id):
    clips = load("clips.json")
    if clip_id not in clips:
        raise HTTPException(404, f"unknown clip {clip_id}")
    return clips[clip_id]


def slot_quads(slot):
    """Per-frame quads: tracked quads for flat surfaces; mask bounding boxes for curved ones (facts only)."""
    d = ROOT / slot["analysis"]
    if (d / "quads.npy").exists():
        return np.load(d / "quads.npy")
    masks = np.load(d / "surface_masks.npz")["masks"]
    quads = np.full((len(masks), 4, 2), np.nan, np.float32)
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if len(xs):
            quads[i] = [[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]]
    return quads


@functools.lru_cache(maxsize=64)
def slots_for(clip_id):
    """Facts for every placement slot in a clip (wide scenes have several)."""
    out = []
    for sl in clip_entry(clip_id)["slots"]:
        quads = slot_quads(sl)
        status = ["ok" if np.isfinite(q).all() else "lost" for q in quads]
        meta = json.loads((ROOT / sl["analysis"] / "meta.json").read_text())
        metrics = discover.slot_metrics(quads, status, tuple(meta["size"]), meta["fps"])
        if sl.get("shape", {}).get("type") == "cylinder":
            # A wrap's artwork is the unrolled label, not the object's outline: judge creatives by that ratio.
            w, h = (int(v) for v in sl["shape"].get("creative", "4x3").split("x"))
            metrics["aspect_ratio"] = w / h
        out.append({"slot_id": sl["slot_id"], "surface": sl["surface"], "metrics": metrics})
    return out


@functools.lru_cache(maxsize=16)
def scene_jpeg(clip_id):
    """A mid-clip frame with every slot outlined and labelled by its slot id, for image-capable rankers (Muse)."""
    entry = clip_entry(clip_id)
    cap = cv2.VideoCapture(str(ROOT / clip_id))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    for sl in entry["slots"]:
        q = slot_quads(sl)
        q = q[60] if np.isfinite(q[60]).all() else next((x for x in q if np.isfinite(x).all()), None)
        if q is None:
            continue
        cv2.polylines(frame, [np.round(q).astype(np.int32)], True, (0, 255, 0), max(2, frame.shape[1] // 400))
        x, y = np.round(q.min(axis=0)).astype(int)
        cv2.putText(frame, sl["slot_id"], (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, frame.shape[1] / 1600,
                    (0, 255, 0), 2, cv2.LINE_AA)
    scale = min(1.0, 1024 / max(frame.shape[:2]))
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


@app.get("/player")
def player():
    return page("player.html")


@app.get("/reels")
def reels():
    return page("reels.html")


class ViewerRenderRequest(BaseModel):
    clip: str          # stem, e.g. "patio_waiter"
    profile_id: str
    assignment: dict[str, str] | None = None   # what this viewer's auction chose; default: the export's choice
    token: str | None = None   # the auction's signature over that assignment (auction_token)


def _secret():
    """This server's signing key, kept across restarts (cache/ is git-ignored)."""
    p = ROOT / "cache" / "auction_secret"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(secrets.token_hex(32))
    return p.read_text().strip().encode()


def auction_token(clip, profile_id, assignment):
    """Signs what an auction this server ran awarded, so a download can only name brands that actually won (a copy
    naming any brand a browser asks for would be an impression nobody bought)."""
    msg = json.dumps([clip, profile_id, sorted(assignment.items())]).encode()
    return hmac.new(_secret(), msg, hashlib.sha256).hexdigest()


class AuctionRequest(BaseModel):
    clip: str          # stem of a player export
    profile_id: str
    jitter: float = 0.15
    fresh: bool = False   # "run the bid again": a new impression; otherwise a viewer's first bid on a video repeats


def player_export(stem):
    d = Path(__file__).parent / "static" / "player" / stem
    # a video the creator removed is gone for viewers too: no settings, auction or download
    if not stem.replace("_", "").replace("-", "").isalnum() or not (d / "track.json").exists() or (d / "removed").exists():
        raise HTTPException(404, "We don't have a player for that video.")
    return d, json.loads((d / "track.json").read_text()), json.loads((d / "assignments.json").read_text())


@app.post("/api/auction")
def run_auction(req: AuctionRequest):
    """The bid when a viewer presses play (pipeline/auction.py): every placement was prepared at upload for any
    brand, so this only picks names and prices, in well under a millisecond."""
    _, track, viewers = player_export(req.clip)
    if req.profile_id not in viewers:
        raise HTTPException(404, f"unknown profile {req.profile_id}")
    entry = load("clips.json").get(track["clip"], {})
    settings = creator.load(player_export_dir(req.clip))
    r = auction.run(creator.allowed(auction.placements_of(track), settings), load("inventory.json"),
                    viewers[req.profile_id].get("preferences", []), entry.get("context_tags", []),
                    jitter=min(max(req.jitter, 0.0), 0.5),
                    # the same viewer reloading the same video sees the same brands; only a deliberate re-bid moves them
                    seed=None if req.fresh else zlib.crc32(f"{req.clip}|{req.profile_id}".encode()))
    return {**r, "context_tags": entry.get("context_tags", []), "creator_enabled": settings["enabled"],
            "token": auction_token(req.clip, req.profile_id, r["assignment"])}


def player_export_dir(stem):
    return Path(__file__).parent / "static" / "player" / stem


def latest_report(stem):
    """The newest upload report for this export, for its declined places and their reasons."""
    best = None
    for p in (ROOT / "runs").glob("*/report.json"):
        if (p.parent / "removed").exists():   # a run the creator removed from their library
            continue
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        t = r.get("finished_at") or p.stat().st_mtime
        if r.get("clip_stem") == stem and (best is None or t > best[0]):
            best = (t, r)
    return best[1] if best else {}


@app.get("/api/creator/{stem}")
def creator_get(stem: str):
    """The creator's settings for one video, with what they're deciding over: the places that passed (approve each)
    and the brands with artwork for them (allow each), plus the places we declined and why."""
    d, track, _ = player_export(stem)
    settings = creator.load(d)
    report = latest_report(stem)
    found = {p["id"]: p for p in report.get("placements", [])}
    campaigns = {c["brand_id"]: c for c in load("inventory.json")["campaigns"]}
    places, brands = [], {}
    centre = {}   # mean x of each place in the frame: "left", "middle" or "right" tells two cups apart
    for sl in track.get("slots", []):
        xs = [pt[0] for q in sl["quads"] if q for pt in q]
        centre[sl["id"]] = sum(xs) / len(xs) if xs else None
    for o in track.get("objects", []):
        xs = [b[0] + b[2] / 2 for b in o["boxes"] if b]
        centre[o["id"]] = sum(xs) / len(xs) if xs else None
    where = lambda x: None if x is None else ("left" if x < track["width"] / 3 else  # noqa: E731
                                               "right" if x > 2 * track["width"] / 3 else "middle")
    for p in auction.placements_of(track):
        r = found.get(p["id"], {})
        places.append({"id": p["id"], "surface": p["surface"], "approved": settings["places"].get(p["id"], True),
                       "on_screen_s": r.get("on_screen_s"), "thumb": r.get("thumb"),
                       "where": where(centre.get(p["id"]))})
        for b in p["brands"]:
            c = campaigns.get(b, {})
            if p["surface"] not in c.get("eligible_surfaces", [p["surface"]]) or \
                    (p.get("contents") and p["contents"] in c.get("unsuitable_contents", [])):
                continue   # artwork exists, but not this kind of surface or drink (no sportswear, no coffee on juice)
            brands.setdefault(b, {"id": b, "category": campaigns.get(b, {}).get("category"),
                                  "allowed": settings["brands"].get(b, True), "places": []})["places"].append(p["id"])
    unsupported = lambda p: any(str(d).startswith("not yet:") for d in p.get("details", []))  # noqa: E731
    declined = [{"id": p["id"], "surface": p.get("surface"), "thumb": p.get("thumb"),
                 "reason": (p.get("reasons") or ["It did not meet one of the checks."])[0]}
                for p in report.get("placements", []) if p.get("decision") != "placed" and not unsupported(p)]
    return {"clip": stem, "enabled": settings["enabled"], "places": places, "brands": list(brands.values()),
            "declined": declined, "not_supported": [p.get("surface") for p in report.get("placements", [])
                                                     if unsupported(p)], "scene": report.get("scene")}


@app.delete("/api/videos/{stem}")
def remove_video(stem: str):
    """Take a video out of the library: placement off, every run of it marked removed. Nothing is deleted from disk,
    so it can be brought back by deleting the runs' `removed` files."""
    if not stem.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(404, "unknown video")
    runs = []
    for rp in (ROOT / "runs").glob("r_*/report.json"):
        try:
            if json.loads(rp.read_text()).get("clip_stem") == stem:
                runs.append(rp.parent)
        except ValueError:
            continue
    if not runs:
        raise HTTPException(404, "unknown video")
    d = player_export_dir(stem)
    if (d / "track.json").exists():
        creator.update(d, {"enabled": False})
        (d / "removed").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    for r in runs:
        (r / "removed").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return {"removed": stem, "runs": len(runs)}


@app.delete("/api/runs/{run_id}")
def remove_run(run_id: str):
    """Clear an upload that failed or is still processing (it has no report, so no video to remove)."""
    d = _run_dir(run_id)
    if (d / "report.json").exists():
        raise HTTPException(409, "this upload finished: remove the video instead")
    (d / "removed").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return {"removed": run_id}


@app.get("/api/players")
def players():
    """The videos a viewer can open in the full player: exports with a placement, not removed, publishable."""
    out = []
    clips, uploads = load("clips.json"), load("uploads.json")
    # real uploads first (the default video), the hand-built clip last
    for d in sorted(PLAYERS.iterdir(), key=lambda d: (not d.name.startswith("upload_"), d.name)):
        if not ((d / "track.json").exists() and (d / "stack.mp4").exists()) or (d / "removed").exists():
            continue
        clip = json.loads((d / "track.json").read_text()).get("clip", "")
        entry = clips.get(clip) or uploads.get(clip) or {}
        if "not for publication" in str(entry.get("source", "")):
            continue
        report = latest_report(d.name)
        if d.name.startswith("upload_") and not report.get("player_url"):
            continue   # an older export whose newest run places nothing
        label = report.get("source_name") or d.name.replace("_", " ").capitalize() + " (set up by hand)"
        out.append({"name": d.name, "label": label})
    return out


@app.get("/api/creator/{stem}/rev")
def creator_rev(stem: str):
    """Cheap: an open player polls this and keeps what it shows within the creator's current settings."""
    d = player_export_dir(stem)
    if not stem.replace("_", "").replace("-", "").isalnum() or not (d / "track.json").exists():
        raise HTTPException(404, "We don't have a player for that video.")
    p = d / creator.NAME
    settings = creator.load(d)
    return {"rev": p.stat().st_mtime_ns if p.exists() else 0, "removed": (d / "removed").exists(),
            "enabled": settings["enabled"] and not (d / "removed").exists(),
            "places": settings["places"], "brands": settings["brands"]}


class CreatorSettings(BaseModel):   # fields left out keep their saved value (the library only flips `enabled`)
    enabled: bool | None = None
    places: dict[str, bool] | None = None
    brands: dict[str, bool] | None = None


@app.put("/api/creator/{stem}")
def creator_put(stem: str, req: CreatorSettings):
    d, track, _ = player_export(stem)
    ids = {p["id"] for p in auction.placements_of(track)}
    names = {b for p in auction.placements_of(track) for b in p["brands"]}
    if set(req.places or {}) - ids or set(req.brands or {}) - names:
        raise HTTPException(400, "settings name a place or brand this video doesn't have")
    return creator.update(d, req.model_dump(exclude_none=True))


@app.post("/api/viewer_render")
def viewer_render(req: ViewerRenderRequest):
    """The downloadable file for one viewer: the same assignment the live player shows, rendered on the server with
    the disclosure label and a Video Seal watermark naming the campaigns (the browser cannot embed the watermark).
    Cached per (clip, assignment). Rendered in a child process: in this process (torch + the app's imports) MKL
    segfaults inside the render, which would take the whole demo server down."""
    d, track, viewers = player_export(req.clip)   # checks the name: it becomes part of the output path
    if req.profile_id not in viewers:
        raise HTTPException(404, f"unknown profile {req.profile_id}")
    clip = track["clip"]
    assignment = viewers[req.profile_id]["assignment"]
    if req.assignment is not None:   # from /api/auction, signed by it: only brands that won this viewer's auction
        if not req.token or not hmac.compare_digest(req.token, auction_token(req.clip, req.profile_id, req.assignment)):
            raise HTTPException(403, "That isn't what this viewer's auction awarded. Run the bid again, then download.")
        brands = {s["id"]: set(s["brands"]) for s in track["slots"] + track.get("objects", [])}
        if any(k not in brands or v not in brands[k] for k, v in req.assignment.items()):
            raise HTTPException(400, "assignment names a placement or brand this export doesn't have")
        assignment = req.assignment
    settings = creator.load(d)
    if not settings["enabled"]:
        raise HTTPException(403, "the creator hasn't turned on product placement for this video")
    if any(not settings["places"].get(k, True) or not settings["brands"].get(v, True) for k, v in assignment.items()):
        raise HTTPException(403, "the creator hasn't approved that place or brand")
    if not assignment:
        raise HTTPException(409, "No ad won for this viewer, so there's nothing to download: the original plays.")
    tag = "_".join(f"{k}-{v}" for k, v in sorted(assignment.items()))
    out = ROOT / "outputs" / "viewers" / f"{req.clip}__{tag}__{render_version(d)}.mp4"
    t0 = time.time()
    with RENDER_LOCKS.setdefault(out.name, threading.Lock()):   # a double click waits for the first render
        return _render_viewer(req, clip, track, assignment, out, t0)


RENDER_LOCKS: dict[str, threading.Lock] = {}


def render_version(d):
    """What a viewer's download depends on besides the assignment: the export and the renderer's code."""
    h = hashlib.sha1()
    for p in [d / "track.json", *sorted((d / "art").glob("*")), ROOT / "scripts" / "render_dynamic.py",
              ROOT / "scripts" / "glass_decal.py"]:
        if p.is_file():
            h.update(p.name.encode() + str(p.stat().st_size).encode() + str(p.stat().st_mtime_ns).encode())
    return h.hexdigest()[:8]


def _render_viewer(req, clip, track, assignment, out, t0):
    cached = out.with_suffix(".json").exists()
    if not cached and not (ROOT / clip).exists():
        raise HTTPException(409, "the original video isn't on this server (a fresh copy of the repo carries the live "
                                 "players, not the source clips), so a watermarked copy can't be rendered here. "
                                 "Upload the video to render one.")
    if not cached:
        clip_entry(clip)
        out.parent.mkdir(parents=True, exist_ok=True)
        run = subprocess.run([sys.executable, str(ROOT / "scripts" / "render_dynamic.py"), "render", clip,
                              "--assign", *(f"{k}={v}" for k, v in assignment.items()), "--out", str(out),
                              "--start", str(track["start"])], cwd=ROOT, capture_output=True, text=True)
        if run.returncode != 0 or not out.with_suffix(".json").exists():
            raise HTTPException(500, f"render failed (exit {run.returncode}): {run.stderr[-500:]}")
    report = json.loads(out.with_suffix(".json").read_text())
    return {"video_url": "/outputs/viewers/" + out.name, "assignment": assignment, "verified": report["verified"],
            "manifest_id": report["manifest_id"], "cached": cached, "seconds": round(time.time() - t0, 1)}


@app.get("/")
def home():
    return page("videos.html")


@app.get("/videos")
def videos_page():
    return page("videos.html")


@app.get("/verify")
def index():
    """The first demo page: pick a clip and viewer, render, and check any file for the Video Seal watermark."""
    return page("index.html")


def failure_text(msg):
    """An older run logged the raw exception ("CalledProcessError: Command '['ffmpeg', ...") as its error: show the
    creator a sentence instead (scripts/ingest.py failure_message writes these now)."""
    if re.match(r"^[A-Za-z]+(Error|Exception|Exit)\b", msg or ""):
        return ("We couldn't read that video file. Try an mp4, mov or webm that plays on your computer."
                if "ffmpeg" in msg else "Something went wrong while processing this video. Please try again.")
    return msg


MAX_RUNNING = 2   # uploads processed at once
STALLED_S = 30 * 60   # a run with no progress for this long has stopped (the process died or the machine slept)


def poster(run_dir, report):
    """A still from the video for its card, made once: a third of the way in, 480 px wide."""
    out = run_dir / "poster.jpg"
    if not out.exists():
        src = next((p for p in [run_dir / "input.mp4", *run_dir.glob("source.*"), ROOT / report.get("clip", "-")]
                    if p.is_file()), None)
        if src is None:
            return None
        cap = cv2.VideoCapture(str(src))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 3))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        frame = cv2.resize(frame, (480, round(frame.shape[0] * 480 / frame.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return f"/runs/{run_dir.name}/poster.jpg"


@app.get("/api/videos")
def videos():
    """The creator's library: every uploaded video (its newest finished run), plus uploads still processing."""
    latest, active = {}, []
    now = time.time()
    for d in (ROOT / "runs").iterdir():
        if not d.is_dir() or not d.name.startswith("r_"):
            continue
        rp, pp = d / "report.json", d / "progress.jsonl"
        if (d / "removed").exists():
            continue
        if rp.exists():
            try:
                r = json.loads(rp.read_text())
            except ValueError:
                continue
            t = r.get("finished_at") or rp.stat().st_mtime
            stem = r.get("clip_stem")
            if stem and (stem not in latest or t > latest[stem][0]):
                latest[stem] = (t, d, r)
        elif pp.exists() and now - pp.stat().st_mtime < STALLED_S:
            lines = pp.read_text().splitlines()
            last = json.loads(lines[-1]) if lines else {"stage": "upload", "pct": 0, "msg": "starting"}
            name = (d / "name.txt").read_text().strip() if (d / "name.txt").exists() else "Your video"
            active.append({"run_id": d.name, "name": name, "status": "failed" if last["stage"] == "error" else "processing",
                           "stage": last["stage"], "pct": last.get("pct", 0), "message": failure_text(last.get("msg", "")),
                           "uploaded": d.stat().st_mtime})
    items = []
    for stem, (t, d, r) in latest.items():
        has_player = (player_export_dir(stem) / "track.json").exists() and bool(r.get("player_url"))
        summary = r.get("summary", {})
        items.append({"run_id": r.get("run_id", d.name), "clip": stem, "status": "ready",
                      "name": r.get("source_name") or stem, "duration_s": r.get("duration_s"),
                      "placed": summary.get("placed", 0) if has_player else 0,
                      "found": sum(1 for p in r.get("placements", []) if not any(
                          str(x).startswith("not yet:") for x in p.get("details", []))), "note": r.get("note"),
                      "enabled": creator.load(player_export_dir(stem))["enabled"] if has_player else False,
                      "has_player": has_player, "poster": poster(d, r), "updated": t,
                      "places": [p["surface"] for p in r.get("placements", []) if p.get("decision") == "placed"]})
    active.sort(key=lambda v: -v["uploaded"])
    items.sort(key=lambda v: (-v["placed"], -v["updated"]))
    # the benchmark and development runs aren't a creator's uploads: they get their own tab, and the 23 unseen
    # videos a funnel (videos -> places measured -> placed), so yield is shown as a measurement, not a library
    unseen = set(load_config("testset.json").get("unseen", []))
    for v in items:
        v["test"] = not v["has_player"] and not v["run_id"].startswith("r_20")
        v["unseen"] = v["clip"] in unseen
    tested = [v for v in items if v["unseen"]]
    funnel = {"videos": len(unseen), "measured": sum(v["found"] for v in tested),
              "placed": sum(v["placed"] for v in tested), "with_ads": sum(1 for v in tested if v["placed"])}
    return {"processing": active, "videos": items, "funnel": funnel}


def load_config(name):
    p = ROOT / "configs" / name
    return json.loads(p.read_text()) if p.exists() else {}


@app.get("/api/catalog")
def catalog():
    clips = load("clips.json")
    return {"clips": [{"id": k, "title": v.get("title", k), "url": "/clips/" + Path(k).name,
                       "context_tags": v["context_tags"], "slots": len(v["slots"])}
                      for k, v in clips.items() if not v.get("hidden")],
            "profiles": load("profiles.json")["profiles"],
            "brands": [c["brand_id"] for c in load("inventory.json")["campaigns"]]}


class ChooseRequest(BaseModel):
    clip: str
    profile_id: str


@app.post("/api/choose")
def choose(req: ChooseRequest):
    """Deterministic facts and filtering, then a model choice (Muse, then Llama), validated, else the rules."""
    entry = clip_entry(req.clip)
    profile = next((p for p in load("profiles.json")["profiles"] if p["profile_id"] == req.profile_id), None)
    if profile is None:
        raise HTTPException(404, f"unknown profile {req.profile_id}")
    inventory = load("inventory.json")
    slots = slots_for(req.clip)
    eligible = discover.eligible_pairs(slots, inventory, entry["context_tags"])
    result = rank.choose(slots, eligible, inventory, profile, entry["context_tags"], image=scene_jpeg(req.clip))
    chosen = next((sl for sl in slots if sl["slot_id"] == result["choice"].get("slot_id")), slots[0])
    return {"slot": chosen, "slots": slots, "eligible": eligible, **result}


class RenderRequest(BaseModel):
    clip: str
    brand: str
    slot_id: str | None = None


@app.post("/api/render")
def render(req: RenderRequest):
    entry = clip_entry(req.clip)
    inventory = load("inventory.json")
    slot = next((sl for sl in entry["slots"] if sl["slot_id"] == req.slot_id), entry["slots"][0])
    eligible = discover.eligible_pairs(slots_for(req.clip), inventory, entry["context_tags"])
    if not any(e["brand_id"] == req.brand and e["slot_id"] == slot["slot_id"] for e in eligible):
        raise HTTPException(409, f"{req.brand} is not an eligible placement for {slot['slot_id']}")
    tag = slot["slot_id"].rsplit("-", 1)[-1] if len(entry["slots"]) > 1 else None
    out = ROOT / "outputs" / f"{Path(req.clip).stem}{'_s' + tag if tag else ''}_{req.brand}.json"
    analysis = ROOT / slot["analysis"]
    newest = max(f.stat().st_mtime for f in analysis.iterdir())
    cached = out.exists() and out.stat().st_mtime > newest
    t0 = time.time()
    report = json.loads(out.read_text()) if cached else render_campaign(
        req.clip, str(analysis), req.brand, slot["emissive"], shape=slot.get("shape"),
        tag=("s" + tag) if tag else None)
    return {**report, "video_url": "/outputs/" + Path(report["output"]).name, "cached": cached,
            "seconds": round(time.time() - t0, 1)}


def named_places(clip_stem, campaigns):
    """c07 -> brewlark-2026-q4 as a person reads it: the kind of place, the brand, the campaign."""
    surfaces = {}
    t = PLAYERS / clip_stem / "track.json"
    if clip_stem.replace("_", "").replace("-", "").isalnum() and t.exists():
        track = json.loads(t.read_text())
        surfaces = {s["id"]: s.get("surface") for s in track.get("slots", []) + track.get("objects", [])}
    brands = {c["campaign_token"]: c["brand_id"] for c in load("inventory.json")["campaigns"]}
    return [{"place": (surfaces.get(pid) or "place").replace("_", " "), "brand": brands.get(tok, tok), "campaign": tok}
            for pid, tok in campaigns.items()]


VERIFY_FRAMES = 96   # every other frame of the first ~8 s at 24 fps: what the download's own check reads


def check_file(path, candidates):
    """Decode a bounded sample (1280 px at most) and look for a watermark. An unreadable file is the sender's
    problem (400), not a server error."""
    cap = cv2.VideoCapture(str(path))
    frames, i = [], 0
    while len(frames) < VERIFY_FRAMES:
        ok, f = cap.read()
        if not ok:
            break
        if i % 2 == 0:
            if max(f.shape[:2]) > 1280:
                s = 1280 / max(f.shape[:2])
                f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            frames.append(f)
        i += 1
    cap.release()
    if len(frames) < 8:
        raise HTTPException(400, "We couldn't read a video in that file. Try an mp4, mov or webm that plays on "
                                 "your computer.")
    return watermark.verify(frames, candidates)


@app.post("/api/verify")
async def verify(file: UploadFile):
    # candidates: every campaign token (a single-ad render) and every scene manifest this server issued (a viewer's
    # download names all its placements at once, scripts/render_scene.py manifest_id)
    tokens = [c["campaign_token"] for c in load("inventory.json")["campaigns"]]
    manifests = {}
    for m in (ROOT / "outputs").glob("**/*.json"):
        try:
            r = json.loads(m.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(r, dict) and str(r.get("manifest_id", "")).startswith("scene-"):
            manifests[r["manifest_id"]] = r.get("manifest")
    suffix = Path(file.filename or "x.mp4").suffix.lower()
    if suffix not in (".mp4", ".mov", ".m4v", ".webm"):
        raise HTTPException(400, "please check an mp4, mov or webm video")
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        size = 0
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > 200 << 20:
                raise HTTPException(413, "videos up to 200 MB")
            tmp.write(chunk)
        tmp.flush()
        r = await run_in_threadpool(check_file, tmp.name, tokens + list(manifests))
    if r["campaign"] in manifests:   # "clip|c01=brewlark-2026-q4|c03=..." -> which campaign is in which place
        clip, *placed = (manifests[r["campaign"]] or "").split("|")
        r.update(manifest_id=r["campaign"], clip=clip, campaigns=dict(x.split("=", 1) for x in placed))
        r["places"] = named_places(clip, r["campaigns"])
    r.pop("scores", None)
    return r


# --- uploads (scripts/ingest.py; contract in INGEST.md) ------------------------------------------------------------

@app.get("/upload")
def upload_page():
    return page("upload.html")


@app.post("/api/ingest")
async def ingest(file: UploadFile):
    """Save the upload and start the pipeline in a child process (it calls the Meta API and takes minutes)."""
    import uuid
    from datetime import datetime
    if not (file.filename or "").lower().endswith((".mp4", ".mov", ".m4v", ".webm")):
        raise HTTPException(400, "please upload an mp4, mov or webm video")
    busy = [d for d in (ROOT / "runs").glob("r_*") if (d / "progress.jsonl").exists() and not (d / "report.json").exists()
            and not (d / "removed").exists() and time.time() - (d / "progress.jsonl").stat().st_mtime < 600
            and '"stage": "error"' not in (d / "progress.jsonl").read_text()]
    if len(busy) >= MAX_RUNNING:   # each run spends Meta API credit and minutes of this machine
        raise HTTPException(429, f"{len(busy)} videos are already processing. Try again when one of them finishes.")
    run_id = f"r_{datetime.now():%Y%m%d_%H%M}_{uuid.uuid4().hex[:8]}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    src = run_dir / ("source" + Path(file.filename).suffix.lower())
    size = 0
    with open(src, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > 200 << 20:
                f.close()
                shutil.rmtree(run_dir, ignore_errors=True)   # no half-uploaded run left behind
                raise HTTPException(413, "videos up to 200 MB")
            f.write(chunk)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type",
                            "-of", "csv=p=0", str(src)], capture_output=True, text=True)
    if "video" not in probe.stdout:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(400, "We couldn't read a video in that file. Try an mp4, mov or webm that plays on your "
                                 "computer.")
    (run_dir / "name.txt").write_text(Path(file.filename).name)   # for the library while it processes
    log = open(run_dir / "ingest.log", "w")
    subprocess.Popen([sys.executable, str(ROOT / "scripts" / "ingest.py"), str(src), "--run-id", run_id],
                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return {"run_id": run_id}


def _run_dir(run_id):
    d = ROOT / "runs" / run_id
    if not run_id.replace("_", "").isalnum() or not d.is_dir():
        raise HTTPException(404, "unknown run")
    return d


@app.get("/api/runs/{run_id}/progress")
def run_progress(run_id: str):
    p = _run_dir(run_id) / "progress.jsonl"
    events = [json.loads(line) for line in p.read_text().splitlines()[-50:]] if p.exists() else []
    events = [{**e, "msg": failure_text(e.get("msg", ""))} if e.get("stage") == "error" else e for e in events]
    if not events and (p.parent / "report.json").exists():   # shipped demo runs keep the report, not the log
        events = [{"stage": "done", "pct": 100, "msg": "finished"}]
    latest = events[-1] if events else {"stage": "upload", "pct": 0, "msg": "waiting to start"}
    return {"events": events, "latest": latest, "done": latest["stage"] in ("done", "error"),
            "error": failure_text(latest["msg"]) if latest["stage"] == "error" else None}


@app.get("/api/runs/{run_id}/report")
def run_report(run_id: str):
    p = _run_dir(run_id) / "report.json"
    if not p.exists():
        raise HTTPException(404, "not finished yet")
    return json.loads(p.read_text())

