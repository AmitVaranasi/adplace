"""Local demo app: pick a clip and a viewer, see which ad that viewer gets and why, watch it, verify any export.

  .venv/bin/uvicorn app.main:app --port 8000      # then open http://127.0.0.1:8000
Analysis (SAM 3 + tracking) is done ahead of time by scripts/analyse.py and scripts/track_eval.py and is shown as
cached. Rendering a (clip, brand) pair runs for real the first time and is reused afterwards.
"""
import functools
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import discover, rank, watermark  # noqa: E402
from pipeline.sam_track import read_frames  # noqa: E402
from scripts.render import render_campaign  # noqa: E402

app = FastAPI(title="AdPlace")
app.mount("/outputs", StaticFiles(directory=ROOT / "outputs"), name="outputs")
app.mount("/clips", StaticFiles(directory=ROOT / "assets" / "clips"), name="clips")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
(ROOT / "runs").mkdir(exist_ok=True)
app.mount("/runs", StaticFiles(directory=ROOT / "runs"), name="runs")


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
    return FileResponse(Path(__file__).parent / "static" / "player.html")


class ViewerRenderRequest(BaseModel):
    clip: str          # stem, e.g. "patio_waiter"
    profile_id: str


@app.post("/api/viewer_render")
def viewer_render(req: ViewerRenderRequest):
    """The downloadable file for one viewer: the same assignment the live player shows, rendered on the server with
    the disclosure label and a Video Seal watermark naming the campaigns (the browser cannot embed the watermark).
    Cached per (clip, assignment). Rendered in a child process: in this process (torch + the app's imports) MKL
    segfaults inside the render, which would take the whole demo server down."""
    d = Path(__file__).parent / "static" / "player" / req.clip
    if not (d / "assignments.json").exists():
        raise HTTPException(404, f"no player export for {req.clip}")
    viewers = json.loads((d / "assignments.json").read_text())
    if req.profile_id not in viewers:
        raise HTTPException(404, f"unknown profile {req.profile_id}")
    track = json.loads((d / "track.json").read_text())
    clip = track["clip"]
    assignment = viewers[req.profile_id]["assignment"]
    tag = "_".join(f"{k}-{v}" for k, v in sorted(assignment.items()))
    out = ROOT / "outputs" / "viewers" / f"{req.clip}__{tag}.mp4"
    t0 = time.time()
    cached = out.with_suffix(".json").exists()
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
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


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


@app.post("/api/verify")
async def verify(file: UploadFile):
    tokens = [c["campaign_token"] for c in load("inventory.json")["campaigns"]]
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "x.mp4").suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        frames, _ = read_frames(tmp.name)
    return watermark.verify(frames, tokens)


# --- uploads (scripts/ingest.py; contract in INGEST.md) ------------------------------------------------------------

@app.get("/upload")
def upload_page():
    return FileResponse(Path(__file__).parent / "static" / "upload.html")


@app.post("/api/ingest")
async def ingest(file: UploadFile):
    """Save the upload and start the pipeline in a child process (it calls the Meta API and takes minutes)."""
    import uuid
    from datetime import datetime
    if not (file.filename or "").lower().endswith((".mp4", ".mov", ".m4v", ".webm")):
        raise HTTPException(400, "please upload an mp4, mov or webm video")
    run_id = f"r_{datetime.now():%Y%m%d_%H%M}_{uuid.uuid4().hex[:4]}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    src = run_dir / ("source" + Path(file.filename).suffix.lower())
    size = 0
    with open(src, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > 200 << 20:
                raise HTTPException(413, "videos up to 200 MB")
            f.write(chunk)
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
    latest = events[-1] if events else {"stage": "upload", "pct": 0, "msg": "waiting to start"}
    return {"events": events, "latest": latest, "done": latest["stage"] in ("done", "error"),
            "error": latest["msg"] if latest["stage"] == "error" else None}


@app.get("/api/runs/{run_id}/report")
def run_report(run_id: str):
    p = _run_dir(run_id) / "report.json"
    if not p.exists():
        raise HTTPException(404, "not finished yet")
    return json.loads(p.read_text())

