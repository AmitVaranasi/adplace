"""Local demo app: pick a clip and a viewer, see which ad that viewer gets and why, watch it, verify any export.

  .venv/bin/uvicorn app.main:app --port 8000      # then open http://127.0.0.1:8000
Analysis (SAM 3 + tracking) is done ahead of time by scripts/analyse.py and scripts/track_eval.py and is shown as
cached. Rendering a (clip, brand) pair runs for real the first time and is reused afterwards.
"""
import functools
import json
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


def load(name):
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
