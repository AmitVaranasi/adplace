"""Local demo app: pick a clip and a viewer, see which ad that viewer gets and why, watch it, verify any export.

  .venv/bin/uvicorn app.main:app --port 8000      # then open http://127.0.0.1:8000
Analysis (SAM 3 + tracking) is done ahead of time by scripts/analyse.py and scripts/track_eval.py and is shown as
cached. Rendering a (clip, brand) pair runs for real the first time and is reused afterwards.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import discover, watermark  # noqa: E402
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


def slot_for(clip_id):
    entry = clip_entry(clip_id)
    quads = np.load(ROOT / entry["analysis"] / "quads.npy")
    status = ["ok" if np.isfinite(q).all() else "lost" for q in quads]
    meta = json.loads((ROOT / entry["analysis"] / "meta.json").read_text())
    metrics = discover.slot_metrics(quads, status, tuple(meta["size"]), meta["fps"])
    return {"slot_id": f"{Path(clip_id).stem}-1", "surface": entry["surface"], "metrics": metrics}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/catalog")
def catalog():
    clips = load("clips.json")
    return {"clips": [{"id": k, "title": v.get("title", k), "url": "/clips/" + Path(k).name,
                       "context_tags": v["context_tags"]} for k, v in clips.items()],
            "profiles": load("profiles.json")["profiles"],
            "brands": [c["brand_id"] for c in load("inventory.json")["campaigns"]]}


class ChooseRequest(BaseModel):
    clip: str
    profile_id: str


@app.post("/api/choose")
def choose(req: ChooseRequest):
    """Deterministic facts and filtering, then a choice. Muse will rank here once connected; the fallback is used now."""
    entry = clip_entry(req.clip)
    profile = next((p for p in load("profiles.json")["profiles"] if p["profile_id"] == req.profile_id), None)
    if profile is None:
        raise HTTPException(404, f"unknown profile {req.profile_id}")
    inventory = load("inventory.json")
    slot = slot_for(req.clip)
    eligible = discover.eligible_pairs([slot], inventory, entry["context_tags"])
    choice = discover.fallback_choice(eligible, [slot], inventory, profile["preferences"])
    if choice.get("status") == "ok":
        tags = next(c["preference_tags"] for c in inventory["campaigns"] if c["brand_id"] == choice["brand_id"])
        shared = sorted(set(tags) & set(profile["preferences"]))
        choice["rationale"] = (f"{choice['brand_id']} matches this viewer's interest in {', '.join(shared)}" if shared
                               else f"{choice['brand_id']} is the best-fitting eligible campaign for this surface")
    return {"slot": slot, "eligible": eligible, "choice": choice, "chooser": "deterministic fallback (Muse pending)"}


class RenderRequest(BaseModel):
    clip: str
    brand: str


@app.post("/api/render")
def render(req: RenderRequest):
    entry = clip_entry(req.clip)
    inventory = load("inventory.json")
    eligible = discover.eligible_pairs([slot_for(req.clip)], inventory, entry["context_tags"])
    if not any(e["brand_id"] == req.brand for e in eligible):
        raise HTTPException(409, f"{req.brand} is not an eligible placement for this clip")
    out = ROOT / "outputs" / f"{Path(req.clip).stem}_{req.brand}.json"
    cached = out.exists() and out.stat().st_mtime > (ROOT / entry["analysis"] / "quads.npy").stat().st_mtime
    t0 = time.time()
    report = json.loads(out.read_text()) if cached else render_campaign(req.clip, str(ROOT / entry["analysis"]),
                                                                        req.brand, entry["emissive"])
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
