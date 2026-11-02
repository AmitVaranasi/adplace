"""Any uploaded video -> per-viewer placements, conservatively (contract: INGEST.md).

  python scripts/ingest.py path/to/clip.mp4            # prints the run id; progress in runs/<id>/progress.jsonl

Stages: upload (normalise) -> shots (keep the longest) -> segment (Muse Spark reads the scene, SAM 3.1 finds the
surfaces it names, plus people) -> track (every candidate, then measured) -> score (pipeline/gates.py) ->
render/export (live player + per-viewer download) -> qa -> done (report.json). A candidate is either placed or
declined with a reason; nothing is placed that the measurements do not support.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API = "https://api.meta.ai/v1"
# What SAM 3.1 is asked for, and the inventory surface each prompt becomes. Muse picks which are in the scene.
SURFACES = {"poster": "poster", "framed picture": "poster", "painting": "poster", "sign": "sign",
            "menu board": "sign", "chalkboard": "sign", "billboard": "billboard", "screen": "screen",
            "television": "screen", "menu": "menu", "pizza box": "box"}
# Things a logo can be printed on (scripts/glass_decal.py): what Muse calls them -> SAM 3.1 prompt. "cup" finds
# drinking glasses and nothing else in a bar scene; "glass" also takes fridge doors and jars.
# Glasses, coffee cups and mugs all come from SAM's "cup" ("coffee cup" finds nothing, "mug" misses teacups,
# "teacup" misses mugs); Muse Spark then says which each object is.
PRINTABLE = {"drinking glass": "cup", "coffee cup": "cup", "mug": "cup"}
OCCLUDERS = ["person"]
STAGES = ["upload", "shots", "segment", "track", "score", "render", "export", "qa", "done"]


class Run:
    def __init__(self, run_id):
        self.id = run_id
        self.dir = ROOT / "runs" / run_id
        (self.dir / "thumbs").mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def progress(self, stage, pct, msg):
        line = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stage": stage, "pct": int(pct),
                "msg": msg}
        with open(self.dir / "progress.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"[{stage} {pct:3.0f}%] {msg}", flush=True)

    def write(self, name, data):
        (self.dir / name).write_text(json.dumps(data, indent=1))


# --- upload / shots ----------------------------------------------------------------------------------------------

def normalise(src, dst):
    """<= 1280 px wide, constant frame rate, H.264."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", "scale='min(1280,iw)':-2",
                    "-vsync", "cfr", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-an", str(dst)],
                   check=True)


def shots(path, thresh=0.55):
    """Hard cuts from the colour-histogram correlation of consecutive frames; [(start, end_exclusive), ...]."""
    cap = cv2.VideoCapture(str(path))
    prev, cuts, i = None, [0], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(h, h)
        if prev is not None and cv2.compareHist(prev, h, cv2.HISTCMP_CORREL) < thresh:
            cuts.append(i)
        prev, i = h, i + 1
    cuts.append(i)
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a > 1]


def trim(src, dst, a, b, fps):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf",
                    f"select='between(n,{a},{b - 1})',setpts=N/({fps}*TB)", "-r", str(fps), "-vsync", "cfr",
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)


# --- segment -----------------------------------------------------------------------------------------------------

def jpeg_url(img, width=1024):
    small = cv2.resize(img, (width, int(width * img.shape[0] / img.shape[1])))
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])[1]).decode()


def muse_json(system, content, tokens=4000, tries=3):
    """A JSON reply from Muse Spark; retried, since a call occasionally comes back empty."""
    from pipeline.rank import chat
    for k in range(tries):
        text = chat(API, os.environ["MODEL_API_KEY"], "muse-spark-1.3", system, content,
                    extra={"max_completion_tokens": tokens}, timeout=300)
        try:
            return json.loads(text[text.index("{"):text.rindex("}") + 1])
        except (AttributeError, TypeError, ValueError):
            time.sleep(2 + 3 * k)
    raise RuntimeError("Muse Spark gave no usable answer")


def read_scene(frames):
    """Muse Spark looks at three keyframes: which surface types are present, is the scene brand-safe, context."""
    keys = [frames[k] for k in (0, frames.n // 2, frames.n - 1)]
    q = ("These are three frames of one video shot. 1) Which of these surface types are clearly visible and large "
         f"enough to carry an ad: {sorted(SURFACES) + sorted(PRINTABLE)}? Only list ones you can actually see. 2) Is the scene safe for "
         "brand advertising (no children as the focus, medical, accident, violence, grief)? 3) Up to 5 context "
         'tags (e.g. food, street, office). Reply JSON: {"surfaces": [...], "scene_safe": bool, '
         '"safety_reason": str, "context_tags": [...]}')
    content = [{"type": "text", "text": q}] + [{"type": "image_url", "image_url": {"url": jpeg_url(k)}} for k in keys]
    # Two independent reads: one read of the same clip said "menu board", another "none". Surfaces either read
    # sees are kept (the gates judge them on measurements later); the scene is safe only if both reads agree.
    reads = [muse_json("You review video scenes for an ad-placement system. Be literal and conservative.", content)
             for _ in range(2)]
    v = dict(reads[0])
    v["surfaces"] = sorted({s for r in reads for s in r.get("surfaces", []) if s in SURFACES or s in PRINTABLE})
    v["scene_safe"] = all(r.get("scene_safe") is True for r in reads)
    v["safety_reason"] = "; ".join(str(r.get("safety_reason", "")) for r in reads if r.get("safety_reason"))
    v["context_tags"] = sorted({t for r in reads for t in r.get("context_tags", [])})[:8]
    return v


def segment(clip, prompts, workers=4):
    """SAM 3.1 through the Meta Model API: upload the clip once, stream every prompt in parallel and record the raw
    events (what object_stores / union_store build masks from). analyse.py also decoded and saved a full mask array
    per prompt that ingest never uses; with the slow mask decoder that was most of a run's time."""
    from pipeline.sam_api import stream_events, upload
    from scripts.analyse import cache_dir
    fid = None

    def one(p):
        base = cache_dir(clip, {"model": "meta/sam-3.1", "prompt": p})
        rec = base.with_name(base.name + "-events.jsonl")
        if rec.exists():
            return p, 0, ""
        try:
            events = list(stream_events(fid, p))
            rec.write_text("\n".join(json.dumps(e) for e in events))
            return p, 0, ""
        except Exception as e:
            return p, 1, str(e)[-400:]
    todo = [p for p in prompts if not (lambda b: b.with_name(b.name + "-events.jsonl").exists())(
        cache_dir(clip, {"model": "meta/sam-3.1", "prompt": p}))]
    if todo:
        fid = upload(str(ROOT / clip))
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, prompts))


# --- track + measure ---------------------------------------------------------------------------------------------

def measure(clip, slot, frames, occ, tr, fps):
    """The candidate's facts (INGEST.md): discover.slot_metrics plus residual, jitter, blur, size, light, occlusion."""
    from pipeline.discover import slot_metrics
    from scripts.render_dynamic import object_stores, quad_mask
    from scripts.render_scene import fit_shape
    quads, valid = np.asarray(tr["quads"], float), np.asarray(tr["valid"], bool)
    status = ["tracked" if v else "lost" for v in valid]

    class Occ:
        def __getitem__(self, i):
            return occ.frame(i) if occ else None

        def __len__(self):
            return frames.n
    m = slot_metrics(quads, status, (frames.w, frames.h), fps, Occ())
    from scripts.render_dynamic import area
    store = object_stores(clip, slot["prompt"], frames)[str(slot["object"])]
    typical = float(np.median([area(store, i) for i in range(frames.n) if store.any(i)] or [0]))
    vis = np.flatnonzero(valid & np.isfinite(quads).all(axis=(1, 2)))
    res, luma, occl, sides = [], [], [], []
    for i in vis:
        q = quads[i]
        sides.append(min(np.linalg.norm(q[k] - q[(k + 1) % 4]) for k in range(4)))
        qm = quad_mask(q, frames.h, frames.w)
        o = occ.frame(i) if occ else None
        if o is not None and qm.sum():
            occl.append(float((o & qm).sum() / qm.sum()))
        if store.any(i) and (o is None or (o & qm).sum() < 0.05 * qm.sum()):
            try:
                fq = np.asarray(fit_shape(store[i], {"type": "quad"})["quad"], float)
                d = np.linalg.norm(fq[:, None, :] - q[None, :, :], axis=2)
                res.append(float(np.median(d.min(1))))
            except Exception:
                pass
        if i % 3 == 0 and qm.sum():
            g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            luma.append(float(g[qm].mean()))
    seq = quads[vis] if len(vis) > 2 else np.zeros((0, 4, 2))
    jit = np.linalg.norm(seq[2:] - 2 * seq[1:-1] + seq[:-2], axis=2).ravel() if len(seq) > 2 else np.array([np.nan])
    speed = np.linalg.norm(np.diff(seq, axis=0), axis=2).mean(1) if len(seq) > 1 else np.array([np.nan])
    nan = float("nan")
    m.update({
        "track_residual_px_median": float(np.median(res)) if res else nan,
        "track_residual_px_p90": float(np.percentile(res, 90)) if res else nan,
        "corner_jitter_px_median": float(np.nanmedian(jit)), "corner_jitter_px_p95": float(np.nanpercentile(jit, 95)),
        "motion_blur_px_median": float(np.nanmedian(speed) * 0.5),
        "min_side_px": float(np.median(sides)) if sides else 0.0,
        "mean_luma": float(np.mean(luma)) if luma else nan,
        "luma_std_over_time": float(np.std(luma)) if luma else nan,
        "mean_occlusion_frac": float(np.mean(occl)) if occl else 0.0,
        "dynamic_content": bool(slot.get("dynamic_content")),
        # frames SAM sees the surface (at least a fifth of its usual size) but the placement does not cover:
        # there the original would pop back into view
        "uncovered_visible_s": float(sum(1 for i in range(frames.n) if not valid[i] and
                                         area(store, i) > 0.2 * typical) / fps),
    })
    return m


def thumb(frames, tr, path):
    quads, valid = np.asarray(tr["quads"], float), np.asarray(tr["valid"], bool)
    i = int(tr["ref"]) if valid[int(tr["ref"])] else int(np.flatnonzero(valid)[len(np.flatnonzero(valid)) // 2])
    f = frames[i].copy()
    cv2.polylines(f, [np.round(quads[i]).astype(np.int32)], True, (60, 220, 120), 3, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.resize(f, (640, int(640 * f.shape[0] / f.shape[1]))), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return f


def plausible(img):
    v = muse_json("You review candidate ad surfaces for an ad-placement system. Be strict.",
                  [{"type": "text", "text": "The green outline marks a candidate surface for inserting an ad after "
                    "filming. Is it one real, flat (or screen) surface where a printed ad, sign or screen picture "
                    "would naturally be, fully inside the outline, not a person, window, floor or sky? Reply JSON: "
                    '{"plausible": bool, "reason": str}'},
                   {"type": "image_url", "image_url": {"url": jpeg_url(img)}}], tokens=2000)
    return bool(v.get("plausible")), str(v.get("reason", ""))[:200]


def fallback_decide(cands):
    """Until pipeline/gates.py lands: the same spirit, fewer facts."""
    out = []
    nan = float("nan")
    for c in cands:
        m, why = c["metrics"], []
        if m.get("error"):
            out.append({"id": c["id"], "decision": "decline", "score": 0.0, "reasons": [m["error"]]})
            continue
        m = {k: m.get(k, nan) for k in ("on_screen_s", "track_residual_px_median", "corner_jitter_px_p95",
                                        "min_side_px")} | m
        if not (m["on_screen_s"] >= 2.0):
            why.append(f"on screen {m['on_screen_s']:.1f} s, needs 2.0 s")
        if not (m["track_residual_px_median"] <= 2.5):
            why.append(f"tracking error {m['track_residual_px_median']:.1f} px, limit 2.5 px")
        if not (m["corner_jitter_px_p95"] <= 3.0):
            why.append(f"corners jitter {m['corner_jitter_px_p95']:.1f} px, limit 3.0 px")
        if not (m["min_side_px"] >= 48):
            why.append(f"too small ({m['min_side_px']:.0f} px side, needs 48)")
        if m.get("uncovered_visible_s", 0) > 0.25:
            why.append(f"would disappear for {m['uncovered_visible_s']:.1f} s while the surface is still in view")
        if not m.get("muse_plausible"):
            why.append("not a plausible ad surface: " + m.get("muse_reason", ""))
        if not m.get("scene_safe", True):
            why.append("scene not brand-safe")
        out.append({"id": c["id"], "decision": "decline" if why else "accept", "score": 0.0 if why else 1.0,
                    "reasons": why})
    return out


def check_placements(run, clip, entry, stem, frames, fps, accepted, samples=8):
    """eval/qa.py on a server render of one viewer's version: stability of each flat ad against the real wall
    around it. (Muse's blind judge, eval/qa.judge, is switched off for now; a person reviews the look.)"""
    from eval.qa import qa_placement, stability
    from scripts.render_dynamic import quad_mask, render, scene_dir
    run.progress("qa", 0, "rendering a check copy")
    viewers = json.loads((ROOT / "app" / "static" / "player" / stem / "assignments.json").read_text())
    assignment = next(iter(viewers.values()))["assignment"]
    out = run.dir / "qa_render.mp4"
    render(clip, entry, assignment, out, label=False)
    cap, rendered = cv2.VideoCapture(str(out)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        rendered.append(f)
    results = []
    for k, c in enumerate(accepted):
        run.progress("qa", 20 + 70 * k / len(accepted), f"checking {c['id']} ({c['surface']})")
        if c["kind"] in ("glass_decal", "cup_decal"):   # measured at the gates; the human reviewer judges the look
            results.append({"id": c["id"], "pass": True, "checks": {}, "reasons": []})
            continue
        with np.load(scene_dir(clip) / f"track_{c['id']}.npz") as z:
            quads, valid = z["quads"], z["valid"]
        # the longest run of frames with the ad on screen
        best, cur = (0, 0), None
        for i in range(len(valid) + 1):
            if i < len(valid) and valid[i]:
                cur = i if cur is None else cur
            elif cur is not None:
                best, cur = max(best, (cur, i), key=lambda r: r[1] - r[0]), None
        idx = list(range(*best))[:len(rendered)]
        checks = {}
        if len(idx) >= 3:
            ref = quads[idx[len(idx) // 2]].astype(np.float32)
            H = [cv2.getPerspectiveTransform(quads[i].astype(np.float32), ref) for i in idx]
            pm = quad_mask(ref, frames.h, frames.w)
            ring = (cv2.dilate(pm.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0) & \
                   ~(cv2.dilate(pm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
            checks["stability"] = stability([rendered[i] for i in idx], H, pm, ring)
        # the blind Muse judge is off for now (it rewarded blurring prints until unreadable); a person reviews the look
        results.append(qa_placement(c["id"], checks, required=("stability",)))
    return results


def glass_candidate(clip, cid, oid, frames, fps, thumbs, prompt="cup"):
    """A drinking glass, cup or mug SAM found ("cup" finds all three): Muse Spark says which it is and whether it's
    plain, then the logo's own placement (scripts/glass_decal.py) is measured like a surface. Returns (metrics,
    frame range, kind, surface)."""
    from scripts.place3d import make
    from scripts.render_dynamic import object_stores
    st = object_stores(clip, prompt, frames)[str(oid)]
    seen = [i for i in range(frames.n) if st.any(i)]
    i = max(seen[len(seen) // 4: 3 * len(seen) // 4 + 1] or seen or [0], key=lambda k: st[k].sum() if st.any(k) else 0)
    f = frames[i].copy()
    if st.any(i):
        cs, _ = cv2.findContours(st[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(f, cs, -1, (60, 220, 120), 3, cv2.LINE_AA)
    cv2.imwrite(str(thumbs / f"{cid}.jpg"), cv2.resize(f, (640, int(640 * f.shape[0] / f.shape[1]))),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    v = muse_json("You review candidate ad placements for an ad-placement system. Be strict.",
                  [{"type": "text", "text": "The green outline marks a candidate object. What is it: \"glass\" (one "
                    "real drinking glass), \"cup\" (one real coffee cup, teacup or mug, ceramic or paper) or "
                    "\"other\" (a jar, bottle, bowl, window, fridge door, several objects, ...)? Is it plain, with no "
                    "print of its own? Does it carry a brand's name or logo (another brand's ad must not be covered "
                    "or sit beside ours)? A decorative pattern or drawing is not a brand. Reply JSON: "
                    '{"vessel": "glass" | "cup" | "other", "plain": bool, "branded": bool, "reason": str}'},
                   {"type": "image_url", "image_url": {"url": jpeg_url(f)}}], tokens=2000)
    vessel = v.get("vessel") if v.get("vessel") in ("glass", "cup") else "other"
    kind, surface = ("glass_decal", "glass") if vessel == "glass" else ("cup_decal", "cup")
    reason = str(v.get("reason", ""))[:200]
    # a cup's own drawing is fine (the print goes on a blank patch of it); a brand's isn't, and a glass's print
    # can't be told from what's seen through it, so a glass must be plain
    if vessel == "other" or v.get("branded") is not False or (vessel == "glass" and not v.get("plain")):
        return ({"on_screen_s": len(seen) / fps, "muse_plausible": False, "muse_reason": reason,
                 "error": "muse_plausible: " + reason}, [seen[0], seen[-1]] if seen else [0, 0], kind, surface)
    g = make(clip, {"id": cid, "kind": kind, "prompt": prompt, "object": oid})
    placed = g.valid
    run, best = 0, 0
    for k in range(frames.n):
        run = run + 1 if k in g.place else 0
        best = max(best, run)
    geo = np.array([g.place[k] for k in placed]) if placed else np.zeros((0, 5))
    fit_ok, on_glass = g.quality()
    m = {"on_screen_s": len(placed) / fps, "longest_run_s": best / fps, "frames": len(placed),
         "radius_px_median": float(np.median(geo[:, 2])) if len(geo) else 0.0,
         "shape_fit_fraction": fit_ok, "print_on_glass_fraction": on_glass,
         "uncovered_visible_s": sum(1 for k in range(frames.n) if g.store.any(k) and k not in g.place) / fps,
         # static objects only: a glass someone picks up needs its logo re-fitted every frame, and that shows
         "travel_r_max": max_travel(g, fps, float(np.median(geo[:, 2])) if len(geo) else 1.0),
         "muse_plausible": True, "muse_reason": reason, "blank_patch": bool(g.blank)}
    return m, [int(placed[0]), int(placed[-1])] if placed else [0, 0], kind, surface


def max_travel(g, fps, r):
    """The farthest the object moves relative to the room (camera motion taken out) within any one second, in
    glass radii. A glass on a table only jiggles when someone touches it; one picked up travels several radii."""
    n, best = int(round(fps)), 0.0
    for s in range(0, g.frames.n - n + 1):
        d = [g.rel[i] for i in range(s, s + n) if i in g.rel]
        if d:
            best = max(best, float(np.linalg.norm(np.cumsum(d, axis=0), axis=1).max()))
    return best / r


def decide_glass(c):
    """The gates for a logo on a glass, in the same spirit as pipeline/gates.py for flat surfaces."""
    m, why = c["metrics"], []
    what = c.get("surface", "glass")
    if m.get("error"):
        return {"id": c["id"], "decision": "decline", "score": 0.0, "reasons": [m["error"]]}
    if not m["on_screen_s"] >= 2.0:
        why.append(f"on screen {m['on_screen_s']:.1f} s, needs 2.0 s")
    if not m["longest_run_s"] >= 1.5:
        why.append(f"longest run {m['longest_run_s']:.1f} s, needs 1.5 s")
    if not m["radius_px_median"] >= 22:
        why.append(f"minimum side: {what} {2 * m['radius_px_median']:.0f} px wide, needs 44 px")
    if not m["shape_fit_fraction"] >= 0.8:
        why.append(f"could not track: the {what}'s shape fitted in {m['shape_fit_fraction']:.0%} of frames, "
                   "needs 80%")
    if not m["print_on_glass_fraction"] >= 0.9:
        why.append(f"track residual: logo well inside the {what} in {m['print_on_glass_fraction']:.0%} of frames, "
                   "needs 90%")
    if m["uncovered_visible_s"] > 0.25:
        why.append(f"visible but not covered for {m['uncovered_visible_s']:.1f} s")
    if not m.get("blank_patch", True):
        why.append(f"no blank patch: no blank area of the {what} (clear of its own design, rim and what covers it) "
                   "is big enough for the logo with a margin round it")
    if m.get("travel_r_max", 0.0) > 0.5:
        why.append(f"moves: travels {m['travel_r_max']:.1f} {what} radii within a second, needs at most 0.5 "
                   "(only still objects carry ads)")
    if not m.get("muse_plausible"):
        why.append("muse_plausible: " + m.get("muse_reason", ""))
    if not m.get("scene_safe", True):
        why.append("scene_safe: false")
    return {"id": c["id"], "decision": "decline" if why else "accept", "score": 0.0 if why else 1.0, "reasons": why}


PLAIN = [  # gate reason (pipeline/gates.py wording) -> what a viewer of the report should read
    ("on screen", "It isn't on screen long enough to register."),
    ("longest run", "It's never visible long enough in one go."),
    ("track residual", "We couldn't follow its outline precisely enough; an ad would drift on it."),
    ("corner jitter", "Its outline wobbles from frame to frame; an ad would shake."),
    ("minimum side", "It's too small to carry readable artwork."),
    ("mean area fraction", "It's too small in the picture."),
    ("edge-clipped", "It's mostly cut off by the edge of the frame."),
    ("maximum occlusion", "People or things cover too much of it at some point."),
    ("mean occlusion", "People or things cover too much of it most of the time."),
    ("motion blur", "The camera moves too fast across it."),
    ("luma standard deviation", "Its lighting changes too much over the shot."),
    ("visible but not covered", "An ad would vanish while the surface is still in view."),
    ("muse_plausible", "Muse Spark judged it not a natural place for an ad."),
    ("scene_safe", "The scene isn't suitable for advertising, so nothing is placed."),
    ("overlap with", "It overlaps a better placement on the same surface."),
    ("already has", "The shot already has as many ads as we allow."),
    ("could not track", "We couldn't track it through the shot."),
    ("moves:", "Someone picks it up or moves it; for now we only place ads on things that stay still."),
    ("no blank patch", "No blank area of it is big enough for a logo with clear space round it, so the logo would "
                       "touch its own design or edges."),
    ("failed final check: stability", "In the rendered video it moved more than the real wall around it."),
    ("failed final check: judge", "A blind reviewer spotted it as inserted much more often than the original."),
    ("failed final check", "It failed the final checks on the rendered video."),
    ("flat placement required", "3D objects need a calibrated table, which this clip doesn't give us."),
]


def plain(reasons):
    """Plain sentences for the report, each once, in order; anything unrecognised is passed through."""
    out = []
    for r in reasons:
        text = next((p for key, p in PLAIN if key in r), r)
        if text not in out:
            out.append(text)
    return out


# --- the run -----------------------------------------------------------------------------------------------------

def ingest(src, run_id=None):
    run = Run(run_id or f"r_{datetime.now():%Y%m%d_%H%M}_{uuid.uuid4().hex[:4]}")
    stem = "upload_" + run.id.split("_")[-1]
    clip = f"assets/clips/uploads/{stem}.mp4"
    try:
        if not Path(src).is_file():
            raise FileNotFoundError(f"no video at {src}")
        run.progress("upload", 0, "normalising the video")
        raw = run.dir / "input.mp4"
        normalise(src, raw)
        cap = cv2.VideoCapture(str(raw))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        run.progress("upload", 100, f"{n} frames at {fps:.2f} fps")

        run.progress("shots", 0, "looking for cuts")
        sh = shots(raw)
        a, b = max(sh, key=lambda s: s[1] - s[0])
        note = "" if len(sh) == 1 else f"{len(sh)} shots; using the longest ({(b - a) / fps:.1f} s)"
        trim(raw, ROOT / clip, a, b, fps) if len(sh) > 1 else normalise(raw, ROOT / clip)
        run.progress("shots", 100, note or "one continuous shot")

        from scripts.render_dynamic import Frames, object_stores, occluders, track_slot, area
        frames = Frames(ROOT / clip)
        run.progress("segment", 5, "Muse Spark is reading the scene")
        scene = read_scene(frames)
        prompts = [p for p in scene["surfaces"] if p in SURFACES]
        printable = sorted({PRINTABLE[p] for p in scene["surfaces"] if p in PRINTABLE})
        run.progress("segment", 15, f"surfaces seen: {', '.join(scene['surfaces']) or 'none'}; "
                                    f"brand-safe: {scene.get('scene_safe')}")
        if not scene.get("scene_safe", False):
            prompts, printable = [], []
        # nothing to place on: skip SAM entirely (segmenting people alone takes minutes)
        # cups' handles too, to cut out of their outlines (scripts/glass_decal.HandleFree)
        results = segment(clip, prompts + printable + (["cup handle"] if printable else []) + OCCLUDERS) \
            if prompts or printable else []
        failed = [p for p, rc, _ in results if rc != 0]
        run.progress("segment", 100, "SAM 3.1 done" + (f" (failed: {', '.join(failed)})" if failed else ""))

        entry = {"hidden": True, "dynamic": True, "upload": True, "source": Path(src).name,
                 # only occluders SAM actually ran for (none when the scene has no surfaces to place on)
                 # glasses too: a glass lifted in front of a poster on the wall stays in front of it
                 "occluder_prompts": [p for p, rc, _ in results if rc == 0 and (p in OCCLUDERS or p in printable)],
                 "context_tags": scene.get("context_tags", []), "dynamic_slots": []}
        occ = occluders(clip, entry, frames)
        cands, k = [], 0
        objs = [(p, oid, st) for p in prompts if p not in failed
                for oid, st in sorted(object_stores(clip, p, frames).items())]
        for j, (p, oid, st) in enumerate(objs):
            seen = sum(st.any(i) for i in range(frames.n))
            peak = max(area(st, i) for i in range(frames.n)) / (frames.w * frames.h)
            if seen < fps or peak < 0.002:   # under a second, or tiny: not worth tracking
                continue
            k += 1
            slot = {"id": f"c{k:02d}", "prompt": p, "object": int(oid), "surface": SURFACES[p],
                    "dynamic_content": SURFACES[p] == "screen", "wall_light": SURFACES[p] != "screen"}

            run.progress("track", 100 * j / max(len(objs), 1), f"tracking {p} #{oid}")
            c = {"id": slot["id"], "shot": 0, "prompt": p, "object": int(oid), "kind": "flat",
                 "surface": SURFACES[p], "fps": fps, "frame_size": [frames.w, frames.h], "slot": slot}
            try:
                from scripts.render_dynamic import static_surface
                if static_surface(object_stores(clip, p, frames)[str(oid)], frames) is not None:
                    slot["track"] = "static"   # locked-off camera: hold SAM's steady outline
                tr = track_slot(clip, slot, frames, occ)
                c["metrics"] = measure(clip, slot, frames, occ, tr, fps)
                c["frames"] = [int(np.flatnonzero(tr["valid"])[0]), int(np.flatnonzero(tr["valid"])[-1])] \
                    if tr["valid"].any() else [0, 0]
                img = thumb(frames, tr, run.dir / "thumbs" / f"{slot['id']}.jpg")
                c["thumb"] = f"thumbs/{slot['id']}.jpg"
                try:
                    c["metrics"]["muse_plausible"], c["metrics"]["muse_reason"] = plausible(img)
                except RuntimeError as e:   # no answer: decline rather than place unchecked
                    c["metrics"]["muse_plausible"], c["metrics"]["muse_reason"] = False, f"Muse unavailable ({e})"
                lo, hi = c["frames"]   # gates.py: quads local to the inclusive frame range
                c["quads"] = [q.tolist() if v else None
                              for q, v in zip(np.asarray(tr["quads"])[lo:hi + 1], tr["valid"][lo:hi + 1])]
            except Exception as e:
                c["metrics"] = {"on_screen_s": 0.0, "error": f"could not track: {e}"}
                c["frames"], c["thumb"] = [0, 0], None
            c["metrics"]["scene_safe"] = bool(scene.get("scene_safe", False))
            cands.append(c)
        for p in printable:
            if p in failed:
                continue
            for oid, st in sorted(object_stores(clip, p, frames).items(), key=lambda kv: int(kv[0])):
                if sum(st.any(i) for i in range(frames.n)) < fps:
                    continue
                k += 1
                cid = f"c{k:02d}"
                run.progress("track", 95, f"following glass or cup #{oid}")
                c = {"id": cid, "shot": 0, "prompt": p, "object": int(oid), "kind": "glass_decal", "surface": "glass",
                     "fps": fps, "frame_size": [frames.w, frames.h], "thumb": f"thumbs/{cid}.jpg"}
                try:
                    c["metrics"], c["frames"], c["kind"], c["surface"] = glass_candidate(
                        clip, cid, int(oid), frames, fps, run.dir / "thumbs", prompt=p)
                except Exception as e:
                    c["metrics"] = {"on_screen_s": 0.0, "error": f"could not track: {e}"}
                    c["frames"], c["thumb"] = [0, 0], None
                c["slot"] = {"id": cid, "kind": c["kind"], "surface": c["surface"], "prompt": p, "object": int(oid)}
                c["metrics"]["scene_safe"] = bool(scene.get("scene_safe", False))
                cands.append(c)
        run.write("candidates.json", [{kk: v for kk, v in c.items() if kk not in ("slot", "quads")} for c in cands])
        run.progress("track", 100, f"{len(cands)} candidates measured")

        run.progress("score", 0, "deciding")
        try:
            from pipeline.gates import decide_all
            # a surface that couldn't be tracked has no measurements for the gates: its error is the reason
            flat = [c for c in cands if c["kind"] == "flat" and not c["metrics"].get("error")]
            by_id = {d["id"]: d for d in decide_all([{kk: v for kk, v in c.items() if kk != "slot"} for c in flat])}
            by_id.update({c["id"]: {"id": c["id"], "decision": "decline", "score": 0.0, "reasons": [c["metrics"]["error"]]}
                          for c in cands if c["kind"] == "flat" and c["metrics"].get("error")})
        except ImportError:
            by_id = {d["id"]: d for d in fallback_decide([c for c in cands if c["kind"] == "flat"])}
        by_id.update({c["id"]: decide_glass(c) for c in cands if c["kind"] in ("glass_decal", "cup_decal")})
        decisions = [by_id[c["id"]] for c in cands]
        run.write("decisions.json", decisions)
        accepted = [c for c, d in zip(cands, decisions) if d["decision"] == "accept"]
        run.progress("score", 100, f"{len(accepted)} of {len(cands)} accepted")

        entry["dynamic_slots"] = [c["slot"] for c in accepted if c["kind"] == "flat"]
        entry["objects3d"] = [c["slot"] for c in accepted if c["kind"] != "flat"]
        from pipeline.clipcfg import save_upload
        save_upload(clip, entry)
        if accepted:
            run.progress("export", 0, "building the live player (artwork per brand, mattes)")
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_player.py"), clip], cwd=ROOT,
                               capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError("export failed: " + r.stderr[-600:])
            run.progress("export", 100, "live player ready")
        qa = check_placements(run, clip, entry, stem, frames, fps, accepted) if accepted else []
        run.write("qa.json", qa)
        failed_qa = {q["id"] for q in qa if not q["pass"]}
        if failed_qa:   # a placement that fails its checks is taken out, and the player rebuilt without it
            entry["dynamic_slots"] = [sl for sl in entry["dynamic_slots"] if sl["id"] not in failed_qa]
            entry["objects3d"] = [sl for sl in entry["objects3d"] if sl["id"] not in failed_qa]
            save_upload(clip, entry)
            accepted = [c for c in accepted if c["id"] not in failed_qa]
            for d in decisions:
                if d["id"] in failed_qa:
                    d["decision"] = "decline"
                    d["reasons"] = d["reasons"] + [f"failed final check: {r}"
                                                   for q in qa if q["id"] == d["id"] for r in q["reasons"]]
            run.write("decisions.json", decisions)
            if accepted:
                run.progress("qa", 90, "rebuilding the live player without the placements that failed")
                subprocess.run([sys.executable, str(ROOT / "scripts" / "export_player.py"), clip], cwd=ROOT,
                               capture_output=True, text=True, check=True)
        run.progress("qa", 100, f"{len(qa) - len(failed_qa)} of {len(qa)} placements passed the final checks")

        viewers = []
        ap = ROOT / "app" / "static" / "player" / stem / "assignments.json"
        if ap.exists():
            viewers = [{"profile_id": pid, "label": v["label"]} for pid, v in json.loads(ap.read_text()).items()]
        if not cands:   # say why there is nothing to show, instead of an empty report
            why = ("The scene isn't suitable for advertising, so nothing was placed."
                   if not scene.get("scene_safe", False) else
                   "Nothing in this video can carry an ad. We look for posters, picture frames, signs, menu boards, "
                   "screens, and plain glasses, cups and mugs that stay still, and found none.")
            note = f"{note}. {why}" if note else why
        report = {"run_id": run.id, "clip_stem": stem, "clip": clip, "source_name": Path(src).name,
                  "duration_s": round(frames.n / fps, 2), "status": "done", "elapsed_s": round(time.time() - run.t0),
                  "note": note, "scene": scene,
                  "summary": {"candidates": len(cands), "placed": len(accepted), "declined": len(cands) - len(accepted)},
                  "placements": sorted([
                      {"id": c["id"], "surface": c["surface"],
                       "decision": "placed" if d["decision"] == "accept" else "declined",
                       "thumb": f"/runs/{run.id}/{c['thumb']}" if c.get("thumb") else None,
                       "on_screen_s": round(c["metrics"].get("on_screen_s", 0.0), 2),
                       "reasons": plain(d["reasons"]), "details": d["reasons"],
                       "qa": next((q for q in qa if q["id"] == c["id"]), None)}
                      for c, d in zip(cands, decisions)], key=lambda p: p["decision"] != "placed"),
                  "player_url": f"/player?clip={stem}" if accepted else None, "viewers": viewers}
        run.write("report.json", report)
        run.progress("done", 100, f"{len(accepted)} placed, {len(cands) - len(accepted)} declined")
    except Exception as e:
        traceback.print_exc()
        run.progress("error", 100, f"{type(e).__name__}: {e}")
    return run.id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--run-id")
    a = ap.parse_args()
    print(ingest(a.video, a.run_id))
