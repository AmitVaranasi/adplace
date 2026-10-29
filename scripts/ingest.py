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
         f"enough to carry an ad: {sorted(SURFACES)}? Only list ones you can actually see. 2) Is the scene safe for "
         "brand advertising (no children as the focus, medical, accident, violence, grief)? 3) Up to 5 context "
         'tags (e.g. food, street, office). Reply JSON: {"surfaces": [...], "scene_safe": bool, '
         '"safety_reason": str, "context_tags": [...]}')
    content = [{"type": "text", "text": q}] + [{"type": "image_url", "image_url": {"url": jpeg_url(k)}} for k in keys]
    v = muse_json("You review video scenes for an ad-placement system. Be literal and conservative.", content)
    v["surfaces"] = [s for s in v.get("surfaces", []) if s in SURFACES]
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
        prompts = scene["surfaces"]
        run.progress("segment", 15, f"surfaces seen: {', '.join(prompts) or 'none'}; brand-safe: {scene.get('scene_safe')}")
        if not prompts or not scene.get("scene_safe", False):
            prompts = []
        # nothing to place on: skip SAM entirely (segmenting people alone takes minutes)
        results = segment(clip, prompts + OCCLUDERS) if prompts else []
        failed = [p for p, rc, _ in results if rc != 0]
        run.progress("segment", 100, "SAM 3.1 done" + (f" (failed: {', '.join(failed)})" if failed else ""))

        entry = {"hidden": True, "dynamic": True, "upload": True, "source": Path(src).name,
                 "occluder_prompts": [p for p in OCCLUDERS if p not in failed],
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
                    "dynamic_content": SURFACES[p] == "screen"}
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
        run.write("candidates.json", [{kk: v for kk, v in c.items() if kk not in ("slot", "quads")} for c in cands])
        run.progress("track", 100, f"{len(cands)} candidates measured")

        run.progress("score", 0, "deciding")
        try:
            from pipeline.gates import decide_all
            decisions = decide_all([{kk: v for kk, v in c.items() if kk != "slot"} for c in cands])
        except ImportError:
            decisions = fallback_decide(cands)
        run.write("decisions.json", decisions)
        accepted = [c for c, d in zip(cands, decisions) if d["decision"] == "accept"]
        run.progress("score", 100, f"{len(accepted)} of {len(cands)} accepted")

        entry["dynamic_slots"] = [c["slot"] for c in accepted]
        from pipeline.clipcfg import save_upload
        save_upload(clip, entry)
        if accepted:
            run.progress("export", 0, "building the live player (artwork per brand, mattes)")
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_player.py"), clip], cwd=ROOT,
                               capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError("export failed: " + r.stderr[-600:])
            run.progress("export", 100, "live player ready")
        qa = []   # eval/qa.py checks land here (GPT Sol); until then every accepted placement is reported unchecked
        run.write("qa.json", qa)
        run.progress("qa", 100, "checks recorded")

        viewers = []
        ap = ROOT / "app" / "static" / "player" / stem / "assignments.json"
        if ap.exists():
            viewers = [{"profile_id": pid, "label": v["label"]} for pid, v in json.loads(ap.read_text()).items()]
        report = {"run_id": run.id, "clip_stem": stem, "clip": clip, "source_name": Path(src).name,
                  "duration_s": round(frames.n / fps, 2), "status": "done", "elapsed_s": round(time.time() - run.t0),
                  "note": note, "scene": scene,
                  "summary": {"candidates": len(cands), "placed": len(accepted), "declined": len(cands) - len(accepted)},
                  "placements": sorted([
                      {"id": c["id"], "surface": c["surface"],
                       "decision": "placed" if d["decision"] == "accept" else "declined",
                       "thumb": f"/runs/{run.id}/{c['thumb']}" if c.get("thumb") else None,
                       "on_screen_s": round(c["metrics"].get("on_screen_s", 0.0), 2), "reasons": d["reasons"],
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
