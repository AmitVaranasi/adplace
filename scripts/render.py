"""Render one campaign into a clip: composite, visible disclosure label, Video Seal watermark, H.264 export, verify.

  .venv/bin/python scripts/render.py assets/clips/billboard_night.mp4 cache/<dir> brewlark --emissive
Writes outputs/<clip>_<brand>.mp4 and outputs/<clip>_<brand>.json (timings and verification).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import composite, exposure, watermark  # noqa: E402
from pipeline.sam_track import read_frames  # noqa: E402

LABEL = "AI product placement"


def campaign(brand_id):
    inv = json.loads((ROOT / "configs" / "inventory.json").read_text())["campaigns"]
    return next(c for c in inv if c["brand_id"] == brand_id), [c["campaign_token"] for c in inv]


def pick_creative(c, quad):
    """The creative whose aspect ratio is closest to the surface's (measured on the reference quad)."""
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    ratios = {k: int(k.split("x")[0]) / int(k.split("x")[1]) for k in c["creatives"]}
    key = min(ratios, key=lambda k: abs(np.log(ratios[k] / (w / h))))
    return cv2.cvtColor(cv2.imread(str(ROOT / c["creatives"][key]), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)


def add_label(frame):
    """Persistent visible disclosure, bottom-left."""
    h = frame.shape[0]
    scale = h / 900
    (tw, th), _ = cv2.getTextSize(LABEL, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = int(16 * scale * 1.5), h - int(24 * scale * 1.5)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - th - 10), (x + tw + 10, y + 10), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    cv2.putText(frame, LABEL, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def encode(frames, fps, path, crf=18):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
                          "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait():
        raise SystemExit(f"ffmpeg failed writing {path}")


def render_cylinder(frames, cache, c, shape):
    """Wrap the creative around an upright cylinder (mug, cup, column) found in each frame's SAM mask."""
    from pipeline.curved_fit import cylinder_from_mask, hold_static, smooth_cylinders
    from pipeline.curved_warp import composite_curved
    masks = np.load(Path(cache) / "surface_masks.npz")["masks"]
    band = tuple(shape.get("band", (0.25, 0.8)))
    top = shape.get("top", "band")  # "rim": start the label at the rim lip found in the frame, as a real print does
    cyls = hold_static(smooth_cylinders([cylinder_from_mask(m, band=band, frame=f, top=top) if m.any() else None
                                         for m, f in zip(masks, frames)]))
    key = shape.get("creative", "4x3")
    art = cv2.cvtColor(cv2.imread(str(ROOT / c["creatives"][key]), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    arc = shape.get("arc_deg", 110)
    return [composite_curved(f, art, cy, arc_deg=arc) if cy is not None else f.copy() for f, cy in zip(frames, cyls)], \
        sum(cy is not None for cy in cyls)


def render_ribbon(frames, cache, c, shape):
    """Bend the creative along a surface curved in one direction (a facade screen wrapping a corner)."""
    from pipeline.ribbon import composite_ribbon, ribbon_from_mask, smooth_edges, smooth_ribbons
    masks = np.load(Path(cache) / "surface_masks.npz")["masks"]
    occ_file = Path(cache) / "occluders.npz"
    occ = np.load(occ_file)["masks"] if occ_file.exists() else [None] * len(masks)
    ribbons = smooth_ribbons([smooth_edges(ribbon_from_mask(m, occluder=o)) if m.any() else None
                              for m, o in zip(masks, occ)])
    if (Path(cache) / "camera_H.npy").exists():
        # Lock to the camera (scripts/lock.py) as one rigid shape: a facade does not bend, and any per-frame freedom
        # lets the ad follow SAM's outline where a passing bus cuts into it (30 px drift on Piccadilly before).
        from pipeline.scene_lock import lock_ribbons
        ribbons = lock_ribbons(frames, ribbons, list(masks), None if occ[0] is None else list(occ),
                               H=np.load(Path(cache) / "camera_H.npy"))
    key = shape.get("creative", "16x9")
    art = cv2.cvtColor(cv2.imread(str(ROOT / c["creatives"][key]), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    out = [composite_ribbon(f, art, r, emissive=shape.get("emissive", True)) if r is not None
           else f.copy() for f, r in zip(frames, ribbons)]
    return soft_occlude(frames, out, occ), sum(r is not None for r in ribbons)


def soft_occlude(frames, placed, occluders, radius=4, gain=2.0):
    """Put the people and vehicles in front back over the ad with edges pulled onto their real outline: SAM's masks are
    blocky and a few px too large, and in that gap the original billboard showed through as ghosting."""
    from pipeline.matte import refine_occluder
    out = []
    for f, p, o in zip(frames, placed, occluders):
        if o is None or not o.any():
            out.append(p)
            continue
        a = refine_occluder(f, o, radius=radius, gain=gain)[..., None]  # wider radii smear a moving head
        out.append(np.rint(a * f + (1 - a) * p).astype(np.uint8))
    return out


def render_campaign(clip, cache, brand, emissive=False, label=True, out_dir=None, shape=None, tag=None):
    """Composite, label, watermark, export and verify one campaign on one analysed clip. Returns the report dict.

    label=False is only for the blinded realism rating (the label would give the edit away); never for distribution.
    """
    frames, fps = read_frames(clip)
    c, tokens = campaign(brand)
    timings = {}
    if shape and shape.get("type") in ("cylinder", "ribbon"):
        t0 = time.time()
        placed, tracked = (render_cylinder if shape["type"] == "cylinder" else render_ribbon)(frames, cache, c, shape)
        timings["composite_ms_per_frame"] = round(1000 * (time.time() - t0) / len(frames), 1)
        return finish(clip, cache, brand, c, tokens, frames, fps, placed, tracked, timings, label, out_dir, tag)
    quads = np.load(Path(cache) / "quads.npy")
    occ_file = Path(cache) / "occluders.npz"
    occluders = list(np.load(occ_file)["masks"]) if occ_file.exists() else None
    ref = int(np.argmax(np.isfinite(quads).all(axis=(1, 2))))
    creative = pick_creative(c, quads[ref])
    if not emissive:
        # Printed surface: bring the creative into the scene's tonal range (no blank reference shot available).
        tone = exposure.scene_tone(frames[ref], quads[ref])
        if tone is not None:
            # The ring around a sign is often bright sky; printed paper is lit by it but never as bright, so cap paper
            # white at 1.6x the frame's mean linear luminance (chosen by eye on the overcast roadside clip).
            gray = cv2.cvtColor(frames[ref], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            frame_mean = float(np.mean(np.where(gray <= 0.04045, gray / 12.92, ((gray + 0.055) / 1.055) ** 2.4)))
            tone["white"] = min(tone["white"], 1.6 * frame_mean / 0.92)
            creative = np.dstack([exposure.fit_creative_tone(creative[:, :, :3], tone), creative[:, :, 3]])
    t0 = time.time()
    placed = composite.render_clip(frames, quads, creative, emissive=emissive)
    if occluders is not None:
        placed = soft_occlude(frames, placed, occluders)
    timings["composite_ms_per_frame"] = round(1000 * (time.time() - t0) / len(frames), 1)
    tracked = int(np.isfinite(quads).all(axis=(1, 2)).sum())
    return finish(clip, cache, brand, c, tokens, frames, fps, placed, tracked, timings, label, out_dir, tag)


def finish(clip, cache, brand, c, tokens, frames, fps, placed, tracked, timings, label, out_dir, tag=None):
    """Label, watermark, export and verify the exact exported file; write the report."""
    if label:
        placed = [add_label(f) for f in placed]
    t0 = time.time()
    marked = watermark.embed(placed, c["campaign_token"])
    timings["watermark_s"] = round(time.time() - t0, 1)

    out = Path(out_dir or ROOT / "outputs") / f"{Path(clip).stem}{'_' + tag if tag else ''}_{brand}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    encode(marked, fps, out)
    exported, _ = read_frames(out)  # verify the exact exported file
    result = watermark.verify(exported, tokens)
    report = {"clip": clip, "brand": brand, "campaign_token": c["campaign_token"], "output": str(out),
              "frames": len(frames), "label": label, "tracked_frames": int(tracked),
              "timings": timings, "verify": result, "verified": result["campaign"] == c["campaign_token"]}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("cache")
    ap.add_argument("brand")
    ap.add_argument("--emissive", action="store_true", help="the surface is a lit screen")
    args = ap.parse_args()

    report = render_campaign(args.clip, args.cache, args.brand, args.emissive)
    print(json.dumps({k: report[k] for k in ("output", "timings", "verified")}, indent=2),
          f"\nbit accuracy {report['verify']['bit_accuracy']:.3f}")


if __name__ == "__main__":
    main()
