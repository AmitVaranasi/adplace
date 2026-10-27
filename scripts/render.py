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
    ratios = {"16x9": 16 / 9, "4x3": 4 / 3, "3x4": 3 / 4, "9x16": 9 / 16}
    key = min(c["creatives"], key=lambda k: abs(np.log(ratios[k] / (w / h))))
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


def render_campaign(clip, cache, brand, emissive=False, label=True, out_dir=None):
    """Composite, label, watermark, export and verify one campaign on one analysed clip. Returns the report dict.

    label=False is only for the blinded realism rating (the label would give the edit away); never for distribution.
    """
    frames, fps = read_frames(clip)
    quads = np.load(Path(cache) / "quads.npy")
    occ_file = Path(cache) / "occluders.npz"
    occluders = list(np.load(occ_file)["masks"]) if occ_file.exists() else None
    c, tokens = campaign(brand)
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
    timings = {}

    t0 = time.time()
    placed = composite.render_clip(frames, quads, creative, occluder_masks=occluders, emissive=emissive)
    timings["composite_ms_per_frame"] = round(1000 * (time.time() - t0) / len(frames), 1)
    if label:
        placed = [add_label(f) for f in placed]
    t0 = time.time()
    marked = watermark.embed(placed, c["campaign_token"])
    timings["watermark_s"] = round(time.time() - t0, 1)

    out = Path(out_dir or ROOT / "outputs") / f"{Path(clip).stem}_{brand}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    encode(marked, fps, out)
    exported, _ = read_frames(out)  # verify the exact exported file
    result = watermark.verify(exported, tokens)
    report = {"clip": clip, "brand": brand, "campaign_token": c["campaign_token"], "output": str(out),
              "frames": len(frames), "label": label, "tracked_frames": int(np.isfinite(quads).all(axis=(1, 2)).sum()),
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
