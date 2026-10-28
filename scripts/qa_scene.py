"""Automatic QA of a multi-ad scene render against the checklist Muse Spark drew up (thresholds are Muse's).

  python scripts/qa_scene.py outputs/piccadilly_long_viewer-a.mp4

Per slot: geometry pops (corner acceleration, px/frame^2; fail > 3 px jump), flicker of the placed ad (mean
|frame-to-frame change| of its mean luma over unoccluded pixels, 5-frame smoothing removed; fail > 3/255),
chunk seams (luma step at render-chunk boundaries vs the clip's typical step; fail > 2/255 above it). Whole clip:
disclosure label present on every frame, Video Seal manifest verified on the exported file.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.render_scene import Frames, Union, canvas, frame_geometry, scene_dir  # noqa: E402
from pipeline.maskstore import MaskStore  # noqa: E402

CHUNK = 48


def slot_mask(shape, T_inv, h, w):
    m = np.zeros((h, w), np.uint8)
    g = frame_geometry(shape, T_inv)
    if shape["type"] == "quad":
        cv2.fillConvexPoly(m, np.round(g).astype(np.int32), 1)
    else:
        poly = np.r_[np.c_[g["x"], g["top"]], np.c_[g["x"], g["bottom"]][::-1]]
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m > 0


def main(out):
    out = Path(out)
    report = json.loads(out.with_suffix(".json").read_text())
    clip = report["clip"]
    entry = json.loads((ROOT / "configs" / "clips.json").read_text())[clip]
    src, dst = Frames(ROOT / clip), Frames(out)
    d = scene_dir(clip)
    H = np.load(d / "camera_H.npy")
    shapes = json.loads((d / "shapes.json").read_text())
    _, offset = canvas(src)
    occ = Union([MaskStore(d / f"occ_{p}.npz") for p in entry.get("occluder_prompts", [])])
    slots = list(report["assignment"])
    T_inv = [np.linalg.inv(offset @ h) for h in H]

    results, ok = {}, True
    for sid in slots:  # pops: corner/edge-sample trajectories
        shape = shapes[sid]
        pts = np.array([np.asarray(frame_geometry(shape, t)) if shape["type"] == "quad" else
                        np.c_[frame_geometry(shape, t)["x"][::8], frame_geometry(shape, t)["top"][::8]]
                        for t in T_inv])
        acc = np.linalg.norm(np.diff(pts, 2, axis=0), axis=2).max(1)
        results[sid] = {"accel_p95": round(float(np.percentile(acc, 95)), 3), "accel_max": round(float(acc.max()), 3)}
        ok &= acc.max() <= 3.0

    luma = {sid: [] for sid in slots}
    label_ok, label_frames = 0, 0
    lh = src.h
    for i, (a, b) in enumerate(zip(src, dst)):
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        o = occ.frame(i) if occ.stores else np.zeros((src.h, src.w), bool)
        for sid in slots:
            m = slot_mask(shapes[sid], T_inv[i], src.h, src.w) & ~o
            luma[sid].append(float(gb[m].mean()) if m.any() else np.nan)
        y0 = lh - int(lh * 0.09)  # label band, bottom-left
        label_frames += 1
        label_ok += np.abs(a[y0:, : src.w // 3].astype(int) - b[y0:, : src.w // 3].astype(int)).max() > 40
    for sid in slots:
        x = np.array(luma[sid])
        step = np.abs(np.diff(x))
        trend = np.convolve(x, np.ones(5) / 5, "same")
        flicker = float(np.nanmean(np.abs(np.diff(x - trend))))
        seams = [step[i - 1] for i in range(CHUNK, len(x), CHUNK)]
        seam_excess = float(np.nanmax(seams) - np.nanmedian(step)) if seams else 0.0
        results[sid].update(flicker=round(flicker, 2), seam_excess=round(seam_excess, 2),
                            luma_step_median=round(float(np.nanmedian(step)), 2),
                            luma_step_max=round(float(np.nanmax(step)), 2))
        # A printed or static ad never jumps in brightness from one frame to the next; the smoothed flicker measure
        # missed 50-level jumps that followed the original screen's content, so bound the raw step as well.
        ok &= flicker <= 3.0 and seam_excess <= 2.0 and np.nanmax(step) <= 8.0
    summary = {"slots": results, "label_frames": f"{label_ok}/{label_frames}",
               "watermark": {"verified": report["verified"], "bit_accuracy": round(report["verify"]["bit_accuracy"], 3)}}
    ok &= label_ok == label_frames and report["verified"]
    summary["pass"] = bool(ok)
    out.with_suffix(".qa.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
