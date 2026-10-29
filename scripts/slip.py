"""Measure how much a placed ad slips against the real surface it sits on, frame to frame ("flinch").

  python scripts/slip.py assets/clips/pizzeria_counter.mp4 counter_sign counter_freezer

For each consecutive pair of frames: track real texture inside the slot in the ORIGINAL video (KLT, forward-backward
checked, people excluded) and compare where it truly moved with where the slot's quads move it. The per-frame slip is
the median disagreement in pixels. Sub-pixel slip reads as glued; 1 px or more on sharp edges reads as wobble.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.render_dynamic import Frames, Union, occluders, quad_mask, scene_dir, union_store  # noqa: E402


def slip(clip, sid, frames, occ, screen=False, glass=None):
    z = np.load(scene_dir(clip) / f"track_{sid}.npz")
    quads, valid = z["quads"], z["valid"]
    out = np.full(frames.n, np.nan)
    prev = None
    for i, f in enumerate(frames):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev is not None and valid[i - 1] and valid[i]:
            m = quad_mask(quads[i - 1], frames.h, frames.w).astype(np.uint8)
            if screen:  # the picture changes: measure on the bezel and the wall right around it
                region = (cv2.dilate(m, np.ones((41, 41), np.uint8)) > 0) & ~(cv2.erode(m, np.ones((13, 13), np.uint8)) > 0)
            else:
                region = cv2.erode(m, np.ones((9, 9), np.uint8)) > 0
            if occ is not None:
                region &= ~occ.frame(i - 1)
            if glass is not None:  # a glass shaker moves with the hand, not the surface
                region &= ~(cv2.dilate(glass.frame(i - 1).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
            region = region.astype(np.uint8)
            p = cv2.goodFeaturesToTrack(prev, 200, 0.01, 5, mask=region.astype(np.uint8))
            if p is not None and len(p) >= 8:
                q, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p, None, winSize=(21, 21), maxLevel=4)
                b, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, q, None, winSize=(21, 21), maxLevel=4)
                ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(b - p, axis=2)[:, 0] < 0.5)
                if ok.sum() >= 8:
                    H = cv2.getPerspectiveTransform(quads[i - 1].astype(np.float32), quads[i].astype(np.float32))
                    pred = cv2.perspectiveTransform(p[ok].astype(np.float64), H)
                    out[i] = float(np.median(np.linalg.norm(pred - q[ok], axis=2)))
        prev = g
    return out


def main(clip, sids):
    frames = Frames(ROOT / clip)
    entry = json.loads((ROOT / "configs" / "clips.json").read_text())[clip]
    occ = occluders(clip, entry, frames)
    for sid in sids:
        slot = next(x for x in entry["dynamic_slots"] if x["id"] == sid)
        glass = Union([union_store(clip, p, frames) for p in slot["glass_prompts"]]) if slot.get("glass_prompts") else None
        s = slip(clip, sid, frames, occ, screen=bool(slot.get("dynamic_content")) or slot.get("texture") == "ring",
                 glass=glass)
        m = s[np.isfinite(s)]
        np.save(scene_dir(clip) / f"slip_{sid}.npy", s)
        print(f"{sid}: {len(m)} measured pairs | slip median {np.median(m):.2f} px, p90 {np.percentile(m, 90):.2f}, "
              f"p99 {np.percentile(m, 99):.2f}, max {m.max():.1f} | frames > 1 px: {(m > 1).mean():.0%}, "
              f"> 2 px: {(m > 2).mean():.0%}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
