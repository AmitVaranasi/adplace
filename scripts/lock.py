"""Lock every slot in a clip to the camera's motion so placements cannot flinch against the scene.

  python scripts/lock.py assets/clips/dundas.mp4

Estimates the camera motion once per clip from the background (all slots and occluders excluded), saves it as
camera_H.npy in each slot's analysis dir, and replaces each flat slot's quads.npy with the scene-locked quads (the
unlocked fit is kept as quads_unlocked.npy). Curved slots (ribbons) are locked at render time from camera_H.npy.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.sam_track import read_frames  # noqa: E402
from pipeline.scene_lock import camera_motion, lock_quads  # noqa: E402


def main(clip):
    entry = json.loads((ROOT / "configs" / "clips.json").read_text())[clip]
    frames, _ = read_frames(ROOT / clip)
    dirs = [ROOT / s["analysis"] for s in entry["slots"]]
    masks = {d: np.load(d / "surface_masks.npz")["masks"] for d in dirs}
    exclude = np.zeros((len(frames),) + frames[0].shape[:2], bool)
    for m in masks.values():
        exclude |= m[:len(frames)]
    occ = None
    for d in dirs:
        if (d / "occluders.npz").exists():
            occ = np.load(d / "occluders.npz")["masks"][:len(frames)]
            exclude |= occ
            break
    H = np.asarray(camera_motion(frames, exclude_masks=list(exclude)))
    moved = np.abs(H - np.eye(3)).max()
    print(f"{Path(clip).stem}: camera motion estimated ({len(H)} frames, max |H - I| {moved:.4f})")
    for sl, d in zip(entry["slots"], dirs):
        np.save(d / "camera_H.npy", H)
        if sl.get("shape") or not (d / "quads.npy").exists():
            continue
        raw = d / "quads_unlocked.npy"
        quads = np.load(raw if raw.exists() else d / "quads.npy")
        if not raw.exists():
            np.save(raw, quads)
        others = [masks[o] for o in dirs if o != d]
        locked = np.asarray(lock_quads(frames, quads, list(masks[d]), None if occ is None else list(occ), H=H,
                                       exclude_masks=list(np.any(others, axis=0)) if others else None))
        np.save(d / "quads.npy", locked)
        step = np.linalg.norm(np.diff(locked, 2, axis=0), axis=2).mean(1)
        print(f"  {sl['slot_id']}: locked; corner accel p95 {np.percentile(step, 95):.3f} px/frame^2, "
              f"mean shift vs unlocked {np.nanmean(np.linalg.norm(locked - quads, axis=2)):.2f} px")


if __name__ == "__main__":
    main(sys.argv[1])
