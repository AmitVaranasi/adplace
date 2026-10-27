"""Track the surface quad through a clip from cached SAM masks and compare with per-frame mask quads.

  python scripts/track_eval.py assets/clips/billboard_night.mp4 cache/<dir> [cache/<dir2> ...]
Masks from several prompts are merged (union of found frames, first cache wins where both exist).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import geometry as g  # noqa: E402
from pipeline.sam_track import read_frames  # noqa: E402


def load_masks(dirs, n):
    masks = [None] * n
    for d in dirs:
        z = np.load(Path(d) / "surface_masks.npz")
        for i in range(n):
            if masks[i] is None and np.isfinite(z["scores"][i]):
                masks[i] = z["masks"][i]
    return masks


def main(clip, *dirs):
    frames, _ = read_frames(clip)
    masks = load_masks(dirs, len(frames))
    found = [i for i, m in enumerate(masks) if m is not None]
    # Reference: the found frame with the largest mask (fully visible surface).
    ref = max(found, key=lambda i: masks[i].sum())
    r = g.propagate_quad(frames, ref, g.quad_from_mask(masks[ref]), surface_masks=masks)
    print("masks :", "".join("#" if m is not None else "." for m in masks))
    print("status:", "".join({"ok": "#", "occluded": "o", "lost": "."}[s] for s in r["status"]))
    errs = [np.linalg.norm(g.quad_from_mask(masks[i]) - r["quads"][i], axis=1).mean() for i in found
            if np.isfinite(r["quads"][i]).all() and g.quad_from_mask(masks[i]) is not None]
    print(f"ref {ref}; tracked vs mask quad: median {np.median(errs):.1f} px, p90 {np.percentile(errs, 90):.1f}, "
          f"max {np.max(errs):.1f} (n={len(errs)}); coverage {np.mean([s != 'lost' for s in r['status']]):.0%}")
    np.save(Path(dirs[0]) / "quads.npy", g.smooth_quads(r["quads"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
