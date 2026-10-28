"""Track the surface quad through a clip from cached SAM masks and compare with per-frame mask quads.

  python scripts/track_eval.py assets/clips/billboard_night.mp4 cache/<dir> [cache/<dir2> ...] [--occluders cache/<dir>] [--masks]
Masks from several prompts are merged (union of found frames, first cache wins where both exist). Occluder masks (e.g.
SAM "person") are passed to the tracker and saved next to the quads for rendering.
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


def fill_occluder_gaps(masks, max_gap=2):
    """SAM sometimes misses a moving occluder for a frame; a missing mask would paint the ad over the person. Fill gaps
    of up to max_gap frames by moving the mask before the gap along the occluder's motion (centroid shift between the
    masks on either side). A union of the two would be too wide for a fast walker and expose the original surface."""
    import cv2

    masks = list(masks)
    for i in range(len(masks)):
        if masks[i] is not None:
            continue
        b = next((j for j in range(i - 1, max(-1, i - 1 - max_gap), -1) if masks[j] is not None), None)
        a = next((j for j in range(i + 1, min(len(masks), i + 1 + max_gap)) if masks[j] is not None), None)
        if b is None or a is None:
            continue
        cb, ca = (np.argwhere(masks[k]).mean(axis=0)[::-1] for k in (b, a))  # (x, y) centroids
        dx, dy = (ca - cb) * (i - b) / (a - b)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        h, w = masks[b].shape
        masks[i] = cv2.warpAffine(masks[b].astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
    return masks


def main(clip, *dirs):
    occ_dir = None
    # --masks: the surface masks are reliable in every frame (e.g. a screen whose content changes, which misleads
    # feature tracking): fit the quad to each mask and smooth (pipeline/mask_track.py, GPT).
    use_masks = "--masks" in dirs
    dirs = tuple(d for d in dirs if d != "--masks")
    if "--occluders" in dirs:
        i = dirs.index("--occluders")
        occ_dir, dirs = dirs[i + 1], dirs[:i] + dirs[i + 2:]
    frames, _ = read_frames(clip)
    masks = load_masks(dirs, len(frames))
    occluders = fill_occluder_gaps(load_masks([occ_dir], len(frames))) if occ_dir else None
    found = [i for i, m in enumerate(masks) if m is not None]
    # Reference: the found frame with the largest mask (fully visible surface).
    ref = max(found, key=lambda i: masks[i].sum())
    if use_masks:
        from pipeline.mask_track import track_from_masks
        r = track_from_masks(masks, occluder_masks=occluders)
    else:
        r = g.propagate_quad(frames, ref, g.quad_from_mask(masks[ref]), surface_masks=masks, occluder_masks=occluders)
    print("masks :", "".join("#" if m is not None else "." for m in masks))
    print("status:", "".join({"ok": "#", "occluded": "o", "lost": "."}[s] for s in r["status"]))
    errs = [np.linalg.norm(g.quad_from_mask(masks[i]) - r["quads"][i], axis=1).mean() for i in found
            if np.isfinite(r["quads"][i]).all() and g.quad_from_mask(masks[i]) is not None]
    print(f"ref {ref}; tracked vs mask quad: median {np.median(errs):.1f} px, p90 {np.percentile(errs, 90):.1f}, "
          f"max {np.max(errs):.1f} (n={len(errs)}); coverage {np.mean([s != 'lost' for s in r['status']]):.0%}")
    np.save(Path(dirs[0]) / "quads.npy", np.asarray(r["quads"]) if use_masks else g.smooth_quads(r["quads"]))
    if occluders is not None:
        empty = np.zeros(frames[0].shape[:2], bool)
        np.savez_compressed(Path(dirs[0]) / "occluders.npz", masks=np.stack([m if m is not None else empty
                                                                             for m in occluders]))


if __name__ == "__main__":
    main(*sys.argv[1:])
