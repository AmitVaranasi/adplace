"""Ads on surfaces in handheld, moving-camera clips: every surface carries its own plane track.

  python scripts/render_dynamic.py objects assets/clips/pizzeria_counter.mp4 television   # list + preview SAM objects
  python scripts/render_dynamic.py render assets/clips/pizzeria_counter.mp4 --assign counter_tv=slicedash ...

render_scene.py maps every slot through one camera homography per frame, which holds when the camera barely moves
(Piccadilly: 0.2 px/frame). Walking through a room it does not: surfaces at different depths move differently
(parallax), so each slot here is defined once at its reference frame and followed by pipeline.planar_track, and the
ad is blurred along its own motion over the shutter interval (pipeline.shutter) so whip pans smear it like the scene.
Slots live in configs/clips.json under the clip's "dynamic_slots".
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
from pipeline.maskstore import MaskStore, MaskWriter  # noqa: E402
from scripts.render_scene import Frames, Union, fit_shape, scene_dir, union_store  # noqa: E402


def object_stores(clip, prompt, frames):
    """One mask store per object SAM 3.1 found for a prompt (a room can hold two TVs), built from the recorded stream."""
    from pipeline.sam_api import _mask_region, parse_masks
    from scripts.analyse import cache_dir
    d = scene_dir(clip)
    index = d / f"objects_{prompt}.json"
    if not index.exists():
        base = cache_dir(clip, {"model": "meta/sam-3.1", "prompt": prompt})
        events = [json.loads(line) for line in base.with_name(base.name + "-events.jsonl").read_text().splitlines()]
        writers = {}
        for (oid, fi), rec in parse_masks(events).items():
            region = _mask_region(rec, frames.h, frames.w) if 0 <= fi < frames.n else None
            if region is not None and region[2].any():
                writers.setdefault(oid, MaskWriter(frames.n, frames.h, frames.w)).add(fi, box=region[:2], local=region[2])
        names = {}
        for oid, w in writers.items():
            names[oid] = f"obj_{prompt}_{oid}.npz"
            w.save(d / names[oid])
        index.write_text(json.dumps(names))
    return {oid: MaskStore(d / name) for oid, name in json.loads(index.read_text()).items()}


def area(store, i):
    if not store.any(i):
        return 0
    x0, y0, x1, y1 = store.box(i)
    return int(store[i][y0:y1, x0:x1].sum())


def cmd_objects(args):
    frames = Frames(ROOT / args.clip)
    stores = object_stores(args.clip, args.prompt, frames)
    tiles = []
    for oid, s in sorted(stores.items(), key=lambda kv: kv[0]):
        areas = np.array([area(s, i) for i in range(frames.n)])
        seen = np.flatnonzero(areas)
        if not len(seen):
            continue
        best = int(np.argmax(areas))
        print(f"object {oid}: frames {seen[0]}-{seen[-1]} ({len(seen)} seen), largest at f{best} "
              f"({100 * areas[best] / (frames.w * frames.h):.1f}% of frame)")
        f = frames[best].copy()
        m = s[best]
        f[m] = (0.5 * f[m] + 0.5 * np.array([0, 255, 255])).astype(np.uint8)
        cv2.putText(f, f"{args.prompt} #{oid} f{best}", (12, 40), 0, 1.2, (0, 255, 255), 3)
        tiles.append(cv2.resize(f, (640, 360)))
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        out = scene_dir(args.clip) / f"objects_{args.prompt}.jpg"
        cv2.imwrite(str(out), np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]))
        print("preview", out)


def occluders(clip, entry, frames):
    stores = [union_store(clip, p, frames) for p in entry.get("occluder_prompts", [])]
    return Union(stores) if stores else None


def inside_area(store, i, frames, margin=6):
    """Mask area, or 0 when the mask touches the image border (a clipped surface gives a wrong reference quad)."""
    if not store.any(i):
        return 0
    x0, y0, x1, y1 = store.box(i)
    if x0 < margin or y0 < margin or x1 > frames.w - margin or y1 > frames.h - margin:
        return 0
    return area(store, i)


def canonical(quad):
    """Corners as TL, TR, BR, BL (clockwise on screen, starting nearest the top-left), so artwork is never rotated
    by however a fit happened to order them."""
    q = np.asarray(quad, float)
    c = q.mean(0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]  # clockwise in image coordinates (y down)
    return np.roll(q, -int(np.argmin(q[:, 0] + q[:, 1])), axis=0)


def full_extent(store, tr, frames, occ, ref, quad, keep=0.5, step=2):
    """The whole surface in the reference frame: every well-tracked frame votes its mask into the plane, only where
    no person covers it. A boy standing in front of the sign at the reference frame hides its right end; frames
    where he has moved show it, so the fitted rectangle grows to the real sign."""
    from scripts.render_scene import PAD
    ox, oy = int(PAD * frames.w), int(PAD * frames.h)
    cw, ch = frames.w + 2 * ox, frames.h + 2 * oy
    offset = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], float)
    hits, seen = np.zeros((ch, cw), np.float32), np.zeros((ch, cw), np.float32)
    full = np.ones((frames.h, frames.w), np.uint8)
    good = {"reference", "anchored", "reacquired"}
    for i in range(0, frames.n, step):
        if tr["status"][i] not in good or not store.any(i):
            continue
        T = offset @ cv2.getPerspectiveTransform(tr["quads"][i].astype(np.float32), quad.astype(np.float32))
        clear = full if occ is None else (~occ.frame(i)).astype(np.uint8)
        seen += cv2.warpPerspective(clear, T, (cw, ch), flags=cv2.INTER_NEAREST)
        hits += cv2.warpPerspective((store[i] & (clear > 0)).astype(np.uint8), T, (cw, ch), flags=cv2.INTER_LINEAR)
    plane = (hits >= keep * np.maximum(seen, 1)) & (seen > 2)
    if not plane.any():  # too few well-tracked frames to vote: keep the reference quad
        return quad, tr["quads"]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(plane.astype(np.uint8), connectivity=8)
    if n > 1:
        plane = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    big = canonical(np.asarray(fit_shape(plane, {"type": "quad"})["quad"], float) - [ox, oy])
    # Each frame: the exact map from the reference quad to its tracked quad, applied to the grown quad.
    out = np.full_like(tr["quads"], np.nan)
    for i in np.flatnonzero(tr["valid"]):
        P = cv2.getPerspectiveTransform(quad.astype(np.float32), tr["quads"][i].astype(np.float32))
        out[i] = cv2.perspectiveTransform(big[None].astype(np.float64), P)[0]
    return big, out


def drop_on_screen(occluders, screen, inside=0.95, grow=6):
    """Remove occluder blobs that lie inside a screen: SAM's "person" also finds the presenter shown ON the TV, and
    cutting him out of the ad punches holes in it. Someone really standing in front of a screen spills past its
    edges, so only blobs at least `inside` within the (slightly grown) screen are treated as its picture."""
    if occluders is None or screen is None or not occluders.any() or not screen.any():
        return occluders
    screen = cv2.dilate(screen.astype(np.uint8), np.ones((2 * grow + 1,) * 2, np.uint8)) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(occluders.astype(np.uint8), connectivity=8)
    if n <= 1:
        return occluders
    within = np.bincount(labels[screen], minlength=n)
    keep = within < inside * stats[:, cv2.CC_STAT_AREA]
    keep[0] = False
    return keep[labels]


def quad_mask(quad, h, w):
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, np.round(quad).astype(np.int32), 1)
    return m > 0


def camera_steps(clip, frames, occ, width=640):
    """Whole-frame motion from each frame to the next (people excluded), cached: (N-1,3,3) in full-res pixels."""
    out = scene_dir(clip) / "camera_steps.npy"
    if out.exists():
        return np.load(out)
    s = width / frames.w
    S = np.diag([s, s, 1.0])
    steps, prev, prev_ok = [], None, None
    for i, f in enumerate(frames):
        g = cv2.cvtColor(cv2.resize(f, (width, int(frames.h * s))), cv2.COLOR_BGR2GRAY)
        ok = np.ones(g.shape, np.uint8) if occ is None else \
            (cv2.resize((~occ.frame(i)).astype(np.uint8), g.shape[::-1], interpolation=cv2.INTER_NEAREST))
        if prev is not None:
            H = np.eye(3)
            p = cv2.goodFeaturesToTrack(prev, 800, 0.01, 7, mask=prev_ok)
            if p is not None and len(p) >= 12:
                q, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p, None, winSize=(31, 31), maxLevel=4)
                good = st[:, 0] == 1
                if good.sum() >= 12:
                    h, inl = cv2.findHomography(p[good], q[good], cv2.RANSAC, 2.0)
                    if h is not None and inl.sum() >= 12:
                        H = np.linalg.inv(S) @ h @ S
            steps.append(H)
        prev, prev_ok = g, ok
    np.save(out, np.array(steps))
    return np.array(steps)


def extend_with_camera(quads, valid, steps, seen, limit=60):
    """Carry a lost plane through frames where SAM still sees the surface, using the whole-frame motion from the
    nearest tracked frame. Without this the original surface pops back when the plane tracker lets go: far away
    behind display glass, or half out of frame on the way out. Only bridges up to `limit` frames per side."""
    quads, valid = quads.copy(), valid.copy()
    n = len(valid)
    for i in np.flatnonzero(valid):
        for direction in (1, -1):
            j, q = i + direction, quads[i]
            while 0 <= j < n and not valid[j] and seen[j] and abs(j - i) <= limit:
                H = steps[j - 1] if direction == 1 else np.linalg.inv(steps[j])
                q = cv2.perspectiveTransform(q[None].astype(np.float64), H)[0]
                quads[j], valid[j] = q, True
                j += direction
    return quads, valid


def keep_screen_shape(quads, valid, edges, steps):
    """A TV half out of frame shows two corners; the tracker's guess for the hidden pair can fold the quad into a
    wedge. Carry the nearest fully seen quad here with the whole-frame camera motion (keeps the screen's true shape),
    then move it by the similarity that puts its predicted corners onto the corners really seen (a corner is seen
    when both its edges are: edge k runs from corner k to k+1)."""
    quads = quads.copy()
    full = np.flatnonzero(valid & edges.all(1))
    if not len(full):
        return quads
    seen_corner = edges & np.roll(edges, 1, axis=1)
    cum = [np.eye(3)]  # cum[i] maps frame 0 to frame i
    for H in steps:
        cum.append(H @ cum[-1])
    for i in np.flatnonzero(valid & ~edges.all(1)):
        j = full[np.argmin(np.abs(full - i))]
        M = cum[i] @ np.linalg.inv(cum[j])
        pred = cv2.perspectiveTransform(quads[j][None].astype(np.float64), M)[0]
        c = np.flatnonzero(seen_corner[i])
        if len(c) >= 2:
            A, _ = cv2.estimateAffinePartial2D(pred[c].astype(np.float32), quads[i][c].astype(np.float32))
            if A is not None:
                pred = pred @ A[:, :2].T + A[:, 2]
        elif len(c) == 1:
            pred = pred + (quads[i][c[0]] - pred[c[0]])
        quads[i] = pred
    return quads


def fit_to_outline(pred, mask, blocked, h, w, reg=0.01, cap=25.0, samples=40):
    """Nudge a predicted screen quad so its edges sit on SAM's outline of the screen. Outline pixels on the image
    border (the screen is cut off there, that is not its edge) or next to a person in front do not count, and edge
    samples outside the image are skipped, so a half-visible TV is fitted on the edges it really shows."""
    from scipy.optimize import least_squares
    m = mask.astype(np.uint8)
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    edge[:4], edge[-4:], edge[:, :4], edge[:, -4:] = False, False, False, False
    edge &= ~blocked
    if edge.sum() < 30:
        return pred, 0
    dt = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5)
    t = np.linspace(0, 1, samples, endpoint=False)
    usable = []
    for k in range(4):
        a, b = pred[k], pred[(k + 1) % 4]
        pts = a + (b - a) * t[:, None]
        ok = (pts[:, 0] > 6) & (pts[:, 0] < w - 7) & (pts[:, 1] > 6) & (pts[:, 1] < h - 7)
        if ok.any():
            usable.append((k, t[ok]))
    n = sum(len(tk) for _, tk in usable)
    if n < 20:
        return pred, n

    def residual(x):
        q = pred + x.reshape(4, 2)
        r = []
        for k, tk in usable:
            pts = q[k] + (q[(k + 1) % 4] - q[k]) * tk[:, None]
            xi = np.clip(pts[:, 0], 0, w - 1).astype(np.float32)
            yi = np.clip(pts[:, 1], 0, h - 1).astype(np.float32)
            d = cv2.remap(dt, xi[None], yi[None], cv2.INTER_LINEAR)[0]
            r.append(np.minimum(d, cap))
        return np.r_[np.concatenate(r), reg * x * np.sqrt(n)]

    # Coarse shift first: camera-motion drift can exceed what the distance field's reach lets the refinement see.
    pts = np.concatenate([pred[k] + (pred[(k + 1) % 4] - pred[k]) * tk[:, None] for k, tk in usable])
    far = np.minimum(cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5), 60.0)
    best, shift = np.inf, np.zeros(2)
    for dx in range(-48, 49, 4):
        for dy in range(-48, 49, 4):
            xi = np.clip(pts[:, 0] + dx, 0, w - 1).astype(np.float32)
            yi = np.clip(pts[:, 1] + dy, 0, h - 1).astype(np.float32)
            cost = cv2.remap(far, xi[None], yi[None], cv2.INTER_LINEAR).mean() + 0.01 * np.hypot(dx, dy)
            if cost < best:
                best, shift = cost, np.array([dx, dy], float)
    pred = pred + shift
    x = least_squares(residual, np.zeros(8), loss="soft_l1", f_scale=2.0, max_nfev=80).x
    return pred + x.reshape(4, 2), n


def track_screen_outline(frames, ref, quad, store, occ, steps, limit=60):
    """Screens: carry the whole quad with the camera, frame to frame from the reference, and pull it onto the
    screen's visible outline each frame (fit_to_outline). The quad keeps a TV's shape when it is half out of frame."""
    n = frames.n
    quads, valid, fitted = np.full((n, 4, 2), np.nan), np.zeros(n, bool), np.zeros(n, int)
    quads[ref], valid[ref] = quad, True
    for direction in (1, -1):
        q, since = np.asarray(quad, float), 0
        i = ref + direction
        while 0 <= i < n:
            H = steps[i - 1] if direction == 1 else np.linalg.inv(steps[i])
            q = cv2.perspectiveTransform(q[None].astype(np.float64), H)[0]
            if store.any(i):
                mask = store[i]
                blocked = np.zeros_like(mask)
                if occ is not None:
                    o = drop_on_screen(occ.frame(i), mask)
                    blocked = cv2.dilate(o.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                q, fitted[i] = fit_to_outline(q, mask, blocked, frames.h, frames.w)
                since = 0
            else:
                since += 1
            if since > limit or not np.isfinite(q).all():
                break
            inside = quad_mask(q, frames.h, frames.w).any() if np.isfinite(q).all() else False
            quads[i], valid[i] = q, inside and since <= limit and store.any(i)
            i += direction
    # Zero-lag smoothing per run: a short quadratic Savitzky-Golay removes outline jitter but keeps hand shake.
    from scipy.signal import savgol_filter
    edges = np.flatnonzero(np.diff(np.r_[0, valid.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= 7:
            quads[a:b] = savgol_filter(quads[a:b], 7, 2, axis=0)
    return {"quads": quads, "valid": valid, "visible": valid.astype(float), "fitted": fitted}


def surface_steps(frames, quads, valid, occ, screen=False, exclude=None):
    """Frame-to-frame motion of the surface itself: KLT on its own texture (forward-backward checked, people
    excluded), one homography per consecutive pair. For a screen the picture changes, so only a band around its
    edge (bezel and the wall right next to it) is used. NaN where there is too little texture."""
    G = np.full((frames.n, 3, 3), np.nan)
    prev = None
    for i, f in enumerate(frames):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev is not None and valid[i - 1] and valid[i]:
            m = quad_mask(quads[i - 1], frames.h, frames.w).astype(np.uint8)
            if screen:
                region = (cv2.dilate(m, np.ones((41, 41), np.uint8)) > 0) & ~(cv2.erode(m, np.ones((13, 13), np.uint8)) > 0)
            else:
                region = cv2.erode(m, np.ones((7, 7), np.uint8)) > 0
            if occ is not None:
                region &= ~(cv2.dilate(occ.frame(i - 1).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
            if exclude is not None:  # glass passing over the surface moves with the hand, not the surface
                region &= ~(cv2.dilate(exclude.frame(i - 1).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
            p = cv2.goodFeaturesToTrack(prev, 300, 0.01, 5, mask=region.astype(np.uint8))
            if p is not None and len(p) >= 10:
                q, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p, None, winSize=(21, 21), maxLevel=4)
                b, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, q, None, winSize=(21, 21), maxLevel=4)
                ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(b - p, axis=2)[:, 0] < 0.5)
                if ok.sum() >= 10:
                    H, inl = cv2.findHomography(p[ok], q[ok], cv2.RANSAC, 1.0)
                    if H is not None and inl.sum() >= 8:
                        G[i] = H
        prev = g
    return G


def glue(quads, valid, G, steps, ref, alpha=0.06):
    """Make the ad move exactly as the surface does from frame to frame, and let the absolute track only pull out
    drift, slowly. Absolute trackers (templates, SAM outlines, keyframe switches, smoothing) are right on average but
    each frame's small correction shows as a flinch; measured on the counter clip, the surface itself follows a
    homography to 0.1 px between frames while our quads disagreed by 1.6-3.7 px (p90). Here each quad is the
    previous one moved by the surface's own motion G (whole-frame camera motion where the texture is too thin), plus
    `alpha` of the remaining gap to the absolute track: a 5 px drift is removed at 0.3 px per frame, below what
    the eye catches. Runs start from the reference frame (or each run's middle) in both directions."""
    out = quads.copy()
    edges = np.flatnonzero(np.diff(np.r_[0, valid.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        start = ref if a <= ref < b else (a + b) // 2
        for direction in (1, -1):
            q = quads[start].astype(np.float64)
            i = start + direction
            while a <= i < b:
                cam = steps[i - 1] if direction == 1 else np.linalg.inv(steps[i])
                step = G[i] if direction == 1 else (np.linalg.inv(G[i + 1]) if np.isfinite(G[i + 1]).all() else None)
                by_cam = cv2.perspectiveTransform(q[None], cam)[0]
                pred = by_cam
                if step is not None and np.isfinite(step).all():
                    by_surface = cv2.perspectiveTransform(q[None], step)[0]
                    # A few features left at the image edge can fit a wild homography (a freezer leaving the frame
                    # blew up to y=-829). Reject only implausible steps: a near surface (a box lid under the phone)
                    # legitimately moves very differently from the whole-frame camera motion through parallax.
                    a0 = cv2.contourArea(q.astype(np.float32))
                    a1 = cv2.contourArea(by_surface.astype(np.float32))
                    jump = np.linalg.norm(by_surface - q, axis=1).max()
                    if (cv2.isContourConvex(by_surface.astype(np.float32)) and a0 > 0 and 0.8 < a1 / a0 < 1.25
                            and jump < 80 + np.linalg.norm(by_cam - q, axis=1).max()):
                        pred = by_surface
                q = pred + alpha * (quads[i] - pred)
                out[i] = q
                i += direction
    return out


def exposure_gain(frames, quads, valid, ref, occ, size=(200, 140), screen=False):
    """Per-frame (gain, veil) of the real surface relative to the reference frame, measured on the same physical
    pixels (rectified through the track, people excluded). A phone re-exposes as it walks: on the counter clip the
    black sign's own paint is 2x brighter than at the reference in a typical frame and up to 8x at the bright
    window, so an ad graded once looks pasted on. Smoothed over ~9 frames; 1 where it cannot be measured. Screens
    are measured on the band around them (their picture changes)."""
    from scipy.ndimage import median_filter
    W, H = size
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    ones = np.ones((frames.h, frames.w), np.uint8)
    rect = {}
    for i, f in enumerate(frames):
        if not valid[i]:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        ok = ones if occ is None else (~occ.frame(i)).astype(np.uint8)
        if screen:  # grow the quad by ~20% and keep only the outer band
            c = quads[i].mean(0)
            q = (c + 1.4 * (quads[i] - c)).astype(np.float32)
            M = cv2.getPerspectiveTransform(q, dst)
            band = np.ones((H, W), bool)
            band[int(H * 0.15):int(H * 0.85), int(W * 0.15):int(W * 0.85)] = False
        else:
            M = cv2.getPerspectiveTransform(quads[i].astype(np.float32), dst)
            band = np.ones((H, W), bool)
        r = cv2.warpPerspective(g, M, (W, H)).astype(np.float32)
        v = (cv2.warpPerspective(ok, M, (W, H), flags=cv2.INTER_NEAREST) > 0) & band
        v &= cv2.warpPerspective(ones, M, (W, H), flags=cv2.INTER_NEAREST) > 0
        rect[i] = (r, v)
    # observed = a * reference + b on the same pixels, from the 10th and 90th percentiles: a is exposure, b is a
    # veil added on top (display-case glass lifts a black sign to grey without making its letters brighter).
    # A plain ratio of near-black medians said 1.5x where the real sign stayed black.
    gain, veil = np.full(frames.n, np.nan), np.full(frames.n, np.nan)
    if ref in rect:
        r0, v0 = rect[ref]
        for i, (r, v) in rect.items():
            k = v & v0
            if k.sum() > 3000 and np.subtract(*np.percentile(r0[k], [90, 10])) < 25:
                # A blank surface (grey wall, white napkin) has no contrast to separate exposure from a veil:
                # the fit ran to its bounds and washed the ad out. A plain brightness ratio is all it supports.
                gain[i], veil[i] = (np.median(r[k]) + 1) / (np.median(r0[k]) + 1), 0.0
            elif k.sum() > 3000 and np.ptp(r0[k]) > 12:
                x, y = r0[k].astype(np.float64), r[k].astype(np.float64)
                keep = np.ones(len(x), bool)
                for _ in range(3):  # least squares, refit without the worst residuals (a hand, a reflection)
                    A = np.c_[x[keep], np.ones(keep.sum())]
                    (g, c), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
                    res = np.abs(y - (g * x + c))
                    keep = res <= max(np.percentile(res, 80), 4.0)
                gain[i], veil[i] = g, c
    out = []
    for x, neutral, lo, hi in ((gain, 1.0, 0.6, 1.8), (veil, 0.0, -40.0, 80.0)):
        ok = np.isfinite(x)
        if ok.sum() >= 2:
            idx = np.arange(frames.n)
            x = np.interp(idx, idx[ok], x[ok])
            x = median_filter(x, 9, mode="nearest")
            x = np.convolve(np.pad(x, 4, mode="edge"), np.ones(9) / 9, "valid")
        else:
            x = np.full(frames.n, neutral)
        out.append(np.clip(x, lo, hi))
    return out


def track_by_mask_ring(frames, ref, quad, store, occ, ring=40, min_pts=12):
    """Blank surfaces with something nearer in view (a grey wall behind a tree): estimate each frame-to-frame step
    only from features on the surface's own plane, inside SAM's mask of it widened by `ring` px (its pillar, bench
    and edges), people excluded, and chain from the reference. Whole-frame motion there is the tree's, whose parallax
    dragged the poster off the wall. Valid while SAM still sees the surface."""
    n = frames.n
    quads, valid = np.full((n, 4, 2), np.nan), np.zeros(n, bool)
    quads[ref], valid[ref] = quad, True
    gray = {}

    def g(i):
        if i not in gray:
            gray[i] = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        return gray[i]

    for direction in (1, -1):
        q, i = np.asarray(quad, float), ref
        while 0 <= i + direction < n and store.any(i) and store.any(i + direction):
            j = i + direction
            region = cv2.dilate(store[i].astype(np.uint8), np.ones((2 * ring + 1,) * 2, np.uint8)) > 0
            if occ is not None:
                region &= ~(cv2.dilate(occ.frame(i).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
            p = cv2.goodFeaturesToTrack(g(i), 400, 0.01, 6, mask=region.astype(np.uint8))
            if p is None or len(p) < min_pts:
                break
            nq, st, _ = cv2.calcOpticalFlowPyrLK(g(i), g(j), p, None, winSize=(21, 21), maxLevel=4)
            bq, st2, _ = cv2.calcOpticalFlowPyrLK(g(j), g(i), nq, None, winSize=(21, 21), maxLevel=4)
            ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(bq - p, axis=2)[:, 0] < 0.7)
            if ok.sum() < min_pts:
                break
            H, inl = cv2.findHomography(p[ok], nq[ok], cv2.RANSAC, 1.5)
            if H is None or inl.sum() < min_pts:
                break
            q = cv2.perspectiveTransform(q[None], H)[0]
            quads[j], valid[j] = q, True
            i = j
    return {"quads": quads, "valid": valid, "visible": valid.astype(float)}


def track_slot(clip, slot, frames, occ, force=False):
    """Plane track for one slot, cached: (N,4,2) quads, valid, visible, and the shutter fraction."""
    from pipeline.planar_track import track_plane, track_screen
    out = scene_dir(clip) / f"track_{slot['id']}.npz"
    if out.exists() and not force:
        z = np.load(out)
        return {k: z[k] for k in z.files}
    s = object_stores(clip, slot["prompt"], frames)[str(slot["object"])]
    ref = slot.get("ref_frame")
    if ref is None:
        # The reference defines the shape, exposure and softness, so it must be sharp: the largest fully visible
        # view among frames where the camera is nearly still (a mid-pan reference made a freezer ad 3x too soft).
        inside = np.array([inside_area(s, i, frames) for i in range(frames.n)], float)
        if not inside.any():
            inside = np.array([area(s, i) for i in range(frames.n)], float)
        steps = camera_steps(clip, frames, occ)
        speed = np.r_[0.0, [np.linalg.norm(cv2.perspectiveTransform(np.array([[[640.0, 360.0]]]), H)[0, 0]
                                           - [640.0, 360.0]) for H in steps]]
        speed = np.maximum(speed, np.r_[speed[1:], 0.0])  # motion on either side of the frame
        still = inside * (speed < 3.0)
        ref = int(np.argmax(still if still.max() > 0.5 * inside.max() else inside))
    quad = canonical(slot["quad"] if "quad" in slot else fit_shape(s[ref], {"type": "quad"})["quad"])
    t0 = time.time()
    masks = dict(surface_masks=lambda i: s[i] if s.any(i) else None,
                 occluder_masks=(lambda i: drop_on_screen(occ.frame(i), s[i] if s.any(i) else None)
                                 if slot.get("dynamic_content") else occ.frame(i)) if occ else None)
    if slot.get("track") == "mask_ring":
        tr = track_by_mask_ring(frames, ref, quad, s, occ)
    elif slot.get("dynamic_content"):  # a TV on a blank wall: nothing around it to track; follow camera + outline
        tr = track_screen_outline(frames, ref, quad, s, occ, camera_steps(clip, frames, occ))
    else:
        tr = track_plane(frames, ref, quad, **masks)
    quads = tr["quads"]
    if "status" in tr:
        print(f"  {slot['id']}: tracker status", {k: tr["status"].count(k) for k in sorted(set(tr["status"]))})
    if "observed_edges" in tr:
        quads = keep_screen_shape(quads, tr["valid"], tr["observed_edges"], camera_steps(clip, frames, occ))
    if slot.get("extent", not slot.get("dynamic_content")) and "quad" not in slot:
        big, quads = full_extent(s, tr, frames, occ, ref, quad)
        print(f"  {slot['id']}: reference quad {np.round(quad).astype(int).tolist()} -> full extent "
              f"{np.round(big).astype(int).tolist()}")
    seen = np.array([s.any(i) for i in range(frames.n)])
    valid = tr["valid"]
    if slot.get("track") != "mask_ring":  # that track is already chained on the surface's own plane
        before = valid.sum()
        quads, valid = extend_with_camera(quads, valid, camera_steps(clip, frames, occ), seen)
        if valid.sum() > before:
            print(f"  {slot['id']}: camera motion bridges {valid.sum() - before} frames the plane tracker lost")
        # Screens (changing picture) and blank surfaces (a white wall: only leaves in front show inside it) are
        # glued by the band around them, which lies in the same wall plane.
        glass = Union([union_store(clip, p, frames) for p in slot["glass_prompts"]]) if slot.get("glass_prompts") else None
        G = surface_steps(frames, quads, valid, occ,
                          screen=bool(slot.get("dynamic_content")) or slot.get("texture") == "ring", exclude=glass)
        quads = glue(quads, valid, G, camera_steps(clip, frames, occ), ref, alpha=float(slot.get("glue_alpha", 0.06)))
        print(f"  {slot['id']}: glued to its own texture on {np.isfinite(G[:, 0, 0]).sum()} of {valid.sum()} frames")
    # Whip pans: a quad carried by motion alone can land on a hand or the floor. Where SAM sees the surface, the
    # quad's on-screen part must mostly be that surface (people in front do not count against it).
    if slot.get("agree", True):
        dropped = 0
        for i in np.flatnonzero(valid):
            qm = quad_mask(quads[i], frames.h, frames.w)
            if occ is not None:
                qm &= ~occ.frame(i)
            if qm.sum() < 400:
                continue
            m = s[i] if s.any(i) else None
            if m is None or (m & qm).sum() / qm.sum() < float(slot.get("agree_min", 0.4)):
                valid[i] = False
                dropped += 1
        if dropped:
            print(f"  {slot['id']}: {dropped} frames dropped where the quad left the surface SAM sees")
    screen = bool(slot.get("dynamic_content")) or slot.get("texture") == "ring"
    gain, veil = exposure_gain(frames, quads, valid, ref, occ, screen=screen)
    print(f"  {slot['id']}: exposure vs reference: gain p5/median/p95 {np.percentile(gain[valid], 5):.2f}/"
          f"{np.median(gain[valid]):.2f}/{np.percentile(gain[valid], 95):.2f}, veil "
          f"{np.percentile(veil[valid], 5):.0f}/{np.median(veil[valid]):.0f}/{np.percentile(veil[valid], 95):.0f}")
    shutter = float(slot.get("shutter", 0.5))  # 180-degree shutter until the blur is measured
    res = {"quads": quads, "valid": valid, "visible": tr["visible"], "ref": np.int64(ref), "gain": gain, "veil": veil,
           "shutter": np.float64(shutter)}
    np.savez(out, **res)
    print(f"  {slot['id']}: tracked in {time.time() - t0:.0f} s, valid {tr['valid'].mean():.0%}, ref f{ref}, "
          f"shutter {shutter:.2f}", flush=True)
    return res


def inset(quad, px):
    """Move each edge inward by px (the ad stops short of a TV bezel or a freezer's metal trim)."""
    q = np.asarray(quad, float)
    if px <= 0:
        return q
    lines = []
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        d = (b - a) / max(np.linalg.norm(b - a), 1e-6)
        nrm = np.array([-d[1], d[0]])
        if np.dot(q.mean(0) - a, nrm) < 0:
            nrm = -nrm
        lines.append((a + px * nrm, d))
    out = []
    for k in range(4):
        (p1, d1), (p2, d2) = lines[k - 1], lines[k]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-9:
            return q
        t = np.linalg.solve(A, p2 - p1)
        out.append(p1 + t[0] * d1)
    return np.array(out)


def carry_shading(placed, original, quad, occluders=None, lo=0.45, hi=1.35):
    """Print, not sticker: ink on a blank surface keeps the surface's own light, so multiply the ad by the original
    surface's shading relative to its median (the crease and the hinge gradient of a box lid, the shadow of a hand
    over it). Only for surfaces that were blank; a printed sign's old artwork would ghost through."""
    h, w = original.shape[:2]
    m = quad_mask(quad, h, w)
    if occluders is not None:
        m &= ~occluders
    if m.sum() < 200:
        return placed
    gray = cv2.GaussianBlur(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    shade = np.clip(gray / max(float(np.median(gray[m])), 1.0), lo, hi)
    out = placed.astype(np.float32)
    out[m] *= shade[m, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def render(clip, entry, assignment, out, chunk=48, label=True, retrack=False, start=0):
    from pipeline import composite, watermark
    from pipeline.shutter import shutter_quads
    from scripts.render import add_label, campaign, pick_creative, soft_occlude
    from scripts.render_scene import grade_slot, manifest_id
    frames = Frames(ROOT / clip)
    inventory = json.loads((ROOT / "configs" / "inventory.json").read_text())
    slots = {s["id"]: s for s in entry["dynamic_slots"]}
    occ = occluders(clip, entry, frames)
    layers, glass_masks = [], None
    for sid, brand in assignment.items():
        slot = slots[sid]
        tr = track_slot(clip, slot, frames, occ, force=retrack)
        c, _ = campaign(brand)
        store = object_stores(clip, slot["prompt"], frames)[str(slot["object"])]
        ref = int(tr["ref"])
        surface = cv2.cvtColor(frames[ref], cv2.COLOR_BGR2GRAY)[store[ref] & ~(occ.frame(ref) if occ else np.zeros_like(store[ref]))]
        tone = slot.get("tone") or ("dark" if np.median(surface) < 70 else "light")
        if tone == "dark":  # the brand's dark artwork, same aspect choice (make_creatives.dark)
            c = {**c, "creatives": {k: v.replace(".png", "_dark.png") for k, v in c["creatives"].items()}}
        art = pick_creative(c, tr["quads"][ref])
        # How soft the real surface is (focus, compression), measured once where it is still and fully seen;
        # per-frame estimates would count motion blur, which the shutter integration already adds.
        blur, grain = composite.estimate_blur_and_grain(frames[ref], tr["quads"][ref])
        blur = min(blur, 1.5)
        print(f"  {sid}: {brand} ({tone} artwork; surface median luma {np.median(surface):.0f})")
        art = grade_slot(art, frames, store, occ)
        print(f"  {sid}: surface softness sigma {blur:.2f} px, grain {grain:.2f}")
        plate = None
        if slot.get("reflections") or slot.get("glass_prompts"):
            from pipeline.clean_plate import CleanPlate
            glass = [union_store(clip, p, frames) for p in slot.get("glass_prompts", [])]
            t1 = time.time()
            plate = CleanPlate(frames, tr["quads"], tr["valid"], tr["gain"], tr["veil"],
                               solid=(lambda i: occ.frame(i)) if occ else None,
                               glass=(lambda i: Union(glass).frame(i)) if glass else None)
            print(f"  {sid}: clean plate {plate.size[0]}x{plate.size[1]} in {time.time() - t1:.0f} s")
        layers.append((slot, tr, art, blur, grain, plate))
        if slot.get("glass_prompts"):
            glass_masks = Union([union_store(clip, p, frames) for p in slot["glass_prompts"]])
    mid, body = manifest_id(clip, assignment, inventory)

    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{frames.w}x{frames.h}", "-r", str(frames.fps), "-i", "-", "-c:v", "libx264",
                            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                           stdin=subprocess.PIPE)
    t0, batch = time.time(), []

    def flush(batch, start_frame):
        placed = []
        for k, f in enumerate(batch):
            i, g = start_frame + k, f
            plates = []
            for slot, tr, art, blur, grain, plate in layers:
                if not tr["valid"][i]:
                    continue
                a = art
                if "veil" in tr and (abs(tr["gain"][i] - 1) > 0.01 or abs(tr["veil"][i]) > 0.5):
                    a = art.copy()  # follow the camera's exposure and any veil in front of the surface
                    a[..., :3] = np.clip(art[..., :3].astype(np.float32) * tr["gain"][i] + tr["veil"][i],
                                         0, 255).astype(art.dtype)
                q = inset(tr["quads"][i], float(slot.get("inset_px", 2)))
                g = composite.composite_frame(
                    g, a, q, emissive=slot.get("dynamic_content", False), match_energy=False,
                    blur_sigma=blur, grain_sigma=grain,
                    motion_quads=[inset(m, float(slot.get("inset_px", 2)))
                                  for m in shutter_quads(tr["quads"], i, shutter=float(tr["shutter"]))])
                if slot.get("shading"):
                    g = carry_shading(g, f, tr["quads"][i], occ.frame(i) if occ else None)
                if plate is not None:  # reflections, glass and the bare surface behind occluder edges
                    add, now = plate.layers(i)
                    A, cover = plate.to_frame(add, i, g.shape)
                    inside = cover & quad_mask(q, frames.h, frames.w)
                    g = g.astype(np.float32)
                    g[inside] += A[inside]
                    g = np.clip(g, 0, 255).astype(np.uint8)
                    P, pc = plate.to_frame(now, i, g.shape)
                    plates.append((P, pc & inside))
                    if glass_masks is not None:  # clear glass shows the ad (softened); cheese and cap stay
                        gm = glass_masks.frame(i) & pc & inside
                        if gm.any():
                            diff = np.abs(f.astype(np.float32) - P).max(-1)
                            al = cv2.GaussianBlur(np.clip((diff - 8.0) / 45.0, 0, 1), (0, 0), 0.8)[..., None]
                            soft = cv2.GaussianBlur(g, (0, 0), 2.0).astype(np.float32)
                            mix = al * f.astype(np.float32) + (1 - al) * soft
                            g = g.copy()
                            g[gm] = np.clip(mix[gm], 0, 255).astype(np.uint8)
            placed.append((g, plates))
        if occ:
            from pipeline.matte import refine_occluder
            done = []
            for k, (f, (p, plates)) in enumerate(zip(batch, placed)):
                i, m = start_frame + k, occ.frame(start_frame + k)
                for slot, tr, *_ in layers:
                    if slot.get("dynamic_content") and tr["valid"][i]:
                        m = drop_on_screen(m, quad_mask(tr["quads"][i], frames.h, frames.w))
                if not m.any():
                    done.append(p)
                    continue
                al = refine_occluder(f, m, radius=4, gain=2.0)[..., None]
                o = al * f + (1 - al) * p
                # Along the soft edge the original mixes person and OLD surface; where the bare surface is known,
                # swap exactly that old part for the ad instead of letting its colour fringe the hand.
                band = ((al[..., 0] > 0.02) & (al[..., 0] < 0.98))
                for P, known in plates:
                    e = band & known
                    swap = f[e] + (1 - al[e]) * (p[e].astype(np.float32) - P[e])
                    # A sharpened matte under-counts the hand in its edge pixels, and subtracting the light old
                    # surface then drew a dark rim; an edge pixel is a blend, so keep it between its two sources.
                    o[e] = np.clip(swap, np.minimum(f[e], p[e]), np.maximum(f[e], p[e]))
                done.append(np.clip(np.rint(o), 0, 255).astype(np.uint8))
            placed = done
        else:
            placed = [p for p, _ in placed]
        if label:
            placed = [add_label(p) for p in placed]
        for k, p in enumerate(watermark.embed(placed, mid)):
            if start_frame + k >= start:  # tracks use the whole clip; the cut starts later
                enc.stdin.write(p.tobytes())

    for i, f in enumerate(frames):
        batch.append(f)
        if len(batch) == chunk:
            flush(batch, i + 1 - chunk)
            batch = []
            print(f"  {i + 1}/{frames.n} frames, {time.time() - t0:.0f} s", flush=True)
    if batch:
        flush(batch, frames.n - len(batch))
    enc.stdin.close()
    if enc.wait():
        raise SystemExit(f"ffmpeg failed writing {out}")

    exported = Frames(out)
    picks = [i for s in (0, exported.n // 2 - 24, exported.n - 48) for i in range(max(s, 0), min(s + 48, exported.n))]
    result = watermark.verify([exported[i] for i in picks[::2]], [mid])
    report = {"clip": clip, "assignment": assignment, "manifest_id": mid, "manifest": body, "output": str(out),
              "frames": frames.n - start, "start_frame": start, "label": label, "render_s": round(time.time() - t0, 1), "verify": result,
              "verified": result["campaign"] == mid,
              "tracks": {slot["id"]: {"valid": float(tr["valid"].mean()), "shutter": float(tr["shutter"])}
                         for slot, tr, *_ in layers}}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def cmd_render(args):
    entry = json.loads((ROOT / "configs" / "clips.json").read_text())[args.clip]
    assignment = dict(a.split("=", 1) for a in args.assign)
    out = Path(args.out) if args.out else ROOT / "outputs" / f"{Path(args.clip).stem}_{args.tag}.mp4"
    report = render(args.clip, entry, assignment, out, label=not args.no_label, retrack=args.retrack,
                    start=args.start)
    print(json.dumps({k: report[k] for k in ("output", "assignment", "render_s", "verified", "tracks")}, indent=2),
          f"\nbit accuracy {report['verify']['bit_accuracy']:.3f}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("objects")
    o.add_argument("clip")
    o.add_argument("prompt")
    r = sub.add_parser("render")
    r.add_argument("clip")
    r.add_argument("--assign", nargs="+", required=True, help="slot_id=brand_id ...")
    r.add_argument("--tag", default="custom")
    r.add_argument("--out")
    r.add_argument("--retrack", action="store_true")
    r.add_argument("--start", type=int, default=0, help="first output frame (tracking still uses the whole clip)")
    r.add_argument("--no-label", action="store_true", help="only for blinded realism rating, never distribution")
    args = p.parse_args()
    {"objects": cmd_objects, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
