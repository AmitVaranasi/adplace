"""3D products standing on a tracked table, rendered into the footage (a can, a bottle, a folded table-tent card),
and prints on real objects already on the table (a logo on a real glass).

Geometry, all from the clip itself (no depth model):
  - table plane tracked through the shot (planar_track, extended back by optical flow; H[i] maps frame i to the
    reference frame);
  - table normal + focal length from the plate rims SAM 3.1 segmented (a circle on a plane images as an ellipse;
    the right normal makes every rim round again) -> scripts/table_calib.py, values below;
  - metric scale from the big plate (dinner plate, 27 cm);
  - per-object camera pose: rotation from table-wide points (smoothed), translation from points around the
    object's foot, so the foot stays locked to the table.
Objects are ray-traced: surfaces of revolution (can, bottle, a real glass) by ray-marching their profile, the tent
card as two planes. Lit like the footage (soft daylight from above-right, one environment for the whole shot, a sky
dome for reflections), graded to the footage, with a contact line and a soft ray-traced cast shadow. People only
cover an object when they reach the table nearer the camera than its foot.

Each object's layer(i, frame, brand) -> box, colour, coverage, table shadow is what render_dynamic (download) and
export_player (live) both use, so the server and the browser show the same thing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.maskstore import MaskStore  # noqa: E402
from scripts.render_dynamic import Frames, union_store  # noqa: E402

CLIP = "assets/clips/patio_waiter.mp4"
REF = 156
F = 1300.0
N_UP = np.array([0.043, -0.950, -0.309])        # table "up" in the reference camera (plate-rim fit at F)
PLATE_D = 0.27                                  # big plate diameter, m
CAN_R, CAN_H = 0.033, 0.122                     # 330 ml can, m
SCENE = ROOT / "cache" / "patio_waiter-scene"


def K():
    return np.array([[F, 0, 640.], [0, F, 360.], [0, 0, 1.]])


def ray(px):
    p = np.c_[np.asarray(px, float).reshape(-1, 2), np.ones(len(np.asarray(px).reshape(-1, 2)))]
    return (np.linalg.inv(K()) @ p.T).T


def on_plane(px, h, n=N_UP):
    """3D points (reference camera) where pixels' rays meet the table n.X = -h."""
    r = ray(px)
    return r * (-h / (r @ n))[:, None]


def table_scale():
    """Camera height above the table, in metres, from the big plate's rim (far half; the near half shows its side)."""
    m = MaskStore(str(SCENE / "obj_plate_2.npz"))[REF].astype(np.uint8)
    c = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea).reshape(-1, 2)
    (_, ey), _, _ = cv2.fitEllipse(c.astype(np.float32))
    c = c[c[:, 1] < ey]
    n = N_UP / np.linalg.norm(N_UP)
    P = on_plane(c, 1.0, n)
    u = np.cross(n, [1, 0, 0]); u /= np.linalg.norm(u); v = np.cross(n, u)
    xy = np.c_[P @ u, P @ v]
    A = np.c_[2 * xy, np.ones(len(xy))]       # algebraic circle fit
    sol = np.linalg.lstsq(A, (xy ** 2).sum(1), rcond=None)[0]
    radius = np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2)
    return PLATE_D / (2 * radius)


def poses(H, h, valid, at=(705, 525), smooth=5.0, n=N_UP, wide=None):
    """R, t (reference camera -> frame i camera) per valid frame, from table points seen through the plane track.

    Rotation: table-wide points (well conditioned), smoothed over time, so objects don't wobble.
    Translation: least squares on points ringing the object's foot with that rotation fixed; the plane track isn't
    perfectly rigid, and a table-wide fit let the foot slide up to 7 px."""
    if wide is None:
        wide = np.array([[x, y] for x in (250, 500, 750, 1000) for y in (470, 560, 650)], float)
    near = np.array([[at[0] + dx, at[1] + dy] for dx in (-160, 0, 160) for dy in (-60, 0, 60)], float)
    Xw, Xn = on_plane(wide, h, n), on_plane(near, h, n)
    see = lambda pts, i: cv2.perspectiveTransform(pts.reshape(-1, 1, 2), np.linalg.inv(H[i])).reshape(-1, 2)
    rv = np.array([cv2.solvePnP(Xw, see(wide, i), K(), None)[1].ravel() for i in valid])
    k = np.exp(-0.5 * (np.arange(-6, 7) / smooth) ** 2)
    pad = np.pad(rv, ((6, 6), (0, 0)), mode="edge")
    rv = np.stack([np.convolve(pad[:, c], k / k.sum(), mode="valid") for c in range(3)], 1)
    Kinv, out = np.linalg.inv(K()), {}
    cross = lambda v: np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    for i, r in zip(valid, rv):
        R = cv2.Rodrigues(r)[0]
        rays = (Kinv @ np.c_[see(near, i), np.ones(len(near))].T).T
        # ray x (R X + t) = 0  ->  [ray]_x t = -[ray]_x R X, stacked over the points
        A = np.vstack([cross(q) for q in rays])
        b = np.concatenate([-cross(q) @ (R @ X) for q, X in zip(rays, Xn)])
        out[i] = (R, np.linalg.lstsq(A, b, rcond=None)[0])
    return out


def srgb_to_lin(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055) * 255


def scene_env(frame):
    lin = srgb_to_lin(frame.astype(np.float32))
    h = frame.shape[0]
    return {"sky": lin[: h // 4].reshape(-1, 3).mean(0) * 1.15, "table": lin[int(h * 0.7):].reshape(-1, 3).mean(0),
            "mean": lin.reshape(-1, 3).mean(0) * 1.25,
            "bright": np.percentile(lin.reshape(-1, 3), 96, axis=0)}  # overcast sky / the bright background


def shot_env(frames, idx):
    """One lighting environment for the whole shot (frame-by-frame light made the can pulse whenever the waiter's
    dark shirt crossed the frame; the real light doesn't change)."""
    envs = [scene_env(frames[i][..., ::-1]) for i in idx]
    return {k: np.mean([e[k] for e in envs], 0) for k in envs[0]}


def extend_back(frames, H, occ, min_inliers=25):
    """Carry the table track back from its first frame while the table stays trackable. planar_track drops the table
    once its big reference region is mostly off-screen, but the can's spot is in view from about frame 88, and the can
    popped in at 111. Consecutive frames are linked by pyramidal optical flow on the lower part of the image (people
    masked out), forward-backward checked to 0.3 px. SIFT matching (2 px) jittered 3.4 px median here; this is 1.5 px
    and aligns the table better at every frame checked."""
    lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
    gray = lambda i: cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

    def allowed(i):
        m = np.zeros((frames.h, frames.w), np.uint8)
        m[int(frames.h * 0.45):] = 255  # the table is the lower part of the frame in this shot
        o = occ.frame(i) if occ else None
        if o is not None:
            m[cv2.dilate(o.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0] = 0
        return m

    H = H.copy()
    first = next(i for i in range(len(H)) if np.isfinite(H[i]).all())
    g1 = gray(first)
    for i in range(first - 1, -1, -1):
        g0 = gray(i)
        p1 = cv2.goodFeaturesToTrack(g1, 1500, 0.005, 7, mask=allowed(i + 1))
        if p1 is None or len(p1) < min_inliers:
            break
        p0, st, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **lk)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **lk)
        q = p0.reshape(-1, 2)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((pb - p1).reshape(-1, 2), axis=1) < 0.3)
        ok &= (q[:, 0] >= 0) & (q[:, 0] < frames.w) & (q[:, 1] >= 0) & (q[:, 1] < frames.h)
        a0 = allowed(i)
        ok[ok] &= a0[q[ok, 1].astype(int), q[ok, 0].astype(int)] > 0
        if ok.sum() < min_inliers:
            break
        G, inl = cv2.findHomography(p0[ok], p1[ok], cv2.USAC_MAGSAC, 1.0)  # frame i -> frame i+1
        if G is None or inl.sum() < min_inliers:
            break
        H[i] = H[i + 1] @ G
        g1 = g0
    return H


def table_track(clip, frames):
    path = ROOT / "cache" / f"{Path(clip).stem}-scene" / "table_track_v3.npz"
    if not path.exists():
        from pipeline.planar_track import track_plane
        from scripts.render_dynamic import occluders
        from pipeline.clipcfg import entry as clip_entry
        entry = clip_entry(clip)
        occ = occluders(clip, entry, frames)
        tr = track_plane(frames, REF, np.float32([[150, 470], [1140, 365], [1125, 700], [150, 700]]),
                         occluder_masks=lambda i: occ.frame(i))
        H0 = np.asarray(tr["H"], float)
        H = extend_back(frames, H0, occ)
        np.savez(path, H=H, planar_first=next(i for i in range(len(H0)) if np.isfinite(H0[i]).all()))
    z = np.load(path)
    return z["H"], int(z["planar_first"])


N_FLOOR = np.array([-0.019, -0.990, -0.141])   # floor "up" in frame 8's camera (tree-pot rim, frames 4/8/12 agree)
FLOOR_REF = 8
POT_D = 0.60                                     # the tree's planter rim, m


def floor_scale():
    """Camera height above the floor, from the tree pot's rim (far half), as table_scale does with the plate."""
    m = MaskStore(str(SCENE / "obj_plant pot_1.npz"))[FLOOR_REF].astype(np.uint8)
    c = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea).reshape(-1, 2)
    c = c[(c[:, 0] > 3) & (c[:, 0] < 1276) & (c[:, 1] > 3) & (c[:, 1] < 716)]
    (_, ey), _, _ = cv2.fitEllipse(c.astype(np.float32))
    c = c[c[:, 1] < ey]
    n = N_FLOOR / np.linalg.norm(N_FLOOR)
    P = on_plane(c, 1.0, n)
    u = np.cross(n, [1, 0, 0]); u /= np.linalg.norm(u); v = np.cross(n, u)
    xy = np.c_[P @ u, P @ v]
    A = np.c_[2 * xy, np.ones(len(xy))]
    sol = np.linalg.lstsq(A, (xy ** 2).sum(1), rcond=None)[0]
    radius = np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2)
    # the rim stands ~0.45 m above the floor: scale the rim's plane distance to the floor's
    return POT_D / (2 * radius) + 0.45


def floor_track(clip, frames, last=90):
    """Floor homographies H[i] (frame i -> FLOOR_REF) by optical flow on the floor (lower part of the picture),
    people, pots and the tree masked out, chained out from the reference frame both ways."""
    path = ROOT / "cache" / f"{Path(clip).stem}-scene" / "floor_track.npz"
    if not path.exists():
        lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
        block = [union_store(clip, p, frames) for p in ("person", "plant pot", "tree", "pole")]

        def allowed(i):
            m = np.zeros((frames.h, frames.w), np.uint8)
            m[int(frames.h * 0.66):] = 255
            for st in block:
                if st.any(i):
                    m[cv2.dilate(st[i].astype(np.uint8), np.ones((21, 21), np.uint8)) > 0] = 0
            return m

        def link(a, b):  # homography frame b -> frame a
            ga, gb = (cv2.cvtColor(frames[k], cv2.COLOR_BGR2GRAY) for k in (a, b))
            pa = cv2.goodFeaturesToTrack(ga, 1500, 0.005, 7, mask=allowed(a))
            if pa is None or len(pa) < 25:
                return None
            pb, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pa, None, **lk)
            pr, st2, _ = cv2.calcOpticalFlowPyrLK(gb, ga, pb, None, **lk)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((pr - pa).reshape(-1, 2), axis=1) < 0.3)
            if ok.sum() < 25:
                return None
            G, inl = cv2.findHomography(pb[ok], pa[ok], cv2.USAC_MAGSAC, 1.0)
            return G if G is not None and inl.sum() >= 20 else None

        H = np.full((frames.n, 3, 3), np.nan)
        H[FLOOR_REF] = np.eye(3)
        for step in (1, -1):
            i = FLOOR_REF
            while 0 <= i + step < min(frames.n, last):
                G = link(i, i + step)
                if G is None:
                    break
                H[i + step] = H[i] @ G
                i += step
        np.savez(path, H=H)
    return np.load(path)["H"]


PLANES = {
    "table": {"ref": REF, "n": N_UP, "occluders": ("person",), "grounded": True,
              "wide": [[x, y] for x in (250, 500, 750, 1000) for y in (470, 560, 650)]},
    "floor": {"ref": FLOOR_REF, "n": N_FLOOR, "occluders": ("person", "pole", "tree", "plant"), "grounded": False,
              "wide": [[x, y] for x in (350, 650, 950, 1200) for y in (520, 610, 700)]},
}


def plane_setup(clip, frames, name):
    """Camera height, track (H[i]: frame i -> the plane's reference frame) and the first frame the main track
    covers, for the table or the floor."""
    if name == "floor":
        return floor_scale(), floor_track(clip, frames), 0
    H, planar_first = table_track(clip, frames)
    return table_scale(), H, planar_first


def label_band(art, front, aspect=2.0, size=1024):
    """A wrap-around label (circumference : printed height = aspect) from flat artwork: only about a third of a
    cylinder faces the camera, so the artwork fills the front `front` of the width and its background colour wraps
    around the back, as on a real can or bottle."""
    border = np.r_[art[:4].reshape(-1, 3), art[-4:].reshape(-1, 3), art[:, :4].reshape(-1, 3), art[:, -4:].reshape(-1, 3)]
    h = int(size / aspect)
    aw = int(round(size * front))
    small = cv2.resize(art, (aw, int(round(aw * art.shape[0] / art.shape[1]))), interpolation=cv2.INTER_AREA)
    if small.shape[0] > h:
        small = cv2.resize(art, (int(round(h * art.shape[1] / art.shape[0])), h), interpolation=cv2.INTER_AREA)
    out = np.empty((h, size, 3), np.uint8)
    out[:] = np.median(border, 0)
    y0, x0 = (h - small.shape[0]) // 2, (size - small.shape[1]) // 2
    out[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return out


def creative(brand, fmt):
    return cv2.imread(str(ROOT / "assets/creatives/gen" / f"{brand}_{fmt}.png"))[..., ::-1]


class TableObject:
    """Something standing on the tracked table at pixel `at` of the reference frame. Subclasses give the shape:
    bounds, intersect(rays) -> surface hits, materials(hits, brand), and the shadow it casts."""

    shadow = True
    sigma = 0.6     # the footage is a touch soft
    grain = 1.6

    def __init__(self, clip, spec):
        self.clip, self.spec, self.id = clip, spec, spec["id"]
        self.at = tuple(spec["at"])
        self.frames = Frames(ROOT / clip)
        self.plane = PLANES[spec.get("plane", "table")]
        self.h, H, planar_first = plane_setup(clip, self.frames, spec.get("plane", "table"))
        lo, hi = spec.get("frames", (0, self.frames.n))
        valid = [i for i in range(lo, hi) if np.isfinite(H[i]).all()]
        n = self.plane["n"]
        self.up = n / np.linalg.norm(n)
        self.base = on_plane([self.at], self.h, n)[0]
        P = poses(H, self.h, valid, self.at, n=n, wide=np.array(self.plane["wide"], float))
        # going back from the planar track's first frame, stop once the object has left the picture: earlier frames
        # are chained through floor and plants, not the table, and could put it somewhere it doesn't belong
        keep = []
        for i in sorted(valid, reverse=True):
            R, t = P[i]
            top = K() @ (R @ (self.base + self.up * self.height) + t)
            if i < planar_first and top[1] / top[2] > self.frames.h + 10:
                break
            keep.append(i)
        self.valid = sorted(keep)
        self.poses = {i: P[i] for i in self.valid}
        # the front (label, card face) faces the camera at mid-shot; the camera orbits the table during the shot
        R, t = self.poses[self.valid[len(self.valid) // 2]]
        to_cam = -R.T @ t - self.base
        self.face = to_cam - (to_cam @ self.up) * self.up
        self.face /= np.linalg.norm(self.face)
        self.env = shot_env(self.frames, [i for i in self.valid if i >= 100][::8] or self.valid[::8])
        from scripts.render_dynamic import object_stores
        self.people = [st for p in self.plane["occluders"] for st in object_stores(clip, p, self.frames).values()]
        self.tex = {}
        self._geom = (None, None)

    # --- shape interface -----------------------------------------------------------------------------------------
    height = CAN_H
    footprint = CAN_R   # radius that bounds the object's foot on the table

    def intersect(self, d, A, a, e1, e2):
        raise NotImplementedError

    def materials(self, g, brand):
        raise NotImplementedError

    def blocks(self, p, L, a, e1, e2):
        """Does the object block light L for table points p (relative to the foot, in the table plane)?"""
        return np.zeros(len(p), bool)

    def contact(self, p, a, e1, e2):
        """Contact darkening of table points p next to the object's foot."""
        return np.zeros(len(p))

    def visibility(self, i, box):
        return None

    # --- shared --------------------------------------------------------------------------------------------------
    def frame_axes(self, i):
        R, t = self.poses[i]
        A = R @ self.base + t
        a = R @ self.up
        a /= np.linalg.norm(a)
        e1 = R @ self.face
        e1 -= (e1 @ a) * a
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        toward = -(A - (A @ a) * a)
        toward /= np.linalg.norm(toward)
        right = np.cross(a, toward)
        right /= np.linalg.norm(right)
        # the light that casts shadows: above and behind the table, a little from the right (the real bowls'
        # shadows fall toward the camera)
        sun = 0.85 * a - 0.45 * toward + 0.2 * right
        sun /= np.linalg.norm(sun)
        return A, a, e1, e2, toward, right, sun

    def geometry(self, i):
        """Everything brand-independent about frame i: rays, surface hits, coverage and the table shadow. Cached, so
        rendering several brands of one frame traces it once."""
        if self._geom[0] == i:
            return self._geom[1]
        A, a, e1, e2, toward, right, sun = self.frame_axes(i)
        run = sun - (sun @ a) * a
        reach = self.height * np.linalg.norm(run) / (sun @ a) if self.shadow else 0.0
        r = 1.6 * self.footprint
        ring = [A + a * z + r * (np.cos(th) * e1 + np.sin(th) * e2)
                for z in (0, self.height) for th in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        if self.shadow:
            tip = A - run / np.linalg.norm(run) * reach * 1.2
            ring += [tip + 1.2 * r * (np.cos(th) * e1 + np.sin(th) * e2) for th in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        pr = (K() @ np.array(ring).T).T
        if (pr[:, 2] <= 0).any():
            self._geom = (i, None)
            return None
        pr = pr[:, :2] / pr[:, 2:]
        x0, y0 = np.floor(pr.min(0) - 6).astype(int)
        x1, y1 = np.ceil(pr.max(0) + 6).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, self.frames.w), min(y1, self.frames.h)
        if x1 <= x0 or y1 <= y0:
            self._geom = (i, None)
            return None
        ss = 3
        us = x0 + (np.arange((x1 - x0) * ss) + 0.5) / ss - 0.5
        vs = y0 + (np.arange((y1 - y0) * ss) + 0.5) / ss - 0.5
        U, V = np.meshgrid(us, vs)
        d = (np.linalg.inv(K()) @ np.stack([U.ravel(), V.ravel(), np.ones(U.size)])).T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        g = self.intersect(d, A, a, e1, e2)
        g.update(d=d, A=A, a=a, e1=e1, e2=e2, toward=toward, right=right, ss=ss, shape=U.shape,
                 box=(x0, y0, x1, y1))
        hh, ww = y1 - y0, x1 - x0
        g["cov"] = g["hit"].reshape(hh, ss, ww, ss).mean((1, 3))
        g["occl"] = self.table_shadow(x0, y0, x1, y1, A, a, e1, e2, toward, sun, reach) if self.shadow \
            else np.zeros((hh, ww))
        self._geom = (i, g)
        return g

    def table_shadow(self, x0, y0, x1, y1, A, a, e1, e2, toward, sun, reach):
        """Contact line plus a soft ray-traced cast shadow, at pixel resolution. Fixed light samples: the shadow
        must not shimmer frame to frame."""
        U, V = np.meshgrid(np.arange(x0, x1, dtype=float), np.arange(y0, y1, dtype=float))
        d = (np.linalg.inv(K()) @ np.stack([U.ravel(), V.ravel(), np.ones(U.size)])).T
        st = (A @ a) / (d @ a)                          # ray parameter to the table plane through the foot
        Pt = d * st[:, None] - A
        p = Pt - np.outer(Pt @ a, a)
        occl = self.contact(p, a, e1, e2)
        near = np.linalg.norm(p, axis=1) < reach * 1.4 + 2 * self.footprint
        rng = np.random.default_rng(7)
        blocked = np.zeros(len(p))
        for _ in range(32):
            L = sun + 0.12 * rng.normal(size=3)
            L /= np.linalg.norm(L)
            blocked[near] += self.blocks(p[near], L, a, e1, e2)
        dist = np.linalg.norm(p, axis=1) / max(reach, 1e-6)
        cast = 0.65 * blocked / 32 * np.exp(-np.maximum(dist - 0.25, 0) / 0.6)  # darkest by the object, fading out
        occl = 1 - (1 - np.clip(occl, 0, 0.92)) * (1 - cast)
        return np.where(st > 0, occl, 0).reshape(U.shape)

    def shade(self, g, brand):
        """Linear premultiplied colour of the object's pixels for one brand."""
        d, a, toward, right, hit = g["d"], g["a"], g["toward"], g["right"], g["hit"]
        nrm = g["nrm"]
        albedo, metal, gloss, alpha = self.materials(g, brand)
        sky, table_col, mean_col = self.env["sky"], self.env["table"], self.env["mean"]
        # Light as the footage has it: soft daylight from above and camera-right with open shadows. (A camera-side
        # key made the front hot and both sides near-black, a studio look; fully even light made it a flat card.)
        key = 0.65 * a + 0.6 * right - 0.25 * toward
        key /= np.linalg.norm(key)
        k = np.clip(nrm @ a, -1, 1)[:, None]
        amb = np.where(k >= 0, mean_col * (1 - k) + sky * k, mean_col * (1 + k) + table_col * (-k))
        wrap = np.clip((nrm @ key + 0.6) / 1.6, 0, 1)
        irr = amb * (0.45 + 0.9 * wrap ** 1.3)[:, None]
        low = np.clip(1 - g["y"] / 0.03, 0, 1) * hit  # warm bounce from the table onto the object's foot
        irr = irr + table_col * (0.25 * low)[:, None]
        # reflections from a simple sky dome: bright overcast above the horizon, the warm table below
        refl = d - 2 * (d * nrm).sum(1, keepdims=True) * nrm
        e = (refl @ a)[:, None]
        above = np.clip(e / 0.5, 0, 1) ** 0.7
        below = np.clip(-e / 0.3, 0, 1)
        spec = np.where(e >= 0, mean_col * (1 - above) + self.env["bright"] * above,
                        mean_col * (1 - below) + table_col * below)
        fres = 0.04 + 0.96 * (1 - np.clip(-(nrm * d).sum(1), 0, 1)) ** 5
        kspec = (metal + (1 - metal) * fres * gloss)[:, None]
        tint = np.where((metal > 0.3)[:, None], albedo / np.maximum(albedo.max(1, keepdims=True), 1e-4) * np.sqrt(albedo.max(1, keepdims=True)), 1.0)  # metals and tinted glass colour their reflections
        col = albedo * irr * (1 - kspec) + spec * kspec * tint
        # the footage's grade: muted chroma, milky blacks (the can was the most saturated thing in the shot)
        lum = (col @ np.array([0.2126, 0.7152, 0.0722]))[:, None]
        col = (lum + (col - lum) * 0.8) * 0.96 + 0.004
        w = (hit * alpha)[:, None]
        return col * w, w[:, 0]

    def texture(self, img, u, v):
        """Bilinear lookup of an RGB(A) image at u, v in [0, 1] (u wraps)."""
        h, w = img.shape[:2]
        U = (u % 1.0) * (w - 1)
        V = np.clip(v, 0, 1) * (h - 1)
        n = len(u)
        side = int(np.ceil(np.sqrt(max(n, 1))))
        pad = side * side - n
        mu = np.pad(U, (0, pad)).astype(np.float32).reshape(side, side)
        mv = np.pad(V, (0, pad)).astype(np.float32).reshape(side, side)
        out = cv2.remap(img.astype(np.float32), mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return out.reshape(side * side, -1)[:n]

    def front(self, i):
        """The people in front of the object this frame. A person is in front only if they reach the table nearer
        the camera than the object's foot (judged within the object's columns, per person): the seated guest in the
        foreground is, the waiter leaning over the pizza behind the can is not."""
        R, t = self.poses[i]
        foot = K() @ (R @ self.base + t)
        foot_y = foot[1] / foot[2]
        ring = np.array([self.base + self.footprint * 1.5 * np.array([np.cos(th), 0, np.sin(th)])
                         for th in np.linspace(0, 6.3, 12)])
        pr = (K() @ (R @ ring.T + t[:, None])).T
        c0 = int(max(pr[:, 0].min() / pr[:, 2].max() - 10, 0))
        c1 = int(min(pr[:, 0].max() / pr[:, 2].min() + 10, self.frames.w))
        out = None
        h = self.frames.h
        for st in self.people:
            if not st.any(i):
                continue
            m = st[i]
            if not m[:, c0:c1].any():
                continue
            # on the floor everything stands on the same ground, so an occluder's lowest point anywhere is where it
            # stands (a branch reaching in from a foreground plant); on the table, judge within the object's columns
            rows = np.flatnonzero((m if not self.plane["grounded"] else m[:, c0:c1]).any(1))
            # nearer than the foot, and standing in the foreground: their outline runs off the bottom of the picture
            # near the object. The waiter behind the table never does (the table hides his legs), even when his hand
            # hangs lower than a card's foot.
            grounded = m[h - 3:, max(c0 - 200, 0):c1 + 200].any() or not self.plane["grounded"]
            # (when the camera has tilted up and the foot is below the picture, someone running off its bottom edge
            # in front of the object is the nearer one: the waiter walking toward the camera past the A-frame)
            if len(rows) and rows.max() > min(foot_y + 4, h - 4) and grounded:
                out = m if out is None else out | m
        if out is not None:
            # whatever covers a person who is in front of the object is in front of it too (a branch across the
            # waiter's shirt): occluder pixels inside a front person's outline join the front mask
            hull = np.zeros_like(out, np.uint8)
            cs, _ = cv2.findContours(out.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                if cv2.contourArea(c) > 500:
                    cv2.fillConvexPoly(hull, cv2.convexHull(c), 1)
            for st in self.people:
                if st.any(i):
                    out = out | (st[i] & (hull > 0))
            # close the thin seams between neighbouring masks (a leaf against the waiter's shirt): pixels that belong
            # to neither let the object show through as dark slivers
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
        return out

    def layer(self, i, frame, brand):
        """What the object adds to frame i, in sRGB: box, colour (unpremultiplied BGR), alpha, table shadow
        multiplier. Server render and player export both use this, so they show the same object."""
        if i not in self.poses:
            return None
        g = self.geometry(i)
        if g is None:
            return None
        x0, y0, x1, y1 = g["box"]
        hh, ww, ss = y1 - y0, x1 - x0, g["ss"]
        col, w = self.shade(g, brand)
        rgb = col.reshape(hh, ss, ww, ss, 3).mean((1, 3))
        cov = w.reshape(hh, ss, ww, ss).mean((1, 3))
        vis = self.visibility(i, g["box"])
        if vis is not None:
            rgb, cov = rgb * vis[..., None], cov * vis
        occl = g["occl"].astype(np.float32)
        colour = lin_to_srgb(rgb / np.maximum(cov, 1e-6)[..., None])[..., ::-1] * cov[..., None]
        if self.sigma > 0:  # blur premultiplied colour, coverage and shadow together
            colour, cov, occl = (cv2.GaussianBlur(x.astype(np.float32), (0, 0), self.sigma) for x in (colour, cov, occl))
        colour = colour / np.maximum(cov, 1e-6)[..., None]
        shadow = (1 - occl) ** (1 / 2.2)  # linear-light occlusion as an sRGB multiplier
        return g["box"], np.clip(colour, 0, 255), np.clip(cov, 0, 1), shadow

    def apply(self, original, composited, i, brand):
        """Put the object into an already composited frame; people in front of it (from the original) stay in front."""
        L = self.layer(i, original, brand)
        if L is None:
            return composited
        (x0, y0, x1, y1), colour, cov, shadow = L
        out = composited.astype(np.float32)
        reg, a = out[y0:y1, x0:x1], cov[..., None]
        noise = np.random.default_rng(i).normal(0, self.grain, colour.shape[:2])[..., None]  # luma grain
        out[y0:y1, x0:x1] = reg * shadow[..., None] * (1 - a) + (colour + noise) * a
        front = self.front(i)
        if front is not None:
            from pipeline.matte import refine_occluder
            k = refine_occluder(original, front, radius=4, gain=2.0)[..., None]
            out = k * original + (1 - k) * out
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)


class Lathe(TableObject):
    """A surface of revolution standing on the table: profile (height, radius) pairs from the foot up, closed by a
    flat top. Rays are marched to the first crossing of the profile, then refined by bisection."""

    profile = np.array([[0.0, CAN_R], [CAN_H, CAN_R]])

    @property
    def height(self):
        return float(self.profile[-1, 0])

    @property
    def footprint(self):
        return float(self.profile[:, 1].max())

    def radius(self, h):
        return np.interp(h, self.profile[:, 0], self.profile[:, 1], left=-1.0, right=-1.0)

    def slope(self, h):
        hs, rs = self.profile[:, 0], self.profile[:, 1]
        dr = np.diff(rs) / np.maximum(np.diff(hs), 1e-9)
        k = np.clip(np.searchsorted(hs, h, side="right") - 1, 0, len(dr) - 1)
        return dr[k]

    def intersect(self, d, A, a, e1, e2):
        n = len(d)
        w = -A
        da, wa = d @ a, w @ a
        dp, wp = d - np.outer(da, a), w - wa * a
        rmax, top = self.footprint, self.height
        # interval inside the bounding cylinder and between the foot and the top
        qa, qb, qc = (dp * dp).sum(1), 2 * (dp @ wp), wp @ wp - rmax ** 2
        disc = qb ** 2 - 4 * qa * qc
        sq = np.sqrt(np.maximum(disc, 0))
        s0, s1 = (-qb - sq) / (2 * qa + 1e-12), (-qb + sq) / (2 * qa + 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            ha, hb = (0 - wa) / da, (top - wa) / da
        lo_h, hi_h = np.minimum(ha, hb), np.maximum(ha, hb)
        s0, s1 = np.maximum(s0, lo_h), np.minimum(s1, hi_h)
        cand = np.flatnonzero((disc > 0) & (s1 > s0) & (s1 > 0))
        hit = np.zeros(n, bool)
        s_hit = np.full(n, np.inf)
        if len(cand):
            c0, c1 = np.maximum(s0[cand], 0), s1[cand]
            steps = 48
            ts = np.linspace(0, 1, steps)
            S = c0[:, None] + (c1 - c0)[:, None] * ts[None, :]
            h = wa + S * da[cand, None]
            rho = np.linalg.norm(wp[None, None, :] + S[..., None] * dp[cand, None, :], axis=2)
            f = rho - self.radius(h)
            inside = (f < 0) & (self.radius(h) > 0)
            first = np.where(inside.any(1), inside.argmax(1), -1)
            got = first > 0
            idx = cand[got]
            lo = S[got, first[got] - 1]
            hi = S[got, first[got]]
            ci = idx
            for _ in range(12):
                mid = (lo + hi) / 2
                hm = wa + mid * da[ci]
                rm = np.linalg.norm(wp[None, :] + mid[:, None] * dp[ci], axis=1)
                ins = (rm - self.radius(hm) < 0) & (self.radius(hm) > 0)
                hi = np.where(ins, mid, hi)
                lo = np.where(ins, lo, mid)
            s_hit[ci] = hi
            hit[ci] = True
            # rays already inside at the start of the interval: they enter through the flat top
            start_in = cand[first == 0]
            s_hit[start_in] = np.maximum(s0[start_in], 0)
            hit[start_in] = True
        P = d * np.where(hit, s_hit, 0)[:, None] - A
        y = P @ a
        radial = P - np.outer(y, a)
        rr = np.linalg.norm(radial, axis=1)
        cap = hit & (y > top - 1e-4)
        nr = radial / np.maximum(rr, 1e-9)[:, None]
        side_n = nr - self.slope(y)[:, None] * a[None, :]
        side_n /= np.linalg.norm(side_n, axis=1, keepdims=True)
        nrm = np.where(cap[:, None], a, side_n)
        ang = np.arctan2(radial @ e2, radial @ e1)
        return {"hit": hit, "cap": cap, "y": y, "rr": rr, "nrm": nrm, "u": ang / (2 * np.pi) + 0.5}

    def blocks(self, p, L, a, e1, e2):
        u = L - (L @ a) * a
        out = np.zeros(len(p), bool)
        for h in np.linspace(0.002, self.height, 14):
            q = p + (h / (L @ a)) * u
            out |= np.linalg.norm(q, axis=1) < self.radius(h)
        return out

    def contact(self, p, a, e1, e2):
        r0 = self.profile[0, 1]
        rt = np.linalg.norm(p, axis=1)
        gap = np.maximum(rt - r0, 0) / r0
        toward = np.zeros(len(p))
        return 0.85 * np.exp(-gap / 0.08) + 0.35 * np.exp(-gap / 0.5) * 0.75 + toward


class Can(Lathe):
    """330 ml can: printed band between a silver neck and base, bright silver lid with a rolled rim."""

    profile = np.array([[0.0, CAN_R * 0.93], [0.006, CAN_R], [CAN_H - 0.011, CAN_R], [CAN_H, CAN_R * 0.88]])
    band = (0.006, CAN_H - 0.011)

    def label(self, brand):
        if brand not in self.tex:
            self.tex[brand] = label_band(creative(brand, "can_wrap"), 0.32)
        return self.tex[brand]

    def materials(self, g, brand):
        y, hit, cap = g["y"], g["hit"], g["cap"]
        v = 1 - (y - self.band[0]) / (self.band[1] - self.band[0])
        in_band = hit & ~cap & (v > 0) & (v < 1)
        tex = self.texture(self.label(brand), g["u"], v)
        silver = np.array([196, 199, 204], np.float32)
        albedo = srgb_to_lin(np.where(in_band[:, None], tex, silver))
        rr = g["rr"] / (CAN_R * 0.88)  # lid: rolled rim, a groove inside it, the recessed panel a touch darker
        lid = np.where(rr > 0.9, 1.25, np.where(rr > 0.82, 0.55, 0.85))
        albedo = np.where(cap[:, None], albedo * lid[:, None], albedo)
        metal = np.where(in_band, 0.05, 0.55)
        return albedo, metal, np.full(len(y), 0.5), np.ones(len(y))


class Bottle(Lathe):
    """A dark amber glass bottle (cold brew, cola, sparkling water) with a paper label and a metal crown cap."""

    profile = np.array([[0.0, 0.029], [0.004, 0.031], [0.120, 0.031], [0.150, 0.024], [0.172, 0.0135],
                        [0.205, 0.0125], [0.212, 0.0140], [0.222, 0.0140]])
    band = (0.030, 0.105)

    def label(self, brand):
        if brand not in self.tex:
            self.tex[brand] = label_band(creative(brand, "bottle_label"), 0.36, aspect=2.6)
        return self.tex[brand]

    def materials(self, g, brand):
        y, hit, u = g["y"], g["hit"], g["u"]
        v = 1 - (y - self.band[0]) / (self.band[1] - self.band[0])
        # the label wraps a little over half way round; the back of the bottle is bare glass
        on_label = hit & (v > 0) & (v < 1) & (np.abs(((u - 0.5 + 0.5) % 1.0) - 0.5) < 0.30)
        lu = (u - 0.5) / 0.60 + 0.5
        tex = self.texture(self.label(brand), lu, v)
        glass = np.array([62, 30, 8], np.float32)        # dark amber glass, seen against its own dark contents
        cap_col = np.array([172, 150, 104], np.float32)  # gold crown cap
        crown = hit & (y > 0.207)
        base = np.where(on_label[:, None], tex, np.where(crown[:, None], cap_col, glass))
        albedo = srgb_to_lin(base)
        # glass tints what it reflects (amber), so the bottle reads brown rather than grey
        metal = np.where(on_label, 0.02, np.where(crown, 0.5, 0.35))
        gloss = np.where(on_label, 0.3, 1.0)               # glass reflects strongly at grazing angles
        return albedo, metal, gloss, np.ones(len(y))


class TentCard(TableObject):
    """A folded paper table-tent card: two faces leaning on each other, the brand's card on the front face."""

    W, D, RIDGE = 0.10, 0.065, 0.135   # width, depth of the footprint, ridge height (m)
    height = RIDGE
    footprint = 0.065

    def faces(self, A, a, e1, e2):
        out = []
        for sgn in (1, -1):   # front face (toward the camera at mid-shot), back face
            Q0 = A + sgn * (self.D / 2) * e1
            v = self.RIDGE * a - sgn * (self.D / 2) * e1
            n = np.cross(e2, v) if sgn > 0 else np.cross(v, e2)
            n /= np.linalg.norm(n)
            if n @ e1 * sgn < 0:
                n = -n
            out.append((Q0, v, n))
        return out

    def intersect(self, d, A, a, e1, e2):
        n_rays = len(d)
        best = np.full(n_rays, np.inf)
        hit, face, xs, ss_ = np.zeros(n_rays, bool), np.zeros(n_rays, int), np.zeros(n_rays), np.zeros(n_rays)
        nrm = np.zeros((n_rays, 3))
        for k, (Q0, v, n) in enumerate(self.faces(A, a, e1, e2)):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (n @ Q0) / (d @ n)
            X = d * t[:, None] - Q0
            x = X @ e2
            s = (X @ v) / (v @ v)
            ok = (t > 0) & (np.abs(x) <= self.W / 2) & (s >= 0) & (s <= 1) & (t < best)
            best = np.where(ok, t, best)
            hit |= ok
            face = np.where(ok, k, face)
            xs, ss_ = np.where(ok, x, xs), np.where(ok, s, ss_)
            facing = np.where(((d @ n) < 0)[:, None], n, -n)   # the side the camera sees
            nrm = np.where(ok[:, None], facing, nrm)
        y = ss_ * self.RIDGE
        return {"hit": hit, "face": face, "x": xs, "s": ss_, "y": y, "nrm": nrm, "cap": np.zeros(n_rays, bool)}

    def card(self, brand):
        if brand not in self.tex:
            self.tex[brand] = creative(brand, "table_tent")
        return self.tex[brand]

    def materials(self, g, brand):
        hit, face = g["hit"], g["face"]
        u = g["x"] / self.W + 0.5
        v = 1 - g["s"]
        tex = self.texture(self.card(brand), u, v)
        paper = np.array([228, 224, 214], np.float32)
        base = np.where(((face == 0) & hit)[:, None], tex, paper)  # the back face is plain card from here
        albedo = srgb_to_lin(base)
        n = len(hit)
        return albedo, np.full(n, 0.02), np.full(n, 0.25), np.ones(n)

    def blocks(self, p, L, a, e1, e2):
        out = np.zeros(len(p), bool)
        for sgn in (1, -1):
            Q0 = sgn * (self.D / 2) * e1
            v = self.RIDGE * a - sgn * (self.D / 2) * e1
            n = np.cross(e2, v)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((Q0 - p) @ n) / (L @ n)
            X = p + t[:, None] * L - Q0
            x = X @ e2
            s = (X @ v) / (v @ v)
            out |= (t > 0) & (np.abs(x) <= self.W / 2) & (s >= 0) & (s <= 1)
        return out

    def contact(self, p, a, e1, e2):
        px, py = np.abs(p @ e2), np.abs(p @ e1)
        edge = np.hypot(np.maximum(px - self.W / 2, 0), py - self.D / 2)   # distance to the two bottom edges
        under = (px < self.W / 2) & (py < self.D / 2)
        return np.maximum(0.8 * np.exp(-edge / 0.004), 0.45 * under)


class AFrameSign(TentCard):
    """A sidewalk A-frame board standing on the patio floor, the brand's poster on its front face."""

    W, D, RIDGE = 0.60, 0.42, 0.90
    height = RIDGE
    footprint = 0.42

    def card(self, brand):
        if brand not in self.tex:
            art = creative(brand, "poster")
            framed = cv2.copyMakeBorder(art, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=(34, 30, 28))  # dark frame
            self.tex[brand] = framed
        return self.tex[brand]

    def contact(self, p, a, e1, e2):
        px, py = np.abs(p @ e2), np.abs(p @ e1)
        edge = np.hypot(np.maximum(px - self.W / 2, 0), py - self.D / 2)
        under = (px < self.W / 2) & (py < self.D / 2)
        return np.maximum(0.7 * np.exp(-edge / 0.02), 0.35 * under)


class GlassPrint(Lathe):
    """A logo printed on a real glass already on the table. The glass's shape comes from its SAM 3.1 mask in the
    reference frame (foot on the table, radius at foot and rim, height); the print is a white enamel decal on its
    front, and the real glass shows through everywhere else. Things in front of the glass hide the print with it."""

    shadow = False
    grain = 1.0

    def __init__(self, clip, spec):
        m = MaskStore(str(SCENE / f"{spec['glass']}.npz"))
        self.mask_store = m
        ys, xs = np.nonzero(m[REF])
        bottom, top_y = ys.max(), ys.min()
        rows = lambda y0, y1: xs[(ys >= y0) & (ys <= y1)]
        bx = rows(bottom - 6, bottom)
        tx = rows(top_y, top_y + 6)
        foot_px = ((bx.min() + bx.max()) / 2, bottom - 2)
        spec = {**spec, "at": foot_px}
        h = table_scale()
        base = on_plane([foot_px], h)[0]
        up = N_UP / np.linalg.norm(N_UP)
        z = base[2]
        r_foot = (bx.max() - bx.min()) / 2 * z / F
        # height: the rim's centre projects to the top of the mask
        hs = np.linspace(0.03, 0.25, 400)
        tops = np.array([(K() @ (base + up * hh))[1] / (K() @ (base + up * hh))[2] for hh in hs])
        Hg = float(hs[np.argmin(np.abs(tops - top_y))])
        r_rim = (tx.max() - tx.min()) / 2 * (base + up * Hg)[2] / F
        self.profile = np.array([[0.0, r_foot], [Hg, r_rim]])
        super().__init__(clip, spec)

    def print_art(self, brand):
        if brand not in self.tex:
            art = creative(brand, "glass_print").astype(np.float32)
            a = np.clip((art.max(2) - 40) / 160, 0, 1)   # white on black: the black is clear glass
            self.tex[brand] = np.dstack([np.full(a.shape + (3,), 240, np.float32), a * 255])
        return self.tex[brand]

    def materials(self, g, brand):
        y, hit, cap = g["y"], g["hit"], g["cap"]
        band = (0.30 * self.height, 0.72 * self.height)
        v = 1 - (y - band[0]) / (band[1] - band[0])
        span = 0.26   # the print covers about a quarter of the way round, centred on the front
        lu = (g["u"] - 0.5) / span + 0.5
        on = hit & ~cap & (v > 0) & (v < 1) & (lu > 0) & (lu < 1)
        tex = self.texture(self.print_art(brand), lu, v)
        alpha = np.where(on, tex[:, 3] / 255.0 * 0.92, 0)
        albedo = srgb_to_lin(tex[:, :3])
        n = len(y)
        return albedo, np.full(n, 0.03), np.full(n, 0.4), alpha

    def visibility(self, i, box):
        """The print is only where the real glass is visible this frame (hands and plates in front hide it)."""
        if not self.mask_store.any(i):
            return np.zeros((box[3] - box[1], box[2] - box[0]), np.float32)
        m = self.mask_store[i][box[1]:box[3], box[0]:box[2]].astype(np.uint8)
        m = cv2.erode(m, np.ones((3, 3), np.uint8))
        return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)


KINDS = {"can": Can, "bottle": Bottle, "tent": TentCard, "glass_print": GlassPrint, "aframe": AFrameSign}


def make(clip, spec):
    if spec.get("kind") == "shirt_print":
        from scripts.shirt_print import ShirtPrint
        return ShirtPrint(clip, spec)
    if spec.get("kind") == "glass_decal":
        from scripts.glass_decal import GlassDecal
        return GlassDecal(clip, spec)
    return KINDS[spec.get("kind", "can")](clip, spec)


def active(entry):
    """The clip's objects3d entries that are switched on (a disabled one says why)."""
    return [s for s in entry.get("objects3d", []) if not s.get("disabled")]


def objects_for(clip, entry, assignment=None):
    """The clip's 3D objects / prints (configs/clips.json objects3d), far to near so nearer ones draw on top."""
    objs = [make(clip, s) for s in active(entry) if assignment is None or s["id"] in assignment]
    return sorted(objs, key=lambda o: -float(getattr(o, "base", np.array([0, 0, 0]))[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=CLIP)
    ap.add_argument("--assign", nargs="+", required=True, help="object_id=brand ...")
    ap.add_argument("--out", default=str(ROOT / "outputs/place3d/objects.mp4"))
    ap.add_argument("--still", type=int, help="only write this frame as a PNG")
    args = ap.parse_args()
    from pipeline.clipcfg import entry as clip_entry
    entry = clip_entry(args.clip)
    assignment = dict(a.split("=", 1) for a in args.assign)
    objs = objects_for(args.clip, entry, assignment)
    frames = Frames(ROOT / args.clip)

    def one(i, f):
        out = f
        for o in objs:
            out = o.apply(f, out, i, assignment[o.id])
        return out

    if args.still is not None:
        cv2.imwrite(str(Path(args.out).with_name(f"objects_{args.still}.png")), one(args.still, frames[args.still]))
        return
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                            f"{frames.w}x{frames.h}", "-r", str(frames.fps), "-i", "-", "-c:v", "libx264",
                            "-crf", "16", "-pix_fmt", "yuv420p", args.out], stdin=subprocess.PIPE)
    for i, f in enumerate(frames):
        enc.stdin.write(one(i, f).tobytes())
    enc.stdin.close()
    enc.wait()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
