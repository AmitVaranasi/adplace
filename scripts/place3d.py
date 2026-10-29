"""Proof of concept: a 3D product (a soda can) standing on a tracked table, rendered into the footage.

Geometry, all from the clip itself (no depth model):
  - table plane tracked through the shot (planar_track, H[i] maps frame i to the reference frame);
  - table normal + focal length from the plate rims SAM 3.1 segmented (a circle on a plane images as an ellipse;
    the right normal makes every rim round again) -> scripts/table_calib.py, values below;
  - metric scale from the big plate (dinner plate, 27 cm);
  - per-frame camera pose: table points through H[i], then solvePnP.
The can is ray-traced analytically (cylinder + top cap), label wrapped from a 2:1 creative, lit from the frame itself
(ambient from the scene, reflections from a blurred copy of the frame), with a contact shadow on the table and the
SAM person matte kept in front. CanPlacement is used by render_dynamic (download) and export_player (live).
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
from scripts.render_dynamic import Frames, Union, union_store  # noqa: E402

CLIP = "assets/clips/patio_waiter.mp4"
REF = 156
F = 1300.0
N_UP = np.array([0.043, -0.950, -0.309])        # table "up" in the reference camera (plate-rim fit at F)
PLATE_D = 0.27                                  # big plate diameter, m
CAN_R, CAN_H = 0.033, 0.122                     # 330 ml can, m


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
    m = MaskStore(str(ROOT / "cache/patio_waiter-scene/obj_plate_2.npz"))[REF].astype(np.uint8)
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


def poses(H, h, valid, at=(705, 525), smooth=5.0):
    """R, t (reference camera -> frame i camera) per valid frame, from table points seen through the plane track.

    Rotation: table-wide points (well conditioned), smoothed over time, so the can doesn't wobble.
    Translation: least squares on points ringing the can's foot with that rotation fixed; the plane track isn't
    perfectly rigid, and a table-wide fit let the foot slide up to 7 px."""
    wide = np.array([[x, y] for x in (250, 500, 750, 1000) for y in (470, 560, 650)], float)
    near = np.array([[at[0] + dx, at[1] + dy] for dx in (-160, 0, 160) for dy in (-60, 0, 60)], float)
    Xw, Xn = on_plane(wide, h), on_plane(near, h)
    see = lambda pts, i: cv2.perspectiveTransform(pts.reshape(-1, 1, 2), np.linalg.inv(H[i])).reshape(-1, 2)
    rv = np.array([cv2.solvePnP(Xw, see(wide, i), K(), None)[1].ravel() for i in valid])
    k = np.exp(-0.5 * (np.arange(-6, 7) / smooth) ** 2)
    pad = np.pad(rv, ((6, 6), (0, 0)), mode="edge")
    rv = np.stack([np.convolve(pad[:, c], k / k.sum(), mode="valid") for c in range(3)], 1)
    Kinv, out = np.linalg.inv(K()), {}
    for i, r in zip(valid, rv):
        R = cv2.Rodrigues(r)[0]
        rays = (Kinv @ np.c_[see(near, i), np.ones(len(near))].T).T
        # ray x (R X + t) = 0  ->  [ray]_x t = -[ray]_x R X, stacked over the points
        cross = lambda v: np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
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


def render_can(frame, R, t, base, up, face, label, env, ss=3):
    """Ray-trace one can for this frame. base/up/face are in the reference camera; R, t move them into this frame.
    Returns (x0, y0, x1, y1), linear premultiplied RGB, coverage and the table's contact-shadow occlusion, or None."""
    h_img, w_img = frame.shape[:2]
    A = R @ base + t
    a = R @ up; a /= np.linalg.norm(a)
    # screen bbox from the cylinder's bounding points
    ring = []
    # label frame fixed on the can: e1 = the side that faced the reference camera (centre of the artwork)
    e1 = R @ face; e1 -= (e1 @ a) * a; e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
    toward = -(A - (A @ a) * a)
    toward /= np.linalg.norm(toward)
    right = np.cross(a, toward)  # camera-right, in the table plane
    right /= np.linalg.norm(right)
    # the light that casts shadows: above and behind the table, a little from the right (the real bowls' shadows
    # fall toward the camera)
    sun = 0.85 * a - 0.45 * toward + 0.2 * right
    sun /= np.linalg.norm(sun)
    run = sun - (sun @ a) * a
    reach = CAN_H * np.linalg.norm(run) / (sun @ a)   # shadow length on the table
    for z in (0, CAN_H):
        for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            ring.append(A + a * z + 1.6 * CAN_R * (np.cos(th) * e1 + np.sin(th) * e2))
    for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        ring.append(A - run / np.linalg.norm(run) * reach * 1.2 + 1.8 * CAN_R * (np.cos(th) * e1 + np.sin(th) * e2))
    ring = np.array(ring)
    pr = (K() @ ring.T).T
    pr = pr[:, :2] / pr[:, 2:]
    x0, y0 = np.floor(pr.min(0) - 6).astype(int)
    x1, y1 = np.ceil(pr.max(0) + 6).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w_img), min(y1, h_img)
    if x1 <= x0 or y1 <= y0:
        return None
    # supersampled pixel grid
    us = x0 + (np.arange((x1 - x0) * ss) + 0.5) / ss - 0.5
    vs = y0 + (np.arange((y1 - y0) * ss) + 0.5) / ss - 0.5
    U, V = np.meshgrid(us, vs)
    d = (np.linalg.inv(K()) @ np.stack([U.ravel(), V.ravel(), np.ones(U.size)])).T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    w = -A                                         # camera origin minus axis base
    da, wa = d @ a, w @ a
    dp, wp = d - np.outer(da, a), w - wa * a
    qa, qb, qc = (dp * dp).sum(1), 2 * (dp @ wp), wp @ wp - CAN_R ** 2
    disc = qb ** 2 - 4 * qa * qc
    s_side = np.where(disc > 0, (-qb - np.sqrt(np.maximum(disc, 0))) / (2 * qa + 1e-12), np.inf)
    y_side = s_side * da + wa
    s_side = np.where((y_side >= 0) & (y_side <= CAN_H) & (s_side > 0), s_side, np.inf)
    s_cap = (CAN_H - wa) / np.where(np.abs(da) > 1e-9, da, 1e-9)
    Pc = np.outer(s_cap, np.ones(3)) * d - A
    rc = np.linalg.norm(Pc - np.outer(Pc @ a, a), axis=1)
    s_cap = np.where((rc <= CAN_R) & (s_cap > 0), s_cap, np.inf)
    hit_side, hit_cap = (s_side < s_cap) & np.isfinite(s_side), (s_cap <= s_side) & np.isfinite(s_cap)
    hit = hit_side | hit_cap
    s = np.where(hit_side, s_side, s_cap)
    P = d * np.where(hit, s, 0)[:, None] - A        # relative to the axis base
    y = P @ a
    radial = P - np.outer(y, a)
    nrm = np.where(hit_cap[:, None], a, radial / CAN_R)
    # label coordinates: angle around the axis (0 = facing the reference camera), height
    ang = np.arctan2(radial @ e2, radial @ e1)
    uu = (ang / (2 * np.pi) + 0.5) % 1.0
    vv = 1 - y / CAN_H
    lab = label.astype(np.float32)
    lh, lw = lab.shape[:2]
    band_top, band_bot = 0.10, 0.94              # silver neck and base outside the printed band
    in_band = (vv > band_top) & (vv < band_bot) & hit_side
    lv = np.clip((vv - band_top) / (band_bot - band_top), 0, 1)
    tex = cv2.remap(lab, (uu * (lw - 1)).astype(np.float32).reshape(U.shape), (lv * (lh - 1)).astype(np.float32).reshape(U.shape),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP).reshape(-1, 3)
    silver = np.array([196, 199, 204], np.float32)
    albedo = srgb_to_lin(np.where(in_band[:, None], tex, silver))
    # lid: the rolled rim catches light, a groove inside it, the recessed panel a touch darker
    rr = np.linalg.norm(radial, axis=1) / CAN_R
    lid = np.where(rr > 0.9, 1.25, np.where(rr > 0.82, 0.55, 0.85))
    albedo = np.where(hit_cap[:, None], albedo * lid[:, None], albedo)
    metal = np.where(in_band, 0.05, 0.55)
    # Light as the footage has it: soft daylight from above and camera-right with open shadows. (A camera-side key
    # made the front hot and both sides near-black, a studio look; fully even light made it a flat card.)
    sky, table_col, mean_col = env["sky"], env["table"], env["mean"]
    # key from above and camera-right, as the real glasses show (bright right side, falloff to the left)
    key = 0.65 * a + 0.6 * right - 0.25 * toward
    key /= np.linalg.norm(key)
    k = np.clip(nrm @ a, -1, 1)[:, None]
    amb = np.where(k >= 0, mean_col * (1 - k) + sky * k, mean_col * (1 + k) + table_col * (-k))
    wrap = np.clip((nrm @ key + 0.6) / 1.6, 0, 1)
    irr = amb * (0.45 + 0.9 * wrap ** 1.3)[:, None]
    low = np.clip(1 - y / (0.25 * CAN_H), 0, 1) * hit_side  # warm bounce from the table onto the can's foot
    irr = irr + table_col * (0.25 * low)[:, None]
    # broad, soft reflections (glossy things in the footage show wide soft highlights, not thin streaks)
    # reflections from a simple sky dome: bright overcast above the horizon, the warm table below, a smooth blend
    # (the flat frame projected onto the can meant nothing for directions pointing up, and drew thin streaks)
    refl = d - 2 * (d * nrm).sum(1, keepdims=True) * nrm
    e = (refl @ a)[:, None]
    above = np.clip(e / 0.5, 0, 1) ** 0.7
    below = np.clip(-e / 0.3, 0, 1)
    spec = np.where(e >= 0, mean_col * (1 - above) + env["bright"] * above, mean_col * (1 - below) + table_col * below)
    fres = 0.04 + 0.96 * (1 - np.clip(-(nrm * d).sum(1), 0, 1)) ** 5
    kspec = (metal + (1 - metal) * fres * 0.5)[:, None]
    col = albedo * irr * (1 - kspec) + spec * kspec * np.where(in_band, 1.0, albedo.mean(1))[:, None]
    # the footage's grade: muted chroma, milky blacks (the can was the most saturated thing in the shot)
    lum = (col @ np.array([0.2126, 0.7152, 0.0722]))[:, None]
    col = (lum + (col - lum) * 0.8) * 0.96 + 0.004
    col = np.where(hit[:, None], col, 0)
    # downsample supersamples: premultiplied colour + coverage
    hh, ww = y1 - y0, x1 - x0
    cov = hit.reshape(hh, ss, ww, ss).mean((1, 3))
    rgb = col.reshape(hh, ss, ww, ss, 3).mean((1, 3))
    # contact shadow: rays that miss the can meet the table; occlude the table near the can's foot
    st = (A @ a) / (d @ a)                          # ray parameter to the table plane through A
    Pt = d * st[:, None] - A
    rt = np.linalg.norm(Pt - np.outer(Pt @ a, a), axis=1)
    toward_cam = -(A - (A @ a) * a); toward_cam /= np.linalg.norm(toward_cam)
    along = Pt @ toward_cam
    gap = np.maximum(rt - CAN_R, 0) / CAN_R
    # a tight dark contact line where it meets the wood (as under the real bowls), plus a soft shadow that leans
    # toward the camera because the light comes from above and behind
    occl = 0.85 * np.exp(-gap / 0.08) + 0.35 * np.exp(-gap / 0.5) * (0.55 + 0.45 * (along > 0))
    # soft cast shadow: from each table point, is the light (sampled over a small area) blocked by the can?
    p = Pt - np.outer(Pt @ a, a)
    rng = np.random.default_rng(7)  # fixed samples: the shadow must not shimmer frame to frame
    blocked = np.zeros(len(p))
    for _ in range(48):
        L = sun + 0.12 * rng.normal(size=3)
        L /= np.linalg.norm(L)
        u = L - (L @ a) * a
        qa2, qb2, qc2 = u @ u, 2 * (p @ u), (p * p).sum(1) - CAN_R ** 2
        disc2 = qb2 ** 2 - 4 * qa2 * qc2
        s_in = (-qb2 - np.sqrt(np.maximum(disc2, 0))) / (2 * qa2)
        s_out = (-qb2 + np.sqrt(np.maximum(disc2, 0))) / (2 * qa2)
        hit2 = (disc2 > 0) & (s_out > 0) & (np.maximum(s_in, 0) * (L @ a) < CAN_H)
        blocked += hit2
    dist = np.linalg.norm(p, axis=1) / max(reach, 1e-6)
    cast = 0.65 * blocked / 48 * np.exp(-np.maximum(dist - 0.25, 0) / 0.6)  # darkest by the can, fading out
    occl = 1 - (1 - np.clip(occl, 0, 0.92)) * (1 - cast)
    occl = np.where(st > 0, occl, 0).reshape(hh, ss, ww, ss).mean((1, 3))
    return (x0, y0, x1, y1), rgb, cov, occl


def scene_env(frame):
    lin = srgb_to_lin(frame.astype(np.float32))
    h = frame.shape[0]
    return {"sky": lin[: h // 4].reshape(-1, 3).mean(0) * 1.15, "table": lin[int(h * 0.7):].reshape(-1, 3).mean(0),
            "mean": lin.reshape(-1, 3).mean(0) * 1.25, "blur": cv2.GaussianBlur(lin, (0, 0), 40),
            "bright": np.percentile(lin.reshape(-1, 3), 96, axis=0)}  # overcast sky / the bright background


def shot_env(frames, idx):
    """One lighting environment for the whole shot (frame-by-frame light made the can pulse whenever the waiter's
    dark shirt crossed the frame; the real light doesn't change)."""
    envs = [scene_env(frames[i][..., ::-1]) for i in idx]
    return {k: np.mean([e[k] for e in envs], 0) for k in envs[0]}


def can_label(brand, front=0.32):
    """Can label texture (2:1 = circumference : printed height) from the brand's can_wrap artwork. Only about a third
    of a can faces the camera, so the artwork fills the front `front` of the width and its background colour wraps
    around the back, as on a real can."""
    art = cv2.imread(str(ROOT / "assets/creatives/gen" / f"{brand}_can_wrap.png"))[..., ::-1]
    border = np.r_[art[:4].reshape(-1, 3), art[-4:].reshape(-1, 3), art[:, :4].reshape(-1, 3), art[:, -4:].reshape(-1, 3)]
    bg = np.median(border, 0)
    aw = int(round(1024 * front))
    small = cv2.resize(art, (aw, int(round(aw * art.shape[0] / art.shape[1]))), interpolation=cv2.INTER_AREA)
    out = np.empty((512, 1024, 3), np.uint8)
    out[:] = bg
    y0, x0 = (512 - small.shape[0]) // 2, (1024 - aw) // 2
    out[y0:y0 + small.shape[0], x0:x0 + aw] = small
    return out


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


class CanPlacement:
    """A can standing at a pixel of the reference frame, for every frame where the table is tracked."""

    def __init__(self, clip=CLIP, at=(705, 525), frames=(0, 157), sigma=0.6, grain=1.6):
        self.clip, self.at, self.sigma, self.grain = clip, tuple(at), sigma, grain
        self.frames = Frames(ROOT / clip)
        H = self.table_track()
        self.valid = [i for i in range(*frames) if np.isfinite(H[i]).all()]
        self.h = table_scale()
        self.up = N_UP / np.linalg.norm(N_UP)
        self.base = on_plane([self.at], self.h)[0]
        self.poses = poses(H, self.h, self.valid, self.at)
        # going back from the planar track's first frame, stop once the can has left the picture: earlier frames are
        # chained through floor and plants, not the table, and could put the can somewhere it doesn't belong
        planar = np.load(ROOT / "cache" / f"{Path(clip).stem}-scene" / "table_track_v3.npz")["planar_first"]
        keep = []
        for i in sorted(self.valid, reverse=True):
            R, t = self.poses[i]
            top = K() @ (R @ (self.base + self.up * CAN_H) + t)
            if i < planar and top[1] / top[2] > self.frames.h + 10:
                break
            keep.append(i)
        self.valid = sorted(keep)
        self.poses = {i: self.poses[i] for i in self.valid}
        # the label's front faces the camera at mid-shot (it orbits the can during the shot)
        R, t = self.poses[self.valid[len(self.valid) // 2]]
        to_cam = -R.T @ t - self.base
        self.face = to_cam - (to_cam @ self.up) * self.up
        self.labels = {}
        self.env = shot_env(self.frames, [i for i in self.valid if i >= 100][::8])
        self.person = union_store(clip, "person", self.frames)  # 'tray' is the pizza pan, behind the can
        from scripts.render_dynamic import object_stores
        self.people = list(object_stores(clip, "person", self.frames).values())  # one store per person

    def table_track(self):
        path = ROOT / "cache" / f"{Path(self.clip).stem}-scene" / "table_track_v3.npz"
        if not path.exists():
            from pipeline.planar_track import track_plane
            from scripts.render_dynamic import occluders
            entry = json.loads((ROOT / "configs" / "clips.json").read_text())[self.clip]
            occ = occluders(self.clip, entry, self.frames)
            tr = track_plane(self.frames, REF, np.float32([[150, 470], [1140, 365], [1125, 700], [150, 700]]),
                             occluder_masks=lambda i: occ.frame(i))
            H0 = np.asarray(tr["H"], float)
            H = extend_back(self.frames, H0, occ)
            np.savez(path, H=H, planar_first=next(i for i in range(len(H0)) if np.isfinite(H0[i]).all()))
        return np.load(path)["H"]

    def front(self, i):
        """The people in front of the can this frame. A person is in front only if they reach the table nearer the
        camera than the can's foot: the seated guest in the foreground is, the waiter leaning over the pizza behind
        it is not (drawing his hand over the can made it look like he grabbed it)."""
        R, t = self.poses[i]
        foot = K() @ (R @ self.base + t)
        foot_y = foot[1] / foot[2]
        # columns the can occupies: depth is judged there (the table edge runs at an angle, so a person's lowest
        # point elsewhere in the frame says nothing about whether they are in front of the can)
        ring = np.array([self.base + CAN_R * 1.5 * np.array([np.cos(th), 0, np.sin(th)]) for th in np.linspace(0, 6.3, 12)])
        pr = (K() @ (R @ ring.T + t[:, None])).T
        c0, c1 = int(max(pr[:, 0].min() / pr[:, 2].max() - 10, 0)), int(min(pr[:, 0].max() / pr[:, 2].min() + 10, self.frames.w))
        out = None
        for st in self.people:  # judged per person: the union mask joins the waiter's arm to the seated guest
            if not st.any(i):
                continue
            m = st[i]
            rows = np.flatnonzero(m[:, c0:c1].any(1))
            if len(rows) and rows.max() > foot_y + 4:
                out = m if out is None else out | m
        return out

    def layer(self, i, frame, brand):
        """What the can adds to frame i, in sRGB: box, colour (unpremultiplied BGR), alpha, table shadow multiplier.
        Server render and player export both use this, so they show the same can."""
        if i not in self.poses:
            return None
        if brand not in self.labels:
            self.labels[brand] = can_label(brand)
        rgb_frame = frame[..., ::-1]
        R, t = self.poses[i]
        r = render_can(rgb_frame, R, t, self.base, self.up, self.face, self.labels[brand], self.env)
        if r is None:
            return None
        box, rgb, cov, occl = r
        colour = lin_to_srgb(rgb / np.maximum(cov, 1e-6)[..., None])[..., ::-1] * cov[..., None]
        if self.sigma > 0:  # the footage is a touch soft: blur premultiplied colour, coverage and shadow together
            colour, cov, occl = (cv2.GaussianBlur(x, (0, 0), self.sigma) for x in (colour, cov.astype(np.float32),
                                                                                   occl.astype(np.float32)))
        colour = colour / np.maximum(cov, 1e-6)[..., None]
        shadow = (1 - occl) ** (1 / 2.2)  # linear-light occlusion as an sRGB multiplier
        return box, np.clip(colour, 0, 255), np.clip(cov, 0, 1), shadow

    def apply(self, original, composited, i, brand):
        """Put the can into an already composited frame; people (from the original) stay in front of it."""
        L = self.layer(i, original, brand)
        if L is None:
            return composited
        (x0, y0, x1, y1), colour, cov, shadow = L
        out = composited.astype(np.float32)
        reg, a = out[y0:y1, x0:x1], cov[..., None]
        noise = np.random.default_rng(i).normal(0, self.grain, colour.shape[:2])[..., None]  # luma grain, like the footage's
        out[y0:y1, x0:x1] = reg * shadow[..., None] * (1 - a) + (colour + noise) * a
        front = self.front(i)
        if front is not None:
            from pipeline.matte import refine_occluder
            k = refine_occluder(original, front, radius=4, gain=2.0)[..., None]
            out = k * original + (1 - k) * out
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="fizzwell")
    ap.add_argument("--at", type=float, nargs=2, default=[705, 525], help="can foot, pixel in the reference frame")
    ap.add_argument("--out", default=str(ROOT / "outputs/place3d/can.mp4"))
    ap.add_argument("--still", type=int, help="only write this frame as a PNG")
    args = ap.parse_args()
    can = CanPlacement(at=args.at)
    print(f"camera {can.h:.2f} m above the table, can at {can.base.round(3)} m, frames {can.valid[0]}-{can.valid[-1]}")
    if args.still is not None:
        f = can.frames[args.still]
        cv2.imwrite(str(Path(args.out).with_name(f"can_{args.still}.png")), can.apply(f, f, args.still, args.brand))
        return
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                            f"{can.frames.w}x{can.frames.h}", "-r", str(can.frames.fps), "-i", "-", "-c:v", "libx264",
                            "-crf", "16", "-pix_fmt", "yuv420p", args.out], stdin=subprocess.PIPE)
    for i, f in enumerate(can.frames):  # the whole clip; the can appears where the table is tracked
        enc.stdin.write(can.apply(f, f, i, args.brand).tobytes())
    enc.stdin.close()
    enc.wait()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
