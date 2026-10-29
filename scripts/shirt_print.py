"""A brand's logo screen-printed on the waiter's T-shirt, in place of the logo he really wears.

SAM 3.1 ("t-shirt logo", through the Meta Model API) finds his chest logo in each frame. The old logo is filled in
from the black fabric around it, and the brand's white print goes on the same spot: its centre and size follow the
mask (smoothed over time so the print doesn't jitter), its brightness follows the fabric's folds, and the tray and
pizza stay in front of it. Same interface as scripts/place3d.TableObject, so render_dynamic and export_player handle
it like the other objects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.render_dynamic import Frames, object_stores, union_store  # noqa: E402


def smooth_runs(values, frames, sigma=1.5):
    """Gaussian-smooth per-frame values within each run of consecutive frames."""
    out = dict(values)
    runs, run = [], []
    for i in sorted(values):
        if run and i != run[-1] + 1:
            runs.append(run)
            run = []
        run.append(i)
    if run:
        runs.append(run)
    k = np.exp(-0.5 * (np.arange(-4, 5) / sigma) ** 2)
    for run in runs:
        v = np.array([values[i] for i in run], float)
        pad = np.pad(v, ((4, 4), (0, 0)), mode="edge")
        sm = np.stack([np.convolve(pad[:, c], k / k.sum(), mode="valid") for c in range(v.shape[1])], 1)
        out.update({i: sm[j] for j, i in enumerate(run)})
    return out


class ShirtPrint:
    shadow = False
    grain = 1.2

    def __init__(self, clip, spec):
        self.clip, self.spec, self.id = clip, spec, spec["id"]
        self.frames = Frames(ROOT / clip)
        self.logo = list(object_stores(clip, spec.get("prompt", "t-shirt logo"), self.frames).values())
        self.front_of = [union_store(clip, p, self.frames) for p in ("tray", "pizza")]
        raw = {}
        for i in range(self.frames.n):
            m = self.mask(i)
            if m is None or m.sum() < 40:
                continue
            ys, xs = np.nonzero(m)
            raw[i] = [xs.mean(), ys.mean(), np.sqrt(m.sum())]
        self.place = smooth_runs(raw, self.frames)
        self.valid = sorted(self.place)
        self.poses = {i: None for i in self.valid}   # same membership test as the table objects
        self.base = np.array([0.0, 0.0, 10.0])        # far: drawn before the objects on the table
        self.tex = {}

    def mask(self, i):
        m = None
        for st in self.logo:
            if st.any(i):
                m = st[i] if m is None else m | st[i]
        return m

    def front(self, i):
        return None

    def art(self, brand):
        if brand not in self.tex:
            a = cv2.imread(str(ROOT / "assets/creatives/gen" / f"{brand}_shirt_print.png")).astype(np.float32)
            alpha = np.clip((a.max(2) - 40) / 160, 0, 1)   # white on black: the black is the shirt
            ys, xs = np.nonzero(alpha > 0.1)
            self.tex[brand] = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return self.tex[brand]

    def layer(self, i, frame, brand):
        if i not in self.place:
            return None
        cx, cy, size = self.place[i]
        m = self.mask(i)
        h, w = frame.shape[:2]
        side = 2.1 * size   # the new print is a little bigger than the old logo, so it covers it
        pad = int(side) + 8
        x0, y0 = max(int(cx) - pad, 0), max(int(cy) - pad, 0)
        x1, y1 = min(int(cx) + pad, w), min(int(cy) + pad, h)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = frame[y0:y1, x0:x1]
        old = np.zeros(frame.shape[:2], np.uint8)
        if m is not None:
            old[m] = 255
        old = cv2.dilate(old, np.ones((7, 7), np.uint8))[y0:y1, x0:x1]
        fabric = cv2.inpaint(crop, old, 6, cv2.INPAINT_TELEA)            # the shirt without its old logo
        # the print, fitted into a side x side square centred on the logo, keeping its aspect
        a = self.art(brand)
        s = side / max(a.shape)
        M = np.array([[s, 0, cx - x0 - a.shape[1] * s / 2], [0, s, cy - y0 - a.shape[0] * s / 2]], np.float32)
        pa = cv2.warpAffine(a, M, (x1 - x0, y1 - y0), flags=cv2.INTER_AREA)
        pa = cv2.GaussianBlur(pa, (0, 0), 0.7) * 0.9                      # ink on cotton: soft edges, not opaque
        pa = self.motion_blur(pa, i)  # he moves: the real shirt smears, so must the print
        # brightness follows the folds: the fabric's own light, relative to its median around the logo
        lum = cv2.GaussianBlur(cv2.cvtColor(fabric, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 3)
        shade = np.clip(lum / max(float(np.median(lum)), 1.0), 0.55, 1.35)
        ink = np.dstack([shade * 215] * 3)                                 # off-white ink, a little below paper
        # the tray and pizza go in front of his chest
        block = np.zeros((y1 - y0, x1 - x0), np.float32)
        for st in self.front_of:
            if st.any(i):
                block = np.maximum(block, st[i][y0:y1, x0:x1].astype(np.float32))
        block = cv2.GaussianBlur(block, (0, 0), 1.0)
        pa = pa * (1 - block)
        ro = cv2.GaussianBlur(old.astype(np.float32) / 255, (0, 0), 1.2) * (1 - block)
        colour = fabric.astype(np.float32) * (1 - pa[..., None]) + ink * pa[..., None]
        cov = np.maximum(ro, pa)
        return (x0, y0, x1, y1), colour, np.clip(cov, 0, 1), np.ones_like(cov)

    def motion_blur(self, img, i, shutter=0.5):
        """Smear along the logo's own motion over the exposure (half a frame interval)."""
        nb = [j for j in (i - 1, i + 1) if j in self.place]
        if len(nb) < 1:
            return img
        v = (self.place[max(nb)][:2] - self.place[min(nb + [i])][:2]) / max(max(nb) - min(nb + [i]), 1)
        length = float(np.hypot(*v)) * shutter
        if length < 1.0:
            return img
        n = int(np.ceil(length)) * 2 + 1
        k = np.zeros((n, n), np.float32)
        c = n // 2
        d = v / np.hypot(*v)
        for t in np.linspace(-length / 2, length / 2, 4 * n):
            k[int(round(c + t * d[1])), int(round(c + t * d[0]))] += 1
        return cv2.filter2D(img, -1, k / k.sum())

    def apply(self, original, composited, i, brand):
        L = self.layer(i, original, brand)
        if L is None:
            return composited
        (x0, y0, x1, y1), colour, cov, _ = L
        out = composited.astype(np.float32)
        noise = np.random.default_rng(i).normal(0, self.grain, colour.shape[:2])[..., None]
        a = cov[..., None]
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a) + (colour + noise) * a
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
