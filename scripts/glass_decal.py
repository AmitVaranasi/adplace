"""A brand's logo printed on a real drinking glass, for any clip: no table calibration, only the glass's own mask.

SAM 3.1 ("cup" / "drinking glass", through the Meta Model API) outlines the glass in every frame. From each outline:
its axis (the glass tilts when someone drinks), centre, radius and the top of the rim, smoothed over time. The print
wraps round the front of that cylinder (narrower towards the sides, with a slight curve for the camera looking
down on it), is white enamel lit like the glass behind it, keeps the glass's own highlights on top, and is only
drawn where the glass itself is visible, so fingers and straws in front of it stay in front. Same interface as
scripts/place3d.TableObject, so render_dynamic and export_player handle it like the other objects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hashlib  # noqa: E402
import pickle  # noqa: E402

from scripts.render_dynamic import Frames, object_stores, scene_dir  # noqa: E402
from scripts.shirt_print import ShirtPrint, smooth_runs  # noqa: E402

HALF_ANGLE = 0.55   # the print covers about +-31 degrees round the front
CENTRE = 0.55       # its centre, as a fraction of the glass's height from the rim
CURVE = 0.12        # rows on the glass bow down at the front: the camera usually looks slightly down


def focus_and_grain(frames, store, keys):
    """How soft the glass is in this shot (Gaussian sigma, px) and how grainy (noise std, grey levels). The blur
    comes from the glass's own silhouette: a step of contrast C blurred by sigma has a steepest slope of
    C / (sigma * sqrt(2 pi)). Without these the print is the sharpest, cleanest thing in a soft, noisy shot."""
    sig, noise = [], []
    for i in keys[:: max(1, len(keys) // 12)][:12]:
        m = store[i].astype(np.uint8)
        g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        k = lambda n: np.ones((n, n), np.uint8)
        ring_in = (cv2.erode(m, k(5)) & ~cv2.erode(m, k(13)) & 1).astype(np.float32)
        ring_out = (cv2.dilate(m, k(13)) & ~cv2.dilate(m, k(5)) & 1).astype(np.float32)
        local = lambda ring: cv2.boxFilter(g * ring, -1, (21, 21)) / np.maximum(cv2.boxFilter(ring, -1, (21, 21)), 1e-3)
        step = np.abs(local(ring_in) - local(ring_out))
        grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) / 8
        grad = cv2.dilate(grad, k(7))
        edge = (m ^ cv2.erode(m, k(3))).astype(bool) & (step > 25) & (grad > 1)
        if edge.sum() > 30:
            sig.append(float(np.median(step[edge] / (grad[edge] * np.sqrt(2 * np.pi)))))
        inner = cv2.erode(m, k(9)).astype(bool)
        if inner.sum() > 200:
            hp = (g - cv2.GaussianBlur(g, (0, 0), 1.0))[inner]
            noise.append(float(1.4826 * np.median(np.abs(hp - np.median(hp)))))
    sigma = float(np.clip(np.median(sig), 0.6, 4.0)) if sig else 1.0
    grain = float(np.clip(np.percentile(noise, 25), 0.8, 4.0)) if noise else 1.0
    return sigma, grain


def glass_geometry(m, min_ratio=1.2):
    """(cx, cy, r, angle, height, bx, by) of a glass mask: centre of the rim, radius, axis angle (0 = upright),
    height, centre of the base. None when the mask is too small or not glass-shaped (shorter than `min_ratio`
    radii: a glass is tall; a teacup can be squat)."""
    if m.sum() < 150:
        return None
    # a straw or a sliver of finger sticking out of the outline swings its axis: open them away first
    opened = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if opened.sum() > 0.6 * m.sum():
        m = opened.astype(bool)
    mo = cv2.moments(m.astype(np.uint8), binaryImage=True)
    ang = 0.5 * np.arctan2(2 * mo["mu11"], mo["mu20"] - mo["mu02"]) - np.pi / 2   # major axis vs vertical
    ang = (ang + np.pi / 2) % np.pi - np.pi / 2
    if min_ratio < 1.2 and abs(ang) > np.pi / 4:   # a cup can be wider than tall: up is the axis nearer vertical
        ang -= np.sign(ang) * np.pi / 2
    if abs(ang) > 1.2:   # tipped further than ~70 degrees (drinking): the outline no longer says which way is up
        return None
    ys, xs = np.nonzero(m)
    c, s = np.cos(ang), np.sin(ang)
    mx, my = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
    u = (xs - mx) * c + (ys - my) * s      # across the glass
    t = -(xs - mx) * s + (ys - my) * c     # along its axis, downwards
    top, bottom = np.percentile(t, 0.5), np.percentile(t, 99.5)
    rows = np.round(t).astype(int)
    lo, hi = rows.min(), rows.max()
    left = np.full(hi - lo + 1, np.inf)
    right = np.full(hi - lo + 1, -np.inf)
    np.minimum.at(left, rows - lo, u)
    np.maximum.at(right, rows - lo, u)
    width = right - left
    ok = np.isfinite(width)
    r = float(np.percentile(width[ok], 85)) / 2
    mid = float(np.median((left[ok] + right[ok]) / 2))
    if r < 6 or (bottom - top) < min_ratio * r:
        return None
    cx, cy = mx + mid * c - top * s, my + mid * s + top * c
    bx, by = mx + mid * c - bottom * s, my + mid * s + bottom * c
    return [cx, cy, r, ang, bottom - top, bx, by]


CODE = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
KEEP_LIVE = {"clip", "spec", "id", "tex", "frames", "store"}   # rebuilt on load, not cached


class GlassDecal(ShirtPrint):
    shadow = False
    grain = 1.0     # both measured per glass (focus_and_grain)
    sigma = 1.0
    min_ratio = 1.2     # height / radius below which an outline isn't this kind of vessel
    ink = 230.0         # white enamel
    ink_alpha = 0.88    # enamel on glass lets a little through
    half = HALF_ANGLE   # the print's half-width round the vessel (radians), its centre's height (fraction of the
    centre = CENTRE     # way from rim to base) and sideways turn from the front (radians): a cup with a design of
    lateral = 0.0       # its own moves and shrinks the print onto a blank patch (CupDecal.choose_spot)
    blank = True        # False: no blank patch big enough

    def choose_spot(self, good):
        """Where on the vessel the print goes; a glass is see-through, so always the default."""

    def mask_store(self, store):
        return store

    def __init__(self, clip, spec):
        self.clip, self.spec, self.id = clip, spec, spec["id"]
        self.tex = {}
        self.frames = Frames(ROOT / clip)
        self.store = self.mask_store(object_stores(clip, spec.get("prompt", "cup"), self.frames)[str(spec["object"])])
        # the measuring below takes about a minute and an upload needs it three times (the gates, the player export,
        # the final check render): keep it per object, keyed by this file's code so any change re-measures
        cache = scene_dir(clip) / f"vessel_{type(self).__name__}_{spec.get('prompt', 'cup')}_{spec['object']}_{CODE}.pkl"
        if cache.exists():
            self.__dict__.update(pickle.loads(cache.read_bytes()))
            return
        self.measure(spec)
        cache.write_bytes(pickle.dumps({k: v for k, v in self.__dict__.items() if k not in KEEP_LIVE}))

    def measure(self, spec):
        h = self.frames.h
        seen = [i for i in range(self.frames.n) if self.store.any(i)]
        geo = {i: glass_geometry(self.store[i], self.min_ratio) for i in seen}
        geo = {i: g for i, g in geo.items() if g is not None}
        areas = {i: float(self.store[i].sum()) for i in geo}
        typical = np.median(list(areas.values())) if areas else 0
        # the whole glass in view: its outline gives every measure. A hand over part of it or the frame edge
        # shortens the outline but leaves its rim and axis right, so those frames keep their own position and
        # take the radius and height from the whole-glass frames. With no usable outline at all (tipped to drink)
        # a frame borrows from its neighbours, if they are close.
        good = {i: g for i, g in geo.items()
                if areas[i] > 0.7 * typical and self.store.box(i)[3] < h - 2}
        self.place, self.clipped = {}, {i for i in seen if self.store.box(i)[3] >= h - 2}
        self.ratio = 2.6
        if good:
            ratio = self.ratio = float(np.median([g[4] / g[2] for g in good.values()]))
            keys = np.array(sorted(good))
            vals = np.array([good[i] for i in keys])
            for i in seen:
                if i in good:
                    self.place[i] = np.array(good[i])
                elif i in geo:
                    p = np.array(geo[i])
                    p[2] = np.interp(i, keys, vals[:, 2])
                    p[4] = max(p[4], ratio * p[2])
                    self.place[i] = p
                elif np.abs(keys - i).min() <= 6:
                    self.place[i] = np.array([np.interp(i, keys, vals[:, k]) for k in range(vals.shape[1])])
        # the tilt from an outline with fingers across it wavers more than the glass turns: smooth it harder
        tilt = smooth_runs({i: p[3:4] for i, p in self.place.items()}, self.frames, sigma=2.5)
        if good:
            self.sigma, self.grain = focus_and_grain(self.frames, self.store, sorted(good))
        if good:
            self.choose_spot(good)
        self.raw_centre = self.print_centres()
        centre = smooth_runs(self.raw_centre, self.frames, sigma=1.2)
        self.place = smooth_runs(self.place, self.frames, sigma=1.2)
        for i in self.place:
            self.place[i][3] = tilt[i][0]
            self.place[i][:2] = centre[i]   # from here on (cx, cy) is the print's centre
        self.good = set(good)   # frames with the whole glass in view
        if good:
            self.fit(good)
        # the glass's overall brightness, steadied over time: measured afresh each frame it moves with every finger
        # that passes and every slosh of the drink, and a white print shows that as flicker
        k = np.ones((5, 5), np.uint8)
        light = {}
        for i in self.place:
            m = cv2.erode(self.store[i].astype(np.uint8), k).astype(bool)
            if m.sum() > 50:
                light[i] = np.array([float(np.median(cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2GRAY)[m]))])
        light = smooth_runs(light, self.frames, sigma=4.0) if light else {}
        fallback = float(np.median([v[0] for v in light.values()])) if light else 128.0
        self.light = {i: float(light[i][0]) if i in light else fallback for i in self.place}

    def room_motion(self, person):
        """Frame i-1 -> i motion of the room: a similarity fitted to corners away from people and this glass."""
        cam, prev = {}, None
        for i in range(self.frames.n):
            g = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2GRAY)
            if prev is not None:
                bg = np.full(g.shape, 255, np.uint8)
                if self.store.any(i - 1):
                    bg[cv2.dilate(self.store[i - 1].astype(np.uint8), np.ones((41, 41), np.uint8)) > 0] = 0
                if person is not None and person.any(i - 1):
                    bg[cv2.dilate(person[i - 1].astype(np.uint8), np.ones((25, 25), np.uint8)) > 0] = 0
                p = cv2.goodFeaturesToTrack(prev, 400, 0.01, 8, mask=bg)
                if p is not None and len(p) >= 20:
                    q, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p, None, winSize=(15, 15), maxLevel=3)
                    ok = st.ravel() == 1
                    if ok.sum() >= 20:
                        M, _ = cv2.estimateAffinePartial2D(p[ok], q[ok], method=cv2.RANSAC, ransacReprojThreshold=1.0)
                        if M is not None:
                            cam[i] = M
            prev = g
        return cam

    def flow_steps(self):
        """Frame i-1 -> i motion of the glass itself: a similarity fitted (RANSAC) to corners inside its outline
        followed by optical flow, checked forwards and backwards. People's masks (hands) are left out. None where
        too few corners agree (a fast, blurred swing)."""
        try:
            from scripts.render_dynamic import union_store
            person = union_store(self.clip, "person", self.frames)
        except Exception:
            person = None
        k5, k9 = np.ones((5, 5), np.uint8), np.ones((9, 9), np.uint8)
        steps, rel, prev = {}, {}, None
        self.rel = rel
        self.camera = self.room_motion(person)
        for i in range(self.frames.n):
            g = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2GRAY)
            if prev is not None and self.store.any(i - 1) and self.store.any(i):
                m = cv2.erode(self.store[i - 1].astype(np.uint8), k5)
                if person is not None and person.any(i - 1):
                    m[cv2.dilate(person[i - 1].astype(np.uint8), k9) > 0] = 0
                p = cv2.goodFeaturesToTrack(prev, 200, 0.01, 4, mask=m)
                if p is not None and len(p) >= 8:
                    q, s1, _ = cv2.calcOpticalFlowPyrLK(prev, g, p, None, winSize=(15, 15), maxLevel=3)
                    b, s2, _ = cv2.calcOpticalFlowPyrLK(g, prev, q, None, winSize=(15, 15), maxLevel=3)
                    ok = (s1.ravel() == 1) & (s2.ravel() == 1) & (np.linalg.norm(b - p, axis=2).ravel() < 0.5)
                    if ok.sum() >= 8:
                        M, inl = cv2.estimateAffinePartial2D(p[ok], q[ok], method=cv2.RANSAC, ransacReprojThreshold=1.0)
                        if M is not None and inl.sum() >= 8:
                            steps[i] = M
                            # how the glass moves relative to the room (the camera's motion taken out)
                            Mc = self.camera.get(i)
                            if Mc is not None:
                                c = np.median(q[ok][:, 0], axis=0)
                                pc = np.median(p[ok][:, 0], axis=0)
                                rel[i] = c - (Mc[:, :2] @ pc + Mc[:, 2])
            prev = g
        # resting: under 3 px of travel relative to the room over any half second around the frame (fingers on a
        # glass jiggle it a little; a glass being moved travels far more)
        half = max(2, int(self.frames.fps / 4))
        resting = set()
        for i in rel:
            window = [rel[j] for j in range(i - half, i + half + 1) if j in rel]
            if len(window) >= half and np.linalg.norm(np.sum(window, axis=0)) < 3.0 and \
                    max(np.linalg.norm(w) for w in window) < 2.0:
                resting.add(i)
        # and still for a while (a second at least): a held glass is often still for a frame or two
        runs, run = set(), []
        for i in range(self.frames.n + 1):
            if i in resting:
                run.append(i)
                continue
            if len(run) >= self.frames.fps:
                runs.update(run)
            run = []
        return steps, runs

    def fit(self, good):
        """The glass is rigid: its silhouette in the frame where it is seen whole and largest is its shape for the
        whole shot. Each frame, that shape is fitted to the parts of the outline that are really the glass's edge,
        not where a hand or the frame edge cuts it: a similarity (move, turn, scale) plus a stretch along the axis
        for the glass tipping towards the camera. The print is fixed in the shape's coordinates, so it can't slide
        over the glass unless the whole fit is wrong. Optical flow of the glass gives each frame's starting guess."""
        try:
            from scripts.render_dynamic import union_store
            person = union_store(self.clip, "person", self.frames)
        except Exception:
            person = None
        H, W = self.frames.h, self.frames.w
        ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        k3, k11 = np.ones((3, 3), np.uint8), np.ones((11, 11), np.uint8)

        def clean(i):   # straws and finger slivers off
            m = self.store[i].astype(np.uint8)
            o = cv2.morphologyEx(m, cv2.MORPH_OPEN, ell)
            return o if o.sum() > 0.6 * m.sum() else m

        def true_edge(i, m):
            """Outline pixels that are the glass's own edge: not next to a person (a hand in front) or the frame."""
            e = (m - cv2.erode(m, k3)).astype(bool)
            if person is not None and person.any(i):
                e &= ~(cv2.dilate(person[i].astype(np.uint8), k11) > 0)
            e[:3], e[-3:], e[:, :3], e[:, -3:] = False, False, False, False
            return e

        ref = max(good, key=lambda i: self.store[i].sum())
        T = clean(ref).astype(bool)
        g0 = glass_geometry(T, self.min_ratio)
        c0 = np.asarray(self.raw_centre[ref], float)
        a0, r0 = g0[3], g0[2]
        eu, et = np.array([np.cos(a0), np.sin(a0)]), np.array([-np.sin(a0), np.cos(a0)])
        # the shape: a clean tapered glass (rim width, base width, height) measured from the reference outline, not
        # the outline itself, which may carry a notch where a finger crossed it
        ys, xs = np.nonzero(T)
        u, t = (xs - c0[0]) * eu[0] + (ys - c0[1]) * eu[1], (xs - c0[0]) * et[0] + (ys - c0[1]) * et[1]
        top, bot = np.percentile(t, 0.5), np.percentile(t, 99.5)
        span = bot - top

        def half_width(lo, hi):
            band = (t >= top + lo * span) & (t <= top + hi * span)
            rows = np.round(t[band]).astype(int)
            w = [np.ptp(u[band][rows == r]) for r in np.unique(rows) if (rows == r).sum() > 3]
            return float(np.percentile(w, 75)) / 2 if w else r0
        rt, rb = half_width(0.05, 0.25), half_width(0.75, 0.95)
        mid_u = float(np.median(u))
        n = 90
        side = np.linspace(0, 1, n)
        Bu = np.concatenate([mid_u - (rt + (rb - rt) * side), mid_u + (rt + (rb - rt) * side),
                             mid_u + np.linspace(-rt, rt, n // 2), mid_u + np.linspace(-rb, rb, n // 2)])
        Bt = np.concatenate([top + span * side, top + span * side, np.full(n // 2, top), np.full(n // 2, bot)])
        poly = np.array([[mid_u - rt, top], [mid_u + rt, top], [mid_u + rb, bot], [mid_u - rb, bot]])
        poly_img = c0 + poly[:, :1] * eu + poly[:, 1:] * et
        T = np.zeros((H, W), np.uint8)
        cv2.fillPoly(T, [np.round(poly_img).astype(np.int32)], 1)
        DT = cv2.distanceTransform((1 - (T - cv2.erode(T, k3))).astype(np.uint8), cv2.DIST_L2, 3)

        def warp(q):        # template (axes) -> frame
            tx, ty, th, sc, sa = q
            a = a0 + th
            u, t = Bu * sc, Bt * sc * sa
            return np.stack([c0[0] + tx + u * np.cos(a) - t * np.sin(a), c0[1] + ty + u * np.sin(a) + t * np.cos(a)], 1)

        def unwarp(P, q):   # frame -> template image coordinates
            tx, ty, th, sc, sa = q
            a = a0 + th
            d = P - (c0 + [tx, ty])
            u = (d[:, 0] * np.cos(a) + d[:, 1] * np.sin(a)) / sc
            t = (-d[:, 0] * np.sin(a) + d[:, 1] * np.cos(a)) / (sc * sa)
            return c0 + np.outer(u, eu) + np.outer(t, et)

        def look(D, P, x0=0, y0=0):
            x = np.clip(np.round(P[:, 0] - x0).astype(int), 0, D.shape[1] - 1)
            y = np.clip(np.round(P[:, 1] - y0).astype(int), 0, D.shape[0] - 1)
            return np.minimum(D[y, x], 20.0)

        steps, resting = self.flow_steps()
        fitted = {ref: np.array([0.0, 0.0, 0.0, 1.0, 1.0])}
        self.fit_cost = {}

        def fit_one(i, q):
            m = clean(i)
            e = true_edge(i, m)
            ys, xs = np.nonzero(e)
            if len(xs) < 40:
                return q, False
            x0, y0 = max(xs.min() - 80, 0), max(ys.min() - 80, 0)
            x1, y1 = min(xs.max() + 80, W), min(ys.max() + 80, H)
            D = cv2.distanceTransform((1 - e[y0:y1, x0:x1]).astype(np.uint8), cv2.DIST_L2, 3)
            hidden = np.zeros((H, W), bool)
            if person is not None and person.any(i):
                hidden = cv2.dilate(person[i].astype(np.uint8), k11) > 0
            sel = np.linspace(0, len(xs) - 1, min(len(xs), 400)).astype(int)
            Q = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)

            def cost(q):
                P = warp(q)
                inside = (P[:, 0] >= 3) & (P[:, 0] < W - 3) & (P[:, 1] >= 3) & (P[:, 1] < H - 3)
                Pi = P[inside]
                vis = ~hidden[np.round(Pi[:, 1]).astype(int), np.round(Pi[:, 0]).astype(int)]
                c1 = look(D, Pi[vis], x0, y0).mean() if vis.sum() > 20 else 20.0
                c2 = look(DT, unwarp(Q, q)).mean()
                # a glass resting on the table isn't tipped: without this a crowded frame (a hand SAM didn't label
                # across its foot) can read it as tipped, and the print squashes
                upright = 30.0 if i in resting else 3.0
                return c1 + c2 + upright * abs(q[4] - 1.0)

            # two starting points: the flow's guess, and the outline's own reading of where the glass is
            starts = [np.array(q, float)]
            raw = glass_geometry(m.astype(bool), self.min_ratio)
            if raw is not None and i in self.raw_centre:
                starts.append(np.array([*(np.asarray(self.raw_centre[i]) - c0), raw[3] - a0, raw[2] / r0, q[4]]))
            best = min(starts, key=cost)
            bc = cost(best)
            step = np.array([12.0, 12.0, 0.12, 0.08, 0.08])
            for _ in range(7):
                improved = True
                while improved:
                    improved = False
                    for k in range(5):
                        for sgn in (1, -1):
                            cand = best.copy()
                            cand[k] += sgn * step[k]
                            cand[3] = np.clip(cand[3], 0.5, 1.6)
                            cand[4] = np.clip(cand[4], 0.25, 1.1)
                            cc = cost(cand)
                            if cc < bc - 1e-4:
                                best, bc, improved = cand, cc, True
                step /= 2
            self.fit_cost[i] = (len(xs), float(bc), [float(cost(st)) for st in starts])
            return best, bc < 8.0

        def walk(order, forward):
            cur = fitted[ref].copy()
            for i in order:
                j = i - 1 if forward else i + 1
                M = steps.get(i) if forward else steps.get(j)
                if M is not None and not forward:
                    M = cv2.invertAffineTransform(M)
                guess = cur.copy()
                if M is not None:   # carry the print centre, turn and scale by the glass's flow
                    c = c0 + cur[:2]
                    guess[:2] = M[:, :2] @ c + M[:, 2] - c0
                    guess[2] += np.arctan2(M[1, 0], M[0, 0])
                    guess[3] *= np.hypot(M[0, 0], M[1, 0])
                if self.store.any(i):
                    q, ok = fit_one(i, guess)
                    cur = q if ok else guess
                else:
                    cur = guess
                fitted[i] = cur
        walk(range(ref + 1, self.frames.n), True)
        walk(range(ref - 1, -1, -1), False)
        self.a0, self.r0 = a0, r0
        final = self.settle(fitted, steps, resting, c0)
        for i in self.place:
            tx, ty, th, sc, sa = final[i]
            self.place[i][:5] = [c0[0] + tx, c0[1] + ty, r0 * sc, a0 + th, sa]
        self.fill_gaps(resting)
        self.fitted = fitted
        self.valid = sorted(self.place)
        self.poses = {i: None for i in self.valid}
        self.base = np.array([0.0, 0.0, 5.0])
        self.tex = {}

    def fill_gaps(self, resting, longest=3.0):
        """A still glass that someone walks in front of: SAM still sees a sliver of it, too little to measure, and the
        logo would drop out while the glass is in view. Its pose there is the nearest measured frame's, moved only as
        the camera moves, if the glass was still at that edge of the gap (resting, or travelling under half a radius
        relative to the room over the second before it: the upload gate's test), or the poses carried in from both
        edges agree (within a third of its radius). The print is still drawn only where the glass is seen, so whoever is in
        front covers it. Gaps longer than `longest` seconds stay empty."""
        def carry(q, M):   # a pose moved by a frame-to-frame similarity of the room
            out = np.array(q, float)
            out[:2] = M[:, :2] @ out[:2] + M[:, 2]
            out[2] *= np.hypot(M[0, 0], M[1, 0])
            out[3] += np.arctan2(M[1, 0], M[0, 0])
            return out
        n = max(2, int(round(self.frames.fps)))

        def still(k, before):
            d = [self.rel[j] for j in (range(k - n + 1, k + 1) if before else range(k + 1, k + n + 1)) if j in self.rel]
            return k in resting or (len(d) >= n // 2 and np.linalg.norm(np.sum(d, axis=0)) < 0.5 * self.place[k][2])
        seen = [i for i in range(self.frames.n) if self.store.any(i)]
        gaps, cur = [], []
        for i in seen:
            if i in self.place:
                if cur:
                    gaps.append(cur)
                cur = []
            elif not cur or i == cur[-1] + 1:
                cur.append(i)
            else:
                gaps.append(cur)
                cur = [i]
        if cur:
            gaps.append(cur)
        for gap in gaps:
            a, b = gap[0] - 1, gap[-1] + 1
            if len(gap) > longest * self.frames.fps or (a not in self.place and b not in self.place):
                continue
            fwd, back = {}, {}
            if a in self.place:
                q = self.place[a]
                for i in range(a + 1, b + 1):
                    if self.camera.get(i) is None:
                        break
                    q = fwd[i] = carry(q, self.camera[i])
            if b in self.place:
                q = self.place[b]
                for i in range(b, a, -1):
                    if self.camera.get(i) is None:
                        break
                    q = back[i - 1] = carry(q, cv2.invertAffineTransform(self.camera[i]))
            both = a in self.place and b in self.place
            if both and b in fwd and np.linalg.norm(fwd[b][:2] - self.place[b][:2]) <= 0.33 * self.place[b][2]:
                for i in gap:
                    w = (i - a) / (b - a)
                    if i in fwd and i in back:
                        self.place[i] = (1 - w) * fwd[i] + w * back[i]
                continue
            # the edges disagree: one is measured from a sliver of the glass; trust the edge that saw more of it
            if both and self.store[a].sum() < self.store[b].sum():
                a = -1
            if a in self.place and still(a, True):
                self.place.update({i: fwd[i] for i in gap if i in fwd})
            elif b in self.place and still(b, False):
                self.place.update({i: back[i] for i in gap if i in back})

    def settle(self, fitted, steps, resting, c0, gain=0.3, ramp=6):
        """From per-frame fits to the print's pose, without the few-pixel wobble each fit carries.
        Resting (still with the room for a second or more): one pose for the whole stay, the median of its fits
        brought back to its first frame, then moved only as the camera moves, as the wall posters are.
        Held: carried frame to frame by the glass's own motion, the fit pulling back only a share each frame
        (`gain`), forwards and backwards through the shot and averaged, so the correction does not lag.
        Where a stay begins or ends the two blend over `ramp` frames."""
        def carry(q, M):
            out = np.array(q, float)
            c = c0 + out[:2]
            out[:2] = M[:, :2] @ c + M[:, 2] - c0
            out[2] += np.arctan2(M[1, 0], M[0, 0])
            out[3] *= np.hypot(M[0, 0], M[1, 0])
            return out

        def full(M):
            return np.vstack([M, [0, 0, 1]])
        good_fit = {i for i, c in self.fit_cost.items() if c[1] < 8.0}
        frames = sorted(self.place)
        # where the shape fit fails (a fast, blurred swing), the outline's simpler reading of the print centre still
        # moves with the glass; it sits a fixed offset (in the glass's own axes) from the fit's, measured where both work
        def axes(q):
            a = self.a0 + q[2]
            return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]), self.r0 * q[3]
        offs = []
        for i in good_fit:
            if i in self.raw_centre and i in fitted:
                Rm, r = axes(fitted[i])
                offs.append(Rm.T @ (c0 + fitted[i][:2] - np.asarray(self.raw_centre[i])) / r)
        offset = np.median(offs, axis=0) if offs else np.zeros(2)

        def outline_pose(i, cur):
            q = np.array(cur, float)
            Rm, r = axes(q)
            q[:2] = np.asarray(self.raw_centre[i]) + Rm @ (offset * r) - c0
            return q

        def held(order, forward):
            out, cur = {}, None
            for i in order:
                if cur is not None:
                    M = steps.get(i) if forward else steps.get(i + 1)
                    if M is not None:
                        cur = carry(cur, M if forward else cv2.invertAffineTransform(M))
                    # a glass moving slowly is followed well by its own motion and the fit's wobble would show; the
                    # fit pulls harder the faster it moves (fast swings blur, and the drink and the room seen
                    # through it then mislead the flow)
                    speed = np.linalg.norm(self.rel.get(i if forward else i + 1, np.zeros(2)))
                    k = gain * float(np.clip(speed / 4.0, 0.1, 1.0))
                    if i in good_fit:
                        cur = cur + k * (fitted[i] - cur)
                    elif i in self.raw_centre:
                        cur = cur + 1.5 * k * (outline_pose(i, cur) - cur)
                else:
                    cur = np.array(fitted[i], float)
                out[i] = cur
            return out
        fwd, bwd = held(frames, True), held(frames[::-1], False)
        final = {i: (fwd[i] + bwd[i]) / 2 for i in frames}

        stays, run = [], []
        for i in frames:
            if i in resting and (not run or i == run[-1] + 1):
                run.append(i)
                continue
            if len(run) > 1:
                stays.append(run)
            run = [i] if i in resting else []
        if len(run) > 1:
            stays.append(run)
        for run in stays:
            a = run[0]
            C = {a: np.eye(3)}   # room motion from the stay's first frame
            for i in run[1:]:
                M = self.camera.get(i)
                C[i] = (full(M) if M is not None else np.eye(3)) @ C[i - 1]
            back = [carry(fitted[i], cv2.invertAffineTransform(C[i][:2])) for i in run if i in good_fit]
            if not back:
                continue
            pose = np.median(np.array(back), axis=0)
            for k, i in enumerate(run):
                pinned = carry(pose, C[i][:2])
                w = min(1.0, (k + 1) / ramp, (len(run) - k) / ramp) if a > frames[0] or k >= ramp else 1.0
                if run[-1] == frames[-1] and len(run) - k <= ramp:
                    w = min(1.0, (k + 1) / ramp) if a > frames[0] else 1.0
                final[i] = w * pinned + (1 - w) * final[i]
        return final

    def quality(self):
        """How well the glass was followed: the share of frames whose shape fit was accepted, and, over the frames
        with the whole glass in view, the share where the print's centre lies at least 0.3 radii inside the glass
        (its outline with the gaps fingers leave filled in)."""
        seen = [i for i in self.valid if self.store.any(i)]
        fit_ok = sum(1 for i in seen if i in self.fit_cost and self.fit_cost[i][1] < 8.0) / max(len(seen), 1)
        inside = []
        for i in self.valid:
            if i in self.good:
                cs, _ = cv2.findContours(self.store[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                hull = cv2.convexHull(np.vstack(cs))
                depth = cv2.pointPolygonTest(hull, (float(self.place[i][0]), float(self.place[i][1])), True)
                inside.append(depth >= 0.3 * self.place[i][2])
        return fit_ok, float(np.mean(inside)) if inside else 0.0

    def print_centres(self):
        """Where on the axis the print's centre is, frame by frame. With the whole glass in view, CENTRE of the way
        from rim to base. When the outline is short, a hand, a mouth or the frame edge hides one end: the print is
        measured from the other, the one that moved on smoothly from the frame before (a hidden end jumps)."""
        out, prev, vel = {}, None, np.zeros(2)
        for i in sorted(self.place):
            cx, cy, r, ang, height, bx, by = self.place[i]
            d = np.array([-np.sin(ang), np.cos(ang)])   # down the axis
            L = self.ratio * r
            from_rim = np.array([cx, cy]) + self.centre * L * d
            from_base = np.array([bx, by]) - (1 - self.centre) * L * d
            if abs(height - L) <= 0.1 * L:
                c = (from_rim + from_base) / 2
            elif i in self.clipped:
                c = from_rim
            elif prev is None:
                c = from_base
            else:
                guess = prev + vel
                c = min((from_rim, from_base), key=lambda q: np.linalg.norm(q - guess))
            if prev is not None and i - 1 in out:
                vel = 0.5 * vel + 0.5 * (c - prev)
            out[i], prev = c, c
        return out

    def front(self, i):
        return None

    def art(self, brand):
        if brand not in self.tex:
            a = cv2.imread(str(ROOT / "assets/creatives/gen" / f"{brand}_glass_print.png")).astype(np.float32)
            alpha = np.clip((a.max(2) - 40) / 160, 0, 1)   # white on black: the black is clear glass
            ys, xs = np.nonzero(alpha > 0.1)
            self.tex[brand] = np.ascontiguousarray(alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
        return self.tex[brand]

    def layer(self, i, frame, brand):
        if i not in self.place or not self.store.any(i):
            return None
        cx, cy, r, ang, stretch = self.place[i][:5]   # stretch: the glass tipped towards the camera
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self.store.box(i)
        x0, y0, x1, y1 = max(x0 - 6, 0), max(y0 - 6, 0), min(x1 + 6, w), min(y1 + 6, h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        X, Y = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
        c, s = np.cos(ang), np.sin(ang)
        u = (X - cx) * c + (Y - cy) * s
        t = -(X - cx) * s + (Y - cy) * c
        sn = np.clip(u / r, -1, 1)
        theta = np.arcsin(sn)
        a = self.art(brand)
        arc = 2 * self.half * r                      # the print's width along the glass
        dh = arc * a.shape[0] / a.shape[1] * stretch
        lu = (theta - self.lateral) / (2 * self.half) + 0.5
        v = (t - CURVE * r * np.cos(theta) + dh / 2) / dh
        alpha = cv2.remap(a, (lu * (a.shape[1] - 1)).astype(np.float32), (v * (a.shape[0] - 1)).astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        alpha *= (np.abs(u) < r) * np.clip((1 - np.abs(sn)) * 4, 0, 1)   # fades where the glass turns away
        alpha = cv2.GaussianBlur(alpha, (0, 0), self.sigma) * self.ink_alpha   # as soft as the glass
        alpha = self.motion_blur(alpha, i)
        # only where the glass itself is seen: fingers and straws in front keep the print behind them
        vis = cv2.erode(self.store[i][y0:y1, x0:x1].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        alpha *= cv2.GaussianBlur(vis, (0, 0), 0.8)
        lum = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        # the glass's light falls on the print: blurred over glass pixels only, so fingers beside it don't darken it
        soft = cv2.GaussianBlur(lum * vis, (0, 0), 4) / np.maximum(cv2.GaussianBlur(vis, (0, 0), 4), 1e-3)
        shade = np.clip((soft / max(self.light[i], 1.0)) ** 0.35, 0.8, 1.12)
        sparkle = np.clip(lum - soft - 6, 0, None)[..., None]                # real highlights stay on top, not noise
        colour = 0.9 * shade[..., None] * self.ink + 0.1 * frame[y0:y1, x0:x1].astype(np.float32) + sparkle
        return (x0, y0, x1, y1), colour, np.clip(alpha, 0, 1), np.ones_like(alpha)


class HandleFree:
    """A cup's mask store without its handle, which would otherwise swing the axis, widen the radius and carry the
    logo off the body. Where SAM 3.1 outlined handles ("cup handle", `handles`), they are cut out, with a pixel to
    spare. Otherwise the cup is opened with a disc about a third of its width, wider than a handle's loop and
    narrower than the body (this misses a handle whose hole SAM filled in). The largest piece is kept. Same interface
    as pipeline.maskstore.MaskStore."""

    def __init__(self, store, handles=()):
        self.store, self.handles, self.n, self.cache = store, list(handles), store.n, {}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if i not in self.cache:
            m = self.store[i]
            if self.store.any(i):
                x0, y0, x1, y1 = self.store.box(i)
                crop = m[y0:y1, x0:x1].astype(np.uint8)
                handle = np.zeros_like(crop)
                for h in self.handles:
                    if h.any(i):
                        handle |= h[i][y0:y1, x0:x1].astype(np.uint8)
                if handle.any():
                    o = crop & (1 - cv2.dilate(handle, np.ones((3, 3), np.uint8)))
                else:
                    k = max(9, int(0.3 * np.sqrt(crop.sum()))) | 1
                    o = cv2.morphologyEx(crop, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
                n, lab, st, _ = cv2.connectedComponentsWithStats(o)
                if n > 1:
                    o = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
                if o.sum() < 0.6 * crop.sum():   # it ate the cup itself (a thin sliver, a handle mask over it)
                    o = crop
                m = np.zeros_like(m)
                m[y0:y1, x0:x1] = o.astype(bool)
            self.cache[i] = m
        return self.cache[i]

    def any(self, i):
        return self.store.any(i) and bool(self[i].any())

    def box(self, i):
        ys, xs = np.nonzero(self[i])
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else (0, 0, 0, 0)


class CupDecal(GlassDecal):
    """A logo printed on a coffee cup or mug: the glass's wrap and tracking, on ceramic. The print is opaque and
    takes a colour that reads on the cup (dark on a light cup, white on a dark one), and the handle is left out."""
    min_ratio = 0.8     # cups are squatter than glasses
    ink_alpha = 0.95

    def mask_store(self, store):
        try:   # handles SAM 3.1 outlined in this clip (ingest asks for them whenever there are cups)
            handles = list(object_stores(self.clip, "cup handle", self.frames).values())
        except (FileNotFoundError, OSError):
            handles = []
        return HandleFree(store, handles)

    def __init__(self, clip, spec):
        super().__init__(clip, spec)
        body = float(np.median(list(self.light.values()))) if self.light else 128.0
        self.ink = 40.0 if body > 140 else 230.0

    def design_map(self, good, step_a=0.05, step_t=0.02):
        """Where the cup carries a design of its own, in its wrapped coordinates: (angle round the cup from the
        front, -1.2..1.2 rad) x (height from rim to base, 0..1). From up to 8 frames with the whole cup in view, a
        pixel is design where its lightness departs from the cup's smooth shading (a blur over the body) or its
        colour from the body's own. Returns (design share, seen) grids; unseen cells are not blank."""
        na, nt = int(2.4 / step_a), int(1 / step_t)
        des, tot = np.zeros((na, nt)), np.zeros((na, nt))
        keys = sorted(good)
        for i in keys[:: max(1, len(keys) // 8)][:8]:
            cx, cy, r, ang, height = good[i][:5]
            m = cv2.erode(self.store[i].astype(np.uint8), np.ones((5, 5), np.uint8))
            if m.sum() < 200:
                continue
            lab = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2LAB).astype(np.float32)
            body = m.astype(bool)
            # the cup's own shading: a wide median over the body (outside it filled with the body's median), which
            # a small dark eye or a letter doesn't drag along the way a blur does
            L = lab[..., 0].copy()
            L[~body] = np.median(L[body])
            smooth = cv2.medianBlur(np.clip(L, 0, 255).astype(np.uint8), max(5, int(r / 2)) | 1).astype(np.float32)
            a0, b0 = np.median(lab[..., 1][body]), np.median(lab[..., 2][body])
            design = (np.abs(lab[..., 0] - smooth) > 14) | (np.hypot(lab[..., 1] - a0, lab[..., 2] - b0) > 12)
            design = cv2.dilate((design & body).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            ys, xs = np.nonzero(body)
            c, s = np.cos(ang), np.sin(ang)
            u = (xs - cx) * c + (ys - cy) * s
            t = -(xs - cx) * s + (ys - cy) * c
            theta = np.arcsin(np.clip(u / r, -1, 1))
            tn = (t - CURVE * r * np.cos(theta)) / (self.ratio * r)
            ia, it = ((theta + 1.2) / step_a).astype(int), (tn / step_t).astype(int)
            ok = (ia >= 0) & (ia < na) & (it >= 0) & (it < nt)
            np.add.at(tot, (ia[ok], it[ok]), 1)
            np.add.at(des, (ia[ok], it[ok]), design[ys, xs][ok])
        return np.where(tot > 0, des / np.maximum(tot, 1), 1.0), tot > 0

    def choose_spot(self, good, step_a=0.05, step_t=0.02):
        """The biggest print, nearest the front and the usual height, whose footprint (and a margin round it) is
        blank on every frame looked at. Sized for the tallest logo in the inventory, so it fits whichever brand wins
        the bid. None found: `blank` is False and the upload declines the cup."""
        share, seen = self.design_map(good, step_a, step_t)
        # a real design is design in nearly every sample of its cells, or in a cluster of them; a lone cell of shading
        # noise or grain is neither
        free = seen & (share < 0.5) & (cv2.blur(share.astype(np.float32), (3, 3)) < 0.12)
        aspect = max(a.shape[0] / a.shape[1] for a in (self.art(b) for b in logo_brands()))
        heights = sorted(np.arange(0.12, 0.88, 0.02), key=lambda c: abs(c - CENTRE))
        for scale in np.arange(1.0, 0.49, -0.05):
            half = HALF_ANGLE * scale
            dh = 2 * half * aspect / self.ratio            # the print's height as a share of the cup's
            for lat in (0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3):
                if abs(lat) + half * 1.25 > 0.95:          # it would wrap out of sight
                    continue
                # a clear margin round it: a quarter of its width each side, a third of its height above and below
                a0, a1 = int((lat - 1.25 * half + 1.2) / step_a), int(np.ceil((lat + 1.25 * half + 1.2) / step_a))
                for c in heights:
                    t0, t1 = c - 0.83 * dh, c + 0.83 * dh
                    if t0 < 0.06 or t1 > 0.94:
                        continue
                    if free[a0:a1, int(t0 / step_t):int(np.ceil(t1 / step_t))].all():
                        self.half, self.lateral, self.centre = half, lat, c
                        return
        self.blank = False


def logo_brands():
    """Brands with a printable logo (white on black, assets/creatives/gen/<brand>_glass_print.png)."""
    return sorted(p.name[:-len("_glass_print.png")] for p in (ROOT / "assets/creatives/gen").glob("*_glass_print.png"))
