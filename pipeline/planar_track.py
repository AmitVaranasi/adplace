"""Offline per-plane tracking. H[i] maps original frame i to ref_idx.

Invalid geometry is NaN. 'interpolated' denotes inferred geometry;
diagnostics['measured'] distinguishes it from observations.
Only reduced gray images and rectified keyframes are retained.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def _project(p, h):
    p = np.asarray(p, np.float64).reshape(-1, 2)
    x = np.c_[p, np.ones(len(p))] @ h.T
    if not np.isfinite(x).all() or np.any(np.abs(x[:, 2]) < 1e-8):
        raise ValueError("projective pole")
    return x[:, :2] / x[:, 2:]


def _matrix(a, b):
    return cv2.getPerspectiveTransform(
        np.asarray(a, np.float32), np.asarray(b, np.float32)
    ).astype(np.float64)


def _poly(shape, quad):
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, np.rint(quad * 16).astype(np.int32), 255, shift=4)
    return m


def _inside(p, mask):
    p = np.asarray(p).reshape(-1, 2)
    ij = np.rint(np.nan_to_num(p, nan=-100)).astype(int)
    h, w = mask.shape
    ok = np.isfinite(p).all(1)
    ok &= (ij[:, 0] >= 0) & (ij[:, 0] < w)
    ok &= (ij[:, 1] >= 0) & (ij[:, 1] < h)
    k = np.flatnonzero(ok)
    ok[k] &= mask[ij[k, 1], ij[k, 0]] != 0
    return ok


def _features(gray, mask, count):
    p = cv2.goodFeaturesToTrack(
        gray, count, .01, 5, mask=mask, blockSize=5
    )
    return np.empty((0, 2), np.float32) if p is None else p.reshape(-1, 2)


def _flow(a, b, p, allowed_a, allowed_b, guess, fb):
    if len(p) == 0:
        return p, p
    p = np.asarray(p, np.float32).reshape(-1, 1, 2)
    q0 = np.asarray(guess, np.float32).reshape(-1, 1, 2)
    kw = dict(
        winSize=(21, 21), maxLevel=4, criteria=(3, 35, .001),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
    )
    q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, q0.copy(), **kw)
    if q is None:
        return np.empty((0, 2)), np.empty((0, 2))
    back, sb, _ = cv2.calcOpticalFlowPyrLK(b, a, q, p.copy(), **kw)
    if back is None:
        return np.empty((0, 2)), np.empty((0, 2))
    x, y = p.reshape(-1, 2), q.reshape(-1, 2)
    keep = (st.ravel() != 0) & (sb.ravel() != 0)
    keep &= np.linalg.norm(back.reshape(-1, 2) - x, axis=1) < fb
    keep &= _inside(x, allowed_a) & _inside(y, allowed_b)
    return x[keep], y[keep]


def _fit(x, y, cq, threshold, minimum, prefer_interior=False):
    """Reject degenerate candidates at every refinement/projection stage."""
    try:
        result = _fit_candidate(
            x, y, cq, threshold, minimum, prefer_interior
        )
        if result is None:
            return None
        h = result[0]
        den = np.c_[cq, np.ones(len(cq))] @ h[2]
        if (
            not np.isfinite(h).all()
            or not np.isfinite(result[1:]).all()
            or den.min() * den.max() <= 0
            or np.linalg.matrix_rank(h) < 3
        ):
            return None
        return result
    except (ValueError, cv2.error, np.linalg.LinAlgError, FloatingPointError):
        return None


def _fit_candidate(x, y, cq, threshold, minimum, prefer_interior=False):
    if len(x) < minimum:
        return None
    if prefer_interior:
        core = (
            (x[:, 0] > cq[0, 0]+3) & (x[:, 0] < cq[2, 0]-3)
            & (x[:, 1] > cq[0, 1]+3) & (x[:, 1] < cq[2, 1]-3)
        )
        anchor = _fit(x[core], y[core], cq, threshold, minimum)
        if anchor is not None:
            # Context joins only if it agrees with the printed plane.
            good = np.linalg.norm(_project(x, anchor[0])-y, axis=1) <= threshold
            return _fit(x[good], y[good], cq, threshold, minimum)

    cv2.setRNGSeed(0)
    h, good = cv2.findHomography(
        x, y, cv2.RANSAC, threshold, maxIters=2000, confidence=.999
    )
    if h is None or good is None:
        return None
    keep = good.ravel() != 0
    if keep.sum() < minimum or keep.mean() < .45:
        return None
    x, y = x[keep], y[keep]
    for _ in range(2):
        e = np.linalg.norm(_project(x, h) - y, axis=1)
        keep = e <= min(threshold, max(.15, 3 * np.median(e)))
        x, y = x[keep], y[keep]
        if len(x) < minimum:
            return None
        h, _ = cv2.findHomography(x, y, 0)
        if h is None:
            return None

    area = cv2.contourArea(cv2.convexHull(x.astype(np.float32)))
    if area < .06 * cv2.contourArea(cq.astype(np.float32)):
        return None
    try:
        q = _project(cq, h)
    except ValueError:
        return None
    if not cv2.isContourConvex(q.astype(np.float32)):
        return None

    e = np.linalg.norm(_project(x, h) - y, axis=1)
    rms = float(np.sqrt(np.mean(e**2)))
    # Propagate correspondence variance to corner uncertainty.
    center, scale = x.mean(0), max(1., np.std(x))
    z = np.array([
        [scale, 0, center[0]], [0, scale, center[1]], [0, 0, 1.]
    ])
    hn = h @ z
    hn /= hn[2, 2]

    def jac(p):
        p = (p - center) / scale
        u, v = _project(p, hn).T
        xx, yy = p.T
        d = p @ hn[2, :2] + 1
        j = np.zeros((len(p), 2, 8))
        j[:, 0, :3] = np.c_[xx, yy, np.ones(len(p))] / d[:, None]
        j[:, 1, 3:6] = j[:, 0, :3]
        j[:, 0, 6:] = -np.c_[u * xx, u * yy] / d[:, None]
        j[:, 1, 6:] = -np.c_[v * xx, v * yy] / d[:, None]
        return j.reshape(-1, 8)

    j, jc = jac(x), jac(cq)
    cov = np.linalg.pinv(j.T @ j) * max(.03**2, rms**2 / 2)
    sigma = float(np.sqrt(max(0., np.trace(jc @ cov @ jc.T) / 8)))
    return h / h[2, 2], rms, len(x), max(.03, sigma)


def _visibility(q, shape, occ):
    h, w = shape
    if not np.isfinite(q).all():
        return 0.
    area = cv2.contourArea(q.astype(np.float32), oriented=True)
    if area <= 4:
        return 0.
    bounds = np.array([
        [-.5, -.5], [w-.5, -.5], [w-.5, h-.5], [-.5, h-.5]
    ], np.float32)
    clipped, _ = cv2.intersectConvexConvex(q.astype(np.float32), bounds)
    m = _poly(shape, q) != 0
    clear = 1. if not m.any() else 1 - np.count_nonzero(m & occ) / m.sum()
    return float(np.clip(clipped / area * clear, 0, 1))


def _iou(q, surface, occ):
    if surface is None or not surface.any():
        return np.nan
    a = (_poly(surface.shape, q) != 0) & ~occ
    b = surface & ~occ
    return np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b))


def _smooth(q, sigma, ref):
    """Local empirical-Bayes acceleration penalty; corrections <= 1 sigma.

    Measurement variance comes from correspondence Jacobians. Process variance
    is observed acceleration energy minus its measurement-noise contribution.
    Contiguous segments are independent; nothing crosses a long gap.
    """
    out = q.copy()
    shifts, ratios = np.zeros(len(q)), np.zeros(len(q))
    valid = np.isfinite(q).all((1, 2))
    ix = np.flatnonzero(valid)
    if not len(ix):
        return out, shifts, ratios

    for run in np.split(ix, np.flatnonzero(np.diff(ix) > 1) + 1):
        n = len(run)
        if n < 3:
            continue
        y = q[run].reshape(n, 8)
        var = np.maximum(sigma[run], .03)**2
        noise = var[:-2] + 4 * var[1:-1] + var[2:]
        acc = np.mean(np.diff(y, n=2, axis=0)**2, axis=1)
        process = np.maximum(acc - noise, noise / 6)
        d = diags(
            [np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)],
            [0, 1, 2], shape=(n-2, n),
        )
        weight = 1 / var
        weight[run == ref] = 1e12
        system = diags(weight) + d.T @ diags(1 / process) @ d
        fit = spsolve(
            system.tocsc(), y * weight[:, None]
        ).reshape(n, 4, 2)

        delta = fit - q[run]
        distance = np.linalg.norm(delta, axis=2).max(1)
        delta *= np.minimum(
            1., sigma[run] / np.maximum(distance, 1e-12)
        )[:, None, None]
        out[run] += delta
        shifts[run] = np.linalg.norm(delta, axis=2).max(1)
        ratios[run[1:-1]] = var[1:-1] / process

    out[ref] = q[ref]
    return out, shifts, ratios


def track_plane(frames, ref_idx, ref_quad, *, surface_masks=None,
                occluder_masks=None, dynamic_content=False, context_px=None,
                **options):
    """Track both directions; positions and residuals use original pixels.

    Options: max_width=960, max_features=350, min_inliers=10, ransac_px=1.5,
    fb_px=.8, min_visible=.04, min_iou=.35, max_gap=6, max_keyframes=8,
    smooth=True, mask_fallback=True. Unknown options raise.

    Context must really be coplanar. Empty/missing masks do not veto good
    appearance evidence. Mask fallback requires four unclipped edges, low
    occlusion, and proximity to a motion prediction.
    """
    cfg = dict(
        max_width=960, max_features=350, min_inliers=10,
        ransac_px=1.5, fb_px=.8, min_visible=.04, min_iou=.35,
        max_gap=6, max_keyframes=8, smooth=True, mask_fallback=True,
    )
    unknown = options.keys() - cfg.keys()
    if unknown:
        raise TypeError("unknown options: " + str(sorted(unknown)))
    cfg.update(options)

    for name in (
        "max_width", "max_features", "min_inliers", "max_keyframes", "max_gap"
    ):
        if (
            not isinstance(cfg[name], (int, np.integer))
            or cfg[name] < (0 if name == "max_gap" else 1)
        ):
            raise ValueError("invalid " + name)
    if cfg["min_inliers"] < 4 or cfg["max_features"] < cfg["min_inliers"]:
        raise ValueError("need at least four features/inliers")
    for name in ("ransac_px", "fb_px", "min_visible", "min_iou"):
        if not np.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError("invalid " + name)
    if cfg["min_visible"] > 1 or cfg["min_iou"] > 1:
        raise ValueError("fractions must be <= 1")

    rq = np.asarray(ref_quad, np.float64)
    if rq.shape != (4, 2) or not np.isfinite(rq).all():
        raise ValueError("ref_quad must be finite (4,2)")
    if (
        not cv2.isContourConvex(rq.astype(np.float32))
        or cv2.contourArea(rq.astype(np.float32), oriented=True) <= 4
    ):
        raise ValueError("ref_quad must have TL,TR,BR,BL winding")

    grays, original = [], None
    for f in frames:
        f = np.asarray(f)
        if f.dtype != np.uint8 or f.ndim != 3 or f.shape[2] != 3:
            raise ValueError("frames must be BGR uint8")
        if original is None:
            original = f.shape[:2]
            oh, ow = original
            sw = min(ow, cfg["max_width"])
            sh = max(1, round(oh * sw / ow))
            sx, sy = sw / ow, sh / oh
            # OpenCV resize transforms pixel centres, including this offset.
            S = np.array([
                [sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1.]
            ])
        if f.shape[:2] != original:
            raise ValueError("frame sizes differ")
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        grays.append(cv2.resize(g, (sw, sh), interpolation=cv2.INTER_AREA))

    n = len(grays)
    if not isinstance(ref_idx, (int, np.integer)) or not 0 <= ref_idx < n:
        raise ValueError("ref_idx is out of range")
    shape, scale = (sh, sw), min(sx, sy)

    def masks(i, margin=True):
        result = []
        for getter in (surface_masks, occluder_masks):
            m = None if getter is None else getter(i)
            if m is not None:
                m = np.asarray(m)
                if m.shape != original or m.dtype != np.bool_:
                    raise ValueError("masks must be full-frame bool arrays")
                m = cv2.resize(
                    m.astype(np.uint8), (sw, sh),
                    interpolation=cv2.INTER_NEAREST,
                ) != 0
            result.append(m)
        surf, occ = result
        occ = (
            np.zeros(shape, np.uint8)
            if occ is None else occ.astype(np.uint8)
        )
        r = max(1, round(3 * scale))
        if margin:
            occ = cv2.dilate(occ, np.ones((2*r+1, 2*r+1), np.uint8))
        return surf, occ != 0

    small_ref = _project(rq, S)
    lengths = np.linalg.norm(
        np.roll(small_ref, -1, axis=0) - small_ref, axis=1
    )
    pw = .5 * (lengths[0]+lengths[2])
    ph = .5 * (lengths[1]+lengths[3])
    context = (
        .35 * min(pw, ph)
        if context_px is None else float(context_px) * scale
    )
    if not np.isfinite(context) or context < 0:
        raise ValueError("context_px must be finite and nonnegative")

    pad = max(2, int(np.ceil(context)))
    cq = np.array([
        [pad, pad], [pad+pw, pad],
        [pad+pw, pad+ph], [pad, pad+ph],
    ], float)
    size = (
        int(np.ceil(pw))+2*pad+1, int(np.ceil(ph))+2*pad+1
    )
    ashape = size[::-1]
    interior = _poly(ashape, cq)
    support = np.full(ashape, 255, np.uint8)
    support[:3] = support[-3:] = 0
    support[:, :3] = support[:, -3:] = 0
    if dynamic_content:
        support[
            cv2.dilate(interior, np.ones((23, 23), np.uint8)) != 0
        ] = 0

    poses = np.full((n, 3, 3), np.nan)
    poses[ref_idx] = _matrix(cq, small_ref)
    quads = np.full((n, 4, 2), np.nan)
    status = ["lost"] * n
    rms = np.full(n, np.nan)
    sigmas = np.ones(n)*3
    counts = np.zeros(n, int)
    ious = np.full(n, np.nan)
    measured = np.zeros(n, bool)
    visible = np.zeros(n)
    sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=.02)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def atlas(i, pose, surf, occ):
        inv = np.linalg.inv(pose)
        image = cv2.warpPerspective(grays[i], inv, size)
        clear = cv2.warpPerspective(
            (~occ).astype(np.uint8)*255, inv, size,
            flags=cv2.INTER_NEAREST,
        )
        allowed = support & clear
        if surf is not None and surf.any() and not dynamic_content:
            sm = cv2.warpPerspective(
                surf.astype(np.uint8), inv, size,
                flags=cv2.INTER_NEAREST,
            )
            sm = cv2.erode(sm, np.ones((3, 3), np.uint8))
            allowed[(interior != 0) & (sm == 0)] = 0
        if dynamic_content:
            image[interior != 0] = 127
        return image, allowed

    def signature(pose):
        q = _project(cq, pose)
        e = np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)
        return np.log(np.maximum(e, 1) / lengths)

    keys = []

    def mask_seed(surf, occ):
        if surf is None or not surf.any():
            return None
        border = (
            surf[0].any() or surf[-1].any()
            or surf[:, 0].any() or surf[:, -1].any()
        )
        if border or np.count_nonzero(surf & occ) > .1*surf.sum():
            return None
        from pipeline.mask_track import _boundary
        fit = _boundary(surf, occ)
        return fit[0] if fit is not None and fit[2] > .6 else None

    def blur_template(image, vector):
        vector = np.asarray(vector) * .5
        if np.linalg.norm(vector) < 2:
            return image
        radius = int(np.ceil(np.max(np.abs(vector))/2))+1
        kernel = np.zeros((2*radius+1, 2*radius+1), np.float32)
        a = np.rint(radius-vector/2).astype(int)
        b = np.rint(radius+vector/2).astype(int)
        cv2.line(kernel, tuple(a), tuple(b), 1., 1, cv2.LINE_AA)
        kernel /= kernel.sum()
        return cv2.filter2D(
            image, -1, kernel, borderType=cv2.BORDER_REFLECT_101
        )

    def add_key(i, pose, surf, occ):
        image, allowed = atlas(i, pose, surf, occ)
        p = _features(image, allowed, cfg["max_features"])
        kp, desc = sift.detectAndCompute(image, allowed)
        if dynamic_content and kp:
            # Descriptor support grows with keypoint scale.
            dist = cv2.distanceTransform(
                (allowed != 0).astype(np.uint8), cv2.DIST_L2, 3
            )
            take = [
                k for k, v in enumerate(kp)
                if dist[int(round(v.pt[1])), int(round(v.pt[0]))]
                > 2 * v.size
            ]
            kp = [kp[k] for k in take]
            desc = None if desc is None or not take else desc[take]
        keys.append(dict(
            image=image, allowed=allowed, points=p,
            kp=kp, desc=desc, signature=signature(pose),
        ))

    def accepted(fit, surf, occ):
        if fit is None:
            return False
        h = fit[0]
        try:
            q = _project(cq, h)
        except ValueError:
            return False
        den = np.c_[cq, np.ones(4)] @ h[2]
        if den.min() * den.max() <= 0:
            return False
        if cv2.contourArea(q.astype(np.float32), oriented=True) <= 4:
            return False
        if np.max(np.abs(q)) > 20 * max(shape):
            return False
        vis = _visibility(q, shape, occ)
        iou = _iou(q, surf, occ)
        return (
            vis >= cfg["min_visible"]
            and (np.isnan(iou) or iou >= cfg["min_iou"])
        )

    surf, occ = masks(ref_idx)
    visible[ref_idx] = _visibility(small_ref, shape, occ)
    if visible[ref_idx] < cfg["min_visible"]:
        raise ValueError("reference plane has insufficient unoccluded area")
    add_key(ref_idx, poses[ref_idx], surf, occ)
    quads[ref_idx], status[ref_idx] = small_ref, "reference"
    rms[ref_idx], sigmas[ref_idx], measured[ref_idx] = 0, .03*scale, True

    for direction in (1, -1):
        previous, before = ref_idx, None
        for i in range(
            ref_idx+direction, n if direction > 0 else -1, direction
        ):
            surf, occ = masks(i)
            fit, seed = None, None
            if previous == i-direction:
                seed = poses[previous].copy()
                if before == previous-direction:
                    seed = (
                        poses[previous] @ np.linalg.inv(poses[before]) @ seed
                    )
                try:
                    _project(cq, seed)
                    ps, po = masks(previous)
                    pa = cv2.warpPerspective(
                        support, poses[previous], (sw, sh),
                        flags=cv2.INTER_NEAREST,
                    )
                    pa[po] = 0
                    if ps is not None and ps.any() and not dynamic_content:
                        inside = _poly(shape, quads[previous]) != 0
                        pa[inside & ~ps] = 0
                    pb = (~occ).astype(np.uint8)*255
                    p = _features(
                        grays[previous], pa, cfg["max_features"]
                    )
                    x, y = _flow(
                        grays[previous], grays[i], p, pa, pb,
                        _project(
                            p, seed @ np.linalg.inv(poses[previous])
                        ),
                        cfg["fb_px"]*scale,
                    )
                    candidate = _fit(
                        _project(x, np.linalg.inv(poses[previous])),
                        y, cq, cfg["ransac_px"]*scale,
                        cfg["min_inliers"], not dynamic_content,
                    )
                    if accepted(candidate, surf, occ):
                        seed = candidate[0]
                except (ValueError, np.linalg.LinAlgError, cv2.error):
                    seed = None

            motion_prediction = (
                None if seed is None else _project(cq, seed)
            )
            coarse = mask_seed(surf, occ)
            if coarse is not None:
                # SAM seeds an appearance search; it does not replace the fit.
                if seed is not None:
                    predicted = _project(cq, seed)
                    coarse = min(
                        (np.roll(coarse, k, axis=0) for k in range(4)),
                        key=lambda a: np.sum((a-predicted)**2),
                    )
                seed = _matrix(cq, coarse)

            velocity = np.zeros(2)
            if seed is not None and before is not None:
                if previous == i-direction:
                    velocity = (
                        cq - _project(
                            cq, np.linalg.inv(seed) @ poses[previous]
                        )
                    ).mean(0)
                else:
                    velocity = (
                        _project(
                            cq, np.linalg.inv(poses[before]) @ poses[previous]
                        ) - cq
                    ).mean(0)

            # Absolute template alignment: try the original anchor first.
            if seed is not None:
                for key in keys:
                    im, allowed = atlas(i, seed, surf, occ)
                    template = blur_template(key["image"], velocity)
                    x, y = _flow(
                        template, im, key["points"], key["allowed"],
                        allowed, key["points"], cfg["fb_px"],
                    )
                    candidate = _fit(
                        x, _project(y, seed), cq,
                        cfg["ransac_px"]*scale, cfg["min_inliers"],
                        not dynamic_content,
                    )
                    if accepted(candidate, surf, occ):
                        fit = candidate
                        break

            label = (
                "anchored" if previous == i-direction else "reacquired"
            )
            if fit is None:
                # Global search requires no temporal prediction.
                allowed = (~occ).astype(np.uint8)*255
                if dynamic_content and surf is not None:
                    allowed[
                        cv2.dilate(
                            surf.astype(np.uint8),
                            np.ones((23, 23), np.uint8),
                        ) != 0
                    ] = 0
                kp, desc = sift.detectAndCompute(grays[i], allowed)
                if desc is not None and len(desc) >= 2:
                    for key in keys:
                        if key["desc"] is None:
                            continue
                        pairs = matcher.knnMatch(key["desc"], desc, k=2)
                        ms = [
                            a for pair in pairs if len(pair) == 2
                            for a, b in [pair]
                            if a.distance < .7*b.distance
                        ]
                        x = np.array([
                            key["kp"][m.queryIdx].pt for m in ms
                        ], np.float32).reshape(-1, 2)
                        y = np.array([
                            kp[m.trainIdx].pt for m in ms
                        ], np.float32).reshape(-1, 2)
                        candidate = _fit(
                            x, y, cq, cfg["ransac_px"]*scale,
                            cfg["min_inliers"], not dynamic_content,
                        )
                        if not accepted(candidate, surf, occ):
                            continue
                        im, am = atlas(i, candidate[0], surf, occ)
                        xx, yy = _flow(
                            key["image"], im, key["points"],
                            key["allowed"], am, key["points"], cfg["fb_px"],
                        )
                        refined = _fit(
                            xx, _project(yy, candidate[0]), cq,
                            cfg["ransac_px"]*scale, cfg["min_inliers"],
                            not dynamic_content,
                        )
                        fit = (
                            refined if accepted(refined, surf, occ)
                            else candidate
                        )
                        label = "reacquired"
                        break

            if (
                fit is None and cfg["mask_fallback"]
                and motion_prediction is not None
                and surf is not None and surf.any()
            ):
                # Masks alone cannot establish identity after a long loss.
                border = (
                    surf[0].any() or surf[-1].any()
                    or surf[:, 0].any() or surf[:, -1].any()
                )
                if not border and np.count_nonzero(surf & occ) < .1*surf.sum():
                    from pipeline.mask_track import _boundary
                    boundary = _boundary(surf, occ)
                    if boundary is not None:
                        q, _, quality = boundary
                        prediction = motion_prediction
                        q = min(
                            (np.roll(q, k, axis=0) for k in range(4)),
                            key=lambda a: np.sum((a-prediction)**2),
                        )
                        distance = np.linalg.norm(
                            q-prediction, axis=1
                        ).max()
                        candidate = (
                            _matrix(cq, q), np.nan, 0, 2.5*scale
                        )
                        if (
                            quality > .6
                            and distance < .12*min(pw, ph)
                            and accepted(candidate, surf, occ)
                        ):
                            fit, label = candidate, "mask"

            if fit is None:
                continue
            poses[i], rms[i], counts[i], sigmas[i] = fit
            quads[i] = _project(cq, poses[i])
            ious[i] = _iou(quads[i], surf, occ)
            visible[i] = _visibility(quads[i], shape, occ)
            measured[i], status[i] = True, label
            sig = signature(poses[i])
            if (
                label != "mask" and visible[i] > .6
                and counts[i] >= 2*cfg["min_inliers"]
                and sigmas[i] < .5*scale
                and len(keys) < cfg["max_keyframes"]
            ):
                change = min(
                    np.max(np.abs(sig-k["signature"])) for k in keys
                )
                if change > np.log(1.35):
                    add_key(i, poses[i], surf, occ)
            before, previous = previous, i

    invS = np.linalg.inv(S)
    for i in np.flatnonzero(measured):
        quads[i] = _project(quads[i], invS)
    sigmas /= scale
    rms /= scale

    missing = np.flatnonzero(~measured)
    if len(missing):
        runs = np.split(
            missing, np.flatnonzero(np.diff(missing) > 1)+1
        )
        for run in runs:
            a, b = run[0]-1, run[-1]+1
            if a < 0 or b >= n or len(run) > cfg["max_gap"]:
                continue
            va = (
                quads[a]-quads[a-1]
                if a > 0 and measured[a-1]
                else (quads[b]-quads[a])/(b-a)
            )
            vb = (
                quads[b+1]-quads[b]
                if b+1 < n and measured[b+1]
                else (quads[b]-quads[a])/(b-a)
            )
            for i in run:
                t = (i-a)/(b-a)
                q = (
                    (2*t**3-3*t*t+1)*quads[a]
                    + (t**3-2*t*t+t)*(b-a)*va
                    + (-2*t**3+3*t*t)*quads[b]
                    + (t**3-t*t)*(b-a)*vb
                )
                surf, occ = masks(i)
                qs = _project(q, S)
                if not cv2.isContourConvex(qs.astype(np.float32)):
                    continue
                v = _visibility(qs, shape, occ)
                if (
                    v < cfg["min_visible"]
                    or (surf is not None and not surf.any())
                ):
                    continue
                iou = _iou(qs, surf, occ)
                if not np.isnan(iou) and iou < cfg["min_iou"]:
                    continue
                quads[i] = q
                visible[i], status[i], sigmas[i] = v, "interpolated", 3.

    if cfg["smooth"]:
        quads, shifts, strengths = _smooth(quads, sigmas, ref_idx)
    else:
        shifts, strengths = np.zeros(n), np.zeros(n)

    valid = np.isfinite(quads).all((1, 2))
    H = np.full((n, 3, 3), np.nan)
    for i in np.flatnonzero(valid):
        qs = _project(quads[i], S)
        _, occ = masks(i, margin=False)
        visible[i] = _visibility(qs, shape, occ)
        if (
            visible[i] < cfg["min_visible"]
            or not cv2.isContourConvex(qs.astype(np.float32))
        ):
            valid[i], status[i], quads[i] = False, "lost", np.nan
            continue
        H[i] = _matrix(quads[i], rq)
        H[i] /= H[i, 2, 2]

    quads[ref_idx], H[ref_idx] = rq, np.eye(3)
    return dict(
        quads=quads, H=H, valid=valid, visible=visible, status=status,
        diagnostics=dict(
            residual_px=rms, corner_sigma_px=sigmas,
            inliers=counts, mask_iou=ious, measured=measured,
            smoothing_shift_px=shifts, smoothing_lambda=strengths,
            keyframes=len(keys),
        ),
    )


def track_screen(frames, ref_idx, ref_quad, **options):
    """Screen-specific tracker; appearance tracking above is unchanged."""
    from pipeline.screen_track import track_screen as implementation
    return implementation(frames, ref_idx, ref_quad, **options)
