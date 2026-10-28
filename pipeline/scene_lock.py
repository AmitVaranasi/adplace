"""Offline scene locking in full-resolution pixel coordinates.

H[t] maps frame t TO the reference. Estimate motion once per clip, excluding
the union of every surface and known foreground mask, then reuse it for slots.
A background homography assumes a common plane or camera rotation/zoom.
Independently moving objects need their own motion model. Dissolves can contain
two incompatible views and should be split before estimating camera motion.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def _project(points, matrix):
    p = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate((p, np.ones((*p.shape[:-1], 1))), axis=-1)
    q = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    if np.any(np.abs(q[..., 2]) < 1e-10):
        raise ValueError("homography projects points to infinity")
    return q[..., :2] / q[..., 2:]


def _gray(frame):
    a = np.asarray(frame)
    if a.dtype != np.uint8 or a.ndim not in (2, 3):
        raise ValueError("frames must be uint8 gray or BGR images")
    if a.ndim == 3:
        if a.shape[2] != 3:
            raise ValueError("color frames must be BGR")
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(a)


def _masks(values, n, shape):
    if values is None:
        return [None] * n
    values = list(values)
    if len(values) != n:
        raise ValueError("masks must have one entry per frame")
    result = []
    for m in values:
        if m is None:
            result.append(None)
        else:
            m = np.asarray(m, dtype=bool)
            if m.shape != shape:
                raise ValueError("mask size must equal frame size")
            result.append(m)
    return result


def _allowed(mask, shape, margin):
    excluded = np.zeros(shape, np.uint8) if mask is None else mask.astype(np.uint8)
    if margin:
        excluded = cv2.dilate(
            excluded, np.ones((2 * margin + 1,) * 2, np.uint8)
        )
    allowed = 255 * (1 - excluded)
    allowed[:margin or 1] = 0
    allowed[-(margin or 1):] = 0
    allowed[:, :margin or 1] = 0
    allowed[:, -(margin or 1):] = 0
    return allowed


def _inside(points, allowed):
    p = np.asarray(points).reshape(-1, 2)
    finite = np.isfinite(p).all(axis=1)
    ij = np.rint(np.where(np.isfinite(p), p, -1)).astype(int)
    h, w = allowed.shape
    good = finite & (ij[:, 0] >= 0) & (ij[:, 0] < w)
    good &= (ij[:, 1] >= 0) & (ij[:, 1] < h)
    idx = np.flatnonzero(good)
    good[idx] &= allowed[ij[idx, 1], ij[idx, 0]] != 0
    return good


def _fit(src, dst, shape, ransac_px, min_inliers):
    if len(src) < min_inliers:
        return None
    matrix, keep = cv2.findHomography(
        src, dst, cv2.RANSAC, ransac_px,
        maxIters=3000, confidence=0.999,
    )
    if matrix is None or keep is None:
        return None
    keep = keep.ravel().astype(bool)
    if keep.sum() < min_inliers or keep.mean() < 0.35:
        return None
    src, dst = src[keep], dst[keep]
    hull = cv2.convexHull(src.astype(np.float32))
    if cv2.contourArea(hull) < 0.02 * np.prod(shape):
        return None

    # Robust trimming followed by a subpixel least-squares refit.
    for _ in range(3):
        residual = np.linalg.norm(_project(src, matrix) - dst, axis=1)
        cutoff = min(ransac_px, max(0.15, 3 * np.median(residual)))
        take = residual <= cutoff
        if take.sum() < min_inliers:
            return None
        matrix, _ = cv2.findHomography(src[take], dst[take], 0)
        if matrix is None:
            return None

    matrix = matrix / matrix[2, 2]
    h, w = shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    warped = _project(corners, matrix)
    area = cv2.contourArea(warped.astype(np.float32), oriented=True)
    if not np.isfinite(matrix).all() or not 0.2 * h * w < area < 5 * h * w:
        return None
    return matrix


def _flow_pair(
    ref, frame, points, ref_allowed, allowed, ransac_px, min_inliers,
    initial=None,
):
    if points is None:
        return None
    if initial is not None:
        # Prewarping removes large scale/rotation differences before KLT.
        # Both the seed and residual still match this frame directly to ref.
        h, w = ref.shape
        flags = cv2.WARP_INVERSE_MAP
        aligned = cv2.warpPerspective(
            frame, initial, (w, h), flags=cv2.INTER_LINEAR | flags
        )
        aligned_allowed = cv2.warpPerspective(
            allowed, initial, (w, h), flags=cv2.INTER_NEAREST | flags
        )
        residual = _flow_pair(
            ref, aligned, points, ref_allowed, aligned_allowed,
            ransac_px, min_inliers,
        )
        return None if residual is None else initial @ residual

    target, status, _ = cv2.calcOpticalFlowPyrLK(
        ref, frame, points, None, winSize=(21, 21), maxLevel=4,
        criteria=(
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.001
        ),
    )
    if target is None:
        return None
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(
        frame, ref, target, points.copy(),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        winSize=(21, 21), maxLevel=4,
        criteria=(
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.001
        ),
    )
    if back is None:
        return None
    src, dst = points.reshape(-1, 2), target.reshape(-1, 2)
    good = status.ravel().astype(bool) & back_status.ravel().astype(bool)
    good &= np.linalg.norm(back.reshape(-1, 2) - src, axis=1) < 0.4
    good &= _inside(src, ref_allowed) & _inside(dst, allowed)
    return _fit(src[good], dst[good], ref.shape, ransac_px, min_inliers)


def _orb_pair(ref, frame, ref_allowed, allowed, ransac_px, min_inliers):
    orb = cv2.ORB_create(nfeatures=6000, fastThreshold=7)
    ka, da = orb.detectAndCompute(ref, ref_allowed)
    kb, db = orb.detectAndCompute(frame, allowed)
    if da is None or db is None or len(db) < 2:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    matches = [
        a for pair in pairs if len(pair) == 2
        for a, b in [pair] if a.distance < 0.7 * b.distance
    ]
    src = np.array(
        [ka[m.queryIdx].pt for m in matches], np.float32
    ).reshape(-1, 2)
    dst = np.array(
        [kb[m.trainIdx].pt for m in matches], np.float32
    ).reshape(-1, 2)
    good = _inside(src, ref_allowed) & _inside(dst, allowed)
    return _fit(src[good], dst[good], ref.shape, ransac_px, min_inliers)


def _smooth(values, weights, strength):
    n = len(values)
    if n < 3 or strength == 0:
        return values.copy()
    d = diags(
        [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
        [0, 1, 2], shape=(n - 2, n),
    )
    a = diags(weights) + strength * (d.T @ d)
    return spsolve(
        a.tocsc(), values.reshape(n, -1) * weights[:, None]
    ).reshape(values.shape)


def camera_motion(
    frames, exclude_masks=None, *, ref_idx=None,
    max_features=2000, mask_margin=12, ransac_px=1.0,
    min_inliers=16, static_threshold=0.3, smoothness=4.0, max_gap=3,
):
    """Return float64 3x3 frame-to-reference homographies.

    Direct-reference KLT with forward/backward checks; direct ORB supplies
    a prewarp when necessary. No adjacent-frame accumulation.

    Missing interior runs up to max_gap are interpolated. Longer failures raise
    instead of silently claiming that the scene is static.

    smoothness penalizes second differences of camera-projected control points.
    Zero preserves unfiltered camera motion. The static threshold measures the
    whole clip's displacement range after denoising, not per-frame velocity.

    exclude_masks must include every surface and known moving foreground.
    """
    frames = [_gray(f) for f in frames]
    n = len(frames)
    if n == 0:
        raise ValueError("at least one frame is required")
    shape = frames[0].shape
    if any(f.shape != shape for f in frames):
        raise ValueError("frame sizes differ")
    if ref_idx is None:
        ref_idx = n // 2
    if not isinstance(ref_idx, (int, np.integer)) or not 0 <= ref_idx < n:
        raise ValueError("ref_idx is out of range")
    if (
        not np.isfinite([ransac_px, static_threshold, smoothness]).all()
        or ransac_px <= 0 or static_threshold < 0 or smoothness < 0
        or max_features < min_inliers or min_inliers < 4
        or not isinstance(mask_margin, (int, np.integer)) or mask_margin < 0
        or not isinstance(max_gap, (int, np.integer)) or max_gap < 0
    ):
        raise ValueError("invalid camera estimation parameters")

    masks = _masks(exclude_masks, n, shape)
    ref_allowed = _allowed(masks[ref_idx], shape, mask_margin)
    points = cv2.goodFeaturesToTrack(
        frames[ref_idx], maxCorners=max_features, qualityLevel=0.01,
        minDistance=8, mask=ref_allowed, blockSize=5,
    )
    h, w = shape
    grid = np.array([
        (x, y)
        for y in np.linspace(0.1 * h, 0.9 * h, 3)
        for x in np.linspace(0.1 * w, 0.9 * w, 3)
    ], float)
    positions = np.full((n, 9, 2), np.nan)
    positions[ref_idx] = grid

    for t, frame in enumerate(frames):
        if t == ref_idx:
            continue
        allowed = _allowed(masks[t], shape, mask_margin)
        matrix = _flow_pair(
            frames[ref_idx], frame, points, ref_allowed,
            allowed, ransac_px, min_inliers,
        )
        if matrix is None:
            seed = _orb_pair(
                frames[ref_idx], frame, ref_allowed,
                allowed, max(3.0, ransac_px), min_inliers,
            )
            if seed is not None:
                matrix = _flow_pair(
                    frames[ref_idx], frame, points, ref_allowed,
                    allowed, ransac_px, min_inliers, initial=seed,
                )
        if matrix is not None:
            positions[t] = _project(grid, matrix)

    valid = np.isfinite(positions).all(axis=(1, 2))
    missing = np.flatnonzero(~valid)
    if len(missing):
        runs = np.split(
            missing, np.flatnonzero(np.diff(missing) > 1) + 1
        )
        if any(
            len(run) > max_gap or run[0] == 0 or run[-1] == n - 1
            for run in runs
        ):
            raise RuntimeError(
                "insufficient background matches; provide another reference"
            )
    if n > 1 and valid.sum() < 2:
        raise RuntimeError("insufficient background evidence")

    extent = np.linalg.norm(
        np.ptp(positions[valid], axis=0), axis=1
    ).max()
    if extent < static_threshold and valid.all():
        return [np.eye(3) for _ in frames]

    observed = np.flatnonzero(valid)
    for j in range(9):
        for c in range(2):
            positions[:, j, c] = np.interp(
                np.arange(n), observed, positions[valid, j, c]
            )
    positions = _smooth(positions, valid.astype(float), smoothness)
    extent = np.linalg.norm(np.ptp(positions, axis=0), axis=1).max()
    if extent < static_threshold and valid.all():
        return [np.eye(3) for _ in frames]

    to_frame = [cv2.findHomography(grid, p, 0)[0] for p in positions]
    # Smoothing slightly moves the reference. Rebase to make H[ref_idx] == I.
    anchor = to_frame[ref_idx]
    result = [anchor @ np.linalg.inv(m) for m in to_frame]
    result = [m / m[2, 2] for m in result]
    result[ref_idx] = np.eye(3)
    return result


def _median(values, weights):
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    sorted_weights = np.take_along_axis(weights, order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    index = np.argmax(cumulative >= 0.5 * cumulative[-1], axis=0)
    return np.take_along_axis(sorted_values, index[None], axis=0)[0]


def lock_points(points_t, H, weights=None, *, deformation_smoothness=None):
    """Lock corresponding (N,K,2) points; weights can be (N,) or (N,K).

    Missing coordinates and zero weights are ignored. Each point needs at least
    one reliable observation. Default: one robust rigid shape in reference space.

    Optional deformation_smoothness >=1000 fits very slow reference residuals.
    Returns float64 with the same shape; inputs are never mutated.
    """
    p = np.asarray(points_t, dtype=float)
    matrices = np.asarray(H, dtype=float)
    if (
        p.ndim != 3 or p.shape[-1] != 2 or min(p.shape[:2]) == 0
        or matrices.shape != (len(p), 3, 3)
        or not np.isfinite(matrices).all()
        or np.any(np.abs(np.linalg.det(matrices)) < 1e-12)
    ):
        raise ValueError(
            "need (N,K,2) points and N finite invertible homographies"
        )
    n, k = p.shape[:2]
    if weights is None:
        w = np.ones((n, k))
    else:
        w = np.asarray(weights, float)
        if w.shape == (n,):
            w = w[:, None]
        try:
            w = np.broadcast_to(w, (n, k)).copy()
        except ValueError as error:
            raise ValueError(
                "weights must have shape (N,) or (N,K)"
            ) from error
        if not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError("weights must be finite and nonnegative")

    finite = np.isfinite(p).all(axis=2)
    w = w * finite
    if np.any(w.sum(axis=0) == 0):
        raise ValueError(
            "each point needs a positive-weight finite observation"
        )
    ref = np.array([
        _project(np.where(finite[t, :, None], p[t], 0), matrices[t])
        for t in range(n)
    ])
    center = np.stack(
        [_median(ref[..., c], w) for c in range(2)], axis=-1
    )
    for _ in range(12):
        residual = np.linalg.norm(ref - center, axis=2)
        scale = np.maximum(0.05, 1.4826 * _median(residual, w))
        robust = w * np.minimum(
            1, 1.5 * scale / np.maximum(residual, 1e-12)
        )
        center = (
            (ref * robust[..., None]).sum(axis=0)
            / robust.sum(axis=0)[:, None]
        )

    canonical = np.broadcast_to(center, p.shape).copy()
    if deformation_smoothness is not None:
        strength = float(deformation_smoothness)
        if not np.isfinite(strength) or strength < 1000:
            raise ValueError(
                "deformation_smoothness must be >=1000 or None"
            )
        for j in range(k):
            weight = robust[:, j] / max(robust[:, j].max(), 1e-12)
            # Ridge anchors missing runs and discourages residual drift.
            ridge = 0.05
            residual = (
                (ref[:, j] - center[j]) * weight[:, None]
                / (weight[:, None] + ridge)
            )
            canonical[:, j] += _smooth(
                residual, weight + ridge, strength
            )

    return np.array([
        _project(canonical[t], np.linalg.inv(matrices[t]))
        for t in range(n)
    ])


def _prepare(
    frames, points, surface_masks, occluder_masks, H, exclude_masks,
    weights, motion_options,
):
    frames = list(frames)
    n = len(frames)
    if n == 0 or len(points) != n:
        raise ValueError(
            "geometry and frames must have the same nonzero length"
        )
    shape = np.asarray(frames[0]).shape[:2]
    surfaces = _masks(surface_masks, n, shape)
    occluders = _masks(occluder_masks, n, shape)
    extra = _masks(exclude_masks, n, shape)
    reliability = np.ones(points.shape[:2], float)

    for t, (p, surface, occ) in enumerate(zip(points, surfaces, occluders)):
        if surface is not None:
            if not surface.any():
                reliability[t] = 0
                continue
            finite = p[np.isfinite(p).all(axis=1)]
            if len(finite) >= 3:
                area = cv2.contourArea(
                    cv2.convexHull(finite.astype(np.float32))
                )
                reliability[t] *= min(
                    1.0, surface.sum() / max(1.0, area)
                ) ** 2
            if (
                surface[:2].any() or surface[-2:].any()
                or surface[:, :2].any() or surface[:, -2:].any()
            ):
                reliability[t] *= 0.1
        if occ is not None and occ.any():
            dilated = cv2.dilate(
                occ.astype(np.uint8), np.ones((9, 9), np.uint8)
            )
            reliability[t, ~_inside(p, 1 - dilated)] *= 0.01
            if surface is not None:
                overlap = (
                    np.count_nonzero(occ & surface)
                    / max(1, surface.sum())
                )
                reliability[t] *= max(0.01, 1 - overlap) ** 2

    if weights is not None:
        supplied = np.asarray(weights, float)
        if supplied.shape == (n,):
            supplied = supplied[:, None]
        try:
            reliability *= np.broadcast_to(
                supplied, reliability.shape
            )
        except ValueError as error:
            raise ValueError(
                "weights must have shape (N,) or (N,K)"
            ) from error

    if H is None:
        union = []
        for masks in zip(surfaces, occluders, extra):
            excluded = np.zeros(shape, bool)
            for m in masks:
                if m is not None:
                    excluded |= m
            union.append(excluded)
        H = camera_motion(frames, union, **motion_options)
    return H, reliability


def lock_quads(
    frames, quads, surface_masks, occluder_masks=None, *,
    H=None, exclude_masks=None, weights=None,
    deformation_smoothness=None, **motion_options,
):
    """Return (N,4,2) float64 quads; pass H to share motion across slots.

    exclude_masks adds other surfaces/foreground to this slot's exclusions.
    Mask clipping/occlusion reduces measurement weight.
    """
    points = np.asarray(quads, float)
    if points.ndim != 3 or points.shape[1:] != (4, 2):
        raise ValueError("quads must have shape (N,4,2)")
    H, weights = _prepare(
        frames, points, surface_masks, occluder_masks, H,
        exclude_masks, weights, motion_options,
    )
    return lock_points(
        points, H, weights,
        deformation_smoothness=deformation_smoothness,
    )


def lock_ribbons(
    frames, ribbons, surface_masks=None, occluder_masks=None, *,
    H=None, exclude_masks=None, weights=None,
    deformation_smoothness=None, **motion_options,
):
    """Return x/top/bottom dicts with unchanged sample counts.

    Samples must correspond across frames; None entries remain None. Project
    both boundaries in 2D, then interpolate y at their shared projected midline
    x. The legacy ribbon format has vertical ends, so it cannot exactly express
    a general projective transform with slanted ends. Use lock_points directly
    with independent 2D boundaries when that distinction matters.
    """
    from pipeline.ribbon import _arrays

    ribbons = list(ribbons)
    arrays = [_arrays(r) for r in ribbons]
    good = [a for a in arrays if a is not None]
    if not good:
        return [None] * len(ribbons)
    k = len(good[0][0])
    if any(len(a[0]) != k for a in good):
        raise ValueError("ribbons must have equal sample counts")

    points = np.full((len(ribbons), 2 * k, 2), np.nan)
    for t, a in enumerate(arrays):
        if a is not None:
            x, top, bottom = a
            points[t] = np.concatenate((
                np.column_stack((x, top)),
                np.column_stack((x, bottom)),
            ))
    H, weights = _prepare(
        frames, points, surface_masks, occluder_masks, H,
        exclude_masks, weights, motion_options,
    )
    locked = lock_points(
        points, H, weights,
        deformation_smoothness=deformation_smoothness,
    )

    result = []
    for t, p in enumerate(locked):
        if arrays[t] is None:
            result.append(None)
            continue
        top, bottom = p[:k], p[k:]
        if (
            np.any(np.diff(top[:, 0]) <= 0)
            or np.any(np.diff(bottom[:, 0]) <= 0)
        ):
            raise ValueError(
                "projected ribbon is not a horizontal x-monotone surface"
            )
        x = 0.5 * (top[:, 0] + bottom[:, 0])
        yt = np.interp(x, top[:, 0], top[:, 1])
        yb = np.interp(x, bottom[:, 0], bottom[:, 1])
        if np.any(yb <= yt):
            raise ValueError("projected ribbon boundaries cross")
        result.append(dict(
            ribbons[t], x=x.copy(), top=yt, bottom=yb
        ))
    return result
