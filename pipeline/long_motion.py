"""H[t] maps original frame t to reference.

Two passes over reiterable sources. Images are reduced to <=640 pixels wide.
Only one keyframe interval, two previous keyframes, and one loop anchor are
cached. Sparse observations and output matrices grow with clip length.
"""
from collections import deque
from collections.abc import Iterator
from itertools import zip_longest

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def _project(p, h):
    p = np.asarray(p, np.float64).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ h.T
    if not np.isfinite(q).all() or np.any(np.abs(q[:, 2]) < 1e-9):
        raise RuntimeError("invalid projective mapping")
    return q[:, :2] / q[:, 2:]


def _normal(h):
    if not np.isfinite(h).all() or abs(h[2, 2]) < 1e-10:
        raise RuntimeError("invalid homography")
    h = h / h[2, 2]
    if abs(np.linalg.det(h)) < 1e-10:
        raise RuntimeError("singular homography")
    return h


def _inside(p, allowed):
    p = np.asarray(p).reshape(-1, 2)
    finite = np.isfinite(p).all(1)
    ij = np.rint(np.where(np.isfinite(p), p, -1)).astype(int)
    h, w = allowed.shape
    ok = finite & (ij[:, 0] >= 0) & (ij[:, 0] < w)
    ok &= (ij[:, 1] >= 0) & (ij[:, 1] < h)
    ix = np.flatnonzero(ok)
    ok[ix] &= allowed[ij[ix, 1], ij[ix, 0]] != 0
    return ok


def _images(frames, masks, width, margin):
    sentinel = object()
    pairs = ((f, None) for f in frames) if masks is None else zip_longest(
        frames, masks, fillvalue=sentinel
    )
    original = None
    for f, mask in pairs:
        if f is sentinel or mask is sentinel:
            raise ValueError("one mask is required per frame")
        f = np.asarray(f)
        if f.dtype != np.uint8 or f.ndim not in (2, 3):
            raise ValueError("frames must be uint8 gray or BGR")
        if f.ndim == 3 and f.shape[2] != 3:
            raise ValueError("color frames must be BGR")
        if original is None:
            original = f.shape[:2]
        if f.shape[:2] != original:
            raise ValueError("frame sizes differ")
        h, w = original
        sw = min(w, width)
        sh = max(1, round(h * sw / w))
        small = cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA)
        if small.ndim == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if mask is None:
            excluded = np.zeros((sh, sw), np.uint8)
        else:
            mask = np.asarray(mask)
            if mask.shape != original:
                raise ValueError("mask size must equal frame size")
            # Preserve small exclusions when reducing the mask.
            excluded = (cv2.resize(
                mask.astype(np.float32), (sw, sh),
                interpolation=cv2.INTER_AREA
            ) > 0).astype(np.uint8)
        if margin:
            excluded = cv2.dilate(
                excluded, np.ones((2 * margin + 1,) * 2, np.uint8)
            )
        allowed = 255 * (1 - excluded)
        edge = max(1, margin)
        allowed[:edge] = allowed[-edge:] = 0
        allowed[:, :edge] = allowed[:, -edge:] = 0
        scale = np.diag([sw / w, sh / h, 1.])
        yield np.ascontiguousarray(small), allowed, scale


def _features(image, allowed, count, diagnostic=False):
    p = cv2.goodFeaturesToTrack(
        image, maxCorners=count, qualityLevel=.02 if diagnostic else .01,
        minDistance=11 if diagnostic else 6, mask=allowed,
        blockSize=7 if diagnostic else 5,
    )
    return np.empty((0, 2), np.float32) if p is None else p.reshape(-1, 2)


def _flow(a, b, p, aa, bb, seed=None):
    if len(p) == 0:
        return p, p
    lk = dict(
        winSize=(21, 21), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, .001),
    )
    if seed is not None:
        # seed maps a->b; prewarp b into a's coordinates.
        h, w = a.shape
        aligned = cv2.warpPerspective(
            b, seed, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        )
        ba = cv2.warpPerspective(
            bb, seed, (w, h),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        )
        x, y = _flow(a, aligned, p, aa, ba)
        y = _project(y, seed).astype(np.float32)
        keep = _inside(y, bb)
        return x[keep], y[keep]
    p = p.astype(np.float32).reshape(-1, 1, 2)
    q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, **lk)
    if q is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    back, sb, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, **lk)
    if back is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    x, y = p.reshape(-1, 2), q.reshape(-1, 2)
    good = (st.ravel() != 0) & (sb.ravel() != 0)
    good &= np.linalg.norm(back.reshape(-1, 2) - x, axis=1) < .3
    good &= _inside(x, aa) & _inside(y, bb)
    return x[good], y[good]


def _fit(x, y, threshold, minimum):
    if len(x) < minimum:
        return None
    h, keep = cv2.findHomography(
        x, y, cv2.RANSAC, threshold, maxIters=3000, confidence=.999
    )
    if h is None or keep is None:
        return None
    good = keep.ravel().astype(bool)
    if good.sum() < minimum or good.mean() < .35:
        return None
    x, y = x[good], y[good]
    for _ in range(3):
        error = np.linalg.norm(_project(x, h) - y, axis=1)
        good = error <= min(threshold, max(.12, 3 * np.median(error)))
        if good.sum() < minimum:
            return None
        x, y = x[good], y[good]
        h, _ = cv2.findHomography(x, y, 0)
        if h is None:
            return None
    if min(cv2.contourArea(cv2.convexHull(z.astype(np.float32)))
           for z in (x, y)) < 100:
        return None
    return _normal(h), x, y


def _pair(a, b, options):
    image, allowed, points = a
    target, target_allowed, _ = b
    threshold, minimum = options["ransac_px"], options["min_inliers"]
    x, y = _flow(image, target, points, allowed, target_allowed)
    fitted = _fit(x, y, threshold, minimum)
    if fitted is not None:
        return fitted

    orb = cv2.ORB_create(nfeatures=4000, fastThreshold=7)
    ka, da = orb.detectAndCompute(image, allowed)
    kb, db = orb.detectAndCompute(target, target_allowed)
    if da is None or db is None or len(db) < 2:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    matches = [
        a for pair in pairs if len(pair) == 2
        for a, b in [pair] if a.distance < .7 * b.distance
    ]
    x = np.array([ka[m.queryIdx].pt for m in matches],
                 np.float32).reshape(-1, 2)
    y = np.array([kb[m.trainIdx].pt for m in matches],
                 np.float32).reshape(-1, 2)
    seed = _fit(x, y, max(3., threshold), minimum)
    if seed is None:
        return None
    x, y = _flow(image, target, points, allowed, target_allowed, seed[0])
    return _fit(x, y, threshold, minimum)


def _joint(edges, count, shape):
    """Joint robust keyframe adjustment with a fixed projective gauge."""
    anchor = count // 2
    poses = [None] * count
    poses[anchor] = np.eye(3)
    # Spanning-tree composition supplies initialization only.
    while any(p is None for p in poses):
        progress = False
        for a, b, matrix, _, _ in edges:
            if poses[a] is not None and poses[b] is None:
                poses[b] = _normal(poses[a] @ np.linalg.inv(matrix))
                progress = True
            elif poses[b] is not None and poses[a] is None:
                poses[a] = _normal(poses[b] @ matrix)
                progress = True
        if not progress:
            raise RuntimeError("disconnected keyframe graph")
    if count == 1:
        return np.array(poses)

    height, width = shape
    c = np.array([
        [2 / width, 0, -1],
        [0, 2 / width, -height / width],
        [0, 0, 1.],
    ])
    ci = np.linalg.inv(c)
    free = [i for i in range(count) if i != anchor]
    slots = {node: j for j, node in enumerate(free)}
    normalized = [_normal(c @ p @ ci) for p in poses]
    observations = [
        (a, b, _project(x, c), _project(y, c))
        for a, b, _, x, y in edges
    ]
    sizes = [2 * len(x) for _, _, x, _ in observations]
    sparsity = lil_matrix((sum(sizes), 8 * len(free)), dtype=int)
    row = 0
    for (a, b, _, _), size in zip(observations, sizes):
        for node in (a, b):
            if node in slots:
                j = 8 * slots[node]
                sparsity[row:row + size, j:j + 8] = 1
        row += size

    def unpack(values):
        result = [np.eye(3) for _ in range(count)]
        for node, chunk in zip(free, values.reshape(-1, 8)):
            result[node] = np.r_[chunk, 1.].reshape(3, 3)
        return result

    def residual(values):
        matrices = unpack(values)
        return np.concatenate([
            ((_project(x, matrices[a]) - _project(y, matrices[b]))
             * width / 2).ravel()
            for a, b, x, y in observations
        ])

    initial = np.concatenate([
        p.ravel()[:8] for i, p in enumerate(normalized) if i != anchor
    ])
    fit = least_squares(
        residual, initial, jac_sparsity=sparsity.tocsr(), method="trf",
        loss="soft_l1", f_scale=.3, x_scale="jac", max_nfev=150,
        ftol=1e-7, xtol=1e-7, gtol=1e-7,
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError("keyframe joint refinement did not converge")
    return np.array([_normal(ci @ p @ c) for p in unpack(fit.x)])


def camera_motion_long(frames, exclude_masks=None, keyframe_every=25, **kw):
    """Return (N,3,3) float64 frame->reference matrices; H[ref_idx] == I.

    frames/masks must support two iterations; mask True means excluded.
    Pixel options apply after resizing. Image memory is O(keyframe_every
    * reduced image size); sparse graph/output storage is O(N).

    Supported kw: ref_idx, max_width, max_features, mask_margin,
    ransac_px, min_inliers. Failed evidence raises; no temporal smoothing
    or missing-frame interpolation is applied.
    """
    defaults = dict(
        ref_idx=None, max_width=640, max_features=1200,
        mask_margin=6, ransac_px=1., min_inliers=16,
    )
    if set(kw) - defaults.keys():
        raise TypeError("unknown options: " +
                        str(sorted(set(kw) - defaults.keys())))
    options = dict(defaults, **kw)
    if (
        not isinstance(keyframe_every, (int, np.integer))
        or keyframe_every < 1
        or not isinstance(options["max_width"], (int, np.integer))
        or not 32 <= options["max_width"] <= 640
        or not isinstance(options["mask_margin"], (int, np.integer))
        or options["mask_margin"] < 0
        or not isinstance(options["min_inliers"], (int, np.integer))
        or not isinstance(options["max_features"], (int, np.integer))
        or not 4 <= options["min_inliers"] <= options["max_features"]
        or not np.isfinite(options["ransac_px"])
        or options["ransac_px"] <= 0
    ):
        raise ValueError("invalid motion options")
    if isinstance(frames, Iterator) or isinstance(exclude_masks, Iterator):
        raise TypeError("frames and masks must support two independent iterations")

    cache = deque(maxlen=2)
    first = None
    key_times, edges = [], []

    def add_key(t, item):
        nonlocal first
        image, allowed, _ = item
        node = (image, allowed,
                _features(image, allowed, options["max_features"]))
        index = len(key_times)
        candidates = list(cache)
        if first is not None and all(i != 0 for i, _ in candidates):
            candidates.append((0, first))
        for previous, source in candidates:
            match = _pair(source, node, options)
            if match is None:
                continue
            matrix, x, y = match
            # Bound each edge's stored correspondence count.
            order = np.lexsort((x[:, 0], x[:, 1]))
            take = order[np.linspace(
                0, len(order) - 1, min(180, len(order))
            ).astype(int)]
            edges.append((previous, index, matrix, x[take], y[take]))
        if first is None:
            first = node
        key_times.append(t)
        cache.append((index, node))

    n = 0
    for t, item in enumerate(_images(
        frames, exclude_masks, options["max_width"], options["mask_margin"]
    )):
        if t == 0:
            scale, shape = item[2], item[0].shape
        if t % keyframe_every == 0:
            add_key(t, item)
        last = item
        n = t + 1
    if not n:
        raise ValueError("at least one frame is required")
    if key_times[-1] != n - 1:
        add_key(n - 1, last)
    ref = options["ref_idx"]
    ref = n // 2 if ref is None else ref
    if not isinstance(ref, (int, np.integer)) or not 0 <= ref < n:
        raise ValueError("ref_idx is out of range")
    poses = _joint(edges, len(key_times), shape)

    cache.clear()
    first = last = item = None
    result = np.empty((n, 3, 3), np.float64)
    left = None
    interval = []
    next_key = 0
    seen = 0
    for t, (image, allowed, current_scale) in enumerate(_images(
        frames, exclude_masks, options["max_width"], options["mask_margin"]
    )):
        if (t >= n or image.shape != shape
                or not np.array_equal(current_scale, scale)):
            raise ValueError("frames changed between passes")
        seen += 1
        node = (image, allowed, None)
        interval.append((t, node))
        if t != key_times[next_key]:
            continue
        right = (image, allowed,
                 _features(image, allowed, options["max_features"]))
        if left is None:
            result[t] = poses[next_key]
            first = right
        else:
            for frame_index, target in interval:
                if frame_index == t:
                    result[t] = poses[next_key]
                    continue
                xs, ys = [], []
                anchors = [(next_key - 1, left), (next_key, right)]
                if next_key > 1:
                    anchors.append((0, first))
                for anchor_index, source in anchors:
                    match = _pair(source, target, options)
                    if match is not None:
                        _, x, y = match
                        xs.append(y)
                        ys.append(_project(x, poses[anchor_index]))
                if not xs:
                    raise RuntimeError(
                        f"no static matches for frame {frame_index}"
                    )
                fit = _fit(
                    np.concatenate(xs), np.concatenate(ys),
                    options["ransac_px"], options["min_inliers"],
                )
                if fit is None:
                    raise RuntimeError(
                        f"insufficient support for frame {frame_index}"
                    )
                result[frame_index] = fit[0]
        interval.clear()
        left = right
        next_key += 1
    if seen != n:
        raise ValueError("frames changed between passes")

    # Rebase exactly to the requested frame, then restore original pixels.
    rebase = np.linalg.inv(result[ref])
    inverse_scale = np.linalg.inv(scale)
    for t in range(n):
        result[t] = _normal(inverse_scale @ rebase @ result[t] @ scale)
    result[ref] = np.eye(3)
    return result


def motion_residuals(
    frames, H, exclude_masks=None, *, ref_idx=None,
    max_width=640, max_features=500, mask_margin=8,
):
    """Independent adjacent KLT tracks anchored at their reference observation.

    Returns per-frame median_px, p95_px, count, max_track_age.
    Errors are original/reference pixels. Tracks must cross ref_idx.
    H never enters detection, tracking, acceptance, or anchor selection.
    Missing evidence produces NaN with count=0.
    """
    H = np.asarray(H, np.float64)
    n = len(H)
    ref = n // 2 if ref_idx is None else ref_idx
    if (
        H.shape != (n, 3, 3) or not n or not np.isfinite(H).all()
        or np.any(np.abs(np.linalg.det(H)) < 1e-12)
        or not isinstance(ref, (int, np.integer)) or not 0 <= ref < n
        or not isinstance(max_width, (int, np.integer))
        or not 32 <= max_width <= 640
        or not isinstance(mask_margin, (int, np.integer))
        or mask_margin < 0
        or not isinstance(max_features, (int, np.integer))
        or max_features < 1
    ):
        raise ValueError("invalid diagnostic inputs")

    observations = []
    points = np.empty((0, 2), np.float32)
    ids = np.empty(0, int)
    births = np.empty(0, int)
    next_id = 0
    previous = previous_allowed = None
    for t, (gray, allowed, scale) in enumerate(
        _images(frames, exclude_masks, max_width, mask_margin)
    ):
        if t >= n:
            raise ValueError("H/frame lengths differ")
        if previous is not None and len(points):
            lk = dict(
                winSize=(19, 19), maxLevel=3, criteria=(3, 40, .001)
            )
            p = points.reshape(-1, 1, 2)
            q, st, _ = cv2.calcOpticalFlowPyrLK(
                previous, gray, p, None, **lk
            )
            if q is None:
                good = np.zeros(len(points), bool)
                q = p.copy()
            else:
                back, sb, _ = cv2.calcOpticalFlowPyrLK(
                    gray, previous, q, None, **lk
                )
                good = np.zeros(len(points), bool) if back is None else (
                    (st.ravel() != 0) & (sb.ravel() != 0)
                    & (np.linalg.norm(
                        back.reshape(-1, 2) - points, axis=1
                    ) < .2)
                    & _inside(q, allowed)
                    & _inside(points, previous_allowed)
                )
            points = q.reshape(-1, 2)[good]
            ids, births = ids[good], births[good]

        if len(points) < max_features // 2 + 1:
            space = allowed.copy()
            for x, y in np.rint(points).astype(int):
                cv2.circle(space, (x, y), 11, 0, -1)
            new = _features(
                gray, space, max_features - len(points), True
            )
            points = np.concatenate((points, new))
            ids = np.r_[ids, np.arange(next_id, next_id + len(new))]
            births = np.r_[births, np.full(len(new), t)]
            next_id += len(new)
        full = _project(points, np.linalg.inv(scale))
        observations.append((ids.copy(), full, t - births))
        previous, previous_allowed = gray, allowed

    if len(observations) != n:
        raise ValueError("H/frame lengths differ")
    anchor_ids, anchor_points, _ = observations[ref]
    anchors = {
        int(i): p for i, p in zip(
            anchor_ids, _project(anchor_points, H[ref])
        )
    }
    output = dict(
        median_px=np.full(n, np.nan),
        p95_px=np.full(n, np.nan),
        count=np.zeros(n, int),
        max_track_age=np.zeros(n, int),
    )
    for t, (identities, p, age) in enumerate(observations):
        keep = np.array([int(i) in anchors for i in identities], bool)
        if not keep.any():
            continue
        target = np.array([anchors[int(i)] for i in identities[keep]])
        error = np.linalg.norm(_project(p[keep], H[t]) - target, axis=1)
        output["median_px"][t], output["p95_px"][t] = np.percentile(
            error, [50, 95]
        )
        output["count"][t] = len(error)
        output["max_track_age"][t] = age[keep].max()
    return output
