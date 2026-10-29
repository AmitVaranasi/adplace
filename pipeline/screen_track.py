"""Bezel tracking for dark rectangular screens on untextured walls.

Camera motion is a prediction, never a screen observation. Segmentation locates
edge searches; only image gradients constrain the bezel. Clipping and people
remove observations rather than moving an edge onto the clipping boundary.
All public coordinates, thresholds and residuals are in original image pixels.
"""
from __future__ import annotations

import cv2
import numpy as np

from pipeline.planar_track import (
    _features, _flow, _iou, _matrix, _project, _smooth, _visibility,
)


def _equations(q):
    edges = np.roll(q, -1, axis=0) - q
    lengths = np.linalg.norm(edges, axis=1)
    if not np.isfinite(lengths).all() or lengths.min() < 1:
        raise ValueError("degenerate bezel")
    tangent = edges / lengths[:, None]
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    return normal, np.einsum("ij,ij->i", normal, q)


def _seed(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    for fraction in (.005, .01, .02, .03, .05):
        q = cv2.approxPolyDP(
            hull, fraction * cv2.arcLength(hull, True), True
        ).reshape(-1, 2).astype(float)
        if len(q) != 4 or not cv2.isContourConvex(q.astype(np.float32)):
            continue
        if cv2.contourArea(q.astype(np.float32), oriented=True) < 0:
            q = q[::-1]
        return np.roll(q, -np.argmin(q.sum(1)), axis=0)
    return None


def _observations(gray, mask, occ, search, radius):
    h, w = gray.shape
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return []
    points = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    ij = points.astype(int)
    points = points[
        (points[:, 0] > 3) & (points[:, 0] < w-4)
        & (points[:, 1] > 3) & (points[:, 1] < h-4)
        & ~occ[ij[:, 1], ij[:, 0]]
    ]
    if len(points) < 12:
        return []
    normals, constants = _equations(search)
    lengths = np.linalg.norm(np.roll(search, -1, axis=0)-search, axis=1)
    distance = np.abs(points @ normals.T - constants)
    owner = distance.argmin(1)
    blurred = cv2.GaussianBlur(gray, (0, 0), .8)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
    offsets = np.arange(-radius, radius+.25, .5)
    observations = []

    for k in range(4):
        pp = points[
            (owner == k)
            & (distance[:, k] < max(radius, .06*lengths[k]))
        ]
        tangent = np.array([normals[k, 1], -normals[k, 0]])
        if len(pp) < 12 or np.ptp(pp @ tangent) < max(14, .12*lengths[k]):
            continue
        vx, vy, x0, y0 = cv2.fitLine(
            pp.astype(np.float32), cv2.DIST_HUBER, 0, .001, .001
        ).ravel()
        normal = np.array([-vy, vx])
        normal *= 1 if normal @ normals[k] > 0 else -1
        if normal @ normals[k] < .98:
            continue
        constant = normal @ [x0, y0]

        # Sample uniformly in distance along the mask edge.
        order = np.argsort(pp @ tangent)
        along = pp[order] @ tangent
        t = np.linspace(
            np.percentile(along, 4), np.percentile(along, 96),
            min(140, len(pp)),
        )
        base = np.column_stack([
            np.interp(t, along, pp[order, j]) for j in range(2)
        ])
        base += (constant-base @ normal)[:, None]*normal
        locations = base[:, None] + offsets[None, :, None]*normal
        x, y = [locations[:, :, j].astype(np.float32) for j in range(2)]
        dx, dy = [
            cv2.remap(a, x, y, cv2.INTER_LINEAR) for a in (gx, gy)
        ]

        # Inward transition from wall to dark outer bezel. The opposite
        # transition at the inner bezel/content boundary is not eligible.
        score = -(dx*normal[0]+dy*normal[1])
        score *= np.exp(-.5*(offsets/max(2., radius/3))**2)[None]
        clear = cv2.remap(
            (~occ).astype(np.uint8), x, y, cv2.INTER_NEAREST
        ) != 0
        clear &= (x > 3) & (x < w-4) & (y > 3) & (y < h-4)
        score[~clear] = 0
        best = score.argmax(1)
        strength = score[np.arange(len(t)), best]
        selected = locations[np.arange(len(t)), best][strength > 6]
        if len(selected) < 12:
            continue

        for _ in range(3):
            vx, vy, x0, y0 = cv2.fitLine(
                selected.astype(np.float32), cv2.DIST_HUBER,
                0, .001, .001,
            ).ravel()
            normal = np.array([-vy, vx])
            normal *= 1 if normal @ normals[k] > 0 else -1
            constant = float(normal @ [x0, y0])
            error = np.abs(selected @ normal-constant)
            selected = selected[error < max(1.5, 2.5*np.median(error))]
            if len(selected) < 8:
                break
        if len(selected) < 8 or normal @ normals[k] < .98:
            continue
        span = np.ptp(selected @ tangent)
        if span < max(14, .12*lengths[k]):
            continue

        # Two small patches at opposite ends of an image border must not
        # masquerade as a supported edge spanning the entire screen.
        u = selected @ tangent
        occupied = np.unique(
            np.minimum(7, ((u-u.min())/span*8).astype(int))
        )
        if len(occupied) < 5:
            continue
        residual = float(np.sqrt(np.mean(
            (selected @ normal-constant)**2
        )))
        observations.append((
            k, normal, constant, residual, len(selected), float(span)
        ))
    return observations


def _update(prediction, observations, prior_sigma):
    # A line constrains only the normal component of its two corners.
    # The motion prediction retains the unconstrained components.
    a = np.eye(8)/prior_sigma
    b = prediction.ravel()/prior_sigma
    information = a.T @ a
    for k, normal, constant, residual, count, span in observations:
        length = np.linalg.norm(prediction[(k+1) % 4]-prediction[k])
        sigma = max(.2, residual/np.sqrt(count))*max(1., length/span)
        for j in (k, (k+1) % 4):
            row = np.zeros(8)
            row[2*j:2*j+2] = normal/sigma
            a = np.vstack([a, row])
            b = np.r_[b, constant/sigma]
            information += np.outer(row, row)
    q = np.linalg.lstsq(a, b, rcond=None)[0].reshape(4, 2)
    sigma = float(np.sqrt(np.trace(np.linalg.pinv(information))/8))
    return q, max(.2, sigma)


def _camera(a, b, allowed_a, allowed_b):
    p = _features(a, allowed_a, 700)
    x, y = _flow(a, b, p, allowed_a, allowed_b, p, 1.)
    if len(x) < 12:
        return np.eye(3), False, len(x)
    cv2.setRNGSeed(0)
    h, good = cv2.findHomography(
        x, y, cv2.RANSAC, 1.5, maxIters=1000, confidence=.995
    )
    height, width = a.shape
    area = cv2.contourArea(cv2.convexHull(x.astype(np.float32)))
    if area < .15*height*width:
        # A small textured patch cannot constrain perspective elsewhere.
        affine, good = cv2.estimateAffinePartial2D(
            x, y, method=cv2.RANSAC, ransacReprojThreshold=1.5,
            maxIters=1000, confidence=.995,
        )
        h = None if affine is None else np.vstack([affine, [0., 0, 1]])
    if h is None or good is None or good.sum() < 12:
        return np.eye(3), False, 0
    corners = np.array([
        [0., 0], [width-1., 0],
        [width-1., height-1.], [0, height-1.]
    ])
    try:
        den = np.c_[corners, np.ones(4)] @ h[2]
        q = _project(corners, h)
        if (
            den.min()*den.max() <= 0
            or not cv2.isContourConvex(q.astype(np.float32))
            or np.max(np.abs(q)) > 4*max(a.shape)
            or np.linalg.matrix_rank(h) < 3
            or not .6 < cv2.contourArea(q.astype(np.float32), oriented=True)
            / max(1, (height-1)*(width-1)) < 1.67
        ):
            return np.eye(3), False, 0
        return h/h[2, 2], True, int(good.sum())
    except (ValueError, cv2.error, np.linalg.LinAlgError):
        return np.eye(3), False, 0


def track_screen(
    frames, ref_idx, ref_quad, *, surface_masks, occluder_masks=None,
    max_width=1280, motion_width=640, search_px=12., min_visible=.01,
    min_iou=.35, max_gap=6, min_edges=2, smooth=True,
):
    """Track a dark bezel; return the same geometry/diagnostics as track_plane.

    surface_masks(i) supplies bool masks or None. Empty/None masks are absence
    of observations. Long gaps remain invalid. A partially observed quad may
    extend outside the image. 'measured' denotes supported bezel evidence,
    not four independently observed corners. By default at least two edges
    are required for validity. observed_edges and motion_valid make that
    distinction explicit. Hidden corners are motion estimates.

    The caller's reference quad is retained exactly. It is attached to the
    refined reference bezel, so it may describe an inset creative rectangle.
    Screen content is excluded from the camera fit by the surface masks.
    """
    for name, value, lower in (
        ("max_width", max_width, 32), ("motion_width", motion_width, 32),
        ("max_gap", max_gap, 0), ("min_edges", min_edges, 1),
    ):
        if not isinstance(value, (int, np.integer)) or value < lower:
            raise ValueError("invalid "+name)
    if min_edges > 4:
        raise ValueError("invalid min_edges")
    if (
        not np.isfinite([search_px, min_visible, min_iou]).all()
        or search_px <= 0 or not 0 < min_visible <= 1
        or not 0 < min_iou <= 1
    ):
        raise ValueError("invalid screen thresholds")
    rq = np.asarray(ref_quad, np.float64)
    if (
        rq.shape != (4, 2) or not np.isfinite(rq).all()
        or not cv2.isContourConvex(rq.astype(np.float32))
        or cv2.contourArea(rq.astype(np.float32), oriented=True) <= 4
    ):
        raise ValueError("ref_quad must have finite TL,TR,BR,BL winding")
    grays, original = [], None
    for frame in frames:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frames must be BGR uint8")
        if original is None:
            original = frame.shape[:2]
            oh, ow = original
            w = min(ow, max_width)
            h = max(1, round(oh*w/ow))
            sx, sy = w/ow, h/oh
            S = np.array([
                [sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1.]
            ])
        if frame.shape[:2] != original:
            raise ValueError("frame sizes differ")
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays.append(cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA))
    n = len(grays)
    if not isinstance(ref_idx, (int, np.integer)) or not 0 <= ref_idx < n:
        raise ValueError("ref_idx is out of range")
    scale = min(sx, sy)

    def masks(i):
        result = []
        for getter in (surface_masks, occluder_masks):
            m = None if getter is None else getter(i)
            if m is None:
                m = np.zeros(original, bool)
            m = np.asarray(m)
            if m.shape != original or m.dtype != np.bool_:
                raise ValueError("masks must be full-frame bool arrays")
            result.append(cv2.resize(
                m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ) != 0)
        surf, occ = result
        radius = max(1, round(3*scale))
        occ = cv2.dilate(
            occ.astype(np.uint8), np.ones((2*radius+1,)*2, np.uint8)
        ) != 0
        return surf, occ

    mw = min(w, motion_width)
    mh = max(1, round(h*mw/w))
    mx, my = mw/w, mh/h
    M = np.array([
        [mx, 0, (mx-1)/2], [0, my, (my-1)/2], [0, 0, 1.]
    ])
    invM = np.linalg.inv(M)
    motion = np.tile(np.eye(3), (n, 1, 1))
    motion_valid = np.zeros(n, bool)
    motion_inliers = np.zeros(n, int)
    prev_gray = prev_allowed = None
    for i, gray in enumerate(grays):
        surf, occ = masks(i)
        exclude = cv2.resize(
            (surf | occ).astype(np.uint8), (mw, mh),
            interpolation=cv2.INTER_NEAREST,
        )
        exclude = cv2.dilate(exclude, np.ones((9, 9), np.uint8))
        allowed = (1-exclude)*255
        reduced = cv2.resize(gray, (mw, mh), interpolation=cv2.INTER_AREA)
        if i:
            g, motion_valid[i], motion_inliers[i] = _camera(
                prev_gray, reduced, prev_allowed, allowed
            )
            motion[i] = invM @ g @ M
        prev_gray, prev_allowed = reduced, allowed

    bezel = np.full((n, 4, 2), np.nan)
    sigma = np.full(n, 3.)
    residual = np.full(n, np.nan)
    counts = np.zeros(n, int)
    observed_edges = np.zeros((n, 4), bool)
    measured = np.zeros(n, bool)
    ious = np.full(n, np.nan)
    visible = np.zeros(n)
    status = ["lost"]*n
    edge_cache = [None]*n

    def observe(i, prediction):
        surf, occ = masks(i)
        if not surf.any():
            edge_cache[i] = []
            return prediction, 3., []
        search = _seed(surf)
        if search is None:
            search = prediction
        else:
            normal, _ = _equations(prediction)
            search = min(
                (np.roll(search, k, axis=0) for k in range(4)),
                key=lambda q: np.sum((_equations(q)[0]-normal)**2),
            )
        if edge_cache[i] is None:
            edge_cache[i] = _observations(
                grays[i], surf, occ, search, max(2., search_px*scale)
            )
        observations = edge_cache[i]

        # Global parallax is absorbed here, rather than accumulated into
        # observed edges. Never substitute the clipped SAM polygon for q.
        q, uncertainty = _update(prediction, observations, 3.*scale)
        if (
            not cv2.isContourConvex(q.astype(np.float32))
            or cv2.contourArea(q.astype(np.float32), oriented=True) <= 4
            or np.max(np.abs(q)) > 20*max(h, w)
        ):
            return prediction, 3., []
        return q, uncertainty, observations

    reference = _project(rq, S)
    bezel[ref_idx], sigma[ref_idx], observations = observe(ref_idx, reference)
    surf, occ = masks(ref_idx)
    if (
        len(observations) != 4
        or _visibility(bezel[ref_idx], (h, w), occ) < min_visible
    ):
        raise ValueError("reference needs four unoccluded bezel edges")

    for direction in (1, -1):
        prediction = bezel[ref_idx].copy()
        velocity = np.zeros((4, 2))
        drift = np.zeros(2)
        for i in range(
            ref_idx if direction == 1 else ref_idx-1,
            n if direction == 1 else -1, direction,
        ):
            if i != ref_idx:
                step = i if direction == 1 else i+1
                if motion_valid[step]:
                    g = (
                        motion[step] if direction == 1
                        else np.linalg.inv(motion[step])
                    )
                    prediction = _project(prediction, g)+drift
                else:
                    prediction = prediction+velocity
            q, uncertainty, observations = observe(i, prediction)
            if i != ref_idx and observations and motion_valid[step]:
                # Learn this plane's translation relative to the global
                # motion, including the correction for unobserved corners.
                a = np.array([v[1] for v in observations])
                b = np.array([
                    v[2]-v[1] @ (
                        (prediction[v[0]]+prediction[(v[0]+1) % 4])/2
                    ) for v in observations
                ])
                delta = np.linalg.solve(a.T @ a+.05*np.eye(2), a.T @ b)
                drift = np.clip(drift+delta, -30*scale, 30*scale)
            previous = i-direction
            if 0 <= previous < n and np.isfinite(bezel[previous]).all():
                velocity = np.clip(q-bezel[previous], -30*scale, 30*scale)
            bezel[i], sigma[i] = q, uncertainty
            prediction = q
            surf, occ = masks(i)
            visible[i] = _visibility(q, (h, w), occ)
            ious[i] = _iou(q, surf, occ)
            if observations:
                residual[i] = np.sqrt(np.mean([v[3]**2 for v in observations]))
                counts[i] = sum(v[4] for v in observations)
                observed_edges[i, [v[0] for v in observations]] = True
            if (
                surf.any() and len(observations) >= min_edges
                and visible[i] >= min_visible and ious[i] >= min_iou
            ):
                measured[i] = True
                status[i] = "bezel"

    # A single vertical edge does not measure vertical corner positions.
    # Later observed top/bottom edges must inform those positions too.
    anchors = np.flatnonzero(observed_edges.sum(1) >= 3)
    if len(anchors) > 1:
        def sweep(start, direction):
            trajectory = np.full_like(bezel, np.nan)
            trajectory[start] = bezel[start]
            prediction = bezel[start].copy()
            drift, velocity = np.zeros(2), np.zeros((4, 2))
            for i in range(
                start+direction, n if direction == 1 else -1, direction
            ):
                step = i if direction == 1 else i+1
                if motion_valid[step]:
                    g = (
                        motion[step] if direction == 1
                        else np.linalg.inv(motion[step])
                    )
                    prediction = _project(prediction, g)+drift
                else:
                    prediction = prediction+velocity
                obs = edge_cache[i] or []
                q, _ = _update(prediction, obs, 3.*scale)
                if (
                    not cv2.isContourConvex(q.astype(np.float32))
                    or cv2.contourArea(q.astype(np.float32), oriented=True) <= 4
                ):
                    q = prediction
                if obs and motion_valid[step]:
                    a = np.array([v[1] for v in obs])
                    b = np.array([
                        v[2]-v[1] @ (
                            (prediction[v[0]]+prediction[(v[0]+1) % 4])/2
                        ) for v in obs
                    ])
                    delta = np.linalg.solve(a.T @ a+.05*np.eye(2), a.T @ b)
                    drift = np.clip(drift+delta, -30*scale, 30*scale)
                velocity = np.clip(
                    q-trajectory[i-direction], -30*scale, 30*scale
                )
                trajectory[i] = prediction = q
            return trajectory

        forward = sweep(int(anchors[0]), 1)
        backward = sweep(int(anchors[-1]), -1)
        for j in range(4):
            times = np.flatnonzero(
                observed_edges[:, (j-1) % 4] & observed_edges[:, j]
            )
            for i in range(n):
                before, after = times[times <= i], times[times >= i]
                wf = 1./(1+i-before[-1])**2 if len(before) else 0.
                wb = 1./(1+after[0]-i)**2 if len(after) else 0.
                if not np.isfinite(forward[i, j]).all():
                    wf = 0.
                if not np.isfinite(backward[i, j]).all():
                    wb = 0.
                if wf+wb:
                    bezel[i, j] = (
                        wf*np.nan_to_num(forward[i, j])
                        + wb*np.nan_to_num(backward[i, j])
                    )/(wf+wb)
        for i in range(n):
            surf, occ = masks(i)
            visible[i], ious[i] = (
                _visibility(bezel[i], (h, w), occ),
                _iou(bezel[i], surf, occ),
            )
            measured[i] = (
                len(edge_cache[i] or []) >= min_edges and surf.any()
                and cv2.isContourConvex(bezel[i].astype(np.float32))
                and visible[i] >= min_visible and ious[i] >= min_iou
            )
            status[i] = "bezel" if measured[i] else "lost"

    for i, obs in enumerate(edge_cache):
        observed_edges[i] = False
        if obs:
            observed_edges[i, [v[0] for v in obs]] = True
            counts[i] = sum(v[4] for v in obs)
            residual[i] = np.sqrt(np.mean([v[3]**2 for v in obs]))

    valid = measured.copy()
    # Only short, bracketed losses may be interpolated.
    missing = np.flatnonzero(~measured)
    for run in np.split(missing, np.flatnonzero(np.diff(missing) > 1)+1):
        if (
            not len(run) or len(run) > max_gap or run[0] == 0
            or run[-1] == n-1
        ):
            continue
        a, b = run[0]-1, run[-1]+1
        if not measured[a] or not measured[b]:
            continue
        for i in run:
            if edge_cache[i] and len(edge_cache[i]) < min_edges:
                continue
            t = (i-a)/(b-a)
            q = (1-t)*bezel[a]+t*bezel[b]
            surf, occ = masks(i)
            iou = _iou(q, surf, occ)
            if (
                cv2.isContourConvex(q.astype(np.float32))
                and _visibility(q, (h, w), occ) >= min_visible
                and (np.isnan(iou) or iou >= min_iou)
            ):
                bezel[i], valid[i] = q, True
                sigma[i], status[i] = 3.*scale, "interpolated"

    # Attach the requested rectangle in reference coordinates.
    quads = np.full_like(bezel, np.nan)
    invS = np.linalg.inv(S)
    for i in np.flatnonzero(valid):
        pose = _matrix(bezel[ref_idx], bezel[i])
        quads[i] = _project(_project(reference, pose), invS)
    sigma /= scale
    residual /= scale
    if smooth:
        quads, shifts, strengths = _smooth(quads, sigma, ref_idx)
    else:
        shifts, strengths = np.zeros(n), np.zeros(n)

    H = np.full((n, 3, 3), np.nan)
    for i in np.flatnonzero(valid):
        q = _project(quads[i], S)
        surf, occ = masks(i)
        v, iou = _visibility(q, (h, w), occ), _iou(q, surf, occ)
        if (
            not cv2.isContourConvex(q.astype(np.float32))
            or cv2.contourArea(q.astype(np.float32), oriented=True) <= 4
            or v < min_visible
        ):
            valid[i], status[i], quads[i] = False, "lost", np.nan
            continue
        visible[i], ious[i] = v, iou
        H[i] = _matrix(quads[i], rq)
        H[i] /= H[i, 2, 2]

    quads[ref_idx], H[ref_idx] = rq, np.eye(3)
    valid[ref_idx], measured[ref_idx], status[ref_idx] = True, True, "reference"
    return dict(
        quads=quads, H=H, valid=valid, visible=visible, status=status,
        diagnostics=dict(
            residual_px=residual, corner_sigma_px=sigma, inliers=counts,
            mask_iou=ious, measured=measured, smoothing_shift_px=shifts,
            smoothing_lambda=strengths, keyframes=1,
            observed_edges=observed_edges, motion_valid=motion_valid,
            motion_inliers=motion_inliers,
        ),
    )
