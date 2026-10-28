"""Offline mask-anchored tracking; screen content is never used."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from pipeline.geometry import _match_corners, _order_quad, _valid_quad


def _intersections(normals, constants):
    return np.array([
        np.linalg.solve(normals[[i - 1, i]], constants[[i - 1, i]])
        for i in range(4)
    ])


def _edge_equations(quad):
    edges = np.roll(quad, -1, axis=0) - quad
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return normals, np.einsum("ij,ij->i", normals, quad)


def _boundary(mask, occluder):
    binary = np.asarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 64:
        return None
    points = contour.reshape(-1, 2).astype(np.float64)
    if occluder is not None:
        excluded = cv2.dilate(
            occluder.astype(np.uint8), np.ones((5, 5), np.uint8)
        )
        pixels = points.astype(int)
        points = points[excluded[pixels[:, 1], pixels[:, 0]] == 0]
    if len(points) < 32:
        return None

    # Clipping boundaries are not evidence of a screen edge.
    h, w = binary.shape
    usable = (
        (points[:, 0] > 1) & (points[:, 0] < w - 2)
        & (points[:, 1] > 1) & (points[:, 1] < h - 2)
    )
    points = points[usable]
    if len(points) < 32:
        return None

    hull = cv2.convexHull(points.astype(np.float32))
    perimeter = cv2.arcLength(hull, True)
    seed = None
    for fraction in (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06):
        approximation = cv2.approxPolyDP(hull, fraction * perimeter, True)
        if len(approximation) == 4:
            candidate = _order_quad(approximation)
            if _valid_quad(candidate):
                seed = candidate
                break
    if seed is None:
        return None

    # Refine onto the blurred mask's 0.5 contour. findContours coordinates
    # are foreground pixel centers; fitting them directly shrinks the quad.
    blurred = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), 0.7)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    samples = points.copy()
    for _ in range(2):
        x = samples[:, 0].astype(np.float32).reshape(-1, 1)
        y = samples[:, 1].astype(np.float32).reshape(-1, 1)
        values, dx, dy = [
            cv2.remap(a, x, y, cv2.INTER_LINEAR).ravel()
            for a in (blurred, gx, gy)
        ]
        gradient = np.column_stack((dx, dy))
        delta = (0.5 - values) / np.maximum(
            np.einsum("ij,ij->i", gradient, gradient), 1e-4
        )
        samples += np.clip(delta, -3, 3)[:, None] * gradient

    edges = np.roll(seed, -1, axis=0) - seed
    lengths = np.linalg.norm(edges, axis=1)
    relative = samples[:, None, :] - seed[None]
    along = np.einsum("nki,ki->nk", relative, edges) / lengths**2
    nearest = seed[None] + np.clip(along, 0, 1)[..., None] * edges[None]
    distance = np.linalg.norm(samples[:, None] - nearest, axis=2)
    owner = distance.argmin(axis=1)
    normals, constants, residuals = [], [], []
    for i in range(4):
        keep = (
            (owner == i) & (along[:, i] > 0.05)
            & (along[:, i] < 0.95)
            & (distance[:, i] < max(4, 0.06 * lengths[i]))
        )
        selected = samples[keep]
        if len(selected) < 8:
            return None
        direction = edges[i] / lengths[i]
        if np.ptp(selected @ direction) < 0.35 * lengths[i]:
            return None
        vx, vy, x0, y0 = cv2.fitLine(
            selected.astype(np.float32), cv2.DIST_HUBER, 0, 0.001, 0.001
        ).ravel()
        normal = np.array([-vy, vx], dtype=np.float64)
        if normal @ (seed.mean(axis=0) - selected.mean(axis=0)) < 0:
            normal *= -1
        constant = float(normal @ [x0, y0])
        normals.append(normal)
        constants.append(constant)
        residuals.extend(np.abs(selected @ normal - constant))
    try:
        quad = _intersections(np.array(normals), np.array(constants))
    except np.linalg.LinAlgError:
        return None
    residual = float(np.percentile(residuals, 90))
    if not _valid_quad(quad) or residual > max(2.5, 0.015 * lengths.min()):
        return None
    return quad, points, 1.0 / (1.0 + residual)


def _reject_outliers(quads, valid, jump_fraction):
    # Leave-one-out Theil-Sen predictions prevent one bad mask from pulling
    # its neighbours' predictions towards itself.
    keep = valid.copy()
    for i in np.flatnonzero(valid):
        indices = np.flatnonzero(valid)
        indices = indices[(np.abs(indices - i) <= 4) & (indices != i)]
        if len(indices) < 3:
            continue
        times = indices - i
        values = quads[indices]
        slopes = [
            (values[b] - values[a]) / (times[b] - times[a])
            for a in range(len(indices)) for b in range(a + 1, len(indices))
        ]
        velocity = np.median(slopes, axis=0)
        prediction = np.median(values - times[:, None, None] * velocity, axis=0)
        expected = np.linalg.norm(
            np.roll(prediction, -1, axis=0) - prediction, axis=1
        )
        actual = np.linalg.norm(
            np.roll(quads[i], -1, axis=0) - quads[i], axis=1
        )
        predicted_area = abs(cv2.contourArea(prediction.astype(np.float32)))
        actual_area = abs(cv2.contourArea(quads[i].astype(np.float32)))
        scale = np.sqrt(predicted_area)
        error = np.linalg.norm(quads[i] - prediction, axis=1).max()
        if (
            not 0.65 <= actual_area / max(predicted_area, 1.0) <= 1.50
            or error > max(6.0, jump_fraction * scale)
            or np.max(np.abs(np.log(actual / np.maximum(expected, 1)))) > 0.30
        ):
            keep[i] = False
    return keep


def _smooth(quads, valid, strength):
    n = len(quads)
    observed = np.flatnonzero(valid)
    flat = quads.reshape(n, 8)
    filled = np.column_stack([
        np.interp(np.arange(n), observed, flat[observed, j])
        for j in range(8)
    ])
    if n < 3 or strength == 0 or len(observed) == 1:
        return filled.reshape(n, 4, 2)
    difference = diags(
        [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
        [0, 1, 2], shape=(n - 2, n),
    )
    weights = valid.astype(float)
    system = diags(weights) + strength * (difference.T @ difference)
    return spsolve(
        system.tocsc(), np.nan_to_num(flat) * weights[:, None]
    ).reshape(n, 4, 2)


def _cover(quads, boundaries, valid, bias, tolerance, strength):
    n = len(quads)
    normals, constants, lower = [], [], []
    for i, quad in enumerate(quads):
        normal, constant = _edge_equations(quad)
        normals.append(normal)
        constants.append(constant)
        needed = np.full(4, bias)
        if tolerance is not None and valid[i]:
            support = (boundaries[i] @ normal.T).min(axis=0)
            # Guard against the final float32 conversion.
            needed = np.maximum(needed, constant - support - tolerance + 1e-3)
        lower.append(needed)
    lower = np.asarray(lower)
    if tolerance is None or n < 3 or strength == 0:
        offsets = lower
    else:
        def objective(flat):
            offsets = flat.reshape(n, 4)
            acceleration = np.diff(offsets, n=2, axis=0)
            error = offsets - bias
            gradient = error.copy()
            gradient[:-2] += strength * acceleration
            gradient[1:-1] -= 2 * strength * acceleration
            gradient[2:] += strength * acceleration
            cost = 0.5 * (
                np.sum(error**2) + strength * np.sum(acceleration**2)
            )
            return cost, gradient.ravel()

        fit = minimize(
            objective, lower.ravel(), jac=True, method="L-BFGS-B",
            bounds=[(float(x), None) for x in lower.ravel()],
            options={"ftol": 1e-11, "gtol": 1e-7, "maxiter": 2000},
        )
        if not fit.success:
            raise RuntimeError("coverage smoothing failed: " + fit.message)
        offsets = np.maximum(fit.x.reshape(n, 4), lower)
    return np.array([
        _intersections(normal, constant - offset)
        for normal, constant, offset in zip(normals, constants, offsets)
    ])


def track_from_masks(
    masks, occluder_masks=None, *, smoothness=0.75, outward_bias=0.25,
    coverage_tolerance=1.0, coverage_smoothness=10.0, jump_fraction=0.06,
):
    """Track an offline sequence using reliable surface masks only.

    Coordinates are pixel centers, initially TL/TR/BR/BL, with positive winding
    and cyclic identities preserved thereafter. None/empty/rejected masks are
    interpolated by the same batch fit. One valid observation gives a held quad.

    Minimize squared measurement error plus smoothness * squared second
    differences: a batch constant-velocity Gaussian smoother with free endpoints.
    No causal filtering delay is introduced. Defaults enforce every retained,
    unoccluded boundary pixel within coverage_tolerance of the output halfplanes.
    coverage_tolerance=None disables that constraint (bias still applies).
    Jitter/error limits depend on mask consistency and cannot be universal.

    Returns float32 quads (N,4,2), confidence (N,), and status (N,) strings:
    ok, occluded, or interpolated. Interpolated confidence is zero.
    """
    parameters = [smoothness, outward_bias, coverage_smoothness, jump_fraction]
    if coverage_tolerance is not None:
        parameters.append(coverage_tolerance)
    if (
        not np.isfinite(parameters).all() or min(parameters) < 0
        or jump_fraction == 0
    ):
        raise ValueError(
            "parameters must be finite and nonnegative; jump_fraction > 0"
        )
    masks = list(masks)
    if not masks:
        raise ValueError("at least one mask is required")
    occluders = (
        [None] * len(masks) if occluder_masks is None else list(occluder_masks)
    )
    if len(occluders) != len(masks):
        raise ValueError("occluder_masks must have one entry per frame")
    shape = next(
        (np.asarray(m).shape for m in masks if m is not None), None
    )
    if shape is None or len(shape) != 2 or 0 in shape:
        raise ValueError("at least one nonempty 2D mask is required")
    quads = np.full((len(masks), 4, 2), np.nan)
    confidence = np.zeros(len(masks), dtype=np.float32)
    boundaries = [None] * len(masks)
    status = np.full(len(masks), "interpolated", dtype="<U12")
    previous = None
    for i, (mask, occluder) in enumerate(zip(masks, occluders)):
        if mask is not None:
            array = np.asarray(mask)
            if array.ndim != 2 or 0 in array.shape:
                raise ValueError("each mask must be a nonempty 2D array")
            if array.shape != shape:
                raise ValueError("mask dimensions must agree")
        if occluder is not None:
            occluder = np.asarray(occluder, dtype=bool)
            if occluder.shape != shape:
                raise ValueError("occluder dimensions must match masks")
        if mask is None or not np.any(mask):
            continue
        fitted = _boundary(array.astype(bool), occluder)
        if fitted is None:
            continue
        quad, points, quality = fitted
        if previous is not None:
            quad = _match_corners(quad, previous)
        quads[i] = previous = quad
        boundaries[i] = points
        confidence[i] = quality
        status[i] = "occluded" if occluder is not None and occluder.any() else "ok"
    valid = _reject_outliers(quads, confidence > 0, jump_fraction)
    if not valid.any():
        raise ValueError("no reliable quadrilateral masks")
    previous = None
    for i in np.flatnonzero(valid):
        if previous is not None:
            quads[i] = _match_corners(quads[i], previous)
        previous = quads[i]
    confidence[~valid] = 0
    status[~valid] = "interpolated"
    result = _smooth(quads, valid, smoothness)
    result = _cover(
        result, boundaries, valid, outward_bias,
        coverage_tolerance, coverage_smoothness,
    )
    if not all(_valid_quad(quad) for quad in result):
        raise ValueError("interpolated trajectory contains a degenerate quad")
    return {
        "quads": result.astype(np.float32),
        "confidence": confidence,
        "status": status,
    }
