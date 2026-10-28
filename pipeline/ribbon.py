"""Mask-anchored horizontal ribbons with straight vertical end segments."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from pipeline.composite import estimate_blur_and_grain


def _mask(value, shape=None, name="mask"):
    array = np.asarray(value)
    if array.ndim != 2 or array.dtype != np.bool_ or array.size == 0:
        raise ValueError(f"{name} must be a nonempty 2D bool array")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return array


def _arrays(ribbon):
    if ribbon is None:
        return None
    try:
        x, top, bottom = (
            np.asarray(ribbon[key], dtype=np.float64)
            for key in ("x", "top", "bottom")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("ribbon needs x, top and bottom arrays") from error
    if (
        x.ndim != 1 or len(x) < 2 or top.shape != x.shape
        or bottom.shape != x.shape
        or not all(np.isfinite(a).all() for a in (x, top, bottom))
        or np.any(np.diff(x) <= 0) or np.any(bottom <= top)
    ):
        raise ValueError("ribbon arrays must be finite, ordered and noncrossing")
    return x, top, bottom


def _fit_boundary(x, y, valid, target, side):
    """Robust quartic seed followed by an IRLS cubic B-spline fit."""
    z = (x - x[0]) / np.ptp(x)
    tz = (target - x[0]) / np.ptp(x)
    polynomial = np.polynomial.legendre.legvander(2 * z - 1, 4)
    weights = valid.astype(float)
    for _ in range(10):
        root = np.sqrt(weights)
        coefficients = np.linalg.lstsq(
            polynomial * root[:, None], y * root, rcond=None
        )[0]
        residual = y - polynomial @ coefficients
        centre = np.median(residual[valid])
        scale = max(0.6, 1.4826 * np.median(abs(residual[valid] - centre)))
        weights = valid * np.minimum(
            1.0, 2 * scale / np.maximum(abs(residual), 1e-6)
        )
        # Inward notches are less credible than an enclosing boundary.
        weights *= np.where(side * residual < -2 * scale, 0.02, 1.0)

    knots = np.r_[np.zeros(4), np.linspace(0, 1, 9)[1:-1], np.ones(4)]
    basis = BSpline.design_matrix(z, knots, 3).toarray()
    evaluation = BSpline.design_matrix(
        tz, knots, 3, extrapolate=True
    ).toarray()
    difference = np.diff(np.eye(basis.shape[1]), n=2, axis=0)
    penalty = 0.01 * difference.T @ difference + 1e-7 * np.eye(basis.shape[1])
    for _ in range(7):
        coefficients = np.linalg.solve(
            basis.T @ (weights[:, None] * basis) + penalty,
            basis.T @ (weights * y),
        )
        residual = y - basis @ coefficients
        centre = np.median(residual[valid])
        scale = max(0.5, 1.4826 * np.median(abs(residual[valid] - centre)))
        weights = valid * np.minimum(
            1.0, 2 * scale / np.maximum(abs(residual), 1e-6)
        )
        weights *= np.where(side * residual < -2 * scale, 0.05, 1.0)
    return evaluation @ coefficients


def ribbon_from_mask(mask, occluder=None, samples=48) -> dict | None:
    """Fit a horizontal ribbon, interpolating occluded boundary columns.

    Pass the union of all foreground occluders. Top and bottom observations are
    excluded independently within two pixels of an occluder. At least 16 usable
    columns per boundary, 16 pixels of width and 64 foreground pixels are needed.

    The outer 1.5% of columns are reserved for straight end segments rather
    than mistaken for curved top/bottom measurements. Frame clipping is retained
    as visible geometry; unseen screen area is not reconstructed.

    Pixel-edge conversion and a 0.5 px outward bias prevent systematic shrinkage.
    Additional expansion, capped at two pixels, targets 99.7% visible coverage.
    """
    if (
        isinstance(samples, (bool, np.bool_))
        or not isinstance(samples, (int, np.integer)) or samples < 4
    ):
        raise ValueError("samples must be an integer >= 4")
    binary = _mask(mask)
    occ = (
        np.zeros(binary.shape, bool) if occluder is None
        else _mask(occluder, binary.shape, "occluder")
    )
    if np.count_nonzero(binary) < 64:
        return None

    # Retain meaningful disconnected pieces, including pieces split by a person.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.r_[False, areas >= max(16, 0.005 * areas.max())]
    visible = keep[labels] & ~occ
    columns = np.flatnonzero(visible.any(axis=0))
    if len(columns) < 16 or np.ptp(columns) < 16:
        return None

    x = np.arange(columns[0], columns[-1] + 1, dtype=float)
    xi = x.astype(int)
    present = visible[:, xi].any(axis=0)
    top = visible[:, xi].argmax(axis=0)
    bottom = binary.shape[0] - 1 - visible[::-1, xi].argmax(axis=0)
    excluded = cv2.dilate(occ.astype(np.uint8), np.ones((5, 5), np.uint8))
    margin = max(3.0, 0.015 * np.ptp(x))
    central = (x >= x[0] + margin) & (x <= x[-1] - margin)
    valid_top = present & central & (excluded[top, xi] == 0)
    valid_bottom = present & central & (excluded[bottom, xi] == 0)
    if min(valid_top.sum(), valid_bottom.sum()) < 16:
        return None

    target = np.linspace(x[0] - 0.5, x[-1] + 0.5, samples)
    try:
        fitted_top = _fit_boundary(x, top - 0.5, valid_top, target, -1) - 0.5
        fitted_bottom = _fit_boundary(
            x, bottom + 0.5, valid_bottom, target, 1
        ) + 0.5
    except np.linalg.LinAlgError:
        return None
    if (
        not np.isfinite([fitted_top, fitted_bottom]).all()
        or np.min(fitted_bottom - fitted_top) < 2
    ):
        return None

    # Cap coverage correction so isolated segmentation errors cannot grow the
    # whole surface arbitrarily. Quantiles ignore the worst 0.3%.
    yy, xx = np.nonzero(binary & ~occ)
    gap = np.maximum(
        np.interp(xx, target, fitted_top) - yy,
        yy - np.interp(xx, target, fitted_bottom),
    )
    gap = np.maximum(gap, np.maximum(target[0] - xx, xx - target[-1]))
    extra = float(np.clip(np.quantile(gap, 0.997) + 0.05, 0, 2))
    target = target + np.linspace(-extra, extra, samples)
    fitted_top -= extra
    fitted_bottom += extra
    return {"x": target, "top": fitted_top, "bottom": fitted_bottom}


def smooth_ribbons(ribbons, smoothness=0.05):
    """Batch least-squares fit with a squared second-difference penalty.

    None frames are interpolated only between observations; leading/trailing
    None frames stay None. Samples must correspond and have equal counts.
    Larger smoothness suppresses more jitter but can lose mask coverage where
    segmentation changes abruptly. Zero interpolates gaps without filtering.
    Inputs are never mutated, and output arrays do not alias input arrays.
    """
    strength = float(smoothness)
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("smoothness must be finite and nonnegative")
    ribbons = list(ribbons)
    arrays = [_arrays(ribbon) for ribbon in ribbons]
    indices = [i for i, array in enumerate(arrays) if array is not None]
    result = [None] * len(ribbons)
    if not indices:
        return result

    samples = len(arrays[indices[0]][0])
    if any(len(arrays[i][0]) != samples for i in indices):
        raise ValueError("ribbons must have equal sample counts")
    start, stop = indices[0], indices[-1] + 1
    n = stop - start
    observed = np.array([a is not None for a in arrays[start:stop]])
    values = np.array([
        np.concatenate(a) if a is not None else np.zeros(3 * samples)
        for a in arrays[start:stop]
    ])
    filled = np.column_stack([
        np.interp(np.arange(n), np.flatnonzero(observed), values[observed, j])
        for j in range(values.shape[1])
    ])
    if n >= 3 and strength > 0:
        difference = diags(
            [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
            [0, 1, 2], shape=(n - 2, n),
        )
        system = diags(observed.astype(float)) + strength * (
            difference.T @ difference
        )
        filled = spsolve(system.tocsc(), values)

    for i, row in enumerate(filled):
        x, top, bottom = np.split(row, 3)
        if np.any(np.diff(x) <= 0) or np.any(bottom <= top):
            continue
        result[start + i] = dict(x=x.copy(), top=top.copy(), bottom=bottom.copy())
    return result


def _creative(value):
    array = np.asarray(value)
    if (
        array.ndim != 3 or array.shape[2] != 4
        or array.dtype != np.uint8 or min(array.shape[:2]) == 0
    ):
        raise ValueError("creative_rgba must be a nonempty RGBA uint8 array")
    return array


def warp_to_ribbon(creative_rgba, ribbon, frame_shape):
    """Inverse-map straight RGBA; u follows projected midline arc length.

    v is linear between the curves. Straight vertical ends close the polygon.
    Feathering extends one pixel inward and never grows support outside the
    supplied ribbon. Premultiplied interpolation prevents transparent RGB halos.
    Arc length is measured in the image plane, not inferred physical depth.
    """
    creative = _creative(creative_rgba)
    if (
        len(frame_shape) not in (2, 3)
        or any(isinstance(v, (bool, np.bool_))
               or not isinstance(v, (int, np.integer)) or v <= 0
               for v in frame_shape[:2])
    ):
        raise ValueError("frame_shape must contain positive integer height/width")
    height, width = map(int, frame_shape[:2])
    output = np.zeros((height, width, 4), np.uint8)
    arrays = _arrays(ribbon)
    if arrays is None:
        return output

    x, top, bottom = arrays
    left = max(0, int(np.ceil(x[0])))
    right = min(width, int(np.floor(x[-1])) + 1)
    upper = max(0, int(np.floor(top.min())))
    lower = min(height, int(np.ceil(bottom.max())) + 1)
    if right <= left or lower <= upper:
        return output

    xx = np.arange(left, right, dtype=float)
    yy = np.arange(upper, lower, dtype=float)[:, None]
    t = np.interp(xx, x, top)
    b = np.interp(xx, x, bottom)
    middle = (top + bottom) / 2
    length = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(middle)))]
    u = np.interp(xx, x, length) / length[-1]
    v = (yy - t[None, :]) / (b - t)[None, :]
    map_x = np.broadcast_to(
        u * (creative.shape[1] - 1), v.shape
    ).astype(np.float32)
    map_y = (v * (creative.shape[0] - 1)).astype(np.float32)

    alpha = creative[..., 3:4].astype(np.float32) / 255
    source = np.concatenate(
        (creative[..., :3].astype(np.float32) * alpha, alpha), axis=2
    )
    mapped = cv2.remap(
        source, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    dt = np.interp(xx, x, np.gradient(top, x))
    db = np.interp(xx, x, np.gradient(bottom, x))
    distance = np.minimum(
        (yy - t[None, :]) / np.hypot(1, dt)[None, :],
        (b[None, :] - yy) / np.hypot(1, db)[None, :],
    )
    distance = np.minimum(
        distance, np.minimum(xx - x[0], x[-1] - xx)[None, :]
    )
    coverage = np.clip(distance, 0, 1)
    colour = np.divide(
        mapped[..., :3], mapped[..., 3:4],
        out=np.zeros_like(mapped[..., :3]), where=mapped[..., 3:4] > 1e-8,
    )
    warped_alpha = np.clip(mapped[..., 3] * coverage, 0, 1)
    rgba = np.concatenate((colour, 255 * warped_alpha[..., None]), axis=2)
    rgba = np.clip(np.rint(rgba), 0, 255).astype(np.uint8)
    rgba[rgba[..., 3] == 0] = 0
    output[upper:lower, left:right] = rgba
    return output


def composite_ribbon(
    frame_bgr, creative_rgba, ribbon, occluder_mask=None, emissive=True,
):
    """Match appearance, preserving every zero-alpha and occluded pixel."""
    frame = np.asarray(frame_bgr)
    if (
        frame.ndim != 3 or frame.shape[2] != 3
        or frame.dtype != np.uint8 or min(frame.shape[:2]) == 0
    ):
        raise ValueError("frame_bgr must be a nonempty BGR uint8 array")
    occ = (
        np.zeros(frame.shape[:2], bool) if occluder_mask is None
        else _mask(occluder_mask, frame.shape[:2], "occluder_mask")
    )
    warped = warp_to_ribbon(creative_rgba, ribbon, frame.shape)
    result = frame.copy()
    active = (warped[..., 3] > 0) & ~occ
    if not active.any():
        return result

    ys, xs = np.nonzero(warped[..., 3])
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    region = np.s_[y0:y1, x0:x1]
    original = frame[region].astype(np.float32)
    alpha = warped[region][..., 3:4].astype(np.float32) / 255
    colour = warped[region][..., :3][..., ::-1].astype(np.float32)
    visible = active[region]
    quad = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32
    )
    blur, grain = estimate_blur_and_grain(frame, quad)
    if blur > 0:
        # Blur colour without spreading alpha outside the ribbon or into holes.
        numerator = cv2.GaussianBlur(colour * alpha, (0, 0), blur)
        denominator = cv2.GaussianBlur(alpha[..., 0], (0, 0), blur)[..., None]
        colour = numerator / np.maximum(denominator, 1e-6)

    target_luma = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if emissive:
        source_luma = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
        weights = alpha[..., 0] * visible
        source_total = float(np.sum(source_luma * weights, dtype=np.float64))
        if source_total > 1e-6:
            gain = float(
                np.sum(target_luma * weights, dtype=np.float64)
            ) / source_total
            peak = float(colour[visible].max())
            colour *= min(gain, 255 / max(peak, 1))
    else:
        # Occluder colours cannot influence the low-frequency shading estimate.
        weight = visible.astype(np.float32)
        sigma = max(1.0, min(original.shape[:2]) / 12)
        shading = cv2.GaussianBlur(target_luma * weight, (0, 0), sigma)
        support = cv2.GaussianBlur(weight, (0, 0), sigma)
        shading /= np.maximum(support, 1e-6)
        mean = float(np.mean(shading[visible]))
        if mean > 1e-6:
            colour *= (shading / mean)[..., None]
            colour *= np.minimum(
                1.0, 255 / np.maximum(colour.max(axis=2, keepdims=True), 1)
            )

    if grain > 0:
        colour += np.random.default_rng(0).normal(
            0, grain, colour.shape
        ).astype(np.float32)
    blended = np.clip(
        np.rint(colour * alpha + original * (1 - alpha)), 0, 255
    ).astype(np.uint8)
    destination = result[region]
    destination[visible] = blended[visible]
    return result
