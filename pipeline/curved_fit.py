"""Fit upright cylinders from silhouettes and smooth independent frame runs."""

import cv2
import numpy as np


def _extents(mask):
    rows = np.flatnonzero(mask.any(axis=1))
    occupied = mask[rows]
    left = occupied.argmax(axis=1)
    right = mask.shape[1] - 1 - occupied[:, ::-1].argmax(axis=1)
    return rows, left, right


def cylinder_from_mask(mask, band=(0.25, 0.8)) -> dict | None:
    """Remove thin handles, then fit the body's axis and label-band radii."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must have shape (H, W)")
    if not (0 <= band[0] < band[1] <= 1):
        raise ValueError("band must satisfy 0 <= start < end <= 1")
    if np.count_nonzero(binary) < 400:
        return None

    _, left, right = _extents(binary)
    size = max(3, int(round(0.35 * np.median(right - left + 1))) | 1)
    opened = cv2.morphologyEx(
        binary.astype(np.uint8), cv2.MORPH_OPEN,
        np.ones((size, size), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if count <= 1:
        return None
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[component, cv2.CC_STAT_AREA] < 400:
        return None

    rows, left, right = _extents(labels == component)
    if len(rows) < 2:
        return None
    centres = (left + right) * 0.5
    a, b = np.polyfit(rows, centres, 1)
    residual = centres - (a * rows + b)
    deviation = np.abs(residual - np.median(residual))
    keep = deviation <= max(1.5, 3 * 1.4826 * np.median(deviation))
    if np.count_nonzero(keep) >= 2:
        a, b = np.polyfit(rows[keep], centres[keep], 1)

    radii = (right - left) * 0.5
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(radii, (2, 2), mode="edge"), 5
    )
    radii = np.median(windows, axis=1)
    top, bottom = rows[0], rows[-1]
    y0, y1 = np.rint(top + np.asarray(band) * (bottom - top)).astype(int)
    if y1 <= y0:
        return None
    band_rows = np.arange(y0, y1 + 1, dtype=np.int32)
    half_width = np.interp(band_rows, rows, radii)
    if half_width.max() <= 0 or half_width.min() / half_width.max() < 0.4:
        return None
    return dict(a=float(a), b=float(b), y0=int(y0), y1=int(y1),
                rows=band_rows, half_width=half_width)


def smooth_cylinders(cyls, alpha=0.6) -> list:
    """Forward/backward EMA; alpha is the current-sample weight, gaps split runs."""
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    cyls = list(cyls)
    result = [None] * len(cyls)
    valid = np.array([c is not None for c in cyls], dtype=bool)
    boundaries = np.diff(np.r_[False, valid, False].astype(np.int8))
    samples = np.linspace(0.0, 1.0, 32)
    starts, stops = np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
    for start, stop in zip(starts, stops):
        states = []
        for cyl in cyls[start:stop]:
            radii = np.interp(
                cyl["y0"] + samples * (cyl["y1"] - cyl["y0"]),
                cyl["rows"], cyl["half_width"],
            )
            states.append(np.r_[cyl["a"], cyl["b"], cyl["y0"], cyl["y1"], radii])
        states = np.asarray(states, dtype=np.float64)
        for i in range(1, len(states)):
            states[i] = alpha * states[i] + (1 - alpha) * states[i - 1]
        for i in range(len(states) - 2, -1, -1):
            states[i] = alpha * states[i] + (1 - alpha) * states[i + 1]
        for index, state in enumerate(states, start):
            y0, y1 = np.rint(state[2:4]).astype(int)
            rows = np.arange(y0, y1 + 1, dtype=np.int32)
            radii = np.interp(np.linspace(0, 1, len(rows)), samples, state[4:])
            result[index] = dict(
                a=float(state[0]), b=float(state[1]), y0=int(y0), y1=int(y1),
                rows=rows, half_width=radii,
            )
    return result
