"""Fit upright cylinders from silhouettes and smooth independent frame runs."""

import cv2
import numpy as np


def _extents(mask):
    rows = np.flatnonzero(mask.any(axis=1))
    occupied = mask[rows]
    left = occupied.argmax(axis=1)
    right = mask.shape[1] - 1 - occupied[:, ::-1].argmax(axis=1)
    return rows, left, right


def _robust_AB(s, y):
    """Robust fit y = A + B*s. Returns (A, B) or None if <8 samples."""
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if s.size < 8:
        return None
    A = np.column_stack([np.ones_like(s), s])
    coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred = coeff[0] + coeff[1] * s
    res = y - pred
    med = float(np.median(res))
    mad = float(np.median(np.abs(res - med)))
    thr = max(1.5, 3 * 1.4826 * mad)
    keep = np.abs(res - med) <= thr
    n_keep = int(np.count_nonzero(keep))
    if n_keep >= 8 and n_keep < s.size:
        A2 = np.column_stack([np.ones(n_keep), s[keep]])
        coeff, _, _, _ = np.linalg.lstsq(A2, y[keep], rcond=None)
    return float(coeff[0]), float(coeff[1])


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

    # Width-variation rejection on the original rows, as before.
    top_old, bottom_old = int(rows[0]), int(rows[-1])
    y0_old, y1_old = np.rint(top_old + np.asarray(band) * (bottom_old - top_old)).astype(int)
    if y1_old <= y0_old:
        return None
    band_rows_old = np.arange(y0_old, y1_old + 1, dtype=np.int32)
    half_width_old = np.interp(band_rows_old, rows, radii)
    if half_width_old.max() <= 0 or half_width_old.min() / half_width_old.max() < 0.4:
        return None

    radius_med = float(np.median(radii))

    # Ellipse sag from ORIGINAL mask restricted to dilated body component.
    comp_open = (labels == component)
    kernel = np.ones((size, size), np.uint8)
    dilated = cv2.dilate(
        comp_open.astype(np.uint8), kernel,
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    body = binary & dilated

    xs = np.flatnonzero(body.any(axis=0))
    s_top, y_top, s_bot, y_bot = [], [], [], []
    for xi in xs:
        x = int(xi)
        col = body[:, x]
        ys = np.flatnonzero(col)
        if ys.size == 0:
            continue
        yt, yb = int(ys[0]), int(ys[-1])
        ct = float(a * yt + b)
        cb = float(a * yb + b)
        if radius_med <= 0:
            continue
        ut = (float(x) - ct) / radius_med
        ub = (float(x) - cb) / radius_med
        if abs(ut) <= 0.9:
            s_top.append(float(np.sqrt(max(0.0, 1.0 - ut * ut))))
            y_top.append(float(yt))
        if abs(ub) <= 0.9:
            s_bot.append(float(np.sqrt(max(0.0, 1.0 - ub * ub))))
            y_bot.append(float(yb))
    fit_top = _robust_AB(s_top, y_top)
    fit_bot = _robust_AB(s_bot, y_bot)
    if fit_top is None:
        A_t = float(rows[0])
        e_top = 0.0
    else:
        A_t, B_t = fit_top
        e_top = -B_t
    if fit_bot is None:
        A_b = float(rows[-1])
        e_bottom = 0.0
    else:
        A_b, B_b = fit_bot
        e_bottom = B_b
    lim = 0.6 * radius_med
    e_top = float(np.clip(e_top, 0.0, lim))
    e_bottom = float(np.clip(e_bottom, 0.0, lim))

    if not (np.isfinite(A_t) and np.isfinite(A_b)):
        return None
    if A_b <= A_t:
        return None
    top, bottom = float(A_t), float(A_b)
    y0, y1 = np.rint(top + np.asarray(band) * (bottom - top)).astype(int)
    if y1 <= y0:
        return None
    if abs(bottom - top) < 1e-9:
        e0 = e1 = (e_top + e_bottom) * 0.5
    else:
        e0 = e_top + (e_bottom - e_top) * (float(y0) - top) / (bottom - top)
        e1 = e_top + (e_bottom - e_top) * (float(y1) - top) / (bottom - top)
    e0 = float(max(0.0, e0))
    e1 = float(max(0.0, e1))
    band_rows = np.arange(y0, y1 + 1, dtype=np.int32)
    half_width = np.interp(band_rows, rows, radii)
    if half_width.max() <= 0 or half_width.min() / half_width.max() < 0.4:
        return None
    return dict(a=float(a), b=float(b), y0=int(y0), y1=int(y1),
                rows=band_rows, half_width=half_width,
                sag_top=float(e_top), sag_bottom=float(e_bottom),
                e0=float(e0), e1=float(e1))


def _scalar(cyl, key):
    try:
        v = float(cyl.get(key, 0.0))
    except Exception:
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return v


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
            states.append(np.r_[cyl["a"], cyl["b"], cyl["y0"], cyl["y1"],
                                _scalar(cyl, "e0"), _scalar(cyl, "e1"),
                                _scalar(cyl, "sag_top"), _scalar(cyl, "sag_bottom"),
                                radii])
        states = np.asarray(states, dtype=np.float64)
        for i in range(1, len(states)):
            states[i] = alpha * states[i] + (1 - alpha) * states[i - 1]
        for i in range(len(states) - 2, -1, -1):
            states[i] = alpha * states[i] + (1 - alpha) * states[i + 1]
        for index, state in enumerate(states, start):
            y0, y1 = np.rint(state[2:4]).astype(int)
            rows = np.arange(y0, y1 + 1, dtype=np.int32)
            radii = np.interp(np.linspace(0, 1, len(rows)), samples, state[8:])
            result[index] = dict(
                a=float(state[0]), b=float(state[1]), y0=int(y0), y1=int(y1),
                rows=rows, half_width=radii,
                e0=float(state[4]), e1=float(state[5]),
                sag_top=float(state[6]), sag_bottom=float(state[7]),
            )
    return result
