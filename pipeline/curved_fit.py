"""Fit upright cylinders from silhouettes and smooth independent frame runs."""

import math

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


def _axis_from_mask(mask):
    rows, left, right = _extents(mask)
    centres = (left + right) * 0.5
    a, b = np.polyfit(rows, centres, 1)
    radii = (right - left) * 0.5
    radius_med = float(np.median(radii))
    return float(a), float(b), radius_med, rows, radii


def fit_rim(frame_bgr, cyl, mask):
    """Find the white-lip lower edge y = A + e*sqrt(1-u^2).

    Returns dict(A=float, e=float) or None.
    """
    try:
        frame = np.asarray(frame_bgr)
        m = np.asarray(mask, dtype=bool)
    except Exception:
        return None
    if m.ndim != 2:
        return None
    if frame.ndim != 3 or frame.shape[2] != 3:
        return None
    if frame.shape[0] != m.shape[0] or frame.shape[1] != m.shape[1]:
        return None
    H, W = m.shape

    # Axis and median half width from cyl if available, else from mask.
    a = b = None
    radius_med = None
    try:
        if isinstance(cyl, dict) and "a" in cyl and "b" in cyl:
            a = float(cyl["a"])
            b = float(cyl["b"])
            if "half_width" in cyl and np.asarray(cyl["half_width"]).size > 0:
                radius_med = float(np.median(np.asarray(cyl["half_width"], dtype=np.float64)))
    except Exception:
        a = b = None
        radius_med = None
    if a is None or b is None or radius_med is None or not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(radius_med) or radius_med <= 0:
        try:
            if np.count_nonzero(m) < 10:
                return None
            a, b, radius_med, _, _ = _axis_from_mask(m)
        except Exception:
            return None
    if not np.isfinite(radius_med) or radius_med <= 0:
        return None

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None

    # Candidate body columns with |u| <= 0.8.
    col_has = np.flatnonzero(m.any(axis=0))
    cand_x = []
    yt_map = {}
    yb_map = {}
    for x in col_has:
        xi = int(x)
        col = m[:, xi]
        ys = np.flatnonzero(col)
        if ys.size == 0:
            continue
        yt = int(ys[0])
        yb = int(ys[-1])
        yt_map[xi] = yt
        yb_map[xi] = yb
        ct = float(a * float(yt) + b)
        u = (float(xi) - ct) / float(radius_med)
        if abs(u) <= 0.8:
            cand_x.append(xi)
    n_cand = len(cand_x)
    if n_cand < 8:
        return None

    # Vertically smoothed profile: mean over small x window + 1-D Gaussian sigma~1.5.
    sigma = 1.5
    rad = 4
    kx = np.arange(-rad, rad + 1, dtype=np.float64)
    kern = np.exp(-0.5 * (kx / sigma) ** 2)
    kern = kern / float(np.sum(kern))

    y_steps = []
    s_list = []
    step_mags = []
    for xi in cand_x:
        yt = yt_map[xi]
        yb = yb_map[xi]
        # Search rows from mask top down to top + 0.3*(y1-top), y1 = column bottom.
        span = float(yb - yt)
        if span < 7:
            continue
        y_end = int(round(float(yt) + 0.3 * span))
        y_end = max(y_end, yt + 6)
        y_end = min(y_end, H - 1)
        y_start = max(0, yt)
        if y_end <= y_start + 4:
            continue
        x0 = max(0, xi - 1)
        x1 = min(W, xi + 2)
        patch = gray[y_start:y_end + 1, x0:x1].astype(np.float64)
        if patch.size == 0:
            continue
        prof_raw = patch.mean(axis=1)
        n = int(prof_raw.size)
        if n < 7:
            continue
        padded = np.pad(prof_raw, (rad, rad), mode="edge")
        prof = np.convolve(padded, kern, mode="valid")
        # Central-difference derivative (grey levels / px).
        d = np.empty(n, dtype=np.float64)
        d[1:-1] = (prof[2:] - prof[:-2]) * 0.5
        d[0] = prof[1] - prof[0]
        d[-1] = prof[-1] - prof[-2]
        idx = int(np.argmin(d))
        d_min = float(d[idx])
        step_mag = float(-d_min)
        # Subpixel refinement: parabola fit to derivative around extremum.
        if 0 < idx < n - 1:
            y0d = float(d[idx - 1])
            y1d = float(d[idx])
            y2d = float(d[idx + 1])
            denom = y0d - 2.0 * y1d + y2d
            if abs(denom) > 1e-9:
                off = 0.5 * (y0d - y2d) / denom
                if off < -0.5:
                    off = -0.5
                elif off > 0.5:
                    off = 0.5
            else:
                off = 0.0
            y_ref = float(y_start + idx + off)
        else:
            y_ref = float(y_start + idx)
        ct2 = float(a * y_ref + b)
        u2 = (float(xi) - ct2) / float(radius_med)
        if abs(u2) > 1.0:
            continue
        s = float(np.sqrt(max(0.0, 1.0 - u2 * u2)))
        y_steps.append(float(y_ref))
        s_list.append(s)
        step_mags.append(step_mag)

    if len(y_steps) < 0.3 * float(n_cand):
        return None
    if len(y_steps) < 8:
        return None
    med_step = float(np.median(np.asarray(step_mags, dtype=np.float64)))
    if not np.isfinite(med_step) or med_step < 8.0:
        return None
    fit = _robust_AB(s_list, y_steps)
    if fit is None:
        return None
    A_fit, B_fit = fit
    e_fit = float(B_fit)
    if not (np.isfinite(A_fit) and np.isfinite(e_fit)):
        return None
    # Signed: the lip's front arc bows up (e < 0) when the camera is below the rim, down when above it.
    if abs(e_fit) > 0.6 * float(radius_med):
        return None
    return dict(A=float(A_fit), e=float(e_fit))


def cylinder_from_mask(mask, band=(0.25, 0.8), frame=None, top="band") -> dict | None:
    """Remove thin handles, then fit the body's axis and label-band radii."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must have shape (H, W)")
    if not (0 <= band[0] < band[1] <= 1):
        raise ValueError("band must satisfy 0 <= start < end <= 1")
    if top not in ("band", "rim"):
        raise ValueError('top must be "band" or "rim"')
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
    top_f, bottom_f = float(A_t), float(A_b)
    y0_band = float(top_f + float(band[0]) * (bottom_f - top_f))
    y1_band = float(top_f + float(band[1]) * (bottom_f - top_f))
    if y1_band <= y0_band:
        return None
    if abs(bottom_f - top_f) < 1e-9:
        e0_band = e1_band = (e_top + e_bottom) * 0.5
    else:
        e0_band = e_top + (e_bottom - e_top) * (y0_band - top_f) / (bottom_f - top_f)
        e1_band = e_top + (e_bottom - e_top) * (y1_band - top_f) / (bottom_f - top_f)
    e0_band = float(max(0.0, e0_band))
    e1_band = float(max(0.0, e1_band))
    r0_b = int(math.floor(y0_band))
    r1_b = int(math.ceil(y1_band))
    band_rows = np.arange(r0_b, r1_b + 1, dtype=np.int32)
    half_width_band = np.interp(band_rows, rows, radii)
    if half_width_band.max() <= 0 or half_width_band.min() / half_width_band.max() < 0.4:
        return None
    base = dict(a=float(a), b=float(b), y0=float(y0_band), y1=float(y1_band),
                rows=band_rows, half_width=half_width_band,
                sag_top=float(e_top), sag_bottom=float(e_bottom),
                e0=float(e0_band), e1=float(e1_band),
                top_source="band")

    if top == "rim" and frame is not None:
        try:
            fr = np.asarray(frame)
            if fr.ndim == 3 and fr.shape[0] == binary.shape[0] and fr.shape[1] == binary.shape[1]:
                rim = fit_rim(fr, base, binary)
                if rim is not None:
                    A_rim = float(rim["A"])
                    e_rim = float(rim["e"])
                    y0_rim = float(A_rim - 0.5)
                    y1_rim = float(y1_band)
                    if np.isfinite(y0_rim) and np.isfinite(e_rim) and y1_rim > y0_rim:
                        r0 = int(math.floor(y0_rim))
                        r1 = int(math.ceil(y1_rim))
                        rows_rim = np.arange(r0, r1 + 1, dtype=np.int32)
                        hw_rim = np.interp(rows_rim, rows, radii)
                        if hw_rim.max() > 0 and hw_rim.min() / hw_rim.max() >= 0.4:
                            return dict(a=float(a), b=float(b), y0=float(y0_rim), y1=float(y1_rim),
                                        rows=rows_rim, half_width=hw_rim,
                                        sag_top=float(e_top), sag_bottom=float(e_bottom),
                                        e0=float(e_rim), e1=float(e1_band),
                                        top_source="rim")
        except Exception:
            pass
        # Fall back to band but record band source.
        base["top_source"] = "band"
        return base
    return base


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
                float(cyl["y0"]) + samples * (float(cyl["y1"]) - float(cyl["y0"])),
                cyl["rows"], cyl["half_width"],
            )
            states.append(np.r_[float(cyl["a"]), float(cyl["b"]), float(cyl["y0"]), float(cyl["y1"]),
                                _scalar(cyl, "e0"), _scalar(cyl, "e1"),
                                _scalar(cyl, "sag_top"), _scalar(cyl, "sag_bottom"),
                                radii])
        states = np.asarray(states, dtype=np.float64)
        for i in range(1, len(states)):
            states[i] = alpha * states[i] + (1 - alpha) * states[i - 1]
        for i in range(len(states) - 2, -1, -1):
            states[i] = alpha * states[i] + (1 - alpha) * states[i + 1]
        for k, state in enumerate(states):
            index = start + k
            y0 = float(state[2])
            y1 = float(state[3])
            r0 = int(math.floor(y0))
            r1 = int(math.ceil(y1))
            if r1 < r0:
                r1 = r0
            rows = np.arange(r0, r1 + 1, dtype=np.int32)
            if y1 > y0 and len(rows) > 1:
                norm = (rows.astype(np.float64) - y0) / (y1 - y0)
                norm = np.clip(norm, 0.0, 1.0)
                radii = np.interp(norm, samples, state[8:])
            else:
                radii = np.interp(np.linspace(0, 1, len(rows)), samples, state[8:])
            orig = cyls[index]
            try:
                ts = orig.get("top_source", "band") if isinstance(orig, dict) else "band"
            except Exception:
                ts = "band"
            if ts not in ("rim", "band"):
                ts = "band"
            result[index] = dict(
                a=float(state[0]), b=float(state[1]), y0=float(y0), y1=float(y1),
                rows=rows, half_width=radii,
                e0=float(state[4]), e1=float(state[5]),
                sag_top=float(state[6]), sag_bottom=float(state[7]),
                top_source=ts,
            )
    return result


def hold_static(cyls, tol_px=3.0) -> list:
    """Give every frame one shared geometry when the object does not move (axis and label rows each vary by less
    than tol_px over the clip): per-frame fit noise would otherwise make the label edges shimmer."""
    good = [c for c in cyls if c is not None]
    if not good:
        return list(cyls)
    keys = ("a", "b", "y0", "y1", "e0", "e1", "sag_top", "sag_bottom")
    vals = {k: np.array([_scalar(c, k) for c in good]) for k in keys}
    if max(np.ptp(vals[k]) for k in ("b", "y0", "y1")) >= tol_px:
        return list(cyls)
    held = {k: float(np.median(v)) for k, v in vals.items()}
    rows = np.arange(int(math.floor(held["y0"])), int(math.ceil(held["y1"])) + 1, dtype=np.int32)
    half_width = np.median([np.interp(rows, c["rows"], c["half_width"]) for c in good], axis=0)
    shared = dict(good[len(good) // 2], **held, rows=rows, half_width=half_width)
    return [None if c is None else dict(shared) for c in cyls]
