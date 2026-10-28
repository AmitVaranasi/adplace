"""Colour-guided spatial refinement of binary person occluder masks."""

from __future__ import annotations

from itertools import zip_longest
from operator import index

import cv2
import numpy as np


def _guided_filter_colour(guide, source, radius, eps):
    """Colour guided filter with an explicit per-pixel covariance inverse."""
    size = (2 * radius + 1, 2 * radius + 1)

    def mean(array):
        return cv2.boxFilter(
            array,
            cv2.CV_32F,
            size,
            normalize=True,
            borderType=cv2.BORDER_REFLECT_101,
        )

    mean_i = mean(guide)
    mean_p = mean(source)
    cov_ip = mean(guide * source[..., None]) - mean_i * mean_p[..., None]

    b, g, r = (guide[..., c] for c in range(3))
    mb, mg, mr = (mean_i[..., c] for c in range(3))

    bb = mean(b * b) - mb * mb + eps
    bg = mean(b * g) - mb * mg
    br = mean(b * r) - mb * mr
    gg = mean(g * g) - mg * mg + eps
    gr = mean(g * r) - mg * mr
    rr = mean(r * r) - mr * mr + eps

    # Adjugate of the symmetric matrix cov(I) + eps * identity.
    c00 = gg * rr - gr * gr
    c01 = br * gr - bg * rr
    c02 = bg * gr - br * gg
    c11 = bb * rr - br * br
    c12 = bg * br - bb * gr
    c22 = bb * gg - bg * bg

    det = bb * c00 + bg * c01 + br * c02
    inv_det = np.float32(1.0) / np.maximum(det, np.finfo(np.float32).tiny)

    pb, pg, pr = (cov_ip[..., c] for c in range(3))
    a = np.empty_like(guide)
    a[..., 0] = (c00 * pb + c01 * pg + c02 * pr) * inv_det
    a[..., 1] = (c01 * pb + c11 * pg + c12 * pr) * inv_det
    a[..., 2] = (c02 * pb + c12 * pg + c22 * pr) * inv_det

    intercept = mean_p - np.sum(a * mean_i, axis=2)
    return np.sum(mean(a) * guide, axis=2) + mean(intercept)


def refine_occluder(
    frame_bgr,
    mask,
    band_px=8,
    radius=4,
    eps=1e-3,
    *,
    sharpen=True,
    gain=1.5,
):
    """Return HxW float32 alpha: 1 preserves the person, 0 reveals the ad.

    frame_bgr must be uint8 BGR; mask is a matching 2-D binary array.
    Elliptical erosion/dilation defines the unknown band. Filtering runs
    on its bounding box plus a 2*radius margin for the two filter passes.
    Inputs are not modified. There is no inter-frame temporal state.
    """
    frame = np.asarray(frame_bgr)
    binary = np.asarray(mask, dtype=bool)

    if (
        frame.ndim != 3
        or frame.shape[2] != 3
        or frame.dtype != np.uint8
        or min(frame.shape[:2]) == 0
    ):
        raise ValueError("frame_bgr must be a nonempty HxWx3 uint8 array")
    if binary.shape != frame.shape[:2]:
        raise ValueError("mask must have shape (H, W)")

    band_px, radius = index(band_px), index(radius)
    if band_px < 0 or radius < 0:
        raise ValueError("band_px and radius must be nonnegative integers")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be finite and positive")
    eps, gain = np.float32(eps), np.float32(gain)

    alpha = binary.astype(np.float32)
    if band_px == 0 or not binary.any() or binary.all():
        return alpha

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1)
    )
    u8 = binary.astype(np.uint8)
    # Replication treats image edges as cropped silhouettes.
    inside = cv2.erode(u8, kernel, borderType=cv2.BORDER_REPLICATE)
    dilated = cv2.dilate(u8, kernel, borderType=cv2.BORDER_REPLICATE)
    band = dilated != inside

    # Reduce along axes instead of allocating coordinates for every band pixel.
    rows = np.flatnonzero(band.any(axis=1))
    cols = np.flatnonzero(band.any(axis=0))
    if not rows.size:
        return alpha

    height, width = binary.shape
    margin = 2 * radius
    y0 = max(0, int(rows[0]) - margin)
    y1 = min(height, int(rows[-1]) + 1 + margin)
    x0 = max(0, int(cols[0]) - margin)
    x1 = min(width, int(cols[-1]) + 1 + margin)
    roi = np.s_[y0:y1, x0:x1]

    guide = frame[roi].astype(np.float32) * np.float32(1.0 / 255.0)
    q = _guided_filter_colour(guide, alpha[roi], radius, eps)
    np.clip(q, 0.0, 1.0, out=q)

    if sharpen:
        q -= np.float32(0.5)
        q *= gain
        q += np.float32(0.5)
        np.clip(q, 0.0, 1.0, out=q)

    # Only unknown pixels change. Sure regions remain exactly binary.
    np.copyto(alpha[roi], q, where=band[roi])
    return alpha


def refine_occluders(frames, masks, **kw):
    """Refine paired iterables in order, preserving None masks."""
    missing = object()
    result = []
    for frame, mask in zip_longest(frames, masks, fillvalue=missing):
        if frame is missing or mask is missing:
            raise ValueError("frames and masks must have the same length")
        result.append(None if mask is None else refine_occluder(frame, mask, **kw))
    return result
