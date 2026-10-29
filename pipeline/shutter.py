"""Centred exposure integration on the projective group, in pixel centres."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.linalg import expm, logm

_UNIT = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)


def _good(q):
    return (
        q.shape == (4, 2) and np.isfinite(q).all()
        and cv2.isContourConvex(q.astype(np.float32))
        and cv2.contourArea(q.astype(np.float32), oriented=True) > 1
    )


def _segment(a, b):
    # Similarity normalization conditions the projective logarithm.
    center = np.r_[a, b].mean(0)
    scale = max(1., np.std(np.r_[a, b]))
    an, bn = (a-center)/scale, (b-center)/scale
    h0 = cv2.getPerspectiveTransform(
        _UNIT, an.astype(np.float32)
    ).astype(float)
    h1 = cv2.getPerspectiveTransform(
        _UNIT, bn.astype(np.float32)
    ).astype(float)
    r = h1 @ np.linalg.inv(h0)
    det = np.linalg.det(r)
    if det <= 0:
        raise ValueError("neighbouring quads have incompatible projective charts")
    r /= np.cbrt(det)
    v = logm(r)
    if np.max(np.abs(v.imag)) > 1e-7:
        raise ValueError("exposure motion has no stable real projective logarithm")
    v = v.real

    def at(t):
        h = expm(t*v) @ h0
        p = np.c_[_UNIT, np.ones(4)] @ h.T
        if (
            np.any(np.abs(p[:, 2]) < 1e-8)
            or p[:, 2].min()*p[:, 2].max() <= 0
        ):
            raise ValueError("exposure crosses a projective pole")
        q = (p[:, :2]/p[:, 2:])*scale+center
        if not _good(q):
            raise ValueError("degenerate exposure quad")
        return q.astype(np.float64)

    return at


def shutter_quads(quads, i, shutter=0.5, samples=None) -> list[np.ndarray]:
    """Equal-weight midpoint samples; automatic spacing <=1 image pixel.

    NaN current pose -> []; never bridge a missing neighbour. At endpoints
    or beside a gap, extrapolate from the available adjacent pose.
    With neither neighbour, return the current pose. Explicit samples
    overrides automatic spacing. shutter is in [0,1].
    """
    q = np.asarray(quads, np.float64)
    if q.ndim != 3 or q.shape[1:] != (4, 2):
        raise ValueError("quads must have shape (N,4,2)")
    if not isinstance(i, (int, np.integer)) or not 0 <= i < len(q):
        raise ValueError("i is out of range")
    if not np.isfinite(shutter) or not 0 <= shutter <= 1:
        raise ValueError("shutter must be in [0,1]")
    if samples is not None and (
        isinstance(samples, (bool, np.bool_))
        or not isinstance(samples, (int, np.integer))
        or samples < 1
    ):
        raise ValueError("samples must be a positive integer")
    if not _good(q[i]):
        return []
    if shutter == 0:
        return [q[i].copy()]

    left = (
        _segment(q[i-1], q[i])
        if i > 0 and _good(q[i-1]) else None
    )
    right = (
        _segment(q[i], q[i+1])
        if i+1 < len(q) and _good(q[i+1]) else None
    )
    if left is None and right is None:
        return [q[i].copy()]

    def at(t):
        if t == 0:
            return q[i].copy()
        if t < 0:
            return left(1+t) if left is not None else right(t)
        return right(t) if right is not None else left(1+t)

    def positions(count):
        times = ((np.arange(count)+.5)/count-.5)*shutter
        return [at(t) for t in times]

    if samples is not None:
        return positions(samples)

    times = np.linspace(-shutter/2, shutter/2, 17)
    probes = np.asarray([at(t) for t in times])
    length = np.linalg.norm(
        np.diff(probes, axis=0), axis=2
    ).max(1).sum()
    if length < .05:
        return [q[i].copy()]

    count = max(1, int(np.ceil(length - 1e-5)))
    while True:
        result = positions(count)
        path = np.asarray([at(-shutter/2), *result, at(shutter/2)])
        step = np.linalg.norm(np.diff(path, axis=0), axis=2).max()
        if step <= 1.001:
            return result
        count = max(count+1, int(np.ceil(count*step)))
