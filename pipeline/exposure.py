"""Tone-match a creative to the scene around its surface (no blank reference shot)."""
import cv2
import numpy as np

_LUMA = (0.2126, 0.7152, 0.0722)  # Rec.709 linear-luminance weights, R/G/B


def _to_linear(u):
    """sRGB in [0, 1] -> linear light."""
    u = np.asarray(u, np.float64)
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


def _to_srgb(y):
    """Linear light in [0, 1] -> sRGB."""
    y = np.clip(np.asarray(y, np.float64), 0.0, 1.0)
    return np.where(y <= 0.0031308, y * 12.92, 1.055 * y ** (1.0 / 2.4) - 0.055)


def _luma(lin):
    """Linear luminance of linear RGB (last axis R, G, B)."""
    return lin[..., 0] * _LUMA[0] + lin[..., 1] * _LUMA[1] + lin[..., 2] * _LUMA[2]


def scene_tone(frame, quad, band=(10, 60)):
    """Tone statistics of the scene around a surface.

    frame: BGR uint8 (H, W, 3). quad: float32 (4, 2) corners TL, TR, BR, BL (x, y).
    Samples the ring of pixels `band` px outside the quad (clipped to the frame) and
    returns {"black": 1st pct, "white": 99.5th pct, "mean"} of linear-light
    luminance in [0, 1]. Returns None if the ring has fewer than 500 pixels.
    """
    h, w = frame.shape[:2]
    quad = np.asarray(quad, np.float32).reshape(4, 2)
    inside = np.zeros((h, w), np.uint8)
    cv2.fillPoly(inside, [np.round(quad).astype(np.int32)], 255)
    if not inside.any():
        return None
    dist = cv2.distanceTransform(255 - inside, cv2.DIST_L2, 5)
    ring = (dist >= band[0]) & (dist <= band[1])
    if ring.sum() < 500:
        return None
    pix = frame[ring].astype(np.float64)[:, ::-1] / 255.0  # BGR -> RGB
    y = _luma(_to_linear(pix))
    return {"black": float(np.percentile(y, 1.0)),
            "white": float(np.percentile(y, 99.5)),
            "mean": float(np.mean(y))}


def fit_creative_tone(creative_rgb, tone, strength=1.0):
    """Map an RGB uint8 creative into the scene's tonal range.

    In linear light, one affine map applied to all three channels (hue-preserving)
    takes the creative's white (99.5th pct) to tone["white"] * 0.92 and its black
    (1st pct) to tone["black"], then back to sRGB uint8. strength in [0, 1]
    blends between the original (0) and the fully mapped result (1).
    """
    orig = np.asarray(creative_rgb).astype(np.float64) / 255.0
    lin = _to_linear(orig)
    y = _luma(lin)
    sw, sb = np.percentile(y, 99.5), np.percentile(y, 1.0)
    if sw - sb < 1e-9:  # flat creative: fall back to the full [0, 1] range
        sb, sw = 0.0, 1.0
    tb, tw = float(tone["black"]), float(tone["white"]) * 0.92
    a = (tw - tb) / (sw - sb)
    b = tb - a * sb
    mapped = _to_srgb(a * lin + b)
    s = min(max(float(strength), 0.0), 1.0)
    out = (1.0 - s) * orig + s * mapped
    return np.clip(np.round(out * 255.0), 0.0, 255.0).astype(np.uint8)
