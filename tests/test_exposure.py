import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from pipeline.exposure import fit_creative_tone, scene_tone

BLACK, WHITE = 0.08, 0.6
QUAD = np.array([[280, 160], [360, 160], [360, 240], [280, 240]], np.float32)


def _srgb(lin):
    lin = np.clip(np.asarray(lin, np.float64), 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def _luma(rgb_u8):
    u = np.asarray(rgb_u8, np.float64) / 255.0
    lin = np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)
    return lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722


def _dim_hazy_frame(h=400, w=640):
    """Gray frame whose linear luminance runs 0.08 -> 0.6 with flat black/white ends."""
    rows = np.arange(h, dtype=np.float64)
    lum = np.full(h, WHITE)
    lum[rows < 130] = BLACK
    ramp = (rows >= 130) & (rows <= 270)
    lum[ramp] = BLACK + (WHITE - BLACK) * (rows[ramp] - 130) / 140.0
    gray = np.round(_srgb(np.tile(lum[:, None], (1, w))) * 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _tone():
    tone = scene_tone(_dim_hazy_frame(), QUAD)
    assert tone is not None
    return tone


def test_dim_hazy_scene_maps_white_black_creative():
    tone = _tone()
    assert set(tone) == {"black", "white", "mean"}
    assert abs(tone["black"] - BLACK) <= 0.05 * BLACK
    assert abs(tone["white"] - WHITE) <= 0.05 * WHITE
    assert tone["black"] < tone["mean"] < tone["white"]

    creative = np.zeros((64, 96, 3), np.uint8)
    creative[:, :48] = 255  # pure white half, pure black half
    mapped = fit_creative_tone(creative, tone)

    mw = _luma(mapped[:, :48]).mean()
    mb = _luma(mapped[:, 48:]).mean()
    assert abs(mw - WHITE * 0.92) <= 0.05 * WHITE * 0.92
    assert abs(mb - BLACK) <= 0.05 * BLACK
    assert mw < tone["white"]  # printed paper never reaches scene highlights


def test_hue_of_saturated_colors_preserved():
    tone = _tone()
    creative = np.zeros((60, 120, 3), np.uint8)
    creative[:, :30] = (255, 0, 0)        # saturated red
    creative[:, 30:60] = (0, 0, 255)      # saturated blue
    creative[:, 60:90] = (255, 255, 255)  # white anchor (black fills the rest)
    mapped = fit_creative_tone(creative, tone)

    for sl in (slice(0, 30), slice(30, 60)):
        h0 = cv2.cvtColor(np.ascontiguousarray(creative[:, sl]), cv2.COLOR_RGB2HSV)[..., 0]
        h1 = cv2.cvtColor(np.ascontiguousarray(mapped[:, sl]), cv2.COLOR_RGB2HSV)[..., 0]
        d = np.abs(h1.astype(int) - h0.astype(int))
        assert np.minimum(d, 180 - d).max() <= 3  # OpenCV hue wraps at 180
    assert not np.array_equal(mapped[:, :30], creative[:, :30])  # really remapped


def test_strength_zero_returns_input_unchanged():
    tone = _tone()
    rng = np.random.default_rng(0)
    creative = (rng.random((32, 48, 3)) * 255).astype(np.uint8)
    assert np.array_equal(fit_creative_tone(creative, tone, strength=0.0), creative)
    assert not np.array_equal(fit_creative_tone(creative, tone, strength=1.0), creative)


def test_band_fully_outside_frame_returns_none():
    frame = _dim_hazy_frame()
    off = np.array([[660, 100], [760, 100], [760, 200], [660, 200]], np.float32)
    assert scene_tone(frame, off) is None
    whole = np.array([[0, 0], [639, 0], [639, 399], [0, 399]], np.float32)
    assert scene_tone(frame, whole) is None  # ring fully clipped away
