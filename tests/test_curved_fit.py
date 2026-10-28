"""Synthetic cylinder fitting and temporal smoothing acceptance tests."""

import math

import cv2
import numpy as np

from pipeline.curved_fit import cylinder_from_mask, fit_rim, hold_static, smooth_cylinders


def _body(a=0.0, handle=False):
    yy, xx = np.indices((260, 320))
    radius = 58.0 - 0.04 * (yy - 30)
    mask = ((yy >= 30) & (yy <= 230)
            & (np.abs(xx - (a * yy + 120.0)) <= radius))
    if handle:
        ring = np.zeros(mask.shape, np.uint8)
        cv2.ellipse(ring, (193, 130), (36, 48), 0, 0, 360, 1, 7)
        mask |= ring.astype(bool)
    return mask


def _cyl(b=100.0, a=0.0, y0=40, y1=160, radius=50.0):
    rows = np.arange(y0, y1 + 1, dtype=np.int32)
    return dict(a=a, b=b, y0=y0, y1=y1, rows=rows,
                half_width=np.full(len(rows), radius, dtype=float))


def _elliptical_mug(radius=80.0, e_t=10.0, e_b=14.0, handle=False):
    H, W = 300, 360
    cx = 180.0
    A_t, A_b = 50.0, 210.0
    yy, xx = np.indices((H, W))
    u = (xx.astype(float) - cx) / radius
    s = np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))
    top = A_t - e_t * s
    bot = A_b + e_b * s
    mask = (np.abs(u) <= 1.0) & (yy >= top) & (yy <= bot)
    if handle:
        ring = np.zeros(mask.shape, np.uint8)
        cv2.ellipse(ring, (int(cx + radius + 8), 130), (28, 40), 0, 0, 360, 1, 7)
        mask |= ring.astype(bool)
    return mask


def _rim_scene(radius=80.0, A=60.0, e=9.0, lip_h=4, A_b=210.0, e_b=14.0,
               cx=180.0, H=300, W=360, lip_bright=240, body_dark=50, bg=128):
    xs = np.arange(W, dtype=np.float64)
    u = (xs - cx) / radius
    s = np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))
    lower = A + e * s
    upper = lower - lip_h
    bottom = A_b + e_b * s
    yy, xx = np.indices((H, W))
    upper_map = np.broadcast_to(upper[None, :], (H, W))
    lower_map = np.broadcast_to(lower[None, :], (H, W))
    bottom_map = np.broadcast_to(bottom[None, :], (H, W))
    valid_map = np.broadcast_to((np.abs(u) <= 1.0)[None, :], (H, W))
    mask = valid_map & (yy >= upper_map) & (yy <= bottom_map)
    frame = np.full((H, W, 3), bg, np.uint8)
    lip = mask & (yy < lower_map)
    body_px = mask & (yy >= lower_map)
    frame[lip] = lip_bright
    frame[body_px] = body_dark
    return frame, mask


def test_tapered_body_ignores_handle():
    mask = _body(handle=True)
    original = mask.copy()
    cyl = cylinder_from_mask(mask)
    assert cyl is not None
    rows = cyl["rows"]
    assert np.max(np.abs(cyl["a"] * rows + cyl["b"] - 120)) < 1.5
    truth = 58 - 0.04 * (rows - 30)
    assert np.max(np.abs(cyl["half_width"] / truth - 1)) < 0.08
    assert (cyl["y0"], cyl["y1"]) == (80, 190)
    np.testing.assert_array_equal(rows, np.arange(80, 191))
    np.testing.assert_array_equal(mask, original)


def test_tilt_is_recovered():
    cyl = cylinder_from_mask(_body(a=0.05))
    assert cyl is not None
    assert abs(cyl["a"] - 0.05) < 0.01
    axis_error = cyl["a"] * cyl["rows"] + cyl["b"] - (0.05 * cyl["rows"] + 120)
    assert np.max(np.abs(axis_error)) < 1.5


def test_empty_and_tiny_masks_are_rejected():
    assert cylinder_from_mask(np.zeros((100, 100), bool)) is None
    tiny = np.zeros((100, 100), bool)
    tiny[10:29, 10:31] = True  # 399 pixels.
    assert cylinder_from_mask(tiny) is None


def test_large_width_variation_is_rejected():
    yy, xx = np.indices((260, 320))
    radius = np.where(yy < 100, 100, 30)
    mask = (yy >= 30) & (yy <= 230) & (np.abs(xx - 160) <= radius)
    assert cylinder_from_mask(mask) is None


def test_smoothing_preserves_none_and_reduces_jitter():
    noise = np.random.default_rng(71).normal(0, 6, 121)
    cyls = [_cyl(b=100 + value) for value in noise]
    cyls[60] = None
    result = smooth_cylinders([None, *cyls, None])
    assert len(result) == 123
    assert [i for i, cyl in enumerate(result) if cyl is None] == [0, 61, 122]
    for start, stop in ((0, 60), (61, 121)):
        before = np.array([c["b"] for c in cyls[start:stop]])
        after = np.array([c["b"] for c in result[start + 1:stop + 1]])
        assert np.std(np.diff(after)) <= 0.6 * np.std(np.diff(before))
        assert np.std(after) <= 0.6 * np.std(before)


def test_smoothing_splits_runs_and_rebuilds_without_mutation():
    left, right = _cyl(b=50), _cyl(b=250)
    left["half_width"] = np.linspace(40, 55, len(left["rows"]))
    original = left["half_width"].copy()
    result = smooth_cylinders([left, left, None, right, right])
    assert result[2] is None
    assert np.allclose([result[i]["b"] for i in (0, 1)], 50)
    assert np.allclose([result[i]["b"] for i in (3, 4)], 250)
    np.testing.assert_array_equal(left["half_width"], original)
    assert not np.shares_memory(result[0]["half_width"], left["half_width"])
    for cyl in (result[0], result[1], result[3], result[4]):
        np.testing.assert_array_equal(cyl["rows"], np.arange(math.floor(cyl["y0"]), math.ceil(cyl["y1"]) + 1))
        assert len(cyl["rows"]) == len(cyl["half_width"])
    np.testing.assert_allclose(result[0]["half_width"], original)
    assert smooth_cylinders([]) == []
    assert smooth_cylinders([None, None]) == [None, None]


def test_smoothing_filters_geometry_and_resampled_radii():
    cyls = [_cyl(a=a, b=b, y0=y0, y1=y1, radius=r)
            for a, b, y0, y1, r in (
                (0.02, 90, 30, 150, 40), (0.06, 110, 50, 170, 60))]
    result = smooth_cylinders(cyls, alpha=0.5)
    for cyl, expected in zip(result, ((0.03, 95, 35, 155, 45),
                                     (0.04, 100, 40, 160, 50))):
        a, b, y0, y1, radius = expected
        np.testing.assert_allclose([cyl["a"], cyl["b"]], [a, b])
        assert (cyl["y0"], cyl["y1"]) == (y0, y1)
        np.testing.assert_allclose(cyl["half_width"], radius)


def test_elliptical_sag_is_recovered():
    cyl = cylinder_from_mask(_elliptical_mug(radius=80.0, e_t=10.0, e_b=14.0))
    assert cyl is not None
    assert abs(float(cyl["sag_top"]) - 10.0) <= 2.0
    assert abs(float(cyl["sag_bottom"]) - 14.0) <= 2.0


def test_flat_rectangle_has_zero_sag():
    cyl = cylinder_from_mask(_elliptical_mug(radius=80.0, e_t=0.0, e_b=0.0))
    assert cyl is not None
    assert abs(float(cyl["sag_top"])) <= 2.0
    assert abs(float(cyl["sag_bottom"])) <= 2.0


def test_handle_does_not_change_sag():
    plain = cylinder_from_mask(_elliptical_mug(radius=80.0, e_t=10.0, e_b=14.0))
    with_handle = cylinder_from_mask(_elliptical_mug(radius=80.0, e_t=10.0, e_b=14.0, handle=True))
    assert plain is not None and with_handle is not None
    assert abs(float(with_handle["sag_top"]) - float(plain["sag_top"])) <= 2.0
    assert abs(float(with_handle["sag_bottom"]) - float(plain["sag_bottom"])) <= 2.0


def test_fit_rim_recovers_lip_edge():
    frame, mask = _rim_scene()
    cyl0 = cylinder_from_mask(mask)
    assert cyl0 is not None
    rim = fit_rim(frame, cyl0, mask)
    assert rim is not None
    assert abs(float(rim["A"]) - 60.0) <= 1.0
    assert abs(float(rim["e"]) - 9.0) <= 1.5  # binary 4 px synthetic lip: ~1 px smoothing bias on the sag


def test_fit_rim_rejects_weak_contrast():
    _frame, mask = _rim_scene()
    flat = np.full((mask.shape[0], mask.shape[1], 3), 128, np.uint8)
    flat[mask] = 50
    cyl0 = cylinder_from_mask(mask)
    assert cyl0 is not None
    assert fit_rim(flat, cyl0, mask) is None


def test_cylinder_top_rim_uses_lip():
    frame, mask = _rim_scene()
    cyl = cylinder_from_mask(mask, frame=frame, top="rim")
    assert cyl is not None
    assert cyl.get("top_source") == "rim"
    assert abs(float(cyl["y0"]) - (60.0 - 0.5)) <= 1.0
    assert isinstance(cyl["y0"], float) and isinstance(cyl["y1"], float)


def test_cylinder_top_band_fallback_without_frame():
    _frame, mask = _rim_scene()
    cyl = cylinder_from_mask(mask, frame=None, top="rim")
    assert cyl is not None
    assert cyl.get("top_source") == "band"
    cyl2 = cylinder_from_mask(mask)
    assert cyl2 is not None
    assert cyl2.get("top_source") == "band"
    assert isinstance(cyl2["y0"], float) and isinstance(cyl2["y1"], float)


def test_y0_y1_are_python_floats_and_rows_cover():
    cyl = cylinder_from_mask(_body())
    assert cyl is not None
    assert isinstance(cyl["y0"], float) and isinstance(cyl["y1"], float)
    np.testing.assert_array_equal(cyl["rows"], np.arange(math.floor(cyl["y0"]), math.ceil(cyl["y1"]) + 1))


def test_hold_static_freezes_a_still_mug_and_leaves_a_moving_one():
    still = [cylinder_from_mask(_elliptical_mug(radius=80.0, e_t=10.0, e_b=14.0)) for _ in range(5)]
    for i, c in enumerate(still):
        c["y0"] += (i % 2) * 0.8  # per-frame fit noise
    held = hold_static(still)
    assert len({round(c["y0"], 6) for c in held}) == 1
    moving = [dict(c, b=c["b"] + 10.0 * i) for i, c in enumerate(still)]
    assert [c["b"] for c in hold_static(moving)] == [c["b"] for c in moving]
    assert hold_static([None, still[0]])[0] is None
