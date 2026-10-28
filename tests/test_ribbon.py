"""Synthetic geometry, mapping, temporal fitting and pixel-preservation tests."""

from unittest.mock import patch

import numpy as np
import pytest

from pipeline.ribbon import (
    composite_ribbon, ribbon_from_mask, smooth_ribbons, warp_to_ribbon,
)


def _top(x):
    return 45 + 0.0008 * (x - 175) ** 2 + 0.025 * (x - 175)


def _bottom(x):
    return _top(x) + 85 + 0.035 * (x - 175)


def _mask():
    yy, xx = np.indices((200, 360))
    return (
        (xx >= 25) & (xx <= 335)
        & (yy >= _top(xx)) & (yy <= _bottom(xx))
    )


def _ribbon(offset=0.0):
    x = np.linspace(25, 335, 48)
    return dict(x=x, top=_top(x) + offset, bottom=_bottom(x) + offset)


def _creative(rgb=(180, 100, 40), alpha=255, shape=(25, 257)):
    image = np.empty((*shape, 4), np.uint8)
    image[..., :3] = rgb
    image[..., 3] = alpha
    return image


def _jitter(ribbons):
    values = np.array([np.stack((r["top"], r["bottom"])) for r in ribbons])
    return np.sqrt(np.mean(np.diff(values, n=2, axis=0) ** 2))


def test_known_curves_recovered_within_one_and_half_pixels():
    mask = _mask()
    original = mask.copy()
    ribbon = ribbon_from_mask(mask)
    assert ribbon is not None
    assert set(ribbon) == {"x", "top", "bottom"}
    assert all(ribbon[key].shape == (48,) for key in ribbon)
    x = ribbon["x"]
    assert np.max(abs(ribbon["top"] - _top(x))) < 1.5
    assert np.max(abs(ribbon["bottom"] - _bottom(x))) < 1.5
    np.testing.assert_array_equal(mask, original)


def test_occluder_notch_does_not_pull_top_boundary():
    mask = _mask()
    yy, xx = np.indices(mask.shape)
    occluder = (xx >= 115) & (xx <= 210) & (yy <= _top(xx) + 36)
    notched = mask & ~occluder
    originals = [a.copy() for a in (notched, occluder)]
    for a in (notched, occluder):
        a.setflags(write=False)
    ribbon = ribbon_from_mask(notched, occluder)
    assert ribbon is not None
    assert np.max(abs(ribbon["top"] - _top(ribbon["x"]))) < 1.5
    assert np.max(abs(ribbon["bottom"] - _bottom(ribbon["x"]))) < 1.5
    for a, expected in zip((notched, occluder), originals):
        np.testing.assert_array_equal(a, expected)


def test_interior_holes_do_not_change_outer_boundaries():
    mask = _mask()
    mask[80:105, 150:180] = False
    a, b = ribbon_from_mask(mask), ribbon_from_mask(_mask())
    for key in ("x", "top", "bottom"):
        np.testing.assert_allclose(a[key], b[key], atol=1e-9)


def test_empty_tiny_and_fully_occluded_masks_are_unusable():
    assert ribbon_from_mask(np.zeros((200, 360), bool)) is None
    tiny = np.zeros((200, 360), bool)
    tiny[5:10, 5:10] = True
    assert ribbon_from_mask(tiny) is None
    mask = _mask()
    assert ribbon_from_mask(mask, np.ones_like(mask)) is None


def test_warp_uses_midline_arc_length_and_linear_v():
    # Segment arc lengths are 40, 50, 40: u(60) = 40 / 130.
    ribbon = dict(
        x=np.array([20., 60., 100., 140.]),
        top=np.array([30., 30., 60., 60.]),
        bottom=np.array([90., 90., 120., 120.]),
    )
    creative = _creative(shape=(101, 257))
    creative[..., 0] = np.rint(np.linspace(0, 255, 257)).astype(np.uint8)
    creative[..., 1] = np.arange(101, dtype=np.uint8)[:, None] * 2
    warped = warp_to_ribbon(creative, ribbon, (160, 180, 3))
    assert warped.shape == (160, 180, 4) and warped.dtype == np.uint8
    assert abs(int(warped[60, 60, 0]) - round(255 * 40 / 130)) <= 2
    assert abs(int(warped[90, 100, 0]) - round(255 * 90 / 130)) <= 2
    assert abs(int(warped[90, 100, 1]) - 100) <= 1
    assert warped[60, 60, 3] == 255


def test_premultiplied_interpolation_has_no_transparent_colour_halo():
    creative = np.array([[[255, 0, 0, 255], [0, 0, 255, 0]]], np.uint8)
    ribbon = dict(
        x=np.array([20., 180.]), top=np.array([20., 20.]),
        bottom=np.array([180., 180.]),
    )
    warped = warp_to_ribbon(creative, ribbon, (200, 200))
    np.testing.assert_array_equal(warped[100, 100], [255, 0, 0, 128])
    one_pixel = warp_to_ribbon(_creative(shape=(1, 1)), ribbon, (200, 200))
    np.testing.assert_array_equal(one_pixel[100, 100], [180, 100, 40, 255])


def test_feather_is_one_pixel_inward_and_never_changes_outside():
    ribbon = dict(
        x=np.array([20.5, 180.5]), top=np.array([20.5, 20.5]),
        bottom=np.array([180.5, 180.5]),
    )
    warped = warp_to_ribbon(_creative(), ribbon, (200, 200))
    assert warped[21, 100, 3] == 128
    assert warped[22, 100, 3] == 255
    assert not warped[:21].any()
    assert not warped[181:].any()
    assert not warped[:, :21].any()
    assert not warped[:, 181:].any()


def test_composite_preserves_outside_occluders_and_inputs_exactly():
    frame = np.random.default_rng(13).integers(
        0, 256, (200, 360, 3), dtype=np.uint8
    )
    creative = _creative()
    occluder = np.zeros(frame.shape[:2], bool)
    occluder[70:145, 130:190] = True
    ribbon = _ribbon()
    originals = [a.copy() for a in (frame, creative, occluder)]
    for a in (frame, creative, occluder):
        a.setflags(write=False)
    support = warp_to_ribbon(creative, ribbon, frame.shape)[..., 3] > 0
    with patch(
        "pipeline.ribbon.estimate_blur_and_grain", return_value=(1.4, 2.0)
    ):
        result = composite_ribbon(frame, creative, ribbon, occluder)
    np.testing.assert_array_equal(result[~support], frame[~support])
    np.testing.assert_array_equal(result[occluder], frame[occluder])
    assert np.any(result[support & ~occluder] != frame[support & ~occluder])
    assert result.shape == frame.shape and result.dtype == np.uint8
    for a, expected in zip((frame, creative, occluder), originals):
        np.testing.assert_array_equal(a, expected)


def test_rgb_to_bgr_and_partial_alpha_blending():
    frame = np.full((200, 360, 3), 100, np.uint8)
    creative = _creative(rgb=(200, 100, 20), alpha=128)
    with patch(
        "pipeline.ribbon.estimate_blur_and_grain", return_value=(0., 0.)
    ):
        result = composite_ribbon(frame, creative, _ribbon(), emissive=False)
    weight = 128 / 255
    expected = np.rint((1 - weight) * 100 + weight * np.array([20, 100, 200]))
    np.testing.assert_array_equal(result[95, 175], expected.astype(np.uint8))


def test_transparent_none_and_offscreen_ribbons_return_independent_copies():
    frame = np.full((200, 360, 3), 100, np.uint8)
    offscreen = _ribbon()
    offscreen["x"] = offscreen["x"] + 1000
    for creative, ribbon in (
        (_creative(alpha=0), _ribbon()), (_creative(), None),
        (_creative(), offscreen),
    ):
        result = composite_ribbon(frame, creative, ribbon)
        np.testing.assert_array_equal(result, frame)
        assert not np.shares_memory(result, frame)


def test_smoothing_reduces_jitter_and_is_invariant_to_time_reversal():
    noise = np.random.default_rng(71).normal(0, 2.5, 101)
    ribbons = [_ribbon(value) for value in noise]
    result = smooth_ribbons(ribbons, smoothness=1.0)
    reverse = smooth_ribbons(ribbons[::-1], smoothness=1.0)[::-1]
    assert _jitter(result) < 0.35 * _jitter(ribbons)
    assert _jitter(smooth_ribbons(ribbons)) < 0.8 * _jitter(ribbons)
    for a, b, original in zip(result, reverse, ribbons):
        for key in ("x", "top", "bottom"):
            np.testing.assert_allclose(a[key], b[key], atol=1e-9)
            assert not np.shares_memory(a[key], original[key])


def test_bracketed_none_frames_are_interpolated_and_linear_motion_preserved():
    observations = [None, _ribbon(0), None, None, _ribbon(6), None]
    for strength in (0.0, 0.05, 1.0):
        result = smooth_ribbons(observations, smoothness=strength)
        assert result[0] is None and result[-1] is None
        for i, offset in enumerate((0, 2, 4, 6), start=1):
            np.testing.assert_allclose(
                result[i]["top"], _ribbon(offset)["top"], atol=1e-8
            )
    assert smooth_ribbons([]) == []
    assert smooth_ribbons([None, None]) == [None, None]
    single = smooth_ribbons([None, _ribbon(), None])
    assert single[0] is None and single[2] is None
    np.testing.assert_array_equal(single[1]["top"], _ribbon()["top"])


def test_invalid_parameters_and_geometry_raise():
    for samples in (True, 3, 4.5):
        with pytest.raises(ValueError):
            ribbon_from_mask(_mask(), samples=samples)
    for strength in (-1, np.nan, np.inf):
        with pytest.raises(ValueError):
            smooth_ribbons([_ribbon()], strength)
    bad = _ribbon()
    bad["x"] = bad["x"][::-1]
    with pytest.raises(ValueError):
        warp_to_ribbon(_creative(), bad, (200, 360))
    with pytest.raises(ValueError):
        ribbon_from_mask(_mask(), np.zeros((10, 10), bool))
    with pytest.raises(ValueError):
        smooth_ribbons([_ribbon(), dict(
            x=np.array([0., 1.]), top=np.array([0., 0.]),
            bottom=np.array([2., 2.]),
        )])


def test_small_boundary_irregularities_are_covered():
    mask = _mask()
    for x in range(145, 165):
        ys = np.flatnonzero(mask[:, x])
        mask[ys[-1] + 1:ys[-1] + 4, x] = True
    ribbon = ribbon_from_mask(mask)
    alpha = warp_to_ribbon(_creative(), ribbon, mask.shape)[..., 3]
    assert np.count_nonzero(mask & (alpha > 0)) / mask.sum() >= 0.99
