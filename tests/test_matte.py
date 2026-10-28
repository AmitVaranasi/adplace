"""Deterministic quality, correctness, crop-equivalence and speed checks."""

from time import perf_counter

import cv2
import numpy as np
import pytest

from pipeline.matte import refine_occluder, refine_occluders


def _synthetic_scene(seed=17):
    height, width = 240, 360
    rng = np.random.default_rng(seed)

    # Random coloured stripes; chroma separates background from the person.
    stripe_blue = rng.integers(20, 236, size=width // 4, dtype=np.uint8)
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[..., 0] = np.repeat(stripe_blue, 4)[None, :]
    frame[..., 1] = 30
    frame[..., 2] = 40

    truth = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(truth, (width // 2, height // 2), 60, 1, -1)
    frame[truth.astype(bool)] = (40, 220, 200)

    inflated = cv2.dilate(
        truth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    small = cv2.resize(
        inflated, (width // 6, height // 6), interpolation=cv2.INTER_NEAREST
    )
    mask = cv2.resize(
        small, (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    return frame, mask, truth


def _sure_regions(mask, band_px=8):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1)
    )
    u8 = mask.astype(np.uint8)
    inside = cv2.erode(
        u8, kernel, borderType=cv2.BORDER_REPLICATE
    ).astype(bool)
    outside = ~cv2.dilate(
        u8, kernel, borderType=cv2.BORDER_REPLICATE
    ).astype(bool)
    return inside, outside


@pytest.mark.parametrize("seed", [3, 17, 29])
def test_refinement_at_least_halves_error_in_unknown_band(seed):
    frame, mask, truth = _synthetic_scene(seed)
    inside, outside = _sure_regions(mask)
    band = ~(inside | outside)

    # Reach across the 4 px outward bias plus the 6 px raster blocks.
    alpha = refine_occluder(frame, mask, radius=16)
    before = np.abs(mask.astype(np.float32) - truth)[band].mean()
    after = np.abs(alpha - truth)[band].mean()

    assert before > 0.0
    assert after <= 0.5 * before, (before, after)


def test_sure_foreground_and_background_are_exact():
    frame, mask, _ = _synthetic_scene()
    inside, outside = _sure_regions(mask)
    alpha = refine_occluder(frame, mask)

    assert inside.any() and outside.any()
    assert np.all(alpha[inside] == 1.0)
    assert np.all(alpha[outside] == 0.0)


@pytest.mark.parametrize("value", [False, True])
def test_empty_and_full_masks_pass_through(value):
    frame = np.full((48, 64, 3), 100, dtype=np.uint8)
    mask = np.full(frame.shape[:2], value, dtype=bool)
    alpha = refine_occluder(frame, mask)

    np.testing.assert_array_equal(alpha, mask.astype(np.float32))
    assert alpha.dtype == np.float32
    assert not np.shares_memory(alpha, mask)


def test_output_shape_dtype_range_and_inputs_unchanged():
    frame, mask, _ = _synthetic_scene()
    original_frame, original_mask = frame.copy(), mask.copy()
    alpha = refine_occluder(frame, mask)

    assert alpha.shape == mask.shape
    assert alpha.dtype == np.float32
    assert np.isfinite(alpha).all()
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    np.testing.assert_array_equal(frame, original_frame)
    np.testing.assert_array_equal(mask, original_mask)


def test_sharpen_is_optional_and_has_requested_gain():
    frame, mask, _ = _synthetic_scene()
    soft = refine_occluder(frame, mask, sharpen=False)
    sharp = refine_occluder(frame, mask, sharpen=True, gain=1.5)
    expected = np.clip((soft - 0.5) * 1.5 + 0.5, 0.0, 1.0)

    np.testing.assert_allclose(sharp, expected, atol=1e-6, rtol=0)


def _full_frame_reference(frame, mask, radius, eps):
    """Independent float64 matrix-solve oracle, without ROI cropping."""
    guide = frame.astype(np.float64) / 255.0
    source = mask.astype(np.float64)

    def mean(array):
        return cv2.boxFilter(
            array,
            -1,
            (2 * radius + 1, 2 * radius + 1),
            normalize=True,
            borderType=cv2.BORDER_REFLECT_101,
        )

    mi, mp = mean(guide), mean(source)
    covariance = np.empty((*mask.shape, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            covariance[..., i, j] = (
                mean(guide[..., i] * guide[..., j])
                - mi[..., i] * mi[..., j]
                + (eps if i == j else 0.0)
            )

    cross = mean(guide * source[..., None]) - mi * mp[..., None]
    a = np.linalg.solve(covariance, cross[..., None])[..., 0]
    b = mp - np.sum(a * mi, axis=2)
    return np.sum(mean(a) * guide, axis=2) + mean(b)


@pytest.mark.parametrize("touches_edge", [False, True])
@pytest.mark.parametrize("constant_guide", [False, True])
def test_colour_inverse_and_crop_match_full_frame_reference(
    touches_edge, constant_guide
):
    rng = np.random.default_rng(31)
    frame = rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)
    if constant_guide:
        frame[:] = (71, 130, 205)  # Singular before regularization.

    mask = np.zeros(frame.shape[:2], dtype=bool)
    if touches_edge:
        mask[:25, :32] = True
    else:
        mask[20:44, 25:55] = True

    radius, eps, band_px = 3, 1e-3, 4
    expected = np.clip(_full_frame_reference(frame, mask, radius, eps), 0, 1)
    inside, outside = _sure_regions(mask, band_px)
    expected[inside], expected[outside] = 1.0, 0.0

    actual = refine_occluder(
        frame,
        mask,
        band_px=band_px,
        radius=radius,
        eps=eps,
        sharpen=False,
    )
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=0)


def test_batch_preserves_none_and_order():
    frame, mask, _ = _synthetic_scene()
    result = refine_occluders(
        iter([frame, None, frame]), iter([mask, None, mask])
    )

    assert len(result) == 3
    assert result[1] is None
    expected = refine_occluder(frame, mask)
    np.testing.assert_array_equal(result[0], expected)
    np.testing.assert_array_equal(result[2], expected)
    assert refine_occluders([], []) == []


def test_batch_rejects_different_lengths():
    with pytest.raises(ValueError, match="same length"):
        refine_occluders([None], [])
    with pytest.raises(ValueError, match="same length"):
        refine_occluders([], [None])


def test_720p_frame_under_half_second():
    rng = np.random.default_rng(123)
    frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)

    # Nearly full-frame ROI makes this a conservative speed check.
    mask = np.zeros((720, 1280), dtype=bool)
    mask[30:690, 30:1250] = True

    refine_occluder(frame, mask)  # Warm OpenCV dispatch.
    start = perf_counter()
    alpha = refine_occluder(frame, mask)
    elapsed = perf_counter() - start

    assert alpha.shape == (720, 1280)
    assert elapsed < 0.5, f"720p refinement took {elapsed:.3f}s"
