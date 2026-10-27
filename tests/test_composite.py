"""Fast synthetic tests for illumination, scene matching, and compositing."""

import cv2
import numpy as np
import pytest

import pipeline.composite as composite


def _wall(blur_sigma=1.6, grain_sigma=3.0):
    height, width = 240, 320
    yy, xx = np.indices((height, width), dtype=np.float32)
    quad = np.array(
        [[96.0, 72.0], [224.0, 72.0], [224.0, 168.0], [96.0, 168.0]],
        dtype=np.float32,
    )
    falloff = np.clip(0.65 + 0.70 * (xx - 96.0) / 128.0, 0.3, 1.6)

    # These straight steps lie in the surrounding band, outside the blank surface.
    texture = (
        -40.0 * ((xx < 84) | (xx >= 236))
        - 30.0 * ((yy < 60) | (yy >= 180))
    )
    base_bgr = np.array([120.0, 135.0, 150.0], dtype=np.float32)
    wall = falloff[..., None] * base_bgr + texture[..., None]
    wall = cv2.GaussianBlur(
        wall.astype(np.float32),
        (0, 0),
        blur_sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    noise = np.random.default_rng(17).normal(0.0, grain_sigma, wall.shape)
    frame = np.clip(np.rint(wall + noise), 0, 255).astype(np.uint8)
    return frame, quad


def _creative(rgb=(160, 80, 40), size=(64, 48)):
    width, height = size
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255
    return rgba


def _luminance(bgr):
    return np.asarray(bgr, dtype=np.float64) @ np.array([0.114, 0.587, 0.299])


@pytest.fixture
def scene():
    return _wall()


def test_illumination_map_recovers_normalized_falloff(scene):
    frame, quad = scene
    original = frame.copy()
    shading = composite.illumination_map(frame, quad, (128, 96))

    expected_row = 0.65 + 0.70 * (np.arange(128) + 0.5) / 128.0
    expected = np.broadcast_to(expected_row, (96, 128))

    assert shading.shape == (96, 128)
    assert shading.dtype == np.float32
    assert np.isfinite(shading).all()
    assert float(shading.mean()) == pytest.approx(1.0, abs=1e-6)
    assert np.mean(np.abs(shading - expected)) < 0.025
    assert np.corrcoef(shading.mean(axis=0), expected_row)[0, 1] > 0.995
    assert np.array_equal(frame, original)


@pytest.mark.parametrize(
    ("blur_sigma", "grain_sigma"),
    [(1.2, 2.5), (2.0, 4.0)],
)
def test_estimate_recovers_known_blur_and_grain(blur_sigma, grain_sigma):
    frame, quad = _wall(blur_sigma, grain_sigma)
    original = frame.copy()

    estimated_blur, estimated_grain = composite.estimate_blur_and_grain(frame, quad)

    assert estimated_blur == pytest.approx(blur_sigma, rel=0.30)
    assert estimated_grain == pytest.approx(grain_sigma, rel=0.30)
    assert np.array_equal(frame, original)


def test_featureless_band_has_safe_estimates():
    frame = np.full((96, 128, 3), 100, dtype=np.uint8)
    quad = np.array([[40, 32], [88, 32], [88, 64], [40, 64]], dtype=np.float32)
    assert composite.estimate_blur_and_grain(frame, quad) == (0.0, 0.0)


def test_creative_preserves_colour_and_follows_shading(scene):
    frame, quad = scene
    creative = _creative()
    shading = composite.illumination_map(frame, quad, (128, 96))
    result = composite.composite_frame(
        frame,
        creative,
        quad,
        illum=shading,
        blur_sigma=0.0,
        grain_sigma=0.0,
        bleed_px=0.0,
    )
    own_bgr = np.array([40.0, 80.0, 160.0])
    own_chromaticity = own_bgr / own_bgr.sum()

    for x in (112, 160, 208):
        pixel = result[120, x].astype(np.float64)
        expected_shading = 0.65 + 0.70 * (x - 96.0) / 128.0
        assert np.allclose(pixel / pixel.sum(), own_chromaticity, atol=0.006)
        assert pixel[2] / own_bgr[2] == pytest.approx(expected_shading, abs=0.025)

    assert _luminance(result[120, 208]) > 1.5 * _luminance(result[120, 112])
    assert result.dtype == np.uint8
    assert result.shape == frame.shape


def test_occluded_pixels_are_exact_and_inputs_unchanged(scene):
    frame, quad = scene
    creative = _creative()
    shading = composite.illumination_map(frame, quad, (64, 48))
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[65:175, 140:164] = True
    inputs = [frame, creative, quad, shading, mask]
    snapshots = [array.copy() for array in inputs]

    kwargs = dict(
        illum=shading,
        occluder_mask=mask,
        blur_sigma=1.1,
        grain_sigma=2.0,
    )
    result = composite.composite_frame(frame, creative, quad, **kwargs)
    repeated = composite.composite_frame(frame, creative, quad, **kwargs)

    assert np.array_equal(result[mask], frame[mask])
    assert np.any(result[100:140, 110:130] != frame[100:140, 110:130])
    assert np.array_equal(result, repeated)
    for array, snapshot in zip(inputs, snapshots):
        assert np.array_equal(array, snapshot)


def test_nan_quad_returns_independent_unchanged_frame(scene):
    frame, _ = scene
    quad = np.full((4, 2), np.nan, dtype=np.float32)
    result = composite.composite_frame(frame, _creative(), quad)

    assert np.array_equal(result, frame)
    assert not np.shares_memory(result, frame)
    result[0, 0] = 255 - frame[0, 0]
    assert np.any(frame[0, 0] != result[0, 0])


def test_emissive_skips_shading_and_matches_brightness_and_contrast(scene):
    _, quad = scene
    frame = np.full((240, 320, 3), 110, dtype=np.uint8)
    creative = _creative()
    creative[:, :32, :3] //= 2
    shading = np.broadcast_to(
        np.linspace(0.2, 2.0, 64, dtype=np.float32), (48, 64)
    ).copy()
    options = dict(
        emissive=True,
        blur_sigma=0.0,
        grain_sigma=0.0,
        bleed_px=0.0,
    )

    with_shading = composite.composite_frame(
        frame, creative, quad, illum=shading, **options
    )
    without_shading = composite.composite_frame(frame, creative, quad, **options)

    assert np.array_equal(with_shading, without_shading)
    patch = with_shading[80:160, 104:216]
    assert float(_luminance(patch).mean()) == pytest.approx(110.0, abs=1.0)

    dark = with_shading[120, 120].astype(np.float64)
    bright = with_shading[120, 200].astype(np.float64)
    assert _luminance(bright) / _luminance(dark) == pytest.approx(2.0, rel=0.025)
    expected_chromaticity = np.array([40.0, 80.0, 160.0]) / 280.0
    assert np.allclose(dark / dark.sum(), expected_chromaticity, atol=0.006)
    assert np.allclose(bright / bright.sum(), expected_chromaticity, atol=0.006)


@pytest.mark.parametrize("transparent_border", [False, True])
def test_antialiased_edges_have_no_dark_halo(transparent_border):
    frame = np.full((112, 128, 3), 60, dtype=np.uint8)
    rgb = (100, 160, 220)
    creative = _creative(rgb, size=(48, 40))
    if transparent_border:
        creative[:4] = 0
        creative[-4:] = 0
        creative[:, :4] = 0
        creative[:, -4:] = 0
    quad = np.array(
        [[22.3, 18.8], [100.7, 24.2], [94.4, 91.6], [16.6, 86.3]],
        dtype=np.float32,
    )
    result = composite.composite_frame(
        frame,
        creative,
        quad,
        blur_sigma=0.0,
        grain_sigma=0.0,
        bleed_px=0.0,
    )
    target = np.array(rgb[::-1], dtype=np.float64)
    assert np.all(result >= 60)
    assert np.all(result <= target)

    inferred_alpha = (result.astype(np.float64) - 60.0) / (target - 60.0)
    edge = (inferred_alpha[..., 0] > 0.03) & (inferred_alpha[..., 0] < 0.97)
    assert np.count_nonzero(edge) > 20
    assert np.max(np.ptp(inferred_alpha[edge], axis=1)) < 0.04


def test_default_bleed_covers_original_rim(scene):
    _, quad = scene
    frame = np.full((240, 320, 3), 240, dtype=np.uint8)
    creative = _creative()
    options = dict(blur_sigma=0.0, grain_sigma=0.0)
    unexpanded = composite.composite_frame(
        frame, creative, quad, bleed_px=0.0, **options
    )
    expanded = composite.composite_frame(frame, creative, quad, **options)
    explicit = composite.composite_frame(
        frame, creative, quad, bleed_px=2.0, **options
    )

    assert np.array_equal(unexpanded[120, 94], frame[120, 94])
    assert np.array_equal(expanded[120, 94], np.array([40, 80, 160], dtype=np.uint8))
    assert np.array_equal(expanded, explicit)


def test_fully_transparent_creative_leaves_frame_unchanged(scene):
    frame, quad = scene
    creative = _creative()
    creative[..., 3] = 0
    result = composite.composite_frame(
        frame,
        creative,
        quad,
        emissive=True,
        blur_sigma=1.5,
        grain_sigma=4.0,
    )
    assert np.array_equal(result, frame)


def test_render_clip_estimates_once_from_first_valid_frame(scene, monkeypatch):
    frame, quad = scene
    frames = [frame.copy() for _ in range(3)]
    quads = [np.full((4, 2), np.nan, dtype=np.float32), quad.copy(), quad.copy()]
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[90:150, 140:160] = True
    masks = [None, mask, None]
    estimates = []
    calls = []
    original_composite = composite.composite_frame

    def estimate(reference_frame, reference_quad):
        estimates.append((reference_frame, reference_quad))
        return 1.25, 2.5

    def record(frame_arg, creative_arg, quad_arg, **kwargs):
        calls.append(kwargs.copy())
        return original_composite(frame_arg, creative_arg, quad_arg, **kwargs)

    monkeypatch.setattr(composite, "estimate_blur_and_grain", estimate)
    monkeypatch.setattr(composite, "composite_frame", record)
    outputs = composite.render_clip(
        frames,
        quads,
        _creative(),
        occluder_masks=masks,
        emissive=True,
    )

    assert len(estimates) == 1
    assert estimates[0][0] is frames[1]
    assert np.array_equal(estimates[0][1], quads[1])
    assert len(outputs) == len(calls) == 3
    for index, call in enumerate(calls):
        assert call["blur_sigma"] == 1.25
        assert call["grain_sigma"] == 2.5
        assert call["emissive"] is True
        assert call["occluder_mask"] is masks[index]
    assert np.array_equal(outputs[0], frames[0])
    assert np.array_equal(outputs[1][mask], frames[1][mask])


def test_render_clip_empty_and_length_validation(scene):
    frame, quad = scene
    creative = _creative()
    assert composite.render_clip([], [], creative) == []
    with pytest.raises(ValueError, match="equal lengths"):
        composite.render_clip([frame], [], creative)
    with pytest.raises(ValueError, match="equal lengths"):
        composite.render_clip([frame], [quad], creative, occluder_masks=[])
