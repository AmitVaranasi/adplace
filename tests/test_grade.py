import cv2
import numpy as np
import pytest

from pipeline.grade import apply_grade, scene_grade


def _lab(rgb):
    rgb = np.ascontiguousarray(rgb, dtype=np.float32) / 255.0
    return cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)


def _bgr(lab):
    lab = np.asarray(lab, np.float32).reshape(-1, 1, 3)
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    rgb = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(rgb.reshape(-1, 3)[:, ::-1])


def _surface(a=0.0, b=0.0):
    return _bgr(np.repeat([[20, a, b], [50, a, b], [90, a, b]], 256, axis=0))


def _creative(rgb, height=16, width=24):
    image = np.empty((height, width, 4), np.uint8)
    image[..., :3] = rgb
    image[..., 3] = 255
    return image


def test_strength_zero_is_identity():
    scene = _surface(4, 8)
    grade = scene_grade(scene, scene, strength=0.0)
    creative = np.random.default_rng(7).integers(
        0, 256, (24, 32, 4), dtype=np.uint8
    )
    output = apply_grade(creative, grade)
    assert np.max(
        np.abs(output.astype(np.int16) - creative.astype(np.int16))
    ) <= 1
    assert not np.shares_memory(output, creative)
    assert grade["black_L"] == 0.0
    assert grade["white_L"] == 100.0
    assert grade["cast_a"] == grade["cast_b"] == 0.0
    assert set(grade) == {
        "black_L", "white_L", "cast_a", "cast_b", "c_max", "strength"
    }
    assert all(type(value) is float for value in grade.values())


def test_saturated_orange_rolls_off_without_rotating_hue():
    grey = _surface()
    grade = scene_grade(grey, grey)
    creative = _creative((255, 128, 0))
    before = _lab(creative[..., :3])
    after = _lab(apply_grade(creative, grade)[..., :3])
    chroma = np.hypot(after[:, 1], after[:, 2])
    assert np.percentile(chroma, 95) <= grade["c_max"] * 1.01
    hue_before = np.degrees(np.arctan2(before[:, 2], before[:, 1]))
    hue_after = np.degrees(np.arctan2(after[:, 2], after[:, 1]))
    hue_error = np.abs((hue_after - hue_before + 180.0) % 360.0 - 180.0)
    assert hue_error.max() <= 3.0


def test_black_and_white_match_tone_endpoints_and_cast():
    surface = _surface(4, 8)
    grade = scene_grade(surface, surface)
    creative = _creative((0, 0, 0), height=2, width=4)
    creative[1, :, :3] = 255
    after = _lab(apply_grade(creative, grade)[..., :3]).reshape(2, 4, 3)
    np.testing.assert_allclose(after[0, :, 0], grade["black_L"], atol=1.5)
    np.testing.assert_allclose(after[1, :, 0], grade["white_L"], atol=1.5)
    np.testing.assert_allclose(after[0, :, 1:3], 0.0, atol=1.5)
    np.testing.assert_allclose(after[1, :, 1], grade["cast_a"], atol=1.5)
    np.testing.assert_allclose(after[1, :, 2], grade["cast_b"], atol=1.5)


def test_alpha_and_inputs_are_unchanged():
    surface = _surface()
    creative = np.random.default_rng(11).integers(
        0, 256, (16, 16, 4), dtype=np.uint8
    )
    creative[..., 3] = np.arange(256, dtype=np.uint8).reshape(16, 16)
    original_surface, original_creative = surface.copy(), creative.copy()
    surface.setflags(write=False)
    creative.setflags(write=False)
    output = apply_grade(creative, scene_grade(surface, surface))
    np.testing.assert_array_equal(output[..., 3], original_creative[..., 3])
    np.testing.assert_array_equal(creative, original_creative)
    np.testing.assert_array_equal(surface, original_surface)
    assert output.dtype == np.uint8
    assert output.shape == creative.shape
    assert np.any(output[..., :3] != creative[..., :3])


def test_warm_light_makes_white_creative_warm():
    warm = _surface(4, 8)
    grade = scene_grade(warm, warm)
    assert grade["cast_a"] == pytest.approx(4.0, abs=0.75)
    assert grade["cast_b"] == pytest.approx(8.0, abs=0.75)
    after = _lab(apply_grade(_creative((255, 255, 255)), grade)[..., :3])
    assert np.median(after[:, 1]) > 2.0
    assert np.median(after[:, 2]) > 5.0


def test_fewer_than_200_replaced_pixels_uses_ring_for_tones():
    ring = _surface()
    ring_L = _lab(ring[:, ::-1])[:, 0]
    for count in (0, 1, 199):
        replaced = np.zeros((count, 3), np.uint8)
        grade = scene_grade(replaced, ring)
        assert grade["black_L"] == pytest.approx(np.percentile(ring_L, 3))
        assert grade["white_L"] == pytest.approx(np.percentile(ring_L, 97))
    grade = scene_grade(np.zeros((200, 3), np.uint8), ring)
    assert grade["black_L"] == 0.0
    assert grade["white_L"] == 30.0


def test_surface_sets_white_darkest_nearby_sets_black_and_range_clamps():
    ring = _surface()
    ring_black = min(35.0, float(np.percentile(_lab(ring[:, ::-1])[:, 0], 3)))
    for value, expected in ((0, (0.0, 30.0)), (255, (ring_black, 100.0))):
        replaced = np.full((256, 3), value, np.uint8)
        grade = scene_grade(replaced, ring)
        assert grade["black_L"] == pytest.approx(expected[0], abs=1e-4)
        assert grade["white_L"] == pytest.approx(expected[1], abs=1e-4)


def test_chroma_ceiling_uses_both_sources_even_with_few_replaced_pixels():
    grey = _surface()
    orange = np.full((199, 3), (0, 128, 255), np.uint8)
    labs = [_lab(pixels[:, ::-1]) for pixels in (grey, orange)]
    expected = 1.25 * max(
        np.percentile(np.hypot(lab[:, 1], lab[:, 2]), 95) for lab in labs
    )
    for replaced, ring in ((orange, grey), (grey, orange)):
        assert scene_grade(replaced, ring)["c_max"] == pytest.approx(expected)
    assert scene_grade(grey, grey)["c_max"] == 12.0


def test_cast_uses_only_bright_low_chroma_pixels():
    # Dark neutral pixels and bright coloured paint must not bias the light cast.
    pixels = _bgr(
        np.repeat(
            [[20, 0, 0], [60, 8, 15], [85, 35, 25], [90, 4, 8]],
            256,
            axis=0,
        )
    )
    grade = scene_grade(pixels, pixels)
    assert grade["cast_a"] == pytest.approx(4.0, abs=0.75)
    assert grade["cast_b"] == pytest.approx(8.0, abs=0.75)


def test_partial_strength_blends_all_lab_effects_once():
    surface = _surface(4, 8)
    full = scene_grade(surface, surface)
    half = scene_grade(surface, surface, strength=0.5)
    assert half["black_L"] == pytest.approx(full["black_L"] / 2.0)
    assert half["white_L"] == pytest.approx((100.0 + full["white_L"]) / 2.0)
    assert half["cast_a"] == pytest.approx(full["cast_a"] / 2.0)
    assert half["cast_b"] == pytest.approx(full["cast_b"] / 2.0)
    creative = _creative((200, 120, 50))
    original = _lab(creative[..., :3])
    target = _lab(apply_grade(creative, full)[..., :3])
    actual = _lab(apply_grade(creative, half)[..., :3])
    np.testing.assert_allclose(actual, (original + target) / 2.0, atol=1.5)


def test_empty_ring_is_allowed_with_enough_surface_pixels():
    grade = scene_grade(_surface(), np.empty((0, 3), np.uint8))
    assert np.isfinite(list(grade.values())).all()
    with pytest.raises(ValueError):
        scene_grade(np.empty((0, 3), np.uint8), np.empty((0, 3), np.uint8))
