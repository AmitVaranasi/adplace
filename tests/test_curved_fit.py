"""Synthetic cylinder fitting and temporal smoothing acceptance tests."""

import cv2
import numpy as np

from pipeline.curved_fit import cylinder_from_mask, smooth_cylinders


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
        np.testing.assert_array_equal(cyl["rows"], np.arange(cyl["y0"], cyl["y1"] + 1))
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
