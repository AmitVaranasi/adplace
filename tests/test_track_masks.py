"""Mask-only tracking acceptance tests; no external clip is required."""

import cv2
import numpy as np
import pytest

from pipeline.mask_track import track_from_masks


@pytest.fixture(scope="module", autouse=True)
def bounded_opencv_threads():
    previous = cv2.getNumThreads()
    cv2.setNumThreads(min(2, max(1, previous)))
    try:
        yield
    finally:
        cv2.setNumThreads(previous)


def _mask(quad, shape=(260, 360)):
    result = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(
        result, np.rint(np.asarray(quad) * 16).astype(np.int32), 1, shift=4
    )
    return result.astype(bool)


def _jitter(quads):
    return float(np.median(
        np.linalg.norm(np.diff(quads, n=2, axis=0), axis=2)
    ))


def _intrusion(quad, mask, occluder=None):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    points = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if occluder is not None:
        exclusion = cv2.dilate(
            occluder.astype(np.uint8), np.ones((5, 5), np.uint8)
        )
        points = points[exclusion[points[:, 1], points[:, 0]] == 0]
    edges = np.roll(quad, -1, axis=0) - quad
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return float(np.max(
        np.einsum("ij,ij->i", normals, quad) - (points @ normals.T).min(axis=0)
    ))


@pytest.fixture(scope="module")
def moving_masks():
    rng = np.random.default_rng(314)
    base = np.array([[65, 55], [260, 65], [250, 195], [50, 185]], dtype=float)
    truth, masks = [], []
    for i in range(45):
        quad = base + [0.8 * i, 0.25 * i]
        quad += [[0, 0], [0.02 * i, 0], [-0.01 * i, 0], [0, 0]]
        truth.append(quad)
        masks.append(_mask(quad + rng.normal(0, 0.28, (4, 2))))
    return np.array(truth), masks


def test_noisy_motion_interpolation_and_outlier_rejection(moving_masks):
    truth, clean = moving_masks
    masks = [m.copy() for m in clean]
    masks[19] = None
    masks[29] = _mask(truth[29] + [48, -36])
    before = [None if m is None else m.copy() for m in masks]
    result = track_from_masks(masks)

    assert set(result) == {"quads", "confidence", "status"}
    assert result["quads"].shape == (45, 4, 2)
    assert result["quads"].dtype == np.float32
    assert result["confidence"].shape == result["status"].shape == (45,)
    assert np.isfinite(result["quads"]).all()
    assert np.isfinite(result["confidence"]).all()
    np.testing.assert_array_equal(
        np.flatnonzero(result["status"] == "interpolated"), [19, 29]
    )
    assert np.all(result["confidence"][[19, 29]] == 0)
    assert np.all((result["confidence"] >= 0) & (result["confidence"] <= 1))
    error = np.linalg.norm(result["quads"] - truth, axis=2)
    assert np.median(error) < 1.5
    assert error.max() < 2.5
    assert _jitter(result["quads"]) < 0.3
    assert error[[0, -1]].max() < 2.0
    for i in range(len(masks)):
        if i not in (19, 29):
            assert _intrusion(result["quads"][i], masks[i]) <= 1.0
        if masks[i] is not None:
            np.testing.assert_array_equal(masks[i], before[i])


def test_occluded_boundary_is_excluded(moving_masks):
    truth, clean = moving_masks
    masks = [m.copy() for m in clean]
    occluders = [None] * len(masks)
    for i in range(8, 15):
        occ = np.zeros(masks[i].shape, dtype=np.uint8)
        cv2.rectangle(occ, (138, 35), (180, 95), 1, cv2.FILLED)
        occluders[i] = occ.astype(bool)
        masks[i][occluders[i]] = False
    result = track_from_masks(masks, occluders)
    assert np.all(result["status"][8:15] == "occluded")
    assert np.max(np.linalg.norm(result["quads"] - truth, axis=2)) < 2.5
    for i in range(8, 15):
        assert _intrusion(result["quads"][i], masks[i], occluders[i]) <= 1.0


def test_corner_identities_survive_canonical_start_change():
    center = np.array([180, 130])
    base = np.array([[-90, -50], [90, -50], [90, 50], [-90, 50]])
    truth = []
    for angle in np.deg2rad(np.linspace(32, 58, 31)):
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        truth.append(base @ rotation.T + center)
    truth = np.asarray(truth)
    result = track_from_masks([_mask(q) for q in truth])
    assert np.max(np.linalg.norm(result["quads"] - truth, axis=2)) < 2.5
    edges = np.roll(result["quads"], -1, axis=1) - result["quads"]
    following = np.roll(edges, -1, axis=1)
    assert np.all(
        edges[..., 0] * following[..., 1]
        - edges[..., 1] * following[..., 0] > 0
    )


def test_empty_frame_area_outlier_and_single_anchor(moving_masks):
    truth, masks = moving_masks
    damaged = list(masks)
    damaged[10] = np.zeros_like(masks[10])
    center = truth[22].mean(axis=0)
    damaged[22] = _mask(center + 0.55 * (truth[22] - center))
    result = track_from_masks(damaged)
    np.testing.assert_array_equal(
        np.flatnonzero(result["status"] == "interpolated"), [10, 22]
    )
    assert np.max(np.linalg.norm(result["quads"] - truth, axis=2)) < 2.5

    single = track_from_masks([None, masks[0], None])
    np.testing.assert_array_equal(single["quads"][0], single["quads"][1])
    np.testing.assert_array_equal(single["quads"][1], single["quads"][2])
    assert single["status"].tolist() == ["interpolated", "ok", "interpolated"]


def test_subpixel_coordinates_and_unconstrained_option(moving_masks):
    truth, masks = moving_masks
    result = track_from_masks(masks, coverage_tolerance=None)
    quads = result["quads"]
    assert np.median(np.linalg.norm(quads - truth, axis=2)) < 1.5
    assert (
        np.count_nonzero(np.abs(quads - np.rint(quads)) > 0.05)
        > quads.size // 2
    )
    assert _jitter(quads) < 0.3


@pytest.mark.parametrize("masks,occluders,options", [
    ([], None, {}),
    ([None, None], None, {}),
    ([np.zeros((30, 30), bool)], None, {}),
    ([np.zeros((30, 30, 3), bool)], None, {}),
    ([np.zeros((30, 30), bool), np.zeros((31, 30), bool)], None, {}),
    ([np.ones((30, 30), bool)], [], {}),
    ([None, np.ones((30, 30), bool)], [np.zeros((20, 20), bool), None], {}),
    ([np.ones((30, 30), bool)], None, {"smoothness": -1}),
    ([np.ones((30, 30), bool)], None, {"smoothness": float("nan")}),
    ([np.ones((30, 30), bool)], None, {"coverage_tolerance": -1}),
])
def test_invalid_inputs(masks, occluders, options):
    with pytest.raises(ValueError):
        track_from_masks(masks, occluders, **options)
