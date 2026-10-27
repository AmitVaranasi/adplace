"""Synthetic geometry acceptance tests; no external assets are required."""

import cv2
import numpy as np
import pytest

import pipeline.geometry as geometry
from pipeline.geometry import propagate_quad, quad_from_mask, smooth_quads


@pytest.fixture(scope="module", autouse=True)
def bounded_opencv_threads():
    previous = cv2.getNumThreads()
    cv2.setNumThreads(min(2, max(1, previous)))
    try:
        yield
    finally:
        cv2.setNumThreads(previous)


def _filled_polygon(quad, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.rint(np.asarray(quad) * 16.0).astype(np.int32)
    cv2.fillConvexPoly(mask, points, 1, lineType=cv2.LINE_8, shift=4)
    return mask.astype(bool)


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(7301)
    height, width = 720, 1280
    world_height, world_width = 1000, 2200

    raw = rng.integers(
        0, 256, size=(world_height, world_width), dtype=np.uint8
    )
    texture = cv2.GaussianBlur(raw, (0, 0), 0.85)
    wall = cv2.merge(
        (texture, cv2.add(texture, 8), cv2.add(texture, 16))
    )
    world_quad = np.array(
        [[680, 320], [1060, 320], [1060, 580], [680, 580]],
        dtype=np.float32,
    )
    world_surface = _filled_polygon(
        world_quad, (world_height, world_width)
    )
    wall[world_surface] = (200, 205, 210)

    crop = np.array(
        [[1.0, 0.0, -460.0], [0.0, 1.0, -90.0], [0.0, 0.0, 1.0]]
    )
    frames, surfaces, occluders, truth = [], [], [], []
    occluded_indices = set(range(8, 13))
    colors = np.array([[25, 65, 175], [150, 45, 30]], dtype=np.uint8)

    for index, time in enumerate(np.linspace(-1.0, 1.0, 15)):
        motion = np.vstack(
            (
                cv2.getRotationMatrix2D(
                    (width / 2, height / 2),
                    float(1.3 * time),
                    float(1.0 + 0.035 * time),
                ),
                [0.0, 0.0, 1.0],
            )
        )
        motion[0, 2] -= 330.0 * time
        motion[1, 2] += 7.0 * time
        motion[2, 0] = 8e-6 * time
        motion[2, 1] = -5e-6 * time
        homography = motion @ crop

        quad = cv2.perspectiveTransform(
            world_quad[None, :, :], homography
        )[0]
        image = cv2.warpPerspective(
            wall,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        surface = cv2.warpPerspective(
            world_surface.astype(np.uint8),
            homography,
            (width, height),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)

        occluder = None
        if index in occluded_indices:
            fraction = (index - 8) / 4.0
            left_center = 0.5 * (quad[0] + quad[3])
            right_center = 0.5 * (quad[1] + quad[2])
            center = (1.0 - fraction) * left_center + fraction * right_center

            blob = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(
                blob,
                tuple(np.rint(center).astype(int)),
                (74, 116),
                8.0,
                0.0,
                360.0,
                1,
                cv2.FILLED,
            )
            occluder = blob.astype(bool)
            rows, columns = np.nonzero(occluder)
            pattern = (columns // 9 + rows // 11 + index) % 2
            image[rows, columns] = colors[pattern]
            surface[occluder] = False

        noise = rng.integers(-2, 3, size=image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(
            np.uint8
        )

        image.setflags(write=False)
        surface.setflags(write=False)
        if occluder is not None:
            occluder.setflags(write=False)
        frames.append(image)
        surfaces.append(surface)
        occluders.append(occluder)
        truth.append(quad)

    truth = np.asarray(truth, dtype=np.float32)
    truth.setflags(write=False)
    return {
        "frames": frames,
        "surfaces": surfaces,
        "occluders": occluders,
        "truth": truth,
        "ref_idx": 7,
        "occluded_indices": occluded_indices,
    }


def _small_clip(count):
    rng = np.random.default_rng(408)
    height, width = 180, 260
    texture = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (0, 0), 0.7)
    wall = cv2.merge((texture, texture, texture))
    base = np.array(
        [[60, 50], [170, 50], [170, 125], [60, 125]], dtype=np.float32
    )
    wall[_filled_polygon(base, (height, width))] = (205, 205, 205)

    frames, masks, truth = [], [], []
    for index in range(count):
        delta = np.array([1.5 * index, 0.75 * index], dtype=np.float32)
        quad = base + delta
        motion = np.array(
            [[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]]],
            dtype=np.float32,
        )
        frame = cv2.warpAffine(
            wall,
            motion,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        # A marker outside the tracking region identifies frames in test doubles.
        frame[0, 0] = index
        frames.append(frame)
        masks.append(_filled_polygon(quad, (height, width)))
        truth.append(quad)
    return frames, masks, np.asarray(truth, dtype=np.float32)


def _exact_pair(first, second, quad, first_exclusion, second_exclusion):
    first_index = int(first[0, 0])
    second_index = int(second[0, 0])
    assert abs(second_index - first_index) == 1
    delta = second_index - first_index
    motion = np.array([1.5 * delta, 0.75 * delta])
    return np.asarray(quad, dtype=np.float64) + motion, 0.9, 40


def test_tracking_blank_poster_pan_zoom_occlusion_and_clipping(scene):
    reference_index = scene["ref_idx"]
    reference = scene["truth"][reference_index]
    result = propagate_quad(
        scene["frames"],
        reference_index,
        reference,
        surface_masks=scene["surfaces"],
        occluder_masks=scene["occluders"],
    )

    count = len(scene["frames"])
    quads = result["quads"]
    confidence = result["confidence"]
    inliers = result["inliers"]
    assert set(result) == {"quads", "confidence", "inliers", "status"}
    assert quads.shape == (count, 4, 2)
    assert quads.dtype == np.float32
    assert confidence.shape == inliers.shape == (count,)
    assert confidence.dtype.kind == "f"
    assert inliers.dtype.kind in "iu"
    assert np.isfinite(quads).all()
    assert np.isfinite(confidence).all()
    assert np.all((confidence > 0.0) & (confidence <= 1.0))
    np.testing.assert_array_equal(quads[reference_index], reference)
    assert inliers[reference_index] == 0
    assert np.all(inliers >= 0)
    # Re-seeding is allowed to have no homography support.
    assert np.count_nonzero(inliers >= 12) >= count // 2

    error = np.linalg.norm(quads - scene["truth"], axis=2)
    assert float(np.median(error)) <= 3.0
    assert float(np.percentile(error, 95)) <= 5.0

    for index, status in enumerate(result["status"]):
        expected = (
            "occluded" if index in scene["occluded_indices"] else "ok"
        )
        assert status == expected

    truth = scene["truth"]
    clipped = (
        (truth[:, :, 0] < 0)
        | (truth[:, :, 0] > 1279)
        | (truth[:, :, 1] < 0)
        | (truth[:, :, 1] > 719)
    ).any(axis=1)
    assert np.count_nonzero(clipped) >= 2
    assert np.isfinite(quads[clipped]).all()
    assert quads[-1, 0, 0] < -20.0

    edges = np.roll(quads, -1, axis=1) - quads
    following = np.roll(edges, -1, axis=1)
    crosses = (
        edges[:, :, 0] * following[:, :, 1]
        - edges[:, :, 1] * following[:, :, 0]
    )
    assert np.all(crosses > 0.0)


def test_empty_mask_outlier_preserves_supported_motion_track(scene):
    indices = [7, 8, 9]
    surfaces = [scene["surfaces"][index] for index in indices]
    surfaces[1] = np.zeros((720, 1280), dtype=bool)

    result = propagate_quad(
        [scene["frames"][index] for index in indices],
        0,
        scene["truth"][7],
        surface_masks=surfaces,
        occluder_masks=[scene["occluders"][index] for index in indices],
    )

    assert result["status"] == ["ok", "occluded", "occluded"]
    assert np.isfinite(result["quads"]).all()
    assert result["inliers"][1] >= 12
    assert result["confidence"][1] > 0.0
    error = np.linalg.norm(
        result["quads"] - scene["truth"][indices], axis=2
    )
    assert float(np.median(error)) <= 3.0


def test_fully_occluded_surface_is_not_negative_mask_evidence(scene):
    full_occlusion = _filled_polygon(scene["truth"][8], (720, 1280))
    full_occlusion |= scene["occluders"][8]
    covered_frame = scene["frames"][8].copy()
    covered_frame[full_occlusion] = (45, 65, 140)

    result = propagate_quad(
        [scene["frames"][7], covered_frame],
        0,
        scene["truth"][7],
        surface_masks=[
            scene["surfaces"][7],
            np.zeros((720, 1280), dtype=bool),
        ],
        occluder_masks=[None, full_occlusion],
    )

    assert result["status"] == ["ok", "occluded"]
    assert np.isfinite(result["quads"][1]).all()
    assert result["inliers"][1] >= 12
    error = np.linalg.norm(result["quads"][1] - scene["truth"][8], axis=1)
    assert float(np.median(error)) <= 3.0


def test_missing_feature_support_drops_untrackable_direction():
    flat = np.full((240, 320, 3), 127, dtype=np.uint8)
    reference = np.array(
        [[80, 60], [240, 60], [240, 180], [80, 180]], dtype=np.float32
    )
    result = propagate_quad([flat.copy() for _ in range(3)], 0, reference)

    assert result["status"] == ["ok", "lost", "lost"]
    np.testing.assert_array_equal(result["quads"][0], reference)
    assert np.isnan(result["quads"][1:]).all()
    assert np.all(result["confidence"][1:] == 0.0)
    assert np.all(result["inliers"] == 0)


@pytest.mark.parametrize("ref_idx", [0, 9])
def test_reacquires_after_forced_noise_failure_in_both_directions(
    monkeypatch, ref_idx
):
    frames, masks, truth = _small_clip(10)
    rng = np.random.default_rng(901)
    damaged = {3, 4, 5}
    for index in damaged:
        frames[index] = rng.integers(
            0, 256, size=frames[index].shape, dtype=np.uint8
        )
        frames[index][0, 0] = index

    original_track_pair = geometry._track_pair

    def forced_failure(first, second, quad, first_exclusion, second_exclusion):
        first_index = int(first[0, 0])
        second_index = int(second[0, 0])
        assert abs(second_index - first_index) == 1
        if first_index in damaged or second_index in damaged:
            return None, 0.0, 0
        return original_track_pair(
            first, second, quad, first_exclusion, second_exclusion
        )

    monkeypatch.setattr(geometry, "_track_pair", forced_failure)
    surfaces = [
        masks[index] if index in {0, 2, 6, 9} else None
        for index in range(10)
    ]
    original_frames = [frame.copy() for frame in frames]
    result = propagate_quad(
        frames, ref_idx, truth[ref_idx], surface_masks=surfaces
    )

    assert result["status"] == ["ok"] * 10
    assert np.isfinite(result["quads"]).all()
    error = np.linalg.norm(result["quads"] - truth, axis=2)
    assert float(np.max(error)) <= 3.5
    assert np.all(result["inliers"][3:6] == 0)
    assert np.all(result["confidence"][3:6] > 0.0)
    assert np.all(result["confidence"][3:6] <= 0.5)

    reacquired = 6 if ref_idx == 0 else 2
    continued = 7 if ref_idx == 0 else 1
    assert result["inliers"][reacquired] == 0
    assert result["confidence"][reacquired] > 0.5
    assert result["inliers"][continued] >= 12
    np.testing.assert_array_equal(result["quads"][ref_idx], truth[ref_idx])
    for frame, original in zip(frames, original_frames):
        np.testing.assert_array_equal(frame, original)


def test_single_shifted_mask_outlier_does_not_reseed_or_drop(monkeypatch):
    frames, masks, truth = _small_clip(5)
    masks[2] = _filled_polygon(
        truth[2] + np.array([45.0, 10.0]), frames[2].shape[:2]
    )
    original_masks = [mask.copy() for mask in masks]
    monkeypatch.setattr(geometry, "_track_pair", _exact_pair)

    result = propagate_quad(frames, 0, truth[0], surface_masks=masks)

    assert result["status"] == ["ok"] * 5
    np.testing.assert_allclose(result["quads"], truth, atol=1e-5, rtol=0)
    assert 0.0 < result["confidence"][2] < result["confidence"][1]
    assert result["inliers"][2] == 40
    for mask, original in zip(masks, original_masks):
        np.testing.assert_array_equal(mask, original)


def test_neighbour_supported_mask_corrects_bad_track_and_continues(monkeypatch):
    frames, masks, truth = _small_clip(5)

    def bad_forward_pair(first, second, quad, first_exclusion, second_exclusion):
        candidate, quality, support = _exact_pair(
            first, second, quad, first_exclusion, second_exclusion
        )
        if int(first[0, 0]) == 1 and int(second[0, 0]) == 2:
            candidate = candidate + np.array([35.0, -8.0])
        return candidate, quality, support

    monkeypatch.setattr(geometry, "_track_pair", bad_forward_pair)
    result = propagate_quad(frames, 0, truth[0], surface_masks=masks)

    assert result["status"] == ["ok"] * 5
    error = np.linalg.norm(result["quads"] - truth, axis=2)
    assert float(np.max(error)) <= 2.0
    assert result["confidence"][2] <= 0.75
    assert result["inliers"][3] == 40


@pytest.mark.parametrize("ref_idx", [0, 2])
def test_many_inliers_and_temporally_agreeing_mask_are_never_lost(
    monkeypatch, ref_idx
):
    frames, masks, truth = _small_clip(3)
    target_index = 2 - ref_idx
    surfaces = [None] * len(frames)
    surfaces[target_index] = masks[target_index]

    seed = quad_from_mask(masks[target_index])
    assert seed is not None
    assert float(np.linalg.norm(seed - truth[1], axis=1).max()) <= 4.0

    def misleading_pair(first, second, quad, first_exclusion, second_exclusion):
        candidate, quality, _ = _exact_pair(
            first, second, quad, first_exclusion, second_exclusion
        )
        if int(second[0, 0]) == target_index:
            candidate = candidate + np.array([18.0, -6.0])
        return candidate, quality, 60

    monkeypatch.setattr(geometry, "_track_pair", misleading_pair)
    result = propagate_quad(
        frames, ref_idx, truth[ref_idx], surface_masks=surfaces
    )

    # The target is an endpoint, so interpolation cannot conceal rejection.
    # Its only neighbour also predicts the misleading homography.
    assert result["status"][target_index] == "ok"
    assert result["status"] == ["ok"] * len(frames)
    assert result["inliers"][target_index] == 60
    assert result["confidence"][target_index] > 0.5
    error = np.linalg.norm(result["quads"] - truth, axis=2)
    assert float(error.max()) <= 2.0


def test_propagate_quad_is_deterministic_across_identical_runs():
    frames, masks, truth = _small_clip(9)
    ref_idx = 4

    cv2.setRNGSeed(17)
    first = propagate_quad(
        frames, ref_idx, truth[ref_idx], surface_masks=masks
    )
    cv2.setRNGSeed(901)
    second = propagate_quad(
        frames, ref_idx, truth[ref_idx], surface_masks=masks
    )

    assert first["status"] == ["ok"] * len(frames)
    assert np.isfinite(first["quads"]).all()
    assert np.count_nonzero(first["inliers"] >= 12) >= len(frames) // 2
    assert first["status"] == second["status"]
    for key in ("quads", "confidence", "inliers"):
        np.testing.assert_array_equal(first[key], second[key])


def test_low_confidence_reseed_preserves_canonical_corner_identity(monkeypatch):
    frames, masks, truth = _small_clip(4)

    def weak_pair(first, second, quad, first_exclusion, second_exclusion):
        candidate, _, _ = _exact_pair(
            first, second, quad, first_exclusion, second_exclusion
        )
        return candidate + np.array([10.0, 0.0]), 0.15, 12

    monkeypatch.setattr(geometry, "_track_pair", weak_pair)
    canonical = np.roll(truth, 2, axis=1)
    result = propagate_quad(frames, 0, canonical[0], surface_masks=masks)

    assert result["status"] == ["ok"] * 4
    error = np.linalg.norm(result["quads"] - canonical, axis=2)
    assert float(np.max(error)) <= 2.0
    assert np.all(result["confidence"][1:] > 0.5)
    np.testing.assert_array_equal(result["quads"][0], canonical[0])


@pytest.mark.parametrize("gap_length", [12, 13])
def test_only_short_bracketed_maskless_gaps_are_interpolated(
    monkeypatch, gap_length
):
    count = gap_length + 2
    frames, masks, truth = _small_clip(count)
    monkeypatch.setattr(
        geometry, "_track_pair", lambda *args: (None, 0.0, 0)
    )
    surfaces = [None] * count
    surfaces[-1] = masks[-1]

    result = propagate_quad(frames, 0, truth[0], surface_masks=surfaces)

    assert result["status"][0] == result["status"][-1] == "ok"
    assert result["confidence"][-1] > 0.5
    assert np.all(result["inliers"] == 0)
    if gap_length == 12:
        assert result["status"] == ["ok"] * count
        assert np.all(result["confidence"][1:-1] > 0.0)
        assert np.all(result["confidence"][1:-1] <= 0.5)
        error = np.linalg.norm(result["quads"] - truth, axis=2)
        assert float(np.max(error)) <= 2.0
    else:
        assert result["status"][1:-1] == ["lost"] * gap_length
        assert np.isnan(result["quads"][1:-1]).all()
        assert np.all(result["confidence"][1:-1] == 0.0)


def test_short_bracketed_gaps_with_masks_and_occlusion_are_interpolated(
    monkeypatch,
):
    count = 14
    frames, masks, truth = _small_clip(count)
    monkeypatch.setattr(
        geometry, "_track_pair", lambda *args: (None, 0.0, 0)
    )
    surfaces = [None] + [
        np.zeros(frames[0].shape[:2], dtype=bool)
        for _ in range(count - 2)
    ] + [masks[-1]]
    occluders = [None] * count
    occluded_index = 6
    occluders[occluded_index] = _filled_polygon(
        truth[occluded_index], frames[0].shape[:2]
    )

    result = propagate_quad(
        frames,
        0,
        truth[0],
        surface_masks=surfaces,
        occluder_masks=occluders,
    )

    expected_status = ["ok"] * count
    expected_status[occluded_index] = "occluded"
    assert result["status"] == expected_status
    assert np.isfinite(result["quads"]).all()
    assert np.all(result["inliers"] == 0)
    assert np.all(result["confidence"][1:-1] > 0.0)
    assert np.all(result["confidence"][1:-1] <= 0.5)
    error = np.linalg.norm(result["quads"] - truth, axis=2)
    assert float(error.max()) <= 2.0


def test_short_gap_with_collapsing_corner_blend_remains_finite_and_convex():
    base = np.array(
        [[10, 10], [110, 10], [110, 90], [10, 90]], dtype=np.float64
    )
    quads = np.stack(
        (base, np.full((4, 2), np.nan), np.roll(base, 2, axis=0))
    )
    original_endpoints = quads[[0, 2]].copy()
    confidence = np.array([1.0, 0.0, 0.8], dtype=np.float32)
    status = ["ok", "lost", "ok"]

    geometry._fill_short_gaps(quads, confidence, status, [None] * 3)

    assert status == ["ok"] * 3
    assert np.isfinite(quads).all()
    assert geometry._valid_quad(quads[1], min_area=1.0)
    assert 0.0 < confidence[1] <= 0.5
    np.testing.assert_array_equal(quads[[0, 2]], original_endpoints)


@pytest.mark.parametrize("kind", ["clipped", "occluded"])
def test_partial_visible_masks_cannot_reseed_canonical_corners(monkeypatch, kind):
    height, width = 180, 260
    frames = [np.full((height, width, 3), 127, dtype=np.uint8) for _ in range(2)]
    reference = np.array(
        [[40, 40], [180, 40], [180, 140], [40, 140]], dtype=np.float32
    )
    occluders = [None, None]
    if kind == "clipped":
        visible_quad = reference - np.array([70.0, 0.0])
        surface = _filled_polygon(visible_quad, (height, width))
    else:
        surface = _filled_polygon(reference, (height, width))
        occluder = np.zeros((height, width), dtype=bool)
        occluder[20:161, 20:111] = True
        surface[occluder] = False
        occluders[1] = occluder

    monkeypatch.setattr(
        geometry, "_track_pair", lambda *args: (None, 0.0, 0)
    )
    result = propagate_quad(
        frames,
        0,
        reference,
        surface_masks=[None, surface],
        occluder_masks=occluders,
    )

    assert result["status"] == ["ok", "lost"]
    assert np.isnan(result["quads"][1]).all()
    assert result["confidence"][1] == 0.0


def test_quad_from_noisy_slightly_rounded_perspective_mask():
    rng = np.random.default_rng(91)
    local = np.zeros((250, 400), dtype=np.uint8)
    radius = 16
    cv2.rectangle(local, (radius, 0), (399 - radius, 249), 1, cv2.FILLED)
    cv2.rectangle(local, (0, radius), (399, 249 - radius), 1, cv2.FILLED)
    for center in (
        (radius, radius),
        (399 - radius, radius),
        (399 - radius, 249 - radius),
        (radius, 249 - radius),
    ):
        cv2.circle(local, center, radius, 1, cv2.FILLED)

    source = np.array(
        [[0, 0], [399, 0], [399, 249], [0, 249]], dtype=np.float32
    )
    truth = np.array(
        [[125, 95], [530, 120], [500, 380], [105, 345]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, truth)
    mask = cv2.warpPerspective(
        local, homography, (640, 480), flags=cv2.INTER_NEAREST
    ).astype(bool)
    mask ^= rng.random(mask.shape) < 0.01
    original = mask.copy()

    quad = quad_from_mask(mask)

    assert quad is not None
    assert quad.shape == (4, 2)
    assert quad.dtype == np.float32
    assert cv2.isContourConvex(quad.reshape(-1, 1, 2))
    error = np.linalg.norm(quad - truth, axis=1)
    assert float(np.median(error)) <= 2.5
    assert float(error.max()) <= 4.0
    np.testing.assert_array_equal(mask, original)


@pytest.mark.parametrize(
    "kind",
    ["empty", "circle", "ellipse", "triangle", "pentagon", "concave", "two", "sliver"],
)
def test_quad_from_mask_rejects_non_quad_blobs(kind):
    mask = np.zeros((400, 500), dtype=np.uint8)
    if kind == "circle":
        cv2.circle(mask, (250, 200), 110, 1, cv2.FILLED)
    elif kind == "ellipse":
        cv2.ellipse(mask, (250, 200), (150, 90), 12, 0, 360, 1, cv2.FILLED)
    elif kind == "triangle":
        cv2.fillPoly(
            mask, [np.array([[90, 325], [250, 60], [410, 325]])], 1
        )
    elif kind == "pentagon":
        angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, 5, endpoint=False)
        vertices = np.column_stack(
            (250 + 140 * np.cos(angles), 200 + 140 * np.sin(angles))
        )
        cv2.fillPoly(mask, [np.rint(vertices).astype(np.int32)], 1)
    elif kind == "concave":
        vertices = np.array(
            [[80, 70], [180, 70], [180, 240],
             [420, 240], [420, 340], [80, 340]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [vertices], 1)
    elif kind == "two":
        cv2.rectangle(mask, (40, 80), (190, 320), 1, cv2.FILLED)
        cv2.rectangle(mask, (300, 80), (450, 320), 1, cv2.FILLED)
    elif kind == "sliver":
        cv2.rectangle(mask, (40, 190), (460, 198), 1, cv2.FILLED)

    assert quad_from_mask(mask.astype(bool)) is None


def test_smoothing_reduces_jitter_without_lag_or_bridging_nan_runs():
    rng = np.random.default_rng(180)
    time = np.arange(61, dtype=np.float32)
    base = np.array(
        [[100, 100], [400, 100], [400, 300], [100, 300]],
        dtype=np.float32,
    )
    motion = np.column_stack((1.8 * time, 0.45 * time))
    truth = base[None, :, :] + motion[:, None, :]
    noisy = (
        truth + rng.normal(0.0, 1.3, size=truth.shape)
    ).astype(np.float32)
    noisy[0] = np.nan
    noisy[20:26] = np.nan
    noisy[43:46] = np.nan
    noisy[-1] = np.nan
    original = noisy.copy()

    result = smooth_quads(noisy, strength=0.75)
    valid = np.isfinite(noisy).all(axis=(1, 2))
    assert result.shape == noisy.shape
    assert result.dtype == np.float32
    assert np.isnan(result[~valid]).all()
    assert np.isfinite(result[valid]).all()

    raw_rmse = np.sqrt(np.mean((noisy[valid] - truth[valid]) ** 2))
    smooth_rmse = np.sqrt(np.mean((result[valid] - truth[valid]) ** 2))
    assert smooth_rmse < 0.85 * raw_rmse
    mean_offset = (result[valid] - truth[valid]).mean(axis=(0, 1))
    assert float(np.linalg.norm(mean_offset)) < 0.45

    np.testing.assert_allclose(
        smooth_quads(truth, strength=1.0), truth, atol=5e-5, rtol=0
    )
    changed = noisy.copy()
    changed[:20] += np.array([60.0, -20.0], dtype=np.float32)
    changed_result = smooth_quads(changed, strength=0.75)
    np.testing.assert_allclose(
        changed_result[26:], result[26:], equal_nan=True, atol=0, rtol=0
    )
    np.testing.assert_allclose(noisy, original, equal_nan=True)

    unchanged = smooth_quads(noisy, strength=0.0)
    np.testing.assert_allclose(unchanged, noisy, equal_nan=True)
    assert not np.shares_memory(unchanged, noisy)


def test_smoothing_leaves_short_runs_and_normalizes_partial_missing_quads():
    base = np.array(
        [[10, 10], [110, 10], [110, 90], [10, 90]], dtype=np.float32
    )
    quads = np.repeat(base[None], 9, axis=0)
    quads[:, :, 0] += np.arange(9, dtype=np.float32)[:, None]
    quads[3:5] = np.nan

    result = smooth_quads(quads, strength=1.0)
    np.testing.assert_allclose(result, quads, equal_nan=True)

    partial = np.repeat(base[None], 3, axis=0)
    partial[1, 0, 0] = np.nan
    result = smooth_quads(partial)
    assert np.isnan(result[1]).all()
    np.testing.assert_array_equal(result[[0, 2]], partial[[0, 2]])


@pytest.mark.parametrize("strength", [-0.1, 1.1, np.nan])
def test_smoothing_rejects_invalid_strength(strength):
    with pytest.raises(ValueError):
        smooth_quads(np.empty((0, 4, 2), dtype=np.float32), strength)
