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


@pytest.mark.parametrize("ref_idx", [0, 32])
@pytest.mark.parametrize("camera_mode", ["unavailable", "misleading"])
def test_dark_blurred_occluder_sweep_with_handheld_jitter(
    monkeypatch, ref_idx, camera_mode
):
    height, width = 360, 640
    count = 34
    base = np.array(
        [[200, 110], [440, 110], [440, 260], [200, 260]],
        dtype=np.float32,
    )
    world_surface = _filled_polygon(base, (height, width))

    # No textured remote background: only a blank bright board on darkness.
    wall = np.full((height, width, 3), 8, dtype=np.uint8)
    wall[world_surface] = (185, 190, 195)

    jitter_rng = np.random.default_rng(1729)
    jitter = np.cumsum(
        jitter_rng.normal(0.0, 3.0, size=(count, 2)), axis=0
    )
    jitter -= jitter[0]

    frames, surfaces, occluders, truth, coverage = [], [], [], [], []
    missing = {12, 19, 20, 21}

    for index in range(count):
        motion = cv2.getRotationMatrix2D(
            (width / 2, height / 2),
            0.035 * index,
            1.0 + 0.0005 * index,
        )
        motion[0, 2] += -0.55 * index + jitter[index, 0]
        motion[1, 2] += 0.20 * index + jitter[index, 1]
        quad = cv2.transform(base[None], motion)[0]
        frame = cv2.warpAffine(
            wall,
            motion,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        surface = cv2.warpAffine(
            world_surface.astype(np.uint8),
            motion,
            (width, height),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)

        body = np.zeros((height, width), dtype=np.uint8)
        present = 4 <= index <= 28
        if present:
            center_x = 65 + 23 * (index - 4)
            cv2.rectangle(
                body,
                (center_x - 96, 78),
                (center_x + 96, 300),
                1,
                cv2.FILLED,
            )
            cv2.ellipse(
                body, (center_x, 66), (31, 32),
                0, 0, 360, 1, cv2.FILLED,
            )
            alpha = cv2.GaussianBlur(
                body.astype(np.float32), (0, 0), 2.0
            )
            alpha = cv2.blur(alpha, (13, 1))
            outer = cv2.dilate(body, np.ones((7, 7), dtype=np.uint8))
            halo = np.maximum(
                cv2.GaussianBlur(
                    outer.astype(np.float32), (0, 0), 2.0
                ) - alpha,
                0.0,
            )
            frame = np.clip(
                (1.0 - alpha[..., None]) * frame.astype(np.float32)
                + 3.0 * alpha[..., None]
                + 7.0 * halo[..., None],
                0,
                255,
            ).astype(np.uint8)

        binary_body = body.astype(bool)
        coverage.append(
            np.count_nonzero(surface & binary_body)
            / np.count_nonzero(surface)
        )
        surface[binary_body] = False
        person_mask = (
            binary_body if present and index not in missing else None
        )

        # Identifies frames to the deliberately incorrect camera test double.
        frame[0, 0] = index
        frame.setflags(write=False)
        surface.setflags(write=False)
        if person_mask is not None:
            person_mask.setflags(write=False)
        frames.append(frame)
        surfaces.append(surface)
        occluders.append(person_mask)
        truth.append(quad)

    truth = np.asarray(truth, dtype=np.float32)
    coverage = np.asarray(coverage)
    assert 0.78 <= float(coverage.max()) <= 0.84
    occluded = coverage > 0.0
    assert np.count_nonzero(occluded) > 12
    assert all(occluded[index] for index in missing)

    covered_indices = np.flatnonzero(occluded)
    left = int(covered_indices[0]) - 1
    right = int(covered_indices[-1]) + 1
    fractions = (
        (covered_indices - left) / (right - left)
    )[:, None, None]
    linear = (
        (1.0 - fractions) * truth[left] + fractions * truth[right]
    )
    linear_errors = np.linalg.norm(linear - truth[occluded], axis=2)
    assert float(np.median(linear_errors)) > 3.0
    assert float(linear_errors.max()) > 6.0

    original_track_pair = geometry._track_pair

    def reject_local_occlusion(
        first, second, quad, first_exclusion, second_exclusion
    ):
        if first_exclusion is not None or second_exclusion is not None:
            return None, 0.0, 0
        return original_track_pair(
            first, second, quad, first_exclusion, second_exclusion
        )

    def camera_pair(first, second, *unused):
        if camera_mode == "unavailable":
            return None, 0.0, 0
        direction = int(second[0, 0]) - int(first[0, 0])
        assert abs(direction) == 1
        motion = np.eye(3, dtype=np.float64)
        motion[:2, 2] = direction * np.array([7.0, -5.0])
        return motion, 0.70, 80

    monkeypatch.setattr(geometry, "_track_pair", reject_local_occlusion)
    monkeypatch.setattr(geometry, "_camera_motion_pair", camera_pair)
    reference = quad_from_mask(surfaces[ref_idx])
    assert reference is not None

    original_frames = [frame.copy() for frame in frames]
    original_surfaces = [mask.copy() for mask in surfaces]
    original_occluders = [
        None if mask is None else mask.copy() for mask in occluders
    ]
    result = propagate_quad(
        frames,
        ref_idx,
        reference,
        surface_masks=surfaces,
        occluder_masks=occluders,
    )

    assert np.isfinite(result["quads"]).all()
    assert "lost" not in result["status"]
    errors = np.linalg.norm(result["quads"] - truth, axis=2)
    assert float(np.median(errors[occluded])) <= 3.0
    assert float(errors[occluded].max()) <= 6.0
    assert all(
        result["status"][index] == "occluded"
        for index in covered_indices
    )
    assert np.all(result["confidence"][occluded] >= 0.30 - 1e-6)
    assert np.all(result["confidence"][occluded] <= 0.70 + 1e-6)
    assert (
        np.count_nonzero(result["inliers"][occluded] >= 8)
        > np.count_nonzero(occluded) // 2
    )
    np.testing.assert_array_equal(result["quads"][ref_idx], reference)
    for index in range(count):
        np.testing.assert_array_equal(frames[index], original_frames[index])
        np.testing.assert_array_equal(surfaces[index], original_surfaces[index])
        if occluders[index] is not None:
            np.testing.assert_array_equal(
                occluders[index], original_occluders[index]
            )


def test_surface_boundary_excludes_occluder_cut_and_dilated_rim():
    shape = (180, 260)
    board = np.array(
        [[40, 30], [220, 30], [220, 150], [40, 150]],
        dtype=np.float64,
    )
    occluder = np.zeros(shape, dtype=bool)
    occluder[10:171, 20:191] = True
    surface = _filled_polygon(board, shape) & ~occluder
    original_surface = surface.copy()
    original_occluder = occluder.copy()

    points = geometry._surface_boundary_points(surface, occluder)

    assert len(points) > 80
    pixels = points.astype(np.int32)
    ignored = cv2.dilate(
        occluder.astype(np.uint8), np.ones((19, 19), dtype=np.uint8)
    )
    assert not np.any(ignored[pixels[:, 1], pixels[:, 0]])
    assert np.all(points[:, 0] > 199)
    assert np.any(points[:, 0] == 220)
    np.testing.assert_array_equal(surface, original_surface)
    np.testing.assert_array_equal(occluder, original_occluder)


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
def test_surface_boundary_excludes_five_pixel_image_rim(quarter_turns):
    shape = (180, 260)
    board = np.array(
        [[-30, 30], [220, 30], [220, 150], [-30, 150]],
        dtype=np.float64,
    )
    surface = np.rot90(
        _filled_polygon(board, shape), quarter_turns
    ).copy()
    original = surface.copy()
    surface.setflags(write=False)

    points = geometry._surface_boundary_points(surface, None)

    height, width = surface.shape
    assert len(points) > 100
    assert np.all(points[:, 0] >= 5)
    assert np.all(points[:, 0] < width - 5)
    assert np.all(points[:, 1] >= 5)
    assert np.all(points[:, 1] < height - 5)
    np.testing.assert_array_equal(surface, original)


def test_partial_edge_assignment_requires_local_contour_direction(monkeypatch):
    shape = (180, 260)
    prediction = np.array(
        [[40, 30], [220, 30], [220, 150], [40, 150]],
        dtype=np.float64,
    )
    surface = _filled_polygon(prediction, shape)

    # These samples are close to the top edge but belong to a perpendicular
    # boundary. A later fit of their positions alone would accept a top line.
    points = np.column_stack(
        (np.linspace(60.0, 200.0, 40), np.full(40, 31.0))
    )
    tangents = np.tile([0.0, 1.0], (len(points), 1))

    monkeypatch.setattr(
        geometry,
        "_surface_boundary_samples",
        lambda *unused: (points.copy(), tangents.copy()),
    )

    candidate, confidence, support = geometry._fit_partial_surface_quad(
        surface, None, prediction, fallback=prediction
    )

    assert candidate is None
    assert confidence == 0.0
    assert support == 0


@pytest.mark.parametrize("corner_shift", [0, 2])
def test_clipping_and_occlusion_require_fallback_for_unobserved_motion(
    corner_shift,
):
    shape = (180, 260)
    truth = np.array(
        [[-30, 30], [150, 30], [150, 150], [-30, 150]],
        dtype=np.float64,
    )
    occluder = np.zeros(shape, dtype=bool)
    occluder[20:161, 120:181] = True
    surface = _filled_polygon(truth, shape) & ~occluder
    prediction = np.roll(
        truth + np.array([12.0, 3.0]), corner_shift, axis=0
    )
    fallback = np.roll(truth, corner_shift, axis=0).copy()
    original_prediction = prediction.copy()
    original_fallback = fallback.copy()
    original_surface = surface.copy()
    original_occluder = occluder.copy()

    # Only top and bottom survive: neither observes horizontal camera motion.
    missing = geometry._fit_partial_surface_quad(
        surface, occluder, prediction
    )
    assert missing[0] is None
    assert missing[1:] == (0.0, 0)

    candidate, confidence, support = geometry._fit_partial_surface_quad(
        surface, occluder, prediction, fallback=fallback
    )

    assert candidate is not None
    assert geometry._valid_quad(candidate)
    assert np.linalg.norm(candidate - fallback, axis=1).max() <= 1.0
    assert np.all(candidate[[0, 3] if corner_shift == 0 else [1, 2], 0] < 0)
    assert 0.30 <= confidence <= 0.70
    assert support >= 80
    np.testing.assert_array_equal(prediction, original_prediction)
    np.testing.assert_array_equal(fallback, original_fallback)
    np.testing.assert_array_equal(surface, original_surface)
    np.testing.assert_array_equal(occluder, original_occluder)


@pytest.mark.parametrize("failure_mode", ["insufficient", "solver"])
def test_partial_edge_failure_preserves_other_fits_and_fallback(
    monkeypatch, failure_mode
):
    shape = (180, 260)
    truth = np.array(
        [[40, 30], [220, 30], [220, 150], [40, 150]],
        dtype=np.float64,
    )
    surface = _filled_polygon(truth, shape)
    prediction = truth + np.array([2.0, -1.0])
    original_fit = geometry._regularized_edge_line
    rejected = []

    def reject_right_edge(samples, start, end):
        if np.ptp(samples[:, 0]) < 1.0 and np.mean(samples[:, 0]) > 200:
            rejected.append(True)
            if failure_mode == "solver":
                raise np.linalg.LinAlgError("forced edge-fit failure")
            return None
        return original_fit(samples, start, end)

    monkeypatch.setattr(
        geometry, "_regularized_edge_line", reject_right_edge
    )
    candidate, confidence, support = geometry._fit_partial_surface_quad(
        surface, None, prediction, fallback=truth
    )

    assert rejected
    assert candidate is not None
    assert geometry._valid_quad(candidate)
    assert np.linalg.norm(candidate - truth, axis=1).max() <= 1.0
    np.testing.assert_allclose(candidate[1:3, 0], truth[1:3, 0], atol=1e-6)
    assert 0.30 <= confidence <= 0.70
    assert support >= 100


def test_clipped_mask_bridge_uses_camera_for_missing_motion(monkeypatch):
    shape = (180, 260)
    count = 3
    base = np.array(
        [[-20, 30], [170, 30], [170, 150], [-20, 150]],
        dtype=np.float64,
    )
    truth = base[None] + np.arange(count)[:, None, None] * np.array(
        [-10.0, 2.0]
    )
    quads = np.full_like(truth, np.nan)
    quads[[0, -1]] = truth[[0, -1]]
    anchors = quads[[0, -1]].copy()
    confidence = np.array([1.0, 0.0, 0.9], dtype=np.float32)
    inliers = np.zeros(count, dtype=np.int32)
    status = ["ok", "occluded", "ok"]

    occluder = np.zeros(shape, dtype=bool)
    occluder[20:171, 130:201] = True
    occluders = [None, occluder, None]
    surfaces = [_filled_polygon(quad, shape) for quad in truth]
    surfaces[1] &= ~occluder
    calls = []

    def camera_pair(first, second, *unused):
        first_index = int(first[0, 0])
        second_index = int(second[0, 0])
        assert abs(second_index - first_index) == 1
        calls.append((first_index, second_index))
        motion = np.eye(3, dtype=np.float64)
        motion[:2, 2] = (
            truth[second_index, 0] - truth[first_index, 0]
        )
        return motion, 0.65, 24

    monkeypatch.setattr(geometry, "_camera_motion_pair", camera_pair)
    geometry._bridge_occlusions(
        quads,
        confidence,
        inliers,
        status,
        occluders,
        [np.full(shape, index, dtype=np.uint8) for index in range(count)],
        occluders,
        surfaces,
        ref_idx=0,
    )

    assert calls
    assert np.linalg.norm(quads - truth, axis=2).max() <= 1.0
    np.testing.assert_array_equal(quads[[0, -1]], anchors)
    assert np.all(quads[:, [0, 3], 0] < 0)
    assert status == ["ok", "occluded", "ok"]
    assert inliers[1] >= 80
    assert 0.30 - 1e-6 <= confidence[1] <= 0.70 + 1e-6


@pytest.mark.parametrize("corner_shift", [0, 2])
def test_partial_edge_fit_recovers_hidden_corners_without_reordering(corner_shift):
    shape = (240, 340)
    base = np.array(
        [[60, 40], [260, 40], [260, 180], [60, 180]],
        dtype=np.float32,
    )
    motion = cv2.getRotationMatrix2D((160, 110), 1.0, 1.004)
    motion[:, 2] += [3.0, -2.0]
    truth = cv2.transform(base[None], motion)[0]
    board = cv2.warpAffine(
        _filled_polygon(base, shape).astype(np.uint8),
        motion,
        (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)

    body = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(body, (30, 10), (220, 210), 1, cv2.FILLED)
    occluder = cv2.warpAffine(
        body,
        motion,
        (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    surface = board & ~occluder
    prediction = np.roll(base, corner_shift, axis=0).astype(np.float64)
    expected = np.roll(truth, corner_shift, axis=0)
    original_prediction = prediction.copy()

    candidate, confidence, support = geometry._fit_partial_surface_quad(
        surface, occluder, prediction
    )

    assert candidate is not None
    assert geometry._valid_quad(candidate)
    assert 0.30 <= confidence <= 0.70
    assert support >= 80
    assert np.linalg.norm(candidate - expected, axis=1).max() <= 3.0
    np.testing.assert_array_equal(prediction, original_prediction)


@pytest.mark.parametrize("surface_kind", ["missing", "empty", "fully_covered"])
def test_partial_edge_fit_requires_actual_boundary_support(surface_kind):
    shape = (180, 260)
    prediction = np.array(
        [[40, 30], [220, 30], [220, 150], [40, 150]],
        dtype=np.float64,
    )
    occluder = np.ones(shape, dtype=bool)
    if surface_kind == "missing":
        surface = None
    elif surface_kind == "empty":
        surface = np.zeros(shape, dtype=bool)
    else:
        surface = _filled_polygon(prediction, shape)

    candidate, confidence, support = geometry._fit_partial_surface_quad(
        surface, occluder, prediction
    )

    assert candidate is None
    assert confidence == 0.0
    assert support == 0


def test_mask_bridge_does_not_consult_camera_when_edges_are_available(monkeypatch):
    shape = (180, 280)
    count = 9
    base = np.array(
        [[60, 40], [230, 40], [230, 140], [60, 140]],
        dtype=np.float64,
    )
    shifts = np.array(
        [[0, 0], [2, -1], [-1, 2], [3, 1], [0, -2],
         [4, 2], [1, -1], [3, 0], [0, 0]],
        dtype=np.float64,
    )
    truth = base[None] + shifts[:, None]
    quads = np.full_like(truth, np.nan)
    quads[[0, -1]] = truth[[0, -1]]
    confidence = np.zeros(count, dtype=np.float32)
    confidence[[0, -1]] = [1.0, 0.9]
    occluders, surfaces = [], []
    for index in range(count):
        board = _filled_polygon(truth[index], shape)
        occluder = None
        if 0 < index < count - 1:
            occluder = np.zeros(shape, dtype=bool)
            occluder[20:161, 30:191] = True
            board &= ~occluder
        occluders.append(occluder)
        surfaces.append(board)

    def forbidden_camera(*unused):
        pytest.fail("Visible mask edges must be measured before camera motion")

    monkeypatch.setattr(geometry, "_camera_motion_pair", forbidden_camera)
    inliers = np.zeros(count, dtype=np.int32)
    status = ["ok"] + ["occluded"] * (count - 2) + ["ok"]
    anchors = quads[[0, -1]].copy()

    geometry._bridge_occlusions(
        quads,
        confidence,
        inliers,
        status,
        occluders,
        [np.full(shape, 8, dtype=np.uint8) for _ in range(count)],
        occluders,
        surfaces,
        ref_idx=0,
    )

    assert np.linalg.norm(quads - truth, axis=2).max() <= 2.0
    np.testing.assert_array_equal(quads[[0, -1]], anchors)
    assert status == ["ok"] + ["occluded"] * (count - 2) + ["ok"]
    assert np.all(inliers[1:-1] >= 8)


def test_partial_masks_do_not_establish_geometry_without_an_anchor(monkeypatch):
    shape = (180, 260)
    board = np.array(
        [[40, 30], [220, 30], [220, 150], [40, 150]],
        dtype=np.float64,
    )
    occluder = np.zeros(shape, dtype=bool)
    occluder[10:171, 20:191] = True
    surface = _filled_polygon(board, shape) & ~occluder
    quads = np.full((3, 4, 2), np.nan, dtype=np.float64)
    confidence = np.zeros(3, dtype=np.float32)
    inliers = np.zeros(3, dtype=np.int32)
    status = ["occluded"] * 3

    def forbidden_camera(*unused):
        pytest.fail("There is no canonical anchor to propagate")

    monkeypatch.setattr(geometry, "_camera_motion_pair", forbidden_camera)
    geometry._bridge_occlusions(
        quads,
        confidence,
        inliers,
        status,
        [occluder] * 3,
        [np.full(shape, 8, dtype=np.uint8)] * 3,
        [occluder] * 3,
        [surface] * 3,
    )

    assert np.isnan(quads).all()
    assert np.all(confidence == 0.0)
    assert np.all(inliers == 0)


def test_camera_regions_separate_background_and_visible_surface():
    shape = (120, 180)
    quad = np.array(
        [[50, 30], [130, 30], [130, 90], [50, 90]],
        dtype=np.float64,
    )
    surface = _filled_polygon(quad, shape)
    occluder = np.zeros(shape, dtype=np.uint8)
    occluder[40:81, 80:111] = 1
    surface[occluder != 0] = False
    exclusion = cv2.dilate(
        occluder, np.ones((11, 11), dtype=np.uint8)
    )

    background, visible = geometry._camera_regions(
        quad, shape, exclusion, surface
    )

    assert background.any()
    assert visible.any()
    assert not np.any(background & _filled_polygon(quad, shape))
    assert not np.any(background[exclusion != 0])
    assert not np.any(visible[exclusion != 0])
    assert np.all(surface[visible != 0])
    assert not np.any(background & visible)


@pytest.mark.parametrize("projective", [False, True])
def test_camera_motion_accepts_eight_inliers_with_appropriate_model(projective):
    quad = np.array(
        [[200, 110], [440, 110], [440, 260], [200, 260]],
        dtype=np.float64,
    )
    if projective:
        source = np.array(
            [
                [20, 20], [320, 20], [620, 20], [20, 180],
                [620, 180], [20, 330], [320, 330], [620, 330],
            ],
            dtype=np.float32,
        )
        expected = np.array(
            [[1.002, -0.005, 2.0],
             [0.004, 0.999, -1.5],
             [0.0003, -0.00015, 1.0]],
            dtype=np.float64,
        )
    else:
        source = np.array(
            [
                [20, 20], [30, 20], [40, 20], [50, 20],
                [20, 24], [30, 24], [40, 24], [50, 24],
            ],
            dtype=np.float32,
        )
        expected = np.vstack(
            (
                cv2.getRotationMatrix2D((0, 0), 0.6, 1.001),
                [0.0, 0.0, 1.0],
            )
        )
        expected[:2, 2] += [2.25, -1.5]

    target = cv2.perspectiveTransform(source[None], expected)[0]
    motion, confidence, support = geometry._fit_camera_motion(
        source, target, quad, (360, 640)
    )

    assert motion is not None
    assert support == 8
    assert 0.30 <= confidence <= 0.70
    actual_quad = cv2.perspectiveTransform(quad[None], motion)[0]
    expected_quad = cv2.perspectiveTransform(quad[None], expected)[0]
    np.testing.assert_allclose(actual_quad, expected_quad, atol=0.02, rtol=0)
    if not projective:
        np.testing.assert_array_equal(motion[2], [0.0, 0.0, 1.0])


@pytest.fixture
def bridge_case():
    count = 19
    shape = (180, 260)
    rng = np.random.default_rng(997)
    walk = np.cumsum(rng.normal(0.0, 2.5, size=(count, 2)), axis=0)
    walk -= walk[0]
    base = np.array(
        [[60, 50], [170, 50], [170, 125], [60, 125]],
        dtype=np.float64,
    )
    truth = base[None] + walk[:, None, :]
    quads = np.full_like(truth, np.nan)
    quads[[0, -1]] = truth[[0, -1]]
    confidence = np.zeros(count, dtype=np.float32)
    confidence[[0, -1]] = [1.0, 0.9]
    occluders = [
        _filled_polygon(quad, shape) if 0 < index < count - 1 else None
        for index, quad in enumerate(truth)
    ]
    return {
        "truth": truth,
        "quads": quads,
        "confidence": confidence,
        "inliers": np.zeros(count, dtype=np.int32),
        "status": ["ok"] + ["occluded"] * (count - 2) + ["ok"],
        "occluders": occluders,
        "grays": [
            np.full(shape, index, dtype=np.uint8)
            for index in range(count)
        ],
        "exclusions": occluders,
        "surfaces": [None] * count,
    }


@pytest.mark.parametrize("finite_one_way_track", [False, True])
def test_bidirectional_bridge_cancels_accumulated_drift_and_pins_anchors(
    monkeypatch, bridge_case, finite_one_way_track
):
    case = bridge_case
    truth = case["truth"]
    count = len(truth)
    bias = np.array([0.40, -0.25])
    if finite_one_way_track:
        case["quads"][1:-1] = (
            truth[1:-1]
            + np.arange(1, count - 1)[:, None, None] * bias
        )
        case["confidence"][1:-1] = 0.6

    def biased_camera_pair(first, second, *unused):
        first_index = int(first[0, 0])
        second_index = int(second[0, 0])
        assert abs(second_index - first_index) == 1
        motion = np.eye(3, dtype=np.float64)
        motion[:2, 2] = (
            truth[second_index, 0] - truth[first_index, 0]
            + (second_index - first_index) * bias
        )
        return motion, 0.65, 8

    monkeypatch.setattr(geometry, "_camera_motion_pair", biased_camera_pair)
    original_anchors = case["quads"][[0, -1]].copy()
    geometry._bridge_occlusions(
        case["quads"],
        case["confidence"],
        case["inliers"],
        case["status"],
        case["occluders"],
        case["grays"],
        case["exclusions"],
        case["surfaces"],
        ref_idx=0,
    )

    np.testing.assert_allclose(case["quads"], truth, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(case["quads"][[0, -1]], original_anchors)
    assert case["status"] == ["ok"] + ["occluded"] * (count - 2) + ["ok"]
    assert np.all(case["inliers"][1:-1] == 8)
    assert np.all(case["confidence"][1:-1] >= 0.30 - 1e-6)
    assert np.all(case["confidence"][1:-1] <= 0.70 + 1e-6)


@pytest.mark.parametrize("failed_edges", [{8}, {5, 12}])
def test_bridge_uses_one_surviving_chain_before_interpolating(
    monkeypatch, bridge_case, failed_edges
):
    case = bridge_case
    truth = case["truth"]

    def camera_pair(first, second, *unused):
        first_index = int(first[0, 0])
        second_index = int(second[0, 0])
        assert abs(second_index - first_index) == 1
        if min(first_index, second_index) in failed_edges:
            return None, 0.0, 0
        motion = np.eye(3, dtype=np.float64)
        motion[:2, 2] = truth[second_index, 0] - truth[first_index, 0]
        return motion, 0.65, 8

    monkeypatch.setattr(geometry, "_camera_motion_pair", camera_pair)
    geometry._bridge_occlusions(
        case["quads"],
        case["confidence"],
        case["inliers"],
        case["status"],
        case["occluders"],
        case["grays"],
        case["exclusions"],
        case["surfaces"],
        ref_idx=0,
    )

    expected = truth.copy()
    if len(failed_edges) == 2:
        for index in range(6, 13):
            fraction = (index - 5) / 8.0
            expected[index] = (
                (1.0 - fraction) * truth[5] + fraction * truth[13]
            )
        assert np.all(case["inliers"][6:13] == 0)
        assert np.all(case["confidence"][6:13] <= 0.40 + 1e-6)
    else:
        assert np.all(case["inliers"][1:-1] == 8)

    np.testing.assert_allclose(case["quads"], expected, atol=1e-6, rtol=0)
    assert "lost" not in case["status"]
    assert case["status"][1:-1] == ["occluded"] * (len(truth) - 2)


@pytest.mark.parametrize("missing_length", [1, 3])
def test_occluder_holes_and_motion_margins_are_conservative(missing_length):
    first = np.zeros((80, 240), dtype=bool)
    first[32:48, 20:36] = True
    last = np.zeros_like(first)
    offset = 20 * (missing_length + 1)
    last[32:48, 20 + offset:36 + offset] = True
    first_original, last_original = first.copy(), last.copy()
    first.setflags(write=False)
    last.setflags(write=False)

    masks = [first] + [None] * missing_length + [last]
    repaired, exclusions = geometry._prepare_occluders(masks)
    edge_occluders = geometry._prepare_edge_occluders(masks)
    union = first | last

    for index in range(1, missing_length + 1):
        assert repaired[index] is not None
        assert np.all(repaired[index][union])
        shifted_left = 20 + 20 * index
        assert repaired[index][40, shifted_left + 8]

        # Edge masks describe the current body, not its whole swept path.
        assert edge_occluders[index][40, shifted_left + 8]
        assert not edge_occluders[index][40, 28]
        assert not edge_occluders[index][40, 28 + offset]
        assert np.count_nonzero(edge_occluders[index]) == np.count_nonzero(first)

    _, stationary = geometry._prepare_occluders([first, first])
    assert exclusions[0][40, 60] != 0
    assert stationary[0][40, 60] == 0
    assert exclusions[0][40, 70] == 0

    np.testing.assert_array_equal(first, first_original)
    np.testing.assert_array_equal(last, last_original)


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
    monkeypatch.setattr(
        geometry, "_camera_motion_pair", lambda *args: (None, 0.0, 0)
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


@pytest.mark.parametrize("ref_idx", [0, 23])
def test_long_occlusion_is_bridged_after_feature_failure(
    monkeypatch, ref_idx
):
    count = 24
    frames, masks, truth = _small_clip(count)
    monkeypatch.setattr(
        geometry, "_track_pair", lambda *args: (None, 0.0, 0)
    )
    monkeypatch.setattr(
        geometry, "_camera_motion_pair", lambda *args: (None, 0.0, 0)
    )
    surfaces = [
        masks[index] if index in {0, count - 1}
        else np.zeros(frames[0].shape[:2], dtype=bool)
        for index in range(count)
    ]
    occluders = [
        masks[index] if 0 < index < count - 1 else None
        for index in range(count)
    ]
    for index in (8, 13, 14, 15):
        occluders[index] = None

    result = propagate_quad(
        frames,
        ref_idx,
        truth[ref_idx],
        surface_masks=surfaces,
        occluder_masks=occluders,
    )

    assert result["status"] == ["ok"] + ["occluded"] * 22 + ["ok"]
    assert np.isfinite(result["quads"]).all()
    assert np.all(result["inliers"] == 0)
    assert np.all(result["confidence"][1:-1] > 0.0)
    assert np.all(result["confidence"][1:-1] <= 0.5)
    errors = np.linalg.norm(result["quads"] - truth, axis=2)
    assert float(errors.max()) <= 2.0
    np.testing.assert_array_equal(result["quads"][ref_idx], truth[ref_idx])


@pytest.mark.parametrize(
    "endpoint_confidence,broken_evidence,should_fill",
    [(0.8, False, True), (0.49, False, False), (0.8, True, False)],
)
def test_long_bridge_requires_confident_ends_and_only_occlusion(
    endpoint_confidence, broken_evidence, should_fill
):
    count = 18
    base = np.array(
        [[40, 30], [140, 30], [140, 100], [40, 100]],
        dtype=np.float64,
    )
    truth = base[None] + np.arange(count)[:, None, None] * np.array(
        [1.0, 0.5]
    )
    quads = np.full((count, 4, 2), np.nan, dtype=np.float64)
    quads[[0, -1]] = truth[[0, -1]]
    confidence = np.zeros(count, dtype=np.float32)
    confidence[0] = 1.0
    confidence[-1] = endpoint_confidence
    status = ["ok"] + ["occluded"] * (count - 2) + ["ok"]
    occluders = [
        _filled_polygon(quad, (160, 220))
        if 0 < index < count - 1 else None
        for index, quad in enumerate(truth)
    ]
    if broken_evidence:
        occluders[8] = np.zeros((160, 220), dtype=bool)
        occluders[8][:10, :10] = True
        status[8] = "lost"

    geometry._fill_short_gaps(quads, confidence, status, occluders)

    if should_fill:
        np.testing.assert_allclose(quads, truth, atol=1e-6, rtol=0)
        assert status[1:-1] == ["occluded"] * (count - 2)
        assert np.all(confidence[1:-1] > 0.0)
    else:
        assert np.isnan(quads[1:-1]).all()
        assert np.all(confidence[1:-1] == 0.0)


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
def test_partial_visible_masks_update_prior_but_cannot_reseed_corners(
    monkeypatch, kind
):
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

    assert geometry._mask_seed(surface, occluders[1]) is None
    monkeypatch.setattr(
        geometry, "_track_pair", lambda *args: (None, 0.0, 0)
    )
    monkeypatch.setattr(
        geometry, "_camera_motion_pair", lambda *args: (None, 0.0, 0)
    )
    result = propagate_quad(
        frames,
        0,
        reference,
        surface_masks=[None, surface],
        occluder_masks=occluders,
    )

    np.testing.assert_array_equal(result["quads"][0], reference)
    if kind == "clipped":
        assert result["status"] == ["ok", "lost"]
        assert np.isnan(result["quads"][1]).all()
        assert result["confidence"][1] == 0.0
    else:
        # Existing canonical geometry supplies the hidden left edge.
        assert result["status"] == ["ok", "occluded"]
        assert np.isfinite(result["quads"][1]).all()
        assert np.linalg.norm(result["quads"][1] - reference, axis=1).max() <= 2.0
        assert 0.30 - 1e-6 <= result["confidence"][1] <= 0.70 + 1e-6
        assert result["inliers"][1] >= 8


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
