"""Canonical quadrilateral extraction, planar tracking, and temporal smoothing."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import savgol_filter


def _signed_area(quad):
    q = np.asarray(quad, dtype=np.float64)
    following = np.roll(q, -1, axis=0)
    return 0.5 * float(
        np.sum(q[:, 0] * following[:, 1] - q[:, 1] * following[:, 0])
    )


def _valid_quad(quad, min_area=64.0):
    q = np.asarray(quad, dtype=np.float64)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        return False

    edges = np.roll(q, -1, axis=0) - q
    lengths = np.linalg.norm(edges, axis=1)
    if np.min(lengths) < 3.0 or _signed_area(q) < min_area:
        return False

    following = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    sines = crosses / (lengths * np.roll(lengths, -1))
    if np.any(sines <= 0.04):
        return False

    aspect = (lengths[0] + lengths[2]) / (lengths[1] + lengths[3])
    return bool(1.0 / 20.0 <= aspect <= 20.0)


def _order_quad(points):
    q = np.asarray(points, dtype=np.float64).reshape(4, 2)
    center = q.mean(axis=0)
    angles = np.arctan2(q[:, 1] - center[1], q[:, 0] - center[0])
    q = q[np.argsort(angles)]
    if _signed_area(q) < 0:
        q = q[::-1]
    return np.roll(q, -int(np.argmin(q.sum(axis=1))), axis=0)


def _polygon_mask(quad, shape):
    result = np.zeros(shape, dtype=np.uint8)
    coordinates = np.rint(np.asarray(quad) * 16.0).astype(np.int32)
    cv2.fillConvexPoly(result, coordinates, 1, lineType=cv2.LINE_8, shift=4)
    return result


def _fit_boundary_lines(contour, seed):
    points = contour.reshape(-1, 2).astype(np.float64)
    edges = np.roll(seed, -1, axis=0) - seed
    lengths = np.linalg.norm(edges, axis=1)
    if np.min(lengths) < 3.0:
        return None

    offsets = points[:, None, :] - seed[None, :, :]
    projections = np.sum(offsets * edges[None, :, :], axis=2) / lengths**2
    closest = (
        seed[None, :, :]
        + np.clip(projections, 0.0, 1.0)[:, :, None] * edges[None, :, :]
    )
    distances = np.linalg.norm(points[:, None, :] - closest, axis=2)
    owners = distances.argmin(axis=1)
    tolerance = max(3.0, 0.05 * float(np.min(lengths)))

    lines = []
    for edge_index in range(4):
        selected = (
            (owners == edge_index)
            & (projections[:, edge_index] >= 0.10)
            & (projections[:, edge_index] <= 0.90)
            & (distances[:, edge_index] <= tolerance)
        )
        samples = points[selected]
        if len(samples) < 8:
            return None

        direction = edges[edge_index] / lengths[edge_index]
        if np.ptp(samples @ direction) < 0.45 * lengths[edge_index]:
            return None

        vx, vy, x0, y0 = cv2.fitLine(
            samples.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
        ).ravel()
        normal = np.array([-vy, vx], dtype=np.float64)
        lines.append((normal, float(normal @ np.array([x0, y0]))))

    corners = []
    for corner_index in range(4):
        previous = lines[corner_index - 1]
        current = lines[corner_index]
        matrix = np.stack((previous[0], current[0]))
        if abs(float(np.linalg.det(matrix))) < 0.08:
            return None
        corners.append(
            np.linalg.solve(matrix, np.array([previous[1], current[1]]))
        )

    return np.asarray(corners, dtype=np.float64)


def quad_from_mask(mask) -> np.ndarray | None:
    """Fit intersecting boundary lines to one sufficiently rectangular component."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must have shape (H, W)")
    if array.size == 0:
        return None

    binary = array.astype(bool).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None

    areas = np.array([cv2.contourArea(contour) for contour in contours])
    largest = int(np.argmax(areas))
    area = float(areas[largest])
    if area < 64.0:
        return None

    significant_area = float(areas[areas >= max(16.0, 0.02 * area)].sum())
    if area / significant_area < 0.90:
        return None

    contour = contours[largest]
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0.0 or area / hull_area < 0.90:
        return None

    filled = np.zeros_like(clean)
    cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED)
    filled_area = int(np.count_nonzero(filled))
    if np.count_nonzero(clean & filled) / filled_area < 0.90:
        return None

    perimeter = cv2.arcLength(hull, True)
    boundary = contour.reshape(-1, 2).astype(np.float64)
    best_quad = None
    best_overlap = 0.0

    for fraction in (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
        approximation = cv2.approxPolyDP(hull, fraction * perimeter, True)
        if len(approximation) != 4:
            continue

        seed = _order_quad(approximation)
        fitted = _fit_boundary_lines(contour, seed)
        if fitted is None or not _valid_quad(fitted):
            continue

        edges = np.roll(fitted, -1, axis=0) - fitted
        lengths = np.linalg.norm(edges, axis=1)
        offsets = boundary[:, None, :] - fitted[None, :, :]
        line_distances = np.abs(
            edges[None, :, 0] * offsets[:, :, 1]
            - edges[None, :, 1] * offsets[:, :, 0]
        ) / lengths[None, :]
        residual = float(np.percentile(line_distances.min(axis=1), 90))
        if residual > max(2.0, 0.012 * float(lengths.min())):
            continue

        fitted_mask = _polygon_mask(fitted, clean.shape)
        intersection = int(np.count_nonzero(fitted_mask & filled))
        union = int(np.count_nonzero(fitted_mask | filled))
        overlap = intersection / max(1, union)
        if overlap >= 0.91 and overlap > best_overlap:
            best_overlap = overlap
            best_quad = fitted

    if best_quad is None:
        return None
    return _order_quad(best_quad).astype(np.float32)


def _prepare_masks(masks, count, original_shape, tracking_size):
    if masks is None:
        return [None] * count

    values = list(masks)
    if len(values) != count:
        raise ValueError("mask sequences must have one entry per frame")

    prepared = []
    original_size = (original_shape[1], original_shape[0])
    for mask in values:
        if mask is None:
            prepared.append(None)
            continue
        array = np.asarray(mask)
        if array.shape != original_shape:
            raise ValueError("each mask must match the frame dimensions")
        binary = array.astype(bool)
        if tracking_size != original_size:
            binary = cv2.resize(
                binary.astype(np.uint8),
                tracking_size,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        prepared.append(binary)
    return prepared


def _tracking_region(quad, shape):
    region = _polygon_mask(quad, shape)
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    band = int(np.clip(0.28 * float(lengths.min()), 14, 96))
    kernel = np.ones((2 * band + 1, 2 * band + 1), dtype=np.uint8)
    return cv2.dilate(region, kernel)


def _track_pair(first_gray, second_gray, quad, first_exclusion, second_exclusion):
    region = _tracking_region(quad, first_gray.shape)
    if first_exclusion is not None:
        region[first_exclusion != 0] = 0

    detected = cv2.goodFeaturesToTrack(
        first_gray,
        maxCorners=600,
        qualityLevel=0.01,
        minDistance=6.0,
        mask=region,
        blockSize=7,
    )
    if detected is None or len(detected) < 12:
        return None, 0.0, 0

    flow_options = {
        "winSize": (25, 25),
        "maxLevel": 3,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }
    try:
        forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            first_gray, second_gray, detected, None, **flow_options
        )
    except cv2.error:
        return None, 0.0, 0

    if forward is None or forward_status is None:
        return None, 0.0, 0

    source = detected.reshape(-1, 2)
    target = forward.reshape(-1, 2)
    height, width = first_gray.shape
    keep = (
        forward_status.ravel().astype(bool)
        & np.isfinite(target).all(axis=1)
        & (target[:, 0] >= 1.0)
        & (target[:, 0] < width - 1.0)
        & (target[:, 1] >= 1.0)
        & (target[:, 1] < height - 1.0)
    )
    source, target = source[keep], target[keep]
    if second_exclusion is not None and len(target):
        pixels = np.rint(target).astype(np.int32)
        keep = second_exclusion[pixels[:, 1], pixels[:, 0]] == 0
        source, target = source[keep], target[keep]
    if len(source) < 12:
        return None, 0.0, 0

    try:
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            second_gray,
            first_gray,
            target.reshape(-1, 1, 2),
            None,
            **flow_options,
        )
    except cv2.error:
        return None, 0.0, 0

    if backward is None or backward_status is None:
        return None, 0.0, 0
    backward = backward.reshape(-1, 2)
    keep = (
        backward_status.ravel().astype(bool)
        & np.isfinite(backward).all(axis=1)
        & (np.linalg.norm(backward - source, axis=1) <= 1.0)
    )
    source, target = source[keep], target[keep]
    if len(source) < 12:
        return None, 0.0, 0

    try:
        homography, consensus = cv2.findHomography(
            source,
            target,
            cv2.RANSAC,
            1.6,
            maxIters=2000,
            confidence=0.995,
        )
        if homography is None or consensus is None:
            return None, 0.0, 0

        accepted = consensus.ravel().astype(bool)
        inlier_count = int(accepted.sum())
        ratio = inlier_count / len(source)
        if inlier_count < 12 or ratio < 0.50:
            return None, 0.0, inlier_count

        support_area = float(
            cv2.contourArea(cv2.convexHull(source[accepted]))
        )
        if support_area < max(64.0, 0.02 * np.count_nonzero(region)):
            return None, 0.0, inlier_count

        homography, _ = cv2.findHomography(source[accepted], target[accepted], 0)
        if homography is None or not np.isfinite(homography).all():
            return None, 0.0, inlier_count

        projected = cv2.perspectiveTransform(
            source[accepted].reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
    except cv2.error:
        return None, 0.0, 0

    residual = float(
        np.median(np.linalg.norm(projected - target[accepted], axis=1))
    )
    if not np.isfinite(residual) or residual > 1.25:
        return None, 0.0, inlier_count

    homogeneous = np.column_stack((quad, np.ones(4))) @ homography.T
    denominators = homogeneous[:, 2]
    if not (
        np.all(denominators > 1e-6) or np.all(denominators < -1e-6)
    ):
        return None, 0.0, inlier_count

    candidate = homogeneous[:, :2] / denominators[:, None]
    if (
        not _valid_quad(candidate)
        or np.max(np.abs(candidate)) > 8.0 * max(height, width)
    ):
        return None, 0.0, inlier_count

    area_ratio = _signed_area(candidate) / _signed_area(quad)
    old_lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    new_lengths = np.linalg.norm(
        np.roll(candidate, -1, axis=0) - candidate, axis=1
    )
    length_ratios = new_lengths / old_lengths
    displacement = float(np.linalg.norm(candidate - quad, axis=1).max())
    if (
        not 0.60 <= area_ratio <= 1.70
        or np.any(length_ratios < 0.65)
        or np.any(length_ratios > 1.55)
        or displacement > 0.45 * np.hypot(width, height)
    ):
        return None, 0.0, inlier_count

    confidence = min(1.0, inlier_count / 80.0) * ratio * np.exp(-residual / 2.0)
    if confidence < 0.12:
        return None, 0.0, inlier_count
    return candidate, float(np.clip(confidence, 0.0, 1.0)), inlier_count


def _assess_frame(quad, shape, surface_mask, occluder_mask):
    expected = _polygon_mask(quad, shape)
    occluded = False
    usable = np.ones(shape, dtype=bool)

    if occluder_mask is not None:
        occluded = bool(np.any((expected != 0) & occluder_mask))
        ignored = cv2.dilate(
            occluder_mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)
        )
        usable = ignored == 0

    if surface_mask is None:
        return 1.0, occluded

    visible_expected = (expected != 0) & usable
    expected_count = int(np.count_nonzero(visible_expected))
    # Invisible or nearly invisible surfaces provide no negative mask evidence.
    if expected_count < 32:
        return 1.0, occluded

    observed = surface_mask & usable
    observed_count = int(np.count_nonzero(observed))
    if observed_count == 0:
        return 0.0, occluded

    tolerance = np.ones((5, 5), dtype=np.uint8)
    expanded_observed = cv2.dilate(observed.astype(np.uint8), tolerance) != 0
    expanded_expected = cv2.dilate(expected, tolerance) != 0
    coverage = np.count_nonzero(visible_expected & expanded_observed)
    precision = np.count_nonzero(observed & expanded_expected)
    agreement = min(coverage / expected_count, precision / observed_count)
    return float(agreement), occluded


def propagate_quad(
    frames, ref_idx, ref_quad, surface_masks=None, occluder_masks=None
) -> dict:
    """Track in both directions; masks validate geometry without fitting it."""
    frames = list(frames)
    count = len(frames)
    if (
        not isinstance(ref_idx, (int, np.integer))
        or not 0 <= ref_idx < count
    ):
        raise ValueError("ref_idx must index a nonempty frame sequence")

    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] != 3 or min(first.shape[:2]) == 0:
        raise ValueError("frames must have shape (H, W, 3)")
    frame_shape = first.shape
    for frame in frames:
        if (
            not isinstance(frame, np.ndarray)
            or frame.shape != frame_shape
            or frame.dtype != np.uint8
        ):
            raise ValueError("frames must be equally sized BGR uint8 arrays")

    reference = np.asarray(ref_quad, dtype=np.float32)
    if not _valid_quad(reference):
        raise ValueError("ref_quad must be a finite clockwise convex quad")

    height, width = frame_shape[:2]
    factor = min(1.0, 960.0 / max(height, width))
    tracking_size = (
        max(1, int(round(width * factor))),
        max(1, int(round(height * factor))),
    )
    scale = np.array(
        [tracking_size[0] / width, tracking_size[1] / height],
        dtype=np.float64,
    )

    grays = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if tracking_size != (width, height):
            gray = cv2.resize(gray, tracking_size, interpolation=cv2.INTER_AREA)
        grays.append(gray)

    surfaces = _prepare_masks(
        surface_masks, count, (height, width), tracking_size
    )
    occluders = _prepare_masks(
        occluder_masks, count, (height, width), tracking_size
    )
    exclusion_kernel = np.ones((21, 21), dtype=np.uint8)
    exclusions = [
        None
        if mask is None
        else cv2.dilate(mask.astype(np.uint8), exclusion_kernel)
        for mask in occluders
    ]

    quads = np.full((count, 4, 2), np.nan, dtype=np.float32)
    confidence = np.zeros(count, dtype=np.float32)
    inliers = np.zeros(count, dtype=np.int32)
    status = ["lost"] * count

    # OpenCV resizing maps pixel centers, rather than pixel-array edges.
    scaled_reference = (reference.astype(np.float64) + 0.5) * scale - 0.5

    def save(index, tracked_quad, quality):
        agreement, occluded = _assess_frame(
            tracked_quad,
            grays[index].shape,
            surfaces[index],
            occluders[index],
        )
        if agreement < 0.60:
            return
        quads[index] = ((tracked_quad + 0.5) / scale - 0.5).astype(np.float32)
        confidence[index] = np.clip(quality * agreement, 0.0, 1.0)
        status[index] = "occluded" if occluded else "ok"

    save(ref_idx, scaled_reference, 1.0)
    for direction in (-1, 1):
        previous = ref_idx
        current_quad = scaled_reference.copy()
        stop = count if direction == 1 else -1
        for index in range(ref_idx + direction, stop, direction):
            candidate, quality, support = _track_pair(
                grays[previous],
                grays[index],
                current_quad,
                exclusions[previous],
                exclusions[index],
            )
            inliers[index] = support
            if candidate is None:
                break

            save(index, candidate, quality)
            # Reliable image motion can bridge a transient segmentation error.
            current_quad = candidate
            previous = index

    return {
        "quads": quads,
        "confidence": confidence,
        "inliers": inliers,
        "status": status,
    }


def smooth_quads(quads, strength=0.5) -> np.ndarray:
    """Blend zero-phase local polynomial fits within finite runs only."""
    array = np.asarray(quads, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (4, 2):
        raise ValueError("quads must have shape (N, 4, 2)")
    strength = float(strength)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be finite and between 0 and 1")

    result = array.copy()
    valid = np.isfinite(array).all(axis=(1, 2))
    result[~valid] = np.nan
    if strength == 0.0 or len(array) == 0:
        return result

    boundaries = np.diff(np.r_[False, valid, False].astype(np.int8))
    starts = np.flatnonzero(boundaries == 1)
    stops = np.flatnonzero(boundaries == -1)
    for start, stop in zip(starts, stops):
        length = int(stop - start)
        if length < 5:
            continue

        window = min(7, length if length % 2 else length - 1)
        run = array[start:stop].astype(np.float64)
        filtered = savgol_filter(
            run, window_length=window, polyorder=2, axis=0, mode="interp"
        )
        candidate = run + strength * (filtered - run)
        for offset, quad in enumerate(candidate):
            if _valid_quad(quad, min_area=1.0):
                result[start + offset] = quad.astype(np.float32)

    return result
