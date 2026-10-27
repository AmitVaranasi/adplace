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
    q = q[np.argsort(angles, kind="stable")]
    if _signed_area(q) < 0:
        q = q[::-1]
    return np.roll(q, -int(np.argmin(q.sum(axis=1))), axis=0)


def _match_corners(quad, previous):
    """Choose the nearest cyclic ordering without changing winding."""
    q = np.asarray(quad, dtype=np.float64)
    choices = np.stack([np.roll(q, shift, axis=0) for shift in range(4)])
    costs = np.sum((choices - previous) ** 2, axis=(1, 2))
    return choices[int(np.argmin(costs))].copy()


def _quads_agree(first, second):
    scale = np.sqrt(min(_signed_area(first), _signed_area(second)))
    tolerance = max(4.0, 0.035 * scale)
    return bool(np.max(np.linalg.norm(first - second, axis=1)) <= tolerance)


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


def _prepare_occluders(occluders):
    """Repair short detection holes and exclude rim, patch, and motion margins."""
    masks = [
        None if mask is None or not np.any(mask) else mask.copy()
        for mask in occluders
    ]
    observed = [index for index, mask in enumerate(masks) if mask is not None]
    centers = {}
    for index in observed:
        moments = cv2.moments(masks[index].astype(np.uint8), binaryImage=True)
        centers[index] = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float64,
        )

    motion = np.zeros(len(masks), dtype=np.float64)
    for left, right in zip(observed, observed[1:]):
        distance = right - left
        # Do not connect unrelated appearances across long absences.
        if distance > 4:
            continue

        displacement = centers[right] - centers[left]
        speed = float(np.linalg.norm(displacement)) / distance
        motion[left:right + 1] = np.maximum(
            motion[left:right + 1], speed
        )

        if distance == 1:
            continue

        first_mask = masks[left]
        last_mask = masks[right]
        height, width = first_mask.shape
        for index in range(left + 1, right):
            fraction = (index - left) / distance
            shift = fraction * displacement
            transform = np.array(
                [[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]],
                dtype=np.float32,
            )
            shifted = cv2.warpAffine(
                first_mask.astype(np.uint8),
                transform,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(bool)
            # The union handles uncertain shape/velocity; the shifted mask
            # also covers intermediate positions during a three-frame hole.
            masks[index] = first_mask | last_mask | shifted

    exclusions = []
    for index, mask in enumerate(masks):
        if mask is None:
            exclusions.append(None)
            continue
        # All distances are in tracking pixels. Protect the 25-pixel LK
        # window as well as a faint rim and a speed-dependent blurred edge.
        radius = int(np.ceil(4.0 + 12.0 + 0.75 * motion[index]))
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        exclusions.append(cv2.dilate(mask.astype(np.uint8), kernel))

    return masks, exclusions


def _mask_seed(surface, occluder):
    """Only complete, unoccluded masks may establish new canonical corners."""
    if surface is None or not surface.any():
        return None
    if (
        surface[:2].any()
        or surface[-2:].any()
        or surface[:, :2].any()
        or surface[:, -2:].any()
    ):
        return None

    quad = quad_from_mask(surface)
    if quad is None:
        return None

    if occluder is not None:
        ignored = cv2.dilate(
            occluder.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)
        )
        if np.any((_polygon_mask(quad, surface.shape) != 0) & (ignored != 0)):
            return None
    return quad.astype(np.float64)


def _tracking_region(quad, shape, expansion=1.0):
    region = _polygon_mask(quad, shape)
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    band = int(round(expansion * np.clip(0.45 * lengths.min(), 24, 128)))
    kernel = np.ones((2 * band + 1, 2 * band + 1), dtype=np.uint8)
    return cv2.dilate(region, kernel)


def _flow_matches(
    first,
    second,
    points,
    second_exclusion,
    initial=None,
    minimum=12,
    min_eigenvalue=1e-4,
):
    empty = np.empty((0, 2), dtype=np.float32)
    if points is None or len(points) == 0:
        return empty, empty

    source = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    options = {
        "winSize": (25, 25),
        "maxLevel": 3,
        "minEigThreshold": min_eigenvalue,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }
    guess = None
    flags = 0
    if initial is not None:
        guess = np.asarray(initial, dtype=np.float32).reshape(-1, 1, 2).copy()
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW

    try:
        forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            first, second, source, guess, flags=flags, **options
        )
    except cv2.error:
        return empty, empty
    if forward is None or forward_status is None:
        return empty, empty

    source = source.reshape(-1, 2)
    target = forward.reshape(-1, 2)
    height, width = first.shape
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
    if len(source) < minimum:
        return empty, empty

    try:
        # A good forward match supplies the reverse initialization.
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            second,
            first,
            target.reshape(-1, 1, 2),
            source.reshape(-1, 1, 2).copy(),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            **options,
        )
    except cv2.error:
        return empty, empty
    if backward is None or backward_status is None:
        return empty, empty

    backward = backward.reshape(-1, 2)
    keep = (
        backward_status.ravel().astype(bool)
        & np.isfinite(backward).all(axis=1)
        & (np.linalg.norm(backward - source, axis=1) <= 1.0)
    )
    return source[keep], target[keep]


def _motion_quad(quad, homography, shape):
    """Apply one adjacent-frame motion, rejecting implausible extrapolation."""
    if homography is None or not np.isfinite(homography).all():
        return None

    homogeneous = np.column_stack((quad, np.ones(4))) @ homography.T
    denominators = homogeneous[:, 2]
    if not (
        np.all(denominators > 1e-6) or np.all(denominators < -1e-6)
    ):
        return None

    candidate = homogeneous[:, :2] / denominators[:, None]
    height, width = shape
    if (
        not _valid_quad(candidate)
        or np.max(np.abs(candidate)) > 8.0 * max(height, width)
    ):
        return None

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
        return None
    return candidate


def _fit_motion(source, target, quad, region, camera=False):
    if len(source) < 12:
        return None, 0.0, 0

    # Canonical correspondence order makes RANSAC independent of enumeration.
    source = np.asarray(source, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float32).reshape(-1, 2)
    order = np.lexsort(
        (target[:, 1], target[:, 0], source[:, 1], source[:, 0])
    )
    source = np.ascontiguousarray(source[order])
    target = np.ascontiguousarray(target[order])

    inlier_count = 0
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
        minimum_inliers = 24 if camera else 12
        minimum_ratio = 0.60 if camera else 0.50
        if inlier_count < minimum_inliers or ratio < minimum_ratio:
            return None, 0.0, inlier_count

        support_area = float(cv2.contourArea(cv2.convexHull(source[accepted])))
        if camera:
            height, width = region.shape
            # Keep online propagation conservative. The separate occlusion
            # bridge can use sparse similarity fits pinned to both anchors.
            span = np.ptp(source[accepted], axis=0)
            if (
                support_area < max(64.0, 0.08 * region.size)
                or np.any(span < 0.25 * np.array([width, height]))
            ):
                return None, 0.0, inlier_count
        elif support_area < max(64.0, 0.02 * np.count_nonzero(region)):
            return None, 0.0, inlier_count

        homography, _ = cv2.findHomography(source[accepted], target[accepted], 0)
        if homography is None or not np.isfinite(homography).all():
            return None, 0.0, inlier_count

        projected = cv2.perspectiveTransform(
            source[accepted].reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
    except cv2.error:
        return None, 0.0, inlier_count

    residual = float(
        np.median(np.linalg.norm(projected - target[accepted], axis=1))
    )
    if not np.isfinite(residual) or residual > 1.25:
        return None, 0.0, inlier_count

    candidate = _motion_quad(quad, homography, region.shape)
    if candidate is None:
        return None, 0.0, inlier_count

    confidence = min(1.0, inlier_count / 80.0) * ratio * np.exp(-residual / 2.0)
    minimum_confidence = 0.30 if camera else 0.12
    if confidence < minimum_confidence:
        return None, 0.0, inlier_count
    return candidate, float(np.clip(confidence, 0.0, 1.0)), inlier_count


def _ordered_features(keypoints, descriptors):
    """Canonicalize ORB features before matching, including descriptor ties."""
    order = sorted(
        range(len(keypoints)),
        key=lambda index: (
            keypoints[index].pt[1],
            keypoints[index].pt[0],
            keypoints[index].octave,
            keypoints[index].size,
            keypoints[index].angle,
            -keypoints[index].response,
            descriptors[index].tobytes(),
        ),
    )
    return [keypoints[index] for index in order], descriptors[order]


def _descriptor_matches(
    first, second, first_region, second_region, minimum=12, fast_threshold=7
):
    empty = np.empty((0, 2), dtype=np.float32)
    try:
        detector = cv2.ORB_create(
            nfeatures=1800,
            edgeThreshold=15,
            fastThreshold=fast_threshold,
        )
        first_keys, first_descriptors = detector.detectAndCompute(
            first, first_region
        )
        second_keys, second_descriptors = detector.detectAndCompute(
            second, second_region
        )
        if (
            first_descriptors is None
            or second_descriptors is None
            or len(first_descriptors) < minimum
            or len(second_descriptors) < minimum
        ):
            return empty, empty

        first_keys, first_descriptors = _ordered_features(
            first_keys, first_descriptors
        )
        second_keys, second_descriptors = _ordered_features(
            second_keys, second_descriptors
        )
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(first_descriptors, second_descriptors, k=2)
        reverse = {
            match.queryIdx: match.trainIdx
            for match in matcher.match(second_descriptors, first_descriptors)
        }
        matches = [
            pair[0]
            for pair in pairs
            if len(pair) == 2
            and pair[0].distance < 0.80 * pair[1].distance
            and reverse.get(pair[0].trainIdx) == pair[0].queryIdx
        ]
        matches.sort(
            key=lambda match: (match.queryIdx, match.trainIdx, match.distance)
        )
    except cv2.error:
        return empty, empty

    source = np.asarray(
        [first_keys[match.queryIdx].pt for match in matches], dtype=np.float32
    ).reshape(-1, 2)
    target = np.asarray(
        [second_keys[match.trainIdx].pt for match in matches], dtype=np.float32
    ).reshape(-1, 2)
    return source, target


def _track_pair(first_gray, second_gray, quad, first_exclusion, second_exclusion):
    # Online propagation remains conservative during an occluder passage.
    camera = first_exclusion is not None or second_exclusion is not None
    region = (
        np.ones(first_gray.shape, dtype=np.uint8)
        if camera
        else _tracking_region(quad, first_gray.shape)
    )
    if first_exclusion is not None:
        region[first_exclusion != 0] = 0

    try:
        detected = cv2.goodFeaturesToTrack(
            first_gray,
            maxCorners=800,
            qualityLevel=0.01,
            minDistance=5.0,
            mask=region,
            blockSize=5,
        )
    except cv2.error:
        detected = None

    if detected is not None:
        points = detected.reshape(-1, 2)
        order = np.lexsort((points[:, 0], points[:, 1]))
        detected = points[order].reshape(-1, 1, 2)

    source, target = _flow_matches(
        first_gray, second_gray, detected, second_exclusion
    )
    flow_result = _fit_motion(source, target, quad, region, camera=camera)
    if flow_result[0] is not None and flow_result[1] >= 0.30:
        return flow_result

    second_region = (
        np.ones(second_gray.shape, dtype=np.uint8)
        if camera
        else _tracking_region(quad, second_gray.shape, expansion=1.8)
    )
    if second_exclusion is not None:
        second_region[second_exclusion != 0] = 0
    source, target = _descriptor_matches(
        first_gray, second_gray, region, second_region
    )
    descriptor_result = _fit_motion(
        source, target, quad, region, camera=camera
    )
    refined_source, refined_target = _flow_matches(
        first_gray, second_gray, source, second_exclusion, initial=target
    )
    refined_result = _fit_motion(
        refined_source, refined_target, quad, region, camera=camera
    )

    results = (flow_result, descriptor_result, refined_result)
    valid = [result for result in results if result[0] is not None]
    if valid:
        return max(valid, key=lambda result: result[1])
    return None, 0.0, max(result[2] for result in results)


def _camera_regions(quad, shape, exclusion, surface=None):
    """Separate static background and visible surface, with patch margins."""
    polygon = _polygon_mask(quad, shape)
    full_surface = polygon.copy()
    if surface is not None:
        full_surface |= surface.astype(np.uint8)

    # Include uncertainty in the interpolated guide and ORB's patch radius.
    outside = cv2.dilate(
        full_surface, np.ones((33, 33), dtype=np.uint8)
    )
    background = (outside == 0).astype(np.uint8)

    observed = polygon if surface is None else polygon & surface.astype(np.uint8)
    visible = cv2.erode(
        observed, np.ones((25, 25), dtype=np.uint8)
    )
    if exclusion is not None:
        background[exclusion != 0] = 0
        visible[exclusion != 0] = 0
    return background, visible


def _fit_camera_motion(source, target, quad, shape):
    """Fit sparse camera motion; prefer similarity unless perspective earns it."""
    if len(source) < 8:
        return None, 0.0, 0

    source = np.asarray(source, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float32).reshape(-1, 2)
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source, target = source[finite], target[finite]
    if len(source) < 8:
        return None, 0.0, 0

    order = np.lexsort(
        (target[:, 1], target[:, 0], source[:, 1], source[:, 0])
    )
    source = np.ascontiguousarray(source[order])
    target = np.ascontiguousarray(target[order])
    maximum_support = 0

    def assess(matrix, consensus, projective):
        nonlocal maximum_support
        if matrix is None or consensus is None:
            return None

        accepted = consensus.ravel().astype(bool)
        support = int(accepted.sum())
        maximum_support = max(maximum_support, support)
        ratio = support / len(source)
        if support < 8 or ratio < 0.50:
            return None

        first = source[accepted].astype(np.float64)
        second = target[accepted].astype(np.float64)
        centered = first - first.mean(axis=0)
        spread = float(np.mean(np.sum(centered**2, axis=1)))
        if spread < 36.0 or np.linalg.norm(np.ptp(first, axis=0)) < 16.0:
            return None

        try:
            if projective:
                # A homography needs two-dimensional support. A small or
                # nearly collinear lamp cluster still permits similarity.
                singular = np.linalg.svd(centered, compute_uv=False)
                area = float(
                    cv2.contourArea(cv2.convexHull(first.astype(np.float32)))
                )
                leverage = float(
                    np.linalg.norm(quad - first.mean(axis=0), axis=1).max()
                    / np.sqrt(spread)
                )
                if (
                    singular[-1] < 0.08 * singular[0]
                    or area < max(64.0, 0.04 * _signed_area(quad))
                    or leverage > 4.0
                ):
                    return None
                matrix, _ = cv2.findHomography(
                    first.astype(np.float32), second.astype(np.float32), 0
                )
            else:
                # Deterministic least-squares similarity refinement.
                first_center = first.mean(axis=0)
                second_center = second.mean(axis=0)
                destination = second - second_center
                denominator = float(np.sum(centered**2))
                a = float(np.sum(centered * destination) / denominator)
                b = float(
                    np.sum(
                        centered[:, 0] * destination[:, 1]
                        - centered[:, 1] * destination[:, 0]
                    ) / denominator
                )
                linear = np.array([[a, -b], [b, a]], dtype=np.float64)
                translation = second_center - linear @ first_center
                matrix = np.eye(3, dtype=np.float64)
                matrix[:2, :2] = linear
                matrix[:2, 2] = translation

            if matrix is None or _motion_quad(quad, matrix, shape) is None:
                return None
            projected = cv2.perspectiveTransform(
                first.reshape(-1, 1, 2), matrix
            ).reshape(-1, 2)
        except (cv2.error, np.linalg.LinAlgError):
            return None

        errors = np.linalg.norm(projected - second, axis=1)
        residual = float(np.median(errors))
        if (
            not np.isfinite(errors).all()
            or residual > 1.25
            or float(np.percentile(errors, 90)) > 2.2
        ):
            return None

        confidence = 0.30 + (
            0.40
            * min(1.0, support / 48.0)
            * ratio
            * np.exp(-residual / 1.5)
        )
        return matrix, float(np.clip(confidence, 0.30, 0.70)), support, residual

    similarity = None
    projective = None
    try:
        affine, consensus = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=1.6,
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )
        similarity = assess(affine, consensus, False)
    except cv2.error:
        pass

    try:
        homography, consensus = cv2.findHomography(
            source,
            target,
            cv2.RANSAC,
            1.6,
            maxIters=2000,
            confidence=0.995,
        )
        projective = assess(homography, consensus, True)
    except cv2.error:
        pass

    best = similarity
    if projective is not None:
        if similarity is None:
            best = projective
        else:
            better_residual = (
                projective[2] >= 0.90 * similarity[2]
                and projective[3] < 0.70 * similarity[3]
                and projective[3] + 0.08 < similarity[3]
            )
            better_support = (
                projective[2] >= 1.25 * similarity[2]
                and projective[3] <= similarity[3] + 0.10
            )
            if better_residual or better_support:
                best = projective

    if best is None:
        return None, 0.0, maximum_support
    return best[:3]


def _camera_motion_pair(
    first_gray,
    second_gray,
    first_quad,
    second_quad,
    first_exclusion,
    second_exclusion,
    first_surface=None,
    second_surface=None,
):
    """Estimate camera motion from dark background and visible surface features."""
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        first = clahe.apply(first_gray)
        second = clahe.apply(second_gray)
    except cv2.error:
        return None, 0.0, 0

    first_regions = _camera_regions(
        first_quad, first.shape, first_exclusion, first_surface
    )
    second_regions = _camera_regions(
        second_quad, second.shape, second_exclusion, second_surface
    )

    results = {}
    maximum_support = 0
    # Evaluate both sources independently: background parallax must not drown
    # out a useful fit on the surviving part of the actual surface.
    for name, region_index in (("surface", 1), ("background", 0)):
        region = first_regions[region_index]
        second_region = second_regions[region_index]
        if (
            np.count_nonzero(region) < 64
            or np.count_nonzero(second_region) < 64
        ):
            continue

        try:
            detected = cv2.goodFeaturesToTrack(
                first,
                maxCorners=1600,
                qualityLevel=0.001,
                minDistance=4.0,
                mask=region,
                blockSize=3,
            )
        except cv2.error:
            detected = None

        if detected is not None:
            points = detected.reshape(-1, 2)
            order = np.lexsort((points[:, 0], points[:, 1]))
            detected = points[order].reshape(-1, 1, 2)

        forbidden = (second_region == 0).astype(np.uint8)
        source, target = _flow_matches(
            first,
            second,
            detected,
            forbidden,
            minimum=8,
            min_eigenvalue=1e-5,
        )
        candidates = [
            _fit_camera_motion(source, target, first_quad, first.shape)
        ]

        if candidates[0][0] is None or candidates[0][1] < 0.52:
            source, target = _descriptor_matches(
                first,
                second,
                region,
                second_region,
                minimum=8,
                fast_threshold=3,
            )
            candidates.append(
                _fit_camera_motion(source, target, first_quad, first.shape)
            )
            refined_source, refined_target = _flow_matches(
                first,
                second,
                source,
                forbidden,
                initial=target,
                minimum=8,
                min_eigenvalue=1e-5,
            )
            candidates.append(
                _fit_camera_motion(
                    refined_source, refined_target, first_quad, first.shape
                )
            )

        maximum_support = max(
            maximum_support, max(candidate[2] for candidate in candidates)
        )
        valid = [candidate for candidate in candidates if candidate[0] is not None]
        if valid:
            results[name] = max(valid, key=lambda candidate: candidate[1])

    surface = results.get("surface")
    background = results.get("background")
    if surface is not None and (
        background is None or surface[1] >= 0.85 * background[1]
    ):
        return surface
    if background is not None:
        return background
    return None, 0.0, maximum_support


def _assess_frame(
    quad, shape, surface_mask, occluder_mask, exclusion_mask=None
):
    expected = _polygon_mask(quad, shape)
    expected_pixels = expected != 0
    occluded = False
    usable = np.ones(shape, dtype=bool)

    if occluder_mask is not None:
        occluded = bool(np.any(expected_pixels & occluder_mask))
        if exclusion_mask is None:
            exclusion_mask = cv2.dilate(
                occluder_mask.astype(np.uint8),
                np.ones((5, 5), dtype=np.uint8),
            )
        usable = exclusion_mask == 0

    if surface_mask is None:
        return 1.0, occluded

    visible_expected = expected_pixels & usable
    expected_count = int(np.count_nonzero(visible_expected))
    total_count = int(np.count_nonzero(expected_pixels))
    # Thin rasterization slivers are not negative evidence of full coverage.
    if expected_count < max(32, 0.08 * total_count):
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


def _neighbour_predictions(
    index, anchor, tracked, seeds, ignored_masks, grays, exclusions
):
    predictions = []
    for neighbour in (index - 1, index + 1):
        if not 0 <= neighbour < len(grays):
            continue

        tracked_neighbour = tracked[neighbour]
        seed = None if ignored_masks[neighbour] else seeds[neighbour]
        if seed is not None:
            seed = _match_corners(seed, anchor)
            if (
                np.isfinite(tracked_neighbour).all()
                and _quads_agree(tracked_neighbour, seed)
            ):
                source_quad = tracked_neighbour
            else:
                source_quad = seed
        elif np.isfinite(tracked_neighbour).all():
            source_quad = tracked_neighbour
        else:
            continue

        source_quad = _match_corners(source_quad, anchor)
        prediction, quality, _ = _track_pair(
            grays[neighbour],
            grays[index],
            source_quad,
            exclusions[neighbour],
            exclusions[index],
        )
        if prediction is not None and quality >= 0.30:
            predictions.append(prediction)
    return predictions


def _mask_is_outlier(seed, predictions):
    return bool(
        len(predictions) == 2
        and _quads_agree(predictions[0], predictions[1])
        and all(not _quads_agree(seed, prediction) for prediction in predictions)
    )


def _interpolate_quad(left, right, fraction):
    quad = (1.0 - fraction) * left + fraction * right
    if not _valid_quad(quad, min_area=1.0):
        endpoint = left if fraction <= 0.5 else right
        quad = endpoint - endpoint.mean(axis=0) + quad.mean(axis=0)
    return quad


def _bridge_occlusions(
    quads,
    confidence,
    inliers,
    status,
    occluders,
    grays,
    exclusions,
    surfaces,
    ref_idx=None,
):
    """Pin occluded camera-motion chains to their available temporal anchors."""
    count = len(quads)
    pending = np.array(
        [
            mask is not None and status[index] != "ok"
            for index, mask in enumerate(occluders)
        ],
        dtype=bool,
    )
    if ref_idx is not None:
        pending[ref_idx] = False
    boundaries = np.diff(np.r_[False, pending, False].astype(np.int8))
    starts = np.flatnonzero(boundaries == 1)
    stops = np.flatnonzero(boundaries == -1)

    def find_anchor(index, direction):
        while 0 <= index < count:
            if (
                np.isfinite(quads[index]).all()
                and (status[index] == "ok" or index == ref_idx)
                and (confidence[index] >= 0.30 or index == ref_idx)
            ):
                return index
            index += direction
        return None

    for start, stop in zip(starts, stops):
        left = find_anchor(int(start) - 1, -1)
        right = find_anchor(int(stop), 1)
        if left is None and right is None:
            continue

        low = int(start) if left is None else left
        high = int(stop) - 1 if right is None else right
        guides = {}
        for index in range(low, high + 1):
            if left is not None and right is not None:
                fraction = (index - left) / (right - left)
                guides[index] = _interpolate_quad(
                    quads[left], quads[right], fraction
                )
            elif np.isfinite(quads[index]).all():
                guides[index] = quads[index].copy()
            else:
                anchor = left if left is not None else right
                guides[index] = quads[anchor].copy()

        # Each edge is estimated once. Inverting an accepted edge preserves
        # the same motion when composing from the right anchor backwards.
        motions = {}
        for index in range(low, high):
            motion, quality, support = _camera_motion_pair(
                grays[index],
                grays[index + 1],
                guides[index],
                guides[index + 1],
                exclusions[index],
                exclusions[index + 1],
                surfaces[index],
                surfaces[index + 1],
            )
            if motion is None:
                reverse, quality, support = _camera_motion_pair(
                    grays[index + 1],
                    grays[index],
                    guides[index + 1],
                    guides[index],
                    exclusions[index + 1],
                    exclusions[index],
                    surfaces[index + 1],
                    surfaces[index],
                )
                if reverse is not None:
                    try:
                        motion = np.linalg.inv(reverse)
                    except np.linalg.LinAlgError:
                        motion = None
            motions[index] = motion, quality, support

        def chain(anchor, direction):
            if anchor is None:
                return {}
            predictions = {
                anchor: (quads[anchor].copy(), float(confidence[anchor]), 0)
            }
            previous = anchor
            total_quality = 0.0
            steps = 0
            end = high + 1 if direction == 1 else low - 1
            for index in range(anchor + direction, end, direction):
                motion, edge_quality, support = motions[min(previous, index)]
                if motion is None:
                    break
                if direction == -1:
                    try:
                        motion = np.linalg.inv(motion)
                    except np.linalg.LinAlgError:
                        break

                candidate = _motion_quad(
                    predictions[previous][0], motion, grays[index].shape
                )
                if candidate is None:
                    break
                steps += 1
                total_quality += edge_quality
                quality = (
                    min(float(confidence[anchor]), total_quality / steps)
                    * np.exp(-steps / 80.0)
                )
                predictions[index] = (
                    candidate,
                    float(np.clip(quality, 0.30, 0.70)),
                    support,
                )
                previous = index
            return predictions

        forward = chain(left, 1)
        backward = chain(right, -1)
        estimates = np.full((high - low + 1, 4, 2), np.nan, dtype=np.float64)
        qualities = np.zeros(high - low + 1, dtype=np.float64)
        supports = np.zeros(high - low + 1, dtype=np.int32)

        for index in range(low, high + 1):
            first = forward.get(index)
            second = backward.get(index)
            offset = index - low
            if first is not None and second is not None:
                fraction = (index - left) / (right - left)
                candidate = (1.0 - fraction) * first[0] + fraction * second[0]
                if not _valid_quad(candidate):
                    candidate = first[0] if fraction <= 0.5 else second[0]
                disagreement = float(
                    np.mean(np.linalg.norm(first[0] - second[0], axis=1))
                )
                quality = (
                    (1.0 - fraction) * first[1] + fraction * second[1]
                ) * np.exp(-disagreement / 6.0)
                support = max(first[2], second[2])
            elif first is not None or second is not None:
                candidate, quality, support = first if first is not None else second
            else:
                continue
            estimates[offset] = candidate
            qualities[offset] = np.clip(quality, 0.30, 0.70)
            supports[offset] = support

        # Anchor values are exact inputs, not outputs of the opposite chain.
        for anchor in (left, right):
            if anchor is not None:
                estimates[anchor - low] = quads[anchor]
                qualities[anchor - low] = confidence[anchor]

        # Only positions reached by neither chain may use interpolation.
        # Use the nearest recovered neighbours, rather than discarding motion
        # estimates on either side of a failed group of edges.
        missing = ~np.isfinite(estimates).all(axis=(1, 2))
        boundaries = np.diff(np.r_[False, missing, False].astype(np.int8))
        missing_starts = np.flatnonzero(boundaries == 1)
        missing_stops = np.flatnonzero(boundaries == -1)
        for gap_start, gap_stop in zip(missing_starts, missing_stops):
            if gap_start == 0 or gap_stop == len(estimates):
                continue
            first = estimates[gap_start - 1]
            second = estimates[gap_stop]
            quality = float(
                np.clip(
                    0.65 * min(qualities[gap_start - 1], qualities[gap_stop]),
                    0.30,
                    0.40,
                )
            )
            for offset in range(gap_start, gap_stop):
                fraction = (offset - gap_start + 1) / (
                    gap_stop - gap_start + 1
                )
                estimates[offset] = _interpolate_quad(first, second, fraction)
                qualities[offset] = quality

        for index in range(int(start), int(stop)):
            candidate = estimates[index - low]
            if not np.isfinite(candidate).all():
                continue
            # A detection elsewhere in the scene cannot explain this loss.
            expected = _polygon_mask(candidate, grays[index].shape) != 0
            if not np.any(expected & occluders[index]):
                continue
            quads[index] = candidate
            confidence[index] = qualities[index - low]
            inliers[index] = supports[index - low]
            status[index] = "occluded"


def _fill_short_gaps(quads, confidence, status, occluders):
    """Fill short losses, or long occlusions bounded by confident geometry."""
    valid = np.isfinite(quads).all(axis=(1, 2))
    boundaries = np.diff(np.r_[False, ~valid, False].astype(np.int8))
    starts = np.flatnonzero(boundaries == 1)
    stops = np.flatnonzero(boundaries == -1)

    for start, stop in zip(starts, stops):
        length = int(stop - start)
        if start == 0 or stop == len(quads):
            continue

        long_gap = length > 12
        if long_gap and (
            not np.isfinite(confidence[[start - 1, stop]]).all()
            or min(confidence[start - 1], confidence[stop]) < 0.50
            or status[start - 1] == "lost"
            or status[stop] == "lost"
            or any(value != "occluded" for value in status[start:stop])
        ):
            continue

        left = quads[start - 1].copy()
        right = quads[stop].copy()
        quality = min(
            0.5, 0.5 * min(confidence[start - 1], confidence[stop])
        )
        interpolated = []
        occluded_flags = []

        for index in range(start, stop):
            fraction = (index - start + 1) / (length + 1)
            quad = _interpolate_quad(left, right, fraction)

            occluded = False
            if occluders[index] is not None:
                expected = _polygon_mask(quad, occluders[index].shape) != 0
                occluded = bool(np.any(expected & occluders[index]))

            interpolated.append(quad)
            occluded_flags.append(occluded)

        if long_gap and not all(occluded_flags):
            continue

        for index, quad, occluded in zip(
            range(start, stop), interpolated, occluded_flags
        ):
            quads[index] = quad
            confidence[index] = quality
            status[index] = "occluded" if occluded else "ok"


def propagate_quad(
    frames, ref_idx, ref_quad, surface_masks=None, occluder_masks=None
) -> dict:
    """Track, re-acquire surfaces, and bridge occlusions with anchored motion."""
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

    # This module does not alter NumPy's global random state.
    cv2.setRNGSeed(0)

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
    occluders, exclusions = _prepare_occluders(occluders)
    seeds = [
        _mask_seed(surface, exclusion)
        for surface, exclusion in zip(surfaces, exclusions)
    ]

    tracked = np.full((count, 4, 2), np.nan, dtype=np.float64)
    quality = np.zeros(count, dtype=np.float64)
    inliers = np.zeros(count, dtype=np.int32)
    ignored_masks = np.zeros(count, dtype=bool)
    supported_masks = np.zeros(count, dtype=bool)

    # OpenCV resizing maps pixel centers, rather than pixel-array edges.
    scaled_reference = (reference.astype(np.float64) + 0.5) * scale - 0.5
    tracked[ref_idx] = scaled_reference
    quality[ref_idx] = 1.0

    for direction in (-1, 1):
        previous = ref_idx
        active_quad = scaled_reference.copy()
        last_known_quad = scaled_reference.copy()
        stop = count if direction == 1 else -1

        for index in range(ref_idx + direction, stop, direction):
            candidate, candidate_quality, support = None, 0.0, 0
            if active_quad is not None:
                candidate, candidate_quality, support = _track_pair(
                    grays[previous],
                    grays[index],
                    active_quad,
                    exclusions[previous],
                    exclusions[index],
                )
            inliers[index] = support

            seed = seeds[index]
            if seed is not None:
                anchor = candidate if candidate is not None else last_known_quad
                seed = _match_corners(seed, anchor)
                needs_seed = candidate is None or candidate_quality < 0.30
                disagrees = (
                    candidate is not None and not _quads_agree(candidate, seed)
                )

                supported_masks[index] = support >= 24 and (
                    _quads_agree(seed, last_known_quad)
                    or (
                        candidate is not None
                        and _quads_agree(candidate, seed)
                    )
                )
                if supported_masks[index]:
                    if needs_seed or disagrees:
                        candidate = seed
                        candidate_quality = 0.75
                elif needs_seed or disagrees:
                    predictions = _neighbour_predictions(
                        index,
                        anchor,
                        tracked,
                        seeds,
                        ignored_masks,
                        grays,
                        exclusions,
                    )
                    if _mask_is_outlier(seed, predictions):
                        ignored_masks[index] = True
                    elif needs_seed or any(
                        _quads_agree(seed, prediction)
                        for prediction in predictions
                    ):
                        candidate = seed
                        candidate_quality = 0.75

            # Continue scanning after a failed chain until a complete mask
            # establishes another adjacent-frame chain.
            previous = index
            active_quad = candidate
            if candidate is not None:
                tracked[index] = candidate
                quality[index] = candidate_quality
                if (
                    seed is None
                    or ignored_masks[index]
                    or _quads_agree(candidate, seed)
                ):
                    last_known_quad = candidate

    resolved = np.full_like(tracked, np.nan)
    confidence = np.zeros(count, dtype=np.float32)
    status = [
        "occluded" if mask is not None else "lost"
        for mask in occluders
    ]

    for index, candidate in enumerate(tracked):
        if not np.isfinite(candidate).all():
            continue

        agreement, occluded = _assess_frame(
            candidate,
            grays[index].shape,
            surfaces[index],
            occluders[index],
            exclusions[index],
        )
        candidate_quality = quality[index]
        seed = seeds[index]
        if seed is not None:
            seed = _match_corners(seed, candidate)

        conflict = agreement < 0.60 or (
            seed is not None and not _quads_agree(candidate, seed)
        )
        if index == ref_idx:
            agreement = 1.0
        elif supported_masks[index]:
            agreement = 1.0
        elif ignored_masks[index]:
            agreement = 1.0
            candidate_quality *= 0.75
        elif conflict:
            predictions = _neighbour_predictions(
                index,
                candidate,
                tracked,
                seeds,
                ignored_masks,
                grays,
                exclusions,
            )
            track_supported = len(predictions) == 2 and all(
                _quads_agree(candidate, prediction)
                for prediction in predictions
            )
            seed_outlier = (
                seed is not None and _mask_is_outlier(seed, predictions)
            )
            if track_supported and (
                seed is None
                or all(
                    not _quads_agree(seed, prediction)
                    for prediction in predictions
                )
            ):
                agreement = 1.0
                candidate_quality *= 0.75
            elif (
                seed is not None
                and not seed_outlier
                and any(
                    _quads_agree(seed, prediction)
                    for prediction in predictions
                )
            ):
                candidate = seed
                candidate_quality = 0.75
                agreement, occluded = _assess_frame(
                    candidate,
                    grays[index].shape,
                    surfaces[index],
                    occluders[index],
                    exclusions[index],
                )
                if agreement < 0.60:
                    continue
            else:
                continue

        resolved[index] = candidate
        confidence[index] = np.clip(candidate_quality * agreement, 0.0, 1.0)
        status[index] = "occluded" if occluded else "ok"

    # This also replaces finite one-way occlusion tracks, preventing their
    # accumulated drift from bypassing the bidirectional bridge.
    _bridge_occlusions(
        resolved,
        confidence,
        inliers,
        status,
        occluders,
        grays,
        exclusions,
        surfaces,
        ref_idx=ref_idx,
    )
    _fill_short_gaps(resolved, confidence, status, occluders)
    for index in range(count):
        if not np.isfinite(resolved[index]).all():
            status[index] = "lost"
            confidence[index] = 0.0

    quads = ((resolved + 0.5) / scale - 0.5).astype(np.float32)
    quads[ref_idx] = reference

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
