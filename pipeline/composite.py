"""Illumination estimation and colour-preserving planar compositing."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


def _frame_array(frame) -> np.ndarray:
    array = np.asarray(frame)
    if (
        array.ndim != 3
        or array.shape[2] != 3
        or min(array.shape[:2]) == 0
        or array.dtype != np.uint8
    ):
        raise ValueError("frame must be a nonempty BGR uint8 array")
    return array


def _quad_array(quad) -> np.ndarray | None:
    points = np.asarray(quad, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("quad must have shape (4, 2)")
    if not np.isfinite(points).all():
        return None
    if not cv2.isContourConvex(points) or cv2.contourArea(points) <= 1.0:
        return None
    return points


def _corners(width: int, height: int) -> np.ndarray:
    # Pixel extents support even a one-pixel creative without a singular warp.
    return np.array(
        [
            [-0.5, -0.5],
            [width - 0.5, -0.5],
            [width - 0.5, height - 0.5],
            [-0.5, height - 0.5],
        ],
        dtype=np.float32,
    )


def _nonnegative(value, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _polygon_mask(shape, quad) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(
        mask,
        np.rint(quad * 256.0).astype(np.int32),
        1,
        lineType=cv2.LINE_8,
        shift=8,
    )
    return mask


def _expanded_quad(quad, bleed_px: float) -> np.ndarray:
    if bleed_px == 0:
        return quad.copy()
    points = quad.astype(np.float64)
    centre = points.mean(axis=0)
    edges = np.roll(points, -1, axis=0) - points
    offsets = centre - points
    distances = np.abs(
        edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    ) / np.linalg.norm(edges, axis=1)
    # A uniform radial expansion moves every supporting edge by at least bleed_px.
    scale = 1.0 + bleed_px / float(distances.min())
    return (centre + scale * (points - centre)).astype(np.float32)


def illumination_map(blank_frame, quad, size) -> np.ndarray:
    """Rectify blank-surface luminance into a smooth, mean-one float32 map."""
    frame = _frame_array(blank_frame)
    points = _quad_array(quad)
    if points is None:
        raise ValueError("illumination_map requires a finite, convex quad")
    if (
        len(size) != 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
            for value in size
        )
    ):
        raise ValueError("size must contain positive integer (width, height)")
    width, height = int(size[0]), int(size[1])
    transform = cv2.getPerspectiveTransform(points, _corners(width, height))
    gray = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2GRAY)
    rectified = cv2.warpPerspective(
        gray,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = cv2.warpPerspective(
        np.ones(frame.shape[:2], dtype=np.float32),
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    weight = float(valid.sum(dtype=np.float64))
    if weight <= 1e-6:
        return np.ones((height, width), dtype=np.float32)

    sigma = max(1.0, min(width, height) / 32.0)
    numerator = cv2.GaussianBlur(
        rectified, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101
    )
    denominator = cv2.GaussianBlur(
        valid, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101
    )
    observed_mean = float(rectified.sum(dtype=np.float64)) / weight
    shading = np.full_like(numerator, observed_mean)
    np.divide(numerator, denominator, out=shading, where=denominator > 1e-6)
    mean = float(shading.mean(dtype=np.float64))
    if mean <= 1e-6:
        return np.ones((height, width), dtype=np.float32)
    return (shading / mean).astype(np.float32)


def _edge_blur(gray, band, gx, gy, magnitude, grain_sigma: float) -> float:
    """Fit Gaussian-blurred step profiles on isolated surrounding edges."""
    gray_noise = grain_sigma * np.sqrt(0.114**2 + 0.587**2 + 0.299**2)
    threshold = max(3.0, 2.0 * gray_noise)
    maxima = cv2.dilate(magnitude, np.ones((3, 3), dtype=np.uint8))
    ys, xs = np.nonzero(
        band & (magnitude >= threshold) & (magnitude >= maxima - 1e-5)
    )
    if len(xs) == 0:
        return 0.0

    order = np.argsort(magnitude[ys, xs])[::-1]
    positions = np.linspace(-10.0, 10.0, 41)
    tangents = np.arange(-2.0, 3.0)
    used = []
    sigmas = []

    def residual(parameters, profile):
        base, amplitude, slope, centre, sigma = parameters
        step = 0.5 * (1.0 + erf((positions - centre) / (np.sqrt(2.0) * sigma)))
        return base + amplitude * step + slope * positions - profile

    band_u8 = band.astype(np.uint8)
    for index in order:
        x, y = int(xs[index]), int(ys[index])
        if any((x - ux) ** 2 + (y - uy) ** 2 < 64 for ux, uy in used):
            continue
        strength = float(magnitude[y, x])
        nx, ny = float(gx[y, x]) / strength, float(gy[y, x]) / strength
        map_x = (
            x + nx * positions[None, :] - ny * tangents[:, None]
        ).astype(np.float32)
        map_y = (
            y + ny * positions[None, :] + nx * tangents[:, None]
        ).astype(np.float32)
        support = cv2.remap(
            band_u8,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if not np.all(support):
            continue

        used.append((x, y))
        samples = cv2.remap(
            gray,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        profile = samples.mean(axis=0, dtype=np.float64)
        difference = float(profile[-6:].mean() - profile[:6].mean())
        if abs(difference) >= 12.0:
            initial = [
                float(profile[:6].mean()),
                difference,
                0.0,
                0.0,
                1.2,
            ]
            fit = least_squares(
                residual,
                initial,
                args=(profile,),
                bounds=(
                    [-255.0, -510.0, -8.0, -2.0, 0.15],
                    [510.0, 510.0, 8.0, 2.0, 5.0],
                ),
                loss="soft_l1",
                f_scale=max(0.8, gray_noise / np.sqrt(len(tangents))),
                max_nfev=60,
            )
            amplitude = abs(float(fit.x[1]))
            error = float(np.sqrt(np.mean(residual(fit.x, profile) ** 2)))
            tolerance = max(
                1.25,
                2.5 * gray_noise / np.sqrt(len(tangents)),
                0.06 * amplitude,
            )
            if (
                fit.success
                and amplitude >= 12.0
                and error <= tolerance
                and fit.x[4] < 4.9
            ):
                # Remove the approximate variance introduced by linear sampling.
                sigmas.append(np.sqrt(max(0.0, float(fit.x[4]) ** 2 - 1.0 / 6.0)))
        if len(sigmas) >= 32 or len(used) >= 80:
            break

    return float(np.median(sigmas)) if sigmas else 0.0


def estimate_blur_and_grain(frame, quad) -> tuple[float, float]:
    """Estimate edge blur in pixels and additive per-channel noise in 0–255 units."""
    frame = _frame_array(frame)
    points = _quad_array(quad)
    if points is None:
        return 0.0, 0.0

    inside = _polygon_mask(frame.shape[:2], points)
    width = max(24, min(64, int(np.ceil(0.15 * np.sqrt(cv2.contourArea(points))))))
    outside = cv2.dilate(
        inside, np.ones((2 * width + 1, 2 * width + 1), dtype=np.uint8)
    )
    band = (outside != 0) & (inside == 0)
    safe = cv2.erode(
        band.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    if np.count_nonzero(safe) < 64:
        return 0.0, 0.0

    floating = frame.astype(np.float32)
    gray = cv2.cvtColor(floating, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), 0.8)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)
    magnitude = cv2.magnitude(gx, gy)
    flat = safe & (magnitude <= np.percentile(magnitude[safe], 65.0))

    kernel = np.array(
        [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
        dtype=np.float32,
    )
    response = cv2.filter2D(floating, cv2.CV_32F, kernel)
    samples = response[flat]
    centre = np.median(samples, axis=0, keepdims=True)
    deviations = np.median(np.abs(samples - centre), axis=0)
    grain = float(np.median(deviations) / (0.67448975 * 6.0))
    blur = _edge_blur(gray, band, gx, gy, magnitude, grain)
    return blur, grain


def composite_frame(
    frame,
    creative_rgba,
    quad,
    illum=None,
    occluder_mask=None,
    emissive=False,
    blur_sigma=None,
    grain_sigma=None,
    bleed_px=2.0,
) -> np.ndarray:
    """Warp premultiplied artwork, match scene appearance, and preserve occluders."""
    frame = _frame_array(frame)
    result = frame.copy()
    points = _quad_array(quad)
    if points is None:
        return result

    creative = np.asarray(creative_rgba)
    if (
        creative.ndim != 3
        or creative.shape[2] != 4
        or min(creative.shape[:2]) == 0
        or creative.dtype != np.uint8
    ):
        raise ValueError("creative_rgba must be a nonempty RGBA uint8 array")

    mask = None
    if occluder_mask is not None:
        mask = np.asarray(occluder_mask)
        if mask.shape != frame.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("occluder_mask must be bool with the frame's height and width")

    bleed = _nonnegative(bleed_px, "bleed_px")
    blur = None if blur_sigma is None else _nonnegative(blur_sigma, "blur_sigma")
    grain = None if grain_sigma is None else _nonnegative(grain_sigma, "grain_sigma")
    if blur is None or grain is None:
        estimated_blur, estimated_grain = estimate_blur_and_grain(frame, points)
        blur = estimated_blur if blur is None else blur
        grain = estimated_grain if grain is None else grain

    height, width = creative.shape[:2]
    colour = creative[..., :3][..., ::-1].astype(np.float32)
    alpha = creative[..., 3:4].astype(np.float32) / 255.0
    if illum is not None and not emissive:
        shading = np.asarray(illum, dtype=np.float32)
        if (
            shading.ndim != 2
            or shading.size == 0
            or not np.isfinite(shading).all()
            or np.any(shading < 0)
        ):
            raise ValueError("illum must be a nonempty, finite, nonnegative 2D map")
        shading = cv2.resize(shading, (width, height), interpolation=cv2.INTER_LINEAR)
        colour *= shading[..., None]
        # Scale overflowing RGB vectors together to preserve chromaticity.
        peak = colour.max(axis=2, keepdims=True)
        colour *= np.minimum(1.0, 255.0 / np.maximum(peak, 1.0))

    source = np.concatenate((colour * alpha, alpha), axis=2)
    expanded = _expanded_quad(points, bleed)
    radius = int(np.ceil(3.0 * blur)) if blur > 0 else 0
    padding = radius + 2
    frame_height, frame_width = frame.shape[:2]
    x0 = max(-padding, int(np.floor(expanded[:, 0].min())) - padding)
    y0 = max(-padding, int(np.floor(expanded[:, 1].min())) - padding)
    x1 = min(frame_width + padding, int(np.ceil(expanded[:, 0].max())) + padding + 1)
    y1 = min(frame_height + padding, int(np.ceil(expanded[:, 1].max())) + padding + 1)
    left, top = max(0, x0), max(0, y0)
    right, bottom = min(frame_width, x1), min(frame_height, y1)
    if right <= left or bottom <= top:
        return result

    roi_width, roi_height = x1 - x0, y1 - y0
    local_quad = expanded - np.array([x0, y0], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(_corners(width, height), local_quad)
    supersample = np.array(
        [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    warped = cv2.warpPerspective(
        source,
        supersample @ transform,
        (2 * roi_width, 2 * roi_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    coverage = np.zeros((2 * roi_height, 2 * roi_width), dtype=np.uint8)
    cv2.fillConvexPoly(
        coverage,
        np.rint((2.0 * local_quad + 0.5) * 256.0).astype(np.int32),
        255,
        lineType=cv2.LINE_AA,
        shift=8,
    )
    # Identical filtering of premultiplied colour and alpha prevents dark halos.
    warped *= coverage[..., None].astype(np.float32) / 255.0
    warped = cv2.resize(warped, (roi_width, roi_height), interpolation=cv2.INTER_AREA)
    if blur > 0:
        warped = cv2.GaussianBlur(
            warped,
            (2 * radius + 1, 2 * radius + 1),
            blur,
            borderType=cv2.BORDER_CONSTANT,
        )

    rows = slice(top - y0, bottom - y0)
    columns = slice(left - x0, right - x0)
    premultiplied = warped[rows, columns, :3]
    alpha = np.clip(warped[rows, columns, 3:4], 0.0, 1.0)
    original = frame[top:bottom, left:right]
    visible = (
        np.ones(original.shape[:2], dtype=bool)
        if mask is None
        else ~mask[top:bottom, left:right]
    )

    if emissive:
        source_luma = cv2.cvtColor(premultiplied, cv2.COLOR_BGR2GRAY)
        target_luma = cv2.cvtColor(original.astype(np.float32), cv2.COLOR_BGR2GRAY)
        source_total = float(source_luma[visible].sum(dtype=np.float64))
        target_total = float(
            (target_luma * alpha[..., 0])[visible].sum(dtype=np.float64)
        )
        support = visible & (alpha[..., 0] > 1e-6)
        if source_total > 1e-6 and np.any(support):
            straight = premultiplied / np.maximum(alpha, 1e-6)
            peak = float(straight[support].max())
            gain = target_total / source_total
            if peak > 0:
                gain = min(gain, 255.0 / peak)
            premultiplied *= gain

    if grain > 0:
        # A local generator makes repeated calls deterministic without global state.
        noise = np.random.default_rng(0).normal(0.0, grain, premultiplied.shape)
        premultiplied += noise.astype(np.float32) * alpha

    visible_alpha = alpha * visible[..., None]
    blended = (
        premultiplied * visible[..., None]
        + original.astype(np.float32) * (1.0 - visible_alpha)
    )
    output_region = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    output_region[~visible] = original[~visible]
    result[top:bottom, left:right] = output_region
    return result


def render_clip(
    frames,
    quads,
    creative_rgba,
    illum=None,
    occluder_masks=None,
    emissive=False,
) -> list:
    """Composite a clip using one appearance estimate from its first valid quad."""
    frames = list(frames)
    quads = list(quads)
    if len(frames) != len(quads):
        raise ValueError("frames and quads must have equal lengths")
    masks = [None] * len(frames) if occluder_masks is None else list(occluder_masks)
    if len(masks) != len(frames):
        raise ValueError("occluder_masks and frames must have equal lengths")

    reference = None
    for index, quad in enumerate(quads):
        if _quad_array(quad) is not None and reference is None:
            reference = index
    blur, grain = (
        (0.0, 0.0)
        if reference is None
        else estimate_blur_and_grain(frames[reference], quads[reference])
    )
    return [
        composite_frame(
            frame,
            creative_rgba,
            quad,
            illum=illum,
            occluder_mask=mask,
            emissive=emissive,
            blur_sigma=blur,
            grain_sigma=grain,
        )
        for frame, quad, mask in zip(frames, quads, masks)
    ]
