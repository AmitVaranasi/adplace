"""Inverse cylindrical RGBA warping and illumination-aware BGR compositing."""

import cv2
import numpy as np


def _geometry(cyl, frame_shape, arc_deg):
    if not np.isfinite(arc_deg) or not 0 < arc_deg <= 180:
        raise ValueError("arc_deg must be finite and in (0, 180]")
    height, width = frame_shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    radii = np.interp(
        np.arange(height), cyl["rows"], cyl["half_width"]
    ).astype(np.float32)[:, None]
    distance = xx - (cyl["a"] * yy + cyl["b"])
    phi = np.arcsin(np.clip(distance / np.maximum(radii, 1e-6), -1, 1))
    half_arc = np.deg2rad(arc_deg) * 0.5
    valid = (
        (yy >= cyl["y0"]) & (yy <= cyl["y1"]) & (radii > 0)
        & (np.abs(distance) < radii) & (np.abs(phi) <= half_arc)
    )
    return yy, phi, valid, half_arc


def warp_to_cylinder(creative_rgba, cyl, frame_shape, arc_deg=110) -> np.ndarray:
    """Map angular/vertical coordinates to the creative; feather angular edges."""
    creative = np.asarray(creative_rgba)
    if (creative.ndim != 3 or creative.shape[2] != 4
            or min(creative.shape[:2]) == 0 or creative.dtype != np.uint8):
        raise ValueError("creative must be a nonempty RGBA uint8 image")
    yy, phi, valid, half_arc = _geometry(cyl, frame_shape, arc_deg)
    height, width = creative.shape[:2]
    map_x = ((phi + half_arc) / (2 * half_arc) * (width - 1)).astype(np.float32)
    map_y = (
        (yy - cyl["y0"]) / max(cyl["y1"] - cyl["y0"], 1) * (height - 1)
    ).astype(np.float32)
    map_x[~valid] = -1
    map_y[~valid] = -1
    warped = cv2.remap(
        creative, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    fade = np.clip((half_arc - np.abs(phi)) / np.deg2rad(8), 0, 1)
    fade = fade * fade * (3 - 2 * fade)
    warped[..., 3] = np.rint(warped[..., 3] * fade).astype(np.uint8)
    warped[~valid] = 0
    return warped


def composite_curved(frame_bgr, creative_rgba, cyl, occluder_mask=None,
                     arc_deg=110, shade=0.7) -> np.ndarray:
    """Shade the wrapped label, preserving every unpainted pixel exactly."""
    frame = np.asarray(frame_bgr)
    if (frame.ndim != 3 or frame.shape[2] != 3
            or min(frame.shape[:2]) == 0 or frame.dtype != np.uint8):
        raise ValueError("frame must be a nonempty BGR uint8 image")
    if not np.isfinite(shade) or shade < 0:
        raise ValueError("shade must be finite and nonnegative")
    warped = warp_to_cylinder(creative_rgba, cyl, frame.shape, arc_deg)
    alpha = warped[..., 3].astype(np.float32) / 255.0
    visible = alpha > 0
    active = visible.copy()
    if occluder_mask is not None:
        occluder = np.asarray(occluder_mask, dtype=bool)
        if occluder.shape != frame.shape[:2]:
            raise ValueError("occluder_mask must match the frame dimensions")
        active &= ~occluder
    result = frame.copy()
    if not np.any(active):
        return result

    luminance = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2GRAY)
    sigma = max(0.15 * float(np.mean(cyl["half_width"])), 1e-3)
    blurred = cv2.GaussianBlur(luminance, (0, 0), sigma)
    median = max(float(np.median(blurred[visible])), 1e-6)
    shading = np.clip((blurred / median) ** shade, 0.3, 1.6)
    _, phi, _, _ = _geometry(cyl, frame.shape, arc_deg)
    shading *= np.sqrt(np.maximum(np.cos(phi), 0))
    label = warped[..., :3][..., ::-1].astype(np.float32) * shading[..., None]
    weight = alpha[active, None]
    mixed = frame[active].astype(np.float32) * (1 - weight) + label[active] * weight
    result[active] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    return result
