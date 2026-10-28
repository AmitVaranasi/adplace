"""Match printed artwork to photographed surface tones in CIE Lab."""
import cv2
import numpy as np


def _unit_strength(strength):
    strength = float(strength)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be finite and in [0, 1]")
    return strength


def _pixels_lab(pixels, name):
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8 or pixels.ndim != 2 or pixels.shape[1] != 3:
        raise ValueError(f"{name} must be an (N, 3) uint8 BGR array")
    if len(pixels) == 0:
        return np.empty((0, 3), np.float32)
    rgb = np.ascontiguousarray(pixels[:, ::-1], dtype=np.float32) / 255.0
    return cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)


def scene_grade(replaced_bgr, scene_bgr, strength=1.0) -> dict:
    """Estimate a grade from original surface pixels and their surrounding ring.

    Tone endpoints and cast are already blended by strength. c_max stores the
    full-strength ceiling; apply_grade blends its compression with identity,
    whose ceiling would otherwise be infinite. Include percentile ties so
    uniform surfaces can provide neutral highlights.
    """
    strength = _unit_strength(strength)
    replaced = _pixels_lab(replaced_bgr, "replaced_bgr")
    scene = _pixels_lab(scene_bgr, "scene_bgr")
    reference = replaced if len(replaced) >= 200 else scene
    if len(reference) == 0:
        raise ValueError("the selected tone reference must contain pixels")

    # The darkest thing nearby, in the same light: the replaced content alone only says how dark that old ad was (a
    # bright yellow board or a lit screen has no black in it, and its p3 greyed our blacks out).
    black = float(np.percentile(reference[:, 0], 3))
    if len(scene):
        black = min(black, float(np.percentile(scene[:, 0], 3)))
    black = float(np.clip(black, 0.0, 35.0))
    white = float(np.clip(np.percentile(reference[:, 0], 97), black + 30.0, 100.0))

    pooled = np.concatenate((replaced, scene), axis=0)
    chroma = np.hypot(pooled[:, 1], pooled[:, 2])
    bright = pooled[:, 0] >= np.percentile(pooled[:, 0], 70)
    neutral = chroma <= np.percentile(chroma, 50)
    if not np.any(bright & neutral):
        # Disjoint cuts (including quantized flat casts): prefer neutral highlights.
        neutral = chroma <= np.percentile(chroma[bright], 50)
    cast = np.median(pooled[bright & neutral, 1:3], axis=0)

    # p99 of the surface: small saturated details (a mug's red print) show what the camera records there; p95 missed
    # them and greyed the brand colours out.
    chroma95 = [
        float(np.percentile(np.hypot(pixels[:, 1], pixels[:, 2]), q))
        for pixels, q in ((replaced, 99), (scene, 95))
        if len(pixels)
    ]
    return {
        "black_L": float(strength * black),
        "white_L": float(100.0 + strength * (white - 100.0)),
        "cast_a": float(strength * float(cast[0])),
        "cast_b": float(strength * float(cast[1])),
        "c_max": float(max(12.0, 1.25 * max(chroma95))),
        "strength": float(strength),
    }


def apply_grade(creative_rgba, grade) -> np.ndarray:
    """Grade straight RGB channels; return a copy with alpha unchanged.

    Compress the artwork chroma before adding the light cast. Weight that cast
    by ORIGINAL creative L, keeping input black neutral and giving input white
    the full cast even when the output tonal range is narrower.
    """
    creative = np.asarray(creative_rgba)
    if (
        creative.dtype != np.uint8
        or creative.ndim != 3
        or creative.shape[2] != 4
        or min(creative.shape[:2]) == 0
    ):
        raise ValueError("creative_rgba must be a nonempty (H, W, 4) uint8 array")

    strength = _unit_strength(grade["strength"])
    out = creative.copy()
    if strength == 0.0:
        return out

    black = float(grade["black_L"])
    white = float(grade["white_L"])
    cast = np.array([grade["cast_a"], grade["cast_b"]], np.float32)
    c_max = float(grade["c_max"])
    if (
        not np.all(np.isfinite([black, white, c_max, *cast]))
        or not 0.0 <= black < white <= 100.0
        or c_max <= 0.0
    ):
        raise ValueError("grade must contain finite, valid Lab parameters")

    rgb = np.ascontiguousarray(creative[..., :3], dtype=np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    original_L = lab[..., 0].copy() / 100.0

    ab = lab[..., 1:3]
    chroma = np.hypot(ab[..., 0], ab[..., 1])
    compressed = c_max * np.tanh(chroma / c_max)
    ratio = np.divide(
        compressed, chroma, out=np.ones_like(chroma), where=chroma > 0.0
    )
    ab *= (1.0 + strength * (ratio - 1.0))[..., None]
    ab += original_L[..., None] * cast
    lab[..., 0] = black + original_L * (white - black)

    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    out[..., :3] = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return out
