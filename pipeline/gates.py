"""Conservative gates for INGEST.md.

Quads are local to the candidate's inclusive ["frames"] range, in TL/TR/BR/BL
order. None means not visible. Missing geometry cannot prove non-overlap.
Only flat placements are supported until 3D calibration evidence is specified.
"""
import math
from collections import Counter
from collections.abc import Mapping
from numbers import Integral, Real

import cv2
import numpy as np

DEFAULTS = {
    "on_screen_s": 2.0,  # Two seconds permits recognition without a fleeting insert.
    "longest_run_s": 1.5,  # Require one uninterrupted readable interval.
    "track_residual_px_median": 2.5,  # Small typical mask/quad errors prevent drifting edges.
    "track_residual_px_p90": 5.0,  # Occasional tracking errors still need a tight bound.
    "corner_jitter_px_p95": 3.0,  # Suppress visible high-frequency corner acceleration.
    "min_side_px": 48.0,  # Smaller surfaces cannot reliably resolve a creative.
    "mean_area_frac": 0.004,  # Require at least 0.4% of the normalized picture.
    "edge_clipped_fraction": 0.3,  # Most tracked frames must contain the whole surface.
    "max_occlusion_frac": 0.6,  # Even the worst measured frame must retain 40%.
    "mean_occlusion_frac": 0.35,  # Typical coverage must leave a readable majority.
    "motion_blur_px_median": 12.0,  # Larger estimated smears obscure placement detail.
    "luma_std_over_time": 25.0,  # Large lighting swings need explicit screen treatment.
    "uncovered_visible_s": 0.25,  # The original must not pop back while its surface is still in view.
    "muse_plausible": True,  # Require an affirmative model observation.
    "scene_safe": True,  # A known unsafe scene vetoes the entire uploaded clip.
    "overlap_iou": 0.2,  # Avoid competing placements occupying the same surface.
}

# key: (direction, human label, units)
RULES = {
    "on_screen_s": ("min", "on screen", " s"),
    "longest_run_s": ("min", "longest run", " s"),
    "track_residual_px_median": ("max", "median track residual", " px"),
    "track_residual_px_p90": ("max", "p90 track residual", " px"),
    "corner_jitter_px_p95": ("max", "p95 corner jitter", " px"),
    "min_side_px": ("min", "minimum side", " px"),
    "mean_area_frac": ("min", "mean area fraction", ""),
    "edge_clipped_fraction": ("max", "edge-clipped fraction", ""),
    "max_occlusion_frac": ("max", "maximum occlusion", ""),
    "mean_occlusion_frac": ("max", "mean occlusion", ""),
    "motion_blur_px_median": ("max", "median motion blur", " px"),
    "luma_std_over_time": ("max", "luma standard deviation", ""),
    "uncovered_visible_s": ("max", "time the surface is visible but not covered", " s"),
}
FRACTIONS = {
    "mean_area_frac", "edge_clipped_fraction",
    "max_occlusion_frac", "mean_occlusion_frac",
}


def _number(value):
    try:
        return (isinstance(value, Real) and not isinstance(value, bool)
                and math.isfinite(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _fmt(value):
    text = f"{float(value):.4g}"
    return text + ".0" if "." not in text and "e" not in text else text


def _config(cfg):
    if not isinstance(cfg, Mapping) or set(cfg) - set(DEFAULTS):
        raise ValueError("unknown or malformed gate configuration")
    out = {**DEFAULTS, **cfg}
    for key in RULES:
        if not _number(out[key]) or out[key] <= 0:
            raise ValueError(f"{key} must have a finite positive limit")
        if key in FRACTIONS and out[key] > 1:
            raise ValueError(f"{key} must be at most 1")
    if out["scene_safe"] is not True or out["muse_plausible"] is not True:
        raise ValueError("safety and plausibility requirements cannot be disabled")
    if not _number(out["overlap_iou"]) or not 0 <= out["overlap_iou"] <= 1:
        raise ValueError("overlap_iou must be in [0, 1]")
    return out


def _metrics(candidate):
    value = candidate.get("metrics")
    return value if isinstance(value, Mapping) else {}


def _unsafe(candidate):
    return (candidate.get("scene_safe") is False
            or _metrics(candidate).get("scene_safe") is False)


def _span(candidate):
    span = candidate.get("frames")
    if (not isinstance(span, (list, tuple)) or len(span) != 2
            or any(not isinstance(x, Integral) or isinstance(x, bool) for x in span)
            or not 0 <= span[0] <= span[1]):
        raise ValueError("frame range missing or invalid; needs inclusive [start, end]")
    return int(span[0]), int(span[1])


def _quad(value):
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ValueError("quads need four finite [x, y] corners")
    with np.errstate(over="ignore", invalid="ignore"):
        q = q.astype(np.float32)
    if (not np.isfinite(q).all() or not cv2.isContourConvex(q)
            or cv2.contourArea(q) <= 0):
        raise ValueError("quads must be ordered, convex and non-degenerate")
    return q


def _quads(candidate):
    start, end = _span(candidate)
    values = candidate.get("quads")
    if values is None:
        return None
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != end - start + 1:
        raise ValueError("quad count must match the inclusive frame range")
    result = {start + i: None if q is None else _quad(q)
              for i, q in enumerate(values)}
    if not any(q is not None for q in result.values()):
        raise ValueError("quads contain no measured visible frame")
    return result


def _decline(row, reason):
    row["decision"], row["score"] = "decline", 0.0
    row["reasons"].append(reason)


def decide(candidate, cfg=DEFAULTS):
    """Apply inclusive limits, then combine normalized passing margins.

    Each margin m >= 0 contributes 0.5 + 0.5*tanh(m). Their geometric
    mean scores a boundary at 0.5 and increases smoothly with improvement.
    The score is a ranking measure, not a calibrated success probability.
    """
    limits = _config(cfg)
    if not isinstance(candidate, Mapping):
        return {"id": None, "decision": "decline", "score": 0.0,
                "reasons": ["candidate missing or invalid"]}

    cid = candidate.get("id")
    row = {"id": cid if isinstance(cid, str) else None,
           "decision": "accept", "score": 0.0, "reasons": []}
    reasons, margins = row["reasons"], []
    m = _metrics(candidate)

    if not isinstance(cid, str) or not cid.strip():
        reasons.append("candidate id missing or invalid")
    shot = candidate.get("shot")
    if not isinstance(shot, Integral) or isinstance(shot, bool) or shot < 0:
        reasons.append("shot missing or invalid; needs a nonnegative integer")
    if candidate.get("kind") != "flat":
        reasons.append("flat placement required; 3D calibration proof is not specified")

    try:
        _quads(candidate)
    except (ValueError, TypeError, OverflowError, cv2.error) as exc:
        reasons.append(str(exc))

    if _unsafe(candidate):
        reasons.append("scene_safe is False; no ads are allowed anywhere in this clip")
    elif m.get("scene_safe") is not True:
        reasons.append("scene_safe missing or invalid; needs True")
    if m.get("muse_plausible") is not True:
        reasons.append(f"muse_plausible is {m.get('muse_plausible')!r}; needs True")

    dynamic = m.get("dynamic_content")
    if type(dynamic) is not bool:
        reasons.append("dynamic_content missing or invalid; needs a measured boolean")
    screen = candidate.get("surface") in ("screen", "television")
    if dynamic is True and not screen:
        reasons.append("dynamic_content is True; luma exemption requires a screen or television")

    values = {}
    for key, (direction, label, unit) in RULES.items():
        value, limit = m.get(key), limits[key]
        need = "at least" if direction == "min" else "at most"
        if not _number(value):
            reasons.append(
                f"{label} missing or non-finite, needs {need} {_fmt(limit)}{unit}"
            )
            continue

        value = float(value)
        values[key] = value
        if value < 0 or (key in FRACTIONS and value > 1):
            physical_limit = "in [0, 1]" if key in FRACTIONS else "at least 0.0"
            reasons.append(f"{label} {_fmt(value)}{unit}, needs {physical_limit}")
            continue
        if key == "luma_std_over_time" and dynamic is True and screen:
            continue  # Require a finite measurement even when granting the exemption.

        margin = ((value - limit) if direction == "min" else (limit - value)) / limit
        if margin < 0:
            reasons.append(
                f"{label} {_fmt(value)}{unit}, needs {need} {_fmt(limit)}{unit}"
            )
        else:
            margins.append(0.5 + 0.5 * math.tanh(margin))

    for left, right in (("longest_run_s", "on_screen_s"),
                        ("track_residual_px_median", "track_residual_px_p90"),
                        ("mean_occlusion_frac", "max_occlusion_frac")):
        if left in values and right in values and values[left] > values[right]:
            reasons.append(
                f"{left} {_fmt(values[left])} exceeds {right} {_fmt(values[right])}"
            )

    if reasons:
        row["decision"] = "decline"
    else:
        row["score"] = math.exp(sum(math.log(x) for x in margins) / len(margins))
    return row


def _overlap(a, b):
    """IoU of coordinate-wise median quads over co-visible shared frames.

    None means unknown. Disjoint time spans or measured absence of
    co-visible frames prove zero overlap. Missing geometry never proves zero.
    """
    alo, ahi = _span(a)
    blo, bhi = _span(b)
    lo, hi = max(alo, blo), min(ahi, bhi)
    if lo > hi:
        return 0.0

    aq, bq = _quads(a), _quads(b)
    if aq is None or bq is None:
        return None
    shared = [i for i in range(lo, hi + 1)
              if aq[i] is not None and bq[i] is not None]
    if not shared:
        return 0.0

    try:
        qa = _quad(np.median([aq[i] for i in shared], axis=0))
        qb = _quad(np.median([bq[i] for i in shared], axis=0))
        intersection, _ = cv2.intersectConvexConvex(qa, qb)
        union = cv2.contourArea(qa) + cv2.contourArea(qb) - intersection
        if not math.isfinite(intersection) or not math.isfinite(union) or union <= 0:
            return None
        return float(np.clip(intersection / union, 0, 1))
    except (ValueError, TypeError, cv2.error):
        return None


def decide_all(candidates, cfg=DEFAULTS, max_per_shot=3):
    """Return input order; keep highest scores, breaking ties by candidate id.

    Suppress overlap before filling the shot quota so suppressed candidates
    do not consume places that another candidate could use.
    """
    limits = _config(cfg)
    if (not isinstance(max_per_shot, Integral) or isinstance(max_per_shot, bool)
            or max_per_shot < 0):
        raise ValueError("max_per_shot must be a nonnegative integer")

    candidates = list(candidates)
    rows = [decide(c, limits) for c in candidates]
    if any(isinstance(c, Mapping) and _unsafe(c) for c in candidates):
        for row in rows:
            reason = "scene_safe is False; no ads are allowed anywhere in this clip"
            if reason not in row["reasons"]:
                _decline(row, reason)
            else:
                row["decision"], row["score"] = "decline", 0.0
        return rows

    counts = Counter(r["id"] for r in rows if r["id"] is not None)
    for row in rows:
        if counts[row["id"]] > 1:
            _decline(row, "duplicate candidate id; placement identity is ambiguous")

    kept = {}
    order = sorted(
        (i for i, r in enumerate(rows) if r["decision"] == "accept"),
        key=lambda i: (-rows[i]["score"], rows[i]["id"]),
    )
    for i in order:
        c, row = candidates[i], rows[i]
        shot_kept = kept.setdefault(c["shot"], [])
        for j in shot_kept:
            overlap = _overlap(c, candidates[j])
            if overlap is None:
                _decline(
                    row,
                    f"overlap with {rows[j]['id']} unknown; needs shared-frame quads",
                )
                break
            if overlap > limits["overlap_iou"]:
                _decline(
                    row,
                    f"overlap with {rows[j]['id']} IoU {_fmt(overlap)}, "
                    f"needs at most {_fmt(limits['overlap_iou'])}",
                )
                break
        if row["decision"] == "accept":
            if len(shot_kept) >= max_per_shot:
                _decline(
                    row,
                    f"shot already has {len(shot_kept)} placements, limit {max_per_shot}",
                )
            else:
                shot_kept.append(i)
    return rows
