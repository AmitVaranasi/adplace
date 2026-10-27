"""Deterministic slot facts and inventory filtering (spec S6).

The language model (Muse) only ranks pairs this module marks eligible, and
its answer is validated here. No model calls in this module.
"""

import math

import cv2
import numpy as np


def _quad_area(quad):
    """Absolute shoelace area of one (4, 2) quad."""
    x = quad[:, 0].astype(np.float64)
    y = quad[:, 1].astype(np.float64)
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _aspect_ratio(quad):
    """Width/height from the mean lengths of the two opposite edge pairs."""
    tl, tr, br, bl = (quad[i].astype(np.float64) for i in range(4))
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    return float(width / height) if height > 0 else float("nan")


def _is_clipped(quad, frame_w, frame_h):
    """True if any corner lies outside the frame extent."""
    x, y = quad[:, 0], quad[:, 1]
    outside = (x < 0) | (x > frame_w) | (y < 0) | (y > frame_h)
    return bool(outside.any())


def _max_occlusion_frac(quads, vis_idx, occluder_masks):
    """Max over visible frames of (occluder pixels inside the quad) / (quad pixels)."""
    if occluder_masks is None:
        return 0.0
    best = 0.0
    for i in vis_idx:
        if i >= len(occluder_masks):
            break
        mask = occluder_masks[i]
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=bool)
        canvas = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(canvas, [np.round(quads[i]).astype(np.int32)], 1)
        inside = canvas.astype(bool)
        area = int(inside.sum())
        if area:
            best = max(best, float((mask & inside).sum()) / area)
    return best


def slot_metrics(quads, status, frame_size, fps, occluder_masks=None) -> dict:
    """Facts about one tracked surface; keys and semantics per the module contract."""
    quads = np.asarray(quads, dtype=np.float32).reshape(-1, 4, 2)
    status = list(status)
    n = quads.shape[0]
    w, h = (float(v) for v in frame_size)

    # A quad is visible when it is tracked (not lost) and finite; "occluded"
    # frames keep their quad and still count as visible.
    visible = (np.array([s != "lost" for s in status], dtype=bool)
               & np.isfinite(quads).all(axis=(1, 2)))
    vis_idx = np.flatnonzero(visible)
    n_vis = len(vis_idx)

    longest = run = 0
    for v in visible:
        run = run + 1 if v else 0
        longest = max(longest, run)

    areas = [_quad_area(quads[i]) for i in vis_idx]
    ratios = [r for r in (_aspect_ratio(quads[i]) for i in vis_idx) if math.isfinite(r)]
    clipped = sum(_is_clipped(quads[i], w, h) for i in vis_idx)

    return {
        "frames": n,
        "visible_frames": int(n_vis),
        "on_screen_s": n_vis / float(fps),
        "visible_fraction": (n_vis / n) if n else 0.0,
        "mean_area_frac": (float(np.mean(areas)) / (w * h)) if areas else 0.0,
        "aspect_ratio": float(np.median(ratios)) if ratios else float("nan"),
        "edge_clipped_fraction": (clipped / n_vis) if n_vis else 0.0,
        "max_occlusion_frac": _max_occlusion_frac(quads, vis_idx, occluder_masks),
        "longest_run_s": longest / float(fps),
    }


def _creative_ratio(key):
    """Parse a creative key like "16x9" into a width/height ratio; None if unusable."""
    try:
        left, right = str(key).lower().split("x")
        ratio = float(left) / float(right)
    except (ValueError, ZeroDivisionError):
        return None
    return ratio if ratio > 0 and math.isfinite(ratio) else None


def _best_creative(creatives, surface_ratio):
    """(key, ratio_error) of the creative closest to the surface aspect; (None, inf)."""
    best_key, best_err = None, math.inf
    if not (math.isfinite(surface_ratio) and surface_ratio > 0):
        return best_key, best_err
    for key in sorted(creatives):
        ratio = _creative_ratio(key)
        if ratio is None:
            continue
        err = abs(math.log(ratio / surface_ratio))
        if err < best_err:
            best_key, best_err = key, err
    return best_key, best_err


def eligible_pairs(slots, inventory, context_tags=(), ratio_tolerance=0.25,
                   min_on_screen_s=1.0, min_area_frac=0.01) -> list[dict]:
    """Allowed (slot, campaign) pairs, sorted by slot_id then brand_id."""
    context = set(context_tags)
    max_ratio_error = math.log(1.0 + ratio_tolerance)
    pairs = []
    for slot in slots:
        m = slot["metrics"]
        if m["on_screen_s"] < min_on_screen_s or m["mean_area_frac"] < min_area_frac:
            continue
        for campaign in inventory.get("campaigns") or []:
            if slot["surface"] not in (campaign.get("eligible_surfaces") or []):
                continue
            if context & set(campaign.get("contextual_exclusions") or []):
                continue
            key, err = _best_creative(campaign.get("creatives") or {}, m["aspect_ratio"])
            if key is None or err > max_ratio_error:
                continue
            pairs.append({"slot_id": slot["slot_id"],
                          "brand_id": campaign["brand_id"],
                          "campaign_token": campaign["campaign_token"],
                          "creative_key": key,
                          "ratio_error": err})
    pairs.sort(key=lambda p: (p["slot_id"], p["brand_id"]))
    return pairs


def validate_choice(choice, eligible) -> dict:
    """Validate a model choice against `eligible`; model IDs are never trusted."""
    if isinstance(choice, dict) and choice.get("none"):
        return {"status": "no_placement", "reason": "model chose no placement"}
    if (not isinstance(choice, dict)
            or not isinstance(choice.get("slot_id"), str)
            or not isinstance(choice.get("brand_id"), str)):
        return {"status": "no_placement", "reason": "malformed choice"}
    for pair in eligible:
        if pair["slot_id"] == choice["slot_id"] and pair["brand_id"] == choice["brand_id"]:
            return {**pair, "status": "ok"}
    return {"status": "no_placement",
            "reason": f"({choice['slot_id']!r}, {choice['brand_id']!r}) is not an eligible pair"}


def fallback_choice(eligible, slots, inventory, preferences=()) -> dict:
    """Deterministic ranking: preference overlap, then prominence, then ratio error."""
    if not eligible:
        return {"status": "no_placement", "reason": "no eligible pairs"}
    metrics = {s["slot_id"]: s["metrics"] for s in slots}
    campaigns = {c["brand_id"]: c for c in inventory.get("campaigns") or []}
    prefs = set(preferences)

    def rank(pair):
        m = metrics.get(pair["slot_id"], {})
        campaign = campaigns.get(pair["brand_id"], {})
        overlap = len(prefs & set(campaign.get("preference_tags") or ()))
        prominence = float(m.get("on_screen_s", 0.0)) * float(m.get("mean_area_frac", 0.0))
        # Negated leading terms let min() pick the highest score; the trailing
        # ids keep ties deterministic.
        return (-overlap, -prominence, pair["ratio_error"], pair["slot_id"], pair["brand_id"])

    return {**min(eligible, key=rank), "status": "ok"}
