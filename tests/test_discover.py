"""Synthetic-data tests for pipeline/discover.py (no files, no network)."""

import math

import numpy as np
import pytest

from pipeline.discover import eligible_pairs, fallback_choice, slot_metrics, validate_choice

W, H, FPS = 320, 240, 10

INVENTORY = {
    "campaigns": [
        {
            "brand_id": "brewlark",
            "campaign_token": "brewlark-2026-q4",
            "category": "coffee",
            "creatives": {
                "16x9": "assets/creatives/brewlark_16x9.png",
                "4x3": "assets/creatives/brewlark_4x3.png",
                "9x16": "assets/creatives/brewlark_9x16.png",
            },
            "eligible_surfaces": ["billboard", "poster", "screen", "sign"],
            "contextual_exclusions": ["children", "hospital"],
            "preference_tags": ["coffee", "food", "morning"],
        },
        {
            "brand_id": "kestrel",
            "campaign_token": "kestrel-2026-q4",
            "category": "sportswear",
            "creatives": {
                "16x9": "assets/creatives/kestrel_16x9.png",
                "4x3": "assets/creatives/kestrel_4x3.png",
                "9x16": "assets/creatives/kestrel_9x16.png",
            },
            "eligible_surfaces": ["billboard", "poster", "sign"],
            "contextual_exclusions": ["funeral"],
            "preference_tags": ["running", "fitness", "outdoors"],
        },
        {
            "brand_id": "paperwhale",
            "campaign_token": "paperwhale-2026-q4",
            "category": "books",
            "creatives": {
                "16x9": "assets/creatives/paperwhale_16x9.png",
                "4x3": "assets/creatives/paperwhale_4x3.png",
                "9x16": "assets/creatives/paperwhale_9x16.png",
            },
            "eligible_surfaces": ["billboard", "poster", "screen"],
            "contextual_exclusions": [],
            "preference_tags": ["reading", "learning", "quiet"],
        },
    ]
}


def make_quad(cx, cy, w, h):
    """Axis-aligned quad (TL, TR, BR, BL) centred on (cx, cy)."""
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


NAN_QUAD = np.full((4, 2), np.nan, dtype=np.float32)


def build_clip():
    """10-frame clip: 8 visible frames (one edge-clipped, one half-occluded), 2 lost."""
    centre = make_quad(160, 120, 80, 45)
    clipped = make_quad(300, 120, 80, 45)  # right corners reach past x = W
    quads = [centre, centre, centre, centre, clipped,
             NAN_QUAD, NAN_QUAD, centre, centre, centre]
    status = ["ok", "ok", "ok", "ok", "ok", "lost", "lost", "ok", "occluded", "ok"]
    occluder = np.zeros((H, W), dtype=bool)
    occluder[:, :160] = True  # covers the left half of the centred quad
    occluders = [None] * 10
    occluders[8] = occluder
    return np.stack(quads), status, occluders


def make_metrics(**overrides):
    """A healthy slot_metrics-shaped dict; tests override individual keys."""
    base = {"frames": 300, "visible_frames": 270, "on_screen_s": 9.0,
            "visible_fraction": 0.9, "mean_area_frac": 0.05, "aspect_ratio": 16 / 9,
            "edge_clipped_fraction": 0.0, "max_occlusion_frac": 0.1, "longest_run_s": 4.0}
    base.update(overrides)
    return base


def make_slot(slot_id, surface, metrics=None):
    return {"slot_id": slot_id, "surface": surface,
            "metrics": metrics if metrics is not None else make_metrics()}


def test_slot_metrics_lost_clipped_occluded():
    quads, status, occluders = build_clip()
    m = slot_metrics(quads, status, (W, H), FPS, occluder_masks=occluders)
    assert m["frames"] == 10
    assert m["visible_frames"] == 8  # the "occluded" frame still counts as visible
    assert m["visible_fraction"] == pytest.approx(0.8)
    assert m["on_screen_s"] == pytest.approx(0.8)
    assert m["mean_area_frac"] == pytest.approx(3600 / (W * H))
    assert m["aspect_ratio"] == pytest.approx(80 / 45)
    assert m["edge_clipped_fraction"] == pytest.approx(1 / 8)
    assert m["max_occlusion_frac"] == pytest.approx(0.5, abs=0.05)
    assert m["longest_run_s"] == pytest.approx(0.5)


def test_slot_metrics_without_masks_or_visibility():
    quads, status, _ = build_clip()
    assert slot_metrics(quads, status, (W, H), FPS)["max_occlusion_frac"] == 0.0
    lost = np.stack([NAN_QUAD] * 3)
    m = slot_metrics(lost, ["lost"] * 3, (W, H), FPS)
    assert m["visible_frames"] == 0
    assert m["visible_fraction"] == 0.0
    assert m["mean_area_frac"] == 0.0
    assert m["edge_clipped_fraction"] == 0.0
    assert m["longest_run_s"] == 0.0
    assert math.isnan(m["aspect_ratio"])


def test_surface_rule_excludes_campaigns():
    pairs = eligible_pairs([make_slot("s1", "screen")], INVENTORY)
    assert [p["brand_id"] for p in pairs] == ["brewlark", "paperwhale"]  # kestrel: no screens


def test_contextual_exclusions_rule():
    slots = [make_slot("s1", "billboard")]
    assert len(eligible_pairs(slots, INVENTORY, context_tags=("running",))) == 3
    pairs = eligible_pairs(slots, INVENTORY, context_tags=("hospital", "funeral"))
    assert [p["brand_id"] for p in pairs] == ["paperwhale"]
    assert len(eligible_pairs(slots, INVENTORY, context_tags=("children",))) == 2


def test_ratio_rule_and_tolerance():
    square = make_slot("s1", "billboard", make_metrics(aspect_ratio=1.0))
    assert eligible_pairs([square], INVENTORY) == []  # best fit 4x3: log(4/3) > log(1.25)
    pairs = eligible_pairs([square], INVENTORY, ratio_tolerance=0.5)
    assert [(p["brand_id"], p["creative_key"]) for p in pairs] == [
        ("brewlark", "4x3"), ("kestrel", "4x3"), ("paperwhale", "4x3")]
    assert pairs[0]["ratio_error"] == pytest.approx(math.log(4 / 3))


def test_min_on_screen_and_area_rules():
    brief = make_slot("s1", "billboard", make_metrics(on_screen_s=0.5))
    assert eligible_pairs([brief], INVENTORY) == []
    assert len(eligible_pairs([brief], INVENTORY, min_on_screen_s=0.4)) == 3
    tiny = make_slot("s1", "billboard", make_metrics(mean_area_frac=0.005))
    assert eligible_pairs([tiny], INVENTORY) == []
    assert len(eligible_pairs([tiny], INVENTORY, min_area_frac=0.001)) == 3


def test_pairs_sorted_and_fields():
    slots = [make_slot("s2", "poster"), make_slot("s1", "billboard")]
    pairs = eligible_pairs(slots, INVENTORY)
    keys = [(p["slot_id"], p["brand_id"]) for p in pairs]
    assert keys == sorted(keys)
    assert keys[0] == ("s1", "brewlark")
    assert set(pairs[0]) == {"slot_id", "brand_id", "campaign_token", "creative_key", "ratio_error"}
    assert pairs[0]["campaign_token"] == "brewlark-2026-q4"
    assert pairs[0]["creative_key"] == "16x9"
    assert pairs[0]["ratio_error"] == pytest.approx(0.0)


def test_creative_aspect_choice():
    for aspect, expected_key in [(16 / 9, "16x9"), (1.35, "4x3"), (0.5, "9x16")]:
        pairs = eligible_pairs(
            [make_slot("s1", "billboard", make_metrics(aspect_ratio=aspect))], INVENTORY)
        assert len(pairs) == 3
        assert {p["creative_key"] for p in pairs} == {expected_key}
    near_four_by_three = eligible_pairs(
        [make_slot("s1", "billboard", make_metrics(aspect_ratio=1.35))], INVENTORY)[0]
    assert near_four_by_three["ratio_error"] == pytest.approx(abs(math.log((4 / 3) / 1.35)))


def test_validate_choice_accepts_eligible_pair():
    eligible = eligible_pairs([make_slot("s1", "billboard")], INVENTORY)
    result = validate_choice({"slot_id": "s1", "brand_id": "kestrel"}, eligible)
    assert result["status"] == "ok"
    pair = next(p for p in eligible if p["brand_id"] == "kestrel")
    assert all(result[k] == pair[k] for k in pair)


def test_validate_choice_rejects_unknown_or_malformed():
    eligible = eligible_pairs([make_slot("s1", "billboard")], INVENTORY)
    for choice in [
        {"slot_id": "s1", "brand_id": "ghost"},        # unknown brand
        {"slot_id": "ghost", "brand_id": "brewlark"},  # unknown slot
        {"slot_id": 1, "brand_id": "brewlark"},        # non-string ids
        {"slot_id": "s1"},                             # missing field
        {},
        None,
        "s1",
        ["s1", "brewlark"],
    ]:
        result = validate_choice(choice, eligible)
        assert result["status"] == "no_placement"
        assert isinstance(result["reason"], str) and result["reason"]


def test_validate_choice_rejects_ineligible_combination():
    slots = [make_slot("scr", "screen"), make_slot("bil", "billboard")]
    eligible = eligible_pairs(slots, INVENTORY)
    ids = {(p["slot_id"], p["brand_id"]) for p in eligible}
    assert ("bil", "kestrel") in ids
    assert ("scr", "kestrel") not in ids  # kestrel is not eligible on screens
    result = validate_choice({"slot_id": "scr", "brand_id": "kestrel"}, eligible)
    assert result["status"] == "no_placement"


def test_validate_choice_none():
    eligible = eligible_pairs([make_slot("s1", "billboard")], INVENTORY)
    result = validate_choice({"none": True}, eligible)
    assert result["status"] == "no_placement"
    assert result["reason"]


def test_fallback_prefers_preference_overlap_over_prominence():
    hero = make_slot("hero", "sign", make_metrics(on_screen_s=10.0, mean_area_frac=0.4))
    side = make_slot("side", "screen", make_metrics(on_screen_s=1.5, mean_area_frac=0.02))
    eligible = eligible_pairs([hero, side], INVENTORY)
    assert {(p["slot_id"], p["brand_id"]) for p in eligible} == {
        ("hero", "brewlark"), ("hero", "kestrel"),
        ("side", "brewlark"), ("side", "paperwhale")}
    chosen = fallback_choice(eligible, [hero, side], INVENTORY, preferences=("reading", "quiet"))
    assert (chosen["slot_id"], chosen["brand_id"]) == ("side", "paperwhale")
    assert chosen["status"] == "ok"


def test_fallback_prefers_prominence_over_ratio_error():
    prominent = make_slot("a", "billboard",
                          make_metrics(on_screen_s=8.0, mean_area_frac=0.5, aspect_ratio=1.45))
    modest = make_slot("b", "billboard",
                      make_metrics(on_screen_s=2.0, mean_area_frac=0.05))
    eligible = eligible_pairs([prominent, modest], INVENTORY)
    chosen = fallback_choice(eligible, [prominent, modest], INVENTORY)
    assert chosen["slot_id"] == "a"  # prominence 4.0 beats 0.1 despite the worse ratio fit
    assert fallback_choice(eligible, [prominent, modest], INVENTORY) == chosen


def test_fallback_ratio_error_breaks_ties():
    inventory = {"campaigns": [
        {"brand_id": "aaa", "campaign_token": "aaa-t", "creatives": {"16x9": "a.png"},
         "eligible_surfaces": ["billboard"], "contextual_exclusions": [],
         "preference_tags": []},
        {"brand_id": "bbb", "campaign_token": "bbb-t", "creatives": {"4x3": "b.png"},
         "eligible_surfaces": ["billboard"], "contextual_exclusions": [],
         "preference_tags": []},
    ]}
    slot = make_slot("s1", "billboard", make_metrics(aspect_ratio=1.5))
    eligible = eligible_pairs([slot], inventory)
    assert {(p["brand_id"], p["creative_key"]) for p in eligible} == {("aaa", "16x9"),
                                                                      ("bbb", "4x3")}
    chosen = fallback_choice(eligible, [slot], inventory)
    assert chosen["brand_id"] == "bbb"  # |log(4/3 / 1.5)| < |log(16/9 / 1.5)|


def test_fallback_no_eligible_pairs():
    assert fallback_choice([], [], INVENTORY)["status"] == "no_placement"
