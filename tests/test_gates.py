import copy
import json
import math

import pytest

from pipeline.gates import DEFAULTS, RULES, decide, decide_all


def quad(x=10, width=60):
    return [[x, 20], [x + width, 20], [x + width, 80], [x, 80]]


def candidate(cid="a", *, shot=0, x=10, start=0, **metrics):
    m = {
        "on_screen_s": 5.0, "longest_run_s": 1.5,
        "track_residual_px_median": 1.0, "track_residual_px_p90": 2.5,
        "corner_jitter_px_p95": 1.0, "min_side_px": 96.0,
        "mean_area_frac": 0.02, "edge_clipped_fraction": 0.0,
        "max_occlusion_frac": 0.4, "mean_occlusion_frac": 0.1,
        "motion_blur_px_median": 2.0, "luma_std_over_time": 5.0, "uncovered_visible_s": 0.0,
        "dynamic_content": False, "muse_plausible": True, "scene_safe": True,
    }
    m.update(metrics)
    return {
        "id": cid, "shot": shot, "kind": "flat", "surface": "poster",
        "frames": [start, start + 59], "fps": 10, "frame_size": [320, 240],
        "metrics": m, "quads": [quad(x) for _ in range(60)],
    }


@pytest.mark.parametrize("key", RULES)
def test_inclusive_boundary_and_just_outside(key):
    limit = DEFAULTS[key]
    assert decide(candidate(**{key: limit}))["decision"] == "accept"
    direction = RULES[key][0]
    outside = math.nextafter(
        limit, -math.inf if direction == "min" else math.inf
    )
    row = decide(candidate(**{key: outside}))
    assert row["decision"] == "decline" and row["score"] == 0
    assert any(RULES[key][1] in r for r in row["reasons"])


@pytest.mark.parametrize("key", RULES)
@pytest.mark.parametrize(
    "bad", [None, math.nan, math.inf, -math.inf, 10 ** 400, True, "2.0"]
)
def test_unknown_or_malformed_measurements_fail(key, bad):
    row = decide(candidate(**{key: bad}))
    assert row["decision"] == "decline"
    assert row["score"] == 0 and row["reasons"]
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize(
    "key", list(RULES) + ["scene_safe", "muse_plausible", "dynamic_content"]
)
def test_missing_measurements_fail(key):
    c = candidate()
    del c["metrics"][key]
    assert decide(c)["decision"] == "decline"


def test_all_numeric_boundaries_together():
    c = candidate(**{key: DEFAULTS[key] for key in RULES})
    row = decide(c)
    assert row["decision"] == "accept"
    assert row["score"] == pytest.approx(0.5)


def test_score_is_bounded_monotone_and_inputs_are_unchanged():
    c = candidate()
    before = copy.deepcopy(c)
    low = decide(c)
    high = decide(candidate(on_screen_s=6.0))
    assert 0 <= low["score"] < high["score"] <= 1
    assert c == before
    assert low["reasons"] == []


def test_dynamic_screen_exemption_still_requires_finite_luma():
    c = candidate(dynamic_content=True, luma_std_over_time=50)
    assert decide(c)["decision"] == "decline"
    c["surface"] = "screen"
    assert decide(c)["decision"] == "accept"
    c["metrics"]["luma_std_over_time"] = math.nan
    assert decide(c)["decision"] == "decline"


@pytest.mark.parametrize("key", ["muse_plausible", "scene_safe", "dynamic_content"])
@pytest.mark.parametrize("bad", [None, "true", 1])
def test_boolean_facts_are_not_coerced(key, bad):
    assert decide(candidate(**{key: bad}))["decision"] == "decline"


def test_physical_ranges_and_inconsistent_metrics_fail():
    for changes in (
        {"mean_occlusion_frac": -0.1},
        {"mean_area_frac": 1.1},
        {"longest_run_s": 6.0},
        {"track_residual_px_median": 2.0, "track_residual_px_p90": 1.0},
        {"mean_occlusion_frac": 0.3, "max_occlusion_frac": 0.2},
    ):
        assert decide(candidate(**changes))["decision"] == "decline"


def test_unsafe_candidate_vetoes_other_shots_and_otherwise_ineligible_rows():
    rows = decide_all([
        candidate("safe"),
        candidate("unsafe", shot=1, scene_safe=False),
        candidate("brief", shot=2, on_screen_s=0.9),
    ])
    assert len(rows) == 3
    assert all(r["decision"] == "decline" and r["score"] == 0 for r in rows)
    assert all(
        any("scene_safe is False" in s for s in r["reasons"]) for r in rows
    )


def test_missing_safety_declines_without_claiming_known_unsafe_scene():
    c = candidate()
    del c["metrics"]["scene_safe"]
    rows = decide_all([c, candidate("other", shot=1)])
    assert [r["decision"] for r in rows] == ["decline", "accept"]


def test_overlap_keeps_better_candidate_and_output_order():
    rows = decide_all([
        candidate("lower", on_screen_s=3),
        candidate("higher", on_screen_s=5),
    ])
    assert [r["id"] for r in rows] == ["lower", "higher"]
    assert [r["decision"] for r in rows] == ["decline", "accept"]
    assert "IoU" in rows[0]["reasons"][0]


def test_overlap_boundary_is_inclusive():
    # Width 60, shift 40: intersection 20*60, union 100*60, IoU exactly 0.2.
    rows = decide_all([candidate("a", x=0), candidate("b", x=40)])
    assert all(r["decision"] == "accept" for r in rows)


def test_use_shared_frame_medians_and_align_nonzero_frame_origins():
    a = candidate("a", start=100)
    b = candidate("b", start=150)
    # Only 150..159 are shared. Overall medians would miss this overlap.
    b["quads"] = [quad(10)] * 10 + [quad(200)] * 50
    assert decide_all([a, b])[1]["decision"] == "decline"
    b["frames"] = [160, 219]
    assert all(r["decision"] == "accept" for r in decide_all([a, b]))


def test_nulls_prove_nonconcurrent_visibility_but_missing_quads_do_not():
    a, b = candidate("a"), candidate("b")
    a["quads"] = [quad()] * 30 + [None] * 30
    b["quads"] = [None] * 30 + [quad()] * 30
    assert all(r["decision"] == "accept" for r in decide_all([a, b]))

    b.pop("quads")
    rows = decide_all([a, b])
    assert rows[1]["decision"] == "decline"
    assert "unknown" in rows[1]["reasons"][0]
    assert decide_all([b])[0]["decision"] == "accept"


def test_cap_is_per_shot_by_score_and_suppressed_rows_do_not_use_quota():
    cs = [
        candidate(str(i), x=75 * i, on_screen_s=3 + i * 0.1)
        for i in range(4)
    ]
    rows = decide_all(cs + [candidate("other-shot", shot=1)])
    assert [r["decision"] for r in rows] == [
        "decline", "accept", "accept", "accept", "accept"
    ]

    rows = decide_all([
        candidate("a", on_screen_s=6),
        candidate("b", on_screen_s=5),
        candidate("c", x=150, on_screen_s=4),
    ], max_per_shot=2)
    assert [r["decision"] for r in rows] == ["accept", "decline", "accept"]
    assert decide_all([candidate()], max_per_shot=0)[0]["decision"] == "decline"


def test_ties_are_deterministic_and_duplicates_decline():
    rows = decide_all([candidate("z"), candidate("a")])
    assert [r["decision"] for r in rows] == ["decline", "accept"]
    assert all(
        r["decision"] == "decline"
        for r in decide_all([candidate(), candidate()])
    )


def test_invalid_geometry_and_unsupported_3d_decline():
    for bad in (
        [[[0, 0]] * 4] * 60,
        [quad()] * 59,
        [[[math.nan, 0], [60, 0], [60, 60], [0, 60]]] * 60,
    ):
        c = candidate()
        c["quads"] = bad
        assert decide(c)["decision"] == "decline"
    c = candidate()
    c["kind"] = "3d"
    assert decide(c)["decision"] == "decline"


def test_config_validation_and_partial_override():
    assert decide(
        candidate(on_screen_s=1.9), {"on_screen_s": 1.8}
    )["decision"] == "accept"
    for cfg in (
        {"scene_safe": False},
        {"on_screen_s": math.nan},
        {"typo": 1},
    ):
        with pytest.raises(ValueError):
            decide(candidate(), cfg)
    with pytest.raises(ValueError):
        decide_all([], max_per_shot=-1)
    assert decide_all([]) == []
