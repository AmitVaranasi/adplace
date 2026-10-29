import json
import math

import cv2
import numpy as np
import pytest

from eval.qa import judge, player_match, qa_placement, stability


def scene(n=5):
    frames = np.zeros((n, 128, 160, 3), np.uint8)
    placement = np.zeros((128, 160), bool)
    real = np.zeros_like(placement)
    placement[30:70, 20:60] = True
    real[30:70, 90:130] = True
    H = np.repeat(np.eye(3)[None], n, axis=0)
    return frames, H, placement, real


def test_static_scene_passes_and_one_pixel_jitter_fails():
    frames, H, placement, real = scene()
    static = frames.copy()
    static[:, placement] = 255
    assert stability(static, H, placement, real) == {
        "value": 0.0, "limit": 1.5, "pass": True
    }

    jitter = frames.copy()
    for i in range(len(jitter)):
        x = 20 + i % 2
        jitter[i, 30:70, x:x + 40] = 255
    assert stability(jitter, H, placement, real)["pass"] is False
    assert not frames.any()


@pytest.mark.parametrize("placed_change,expected", [(3, True), (4, False)])
def test_stability_ratio_boundary(placed_change, expected):
    frames, H, placement, real = scene()
    for i in range(len(frames)):
        frames[i, placement] = 20 + placed_change * i
        frames[i, real] = 20 + 2 * i
    result = stability(frames, H, placement, real)
    assert result["value"] == pytest.approx(placed_change / 2)
    assert result["pass"] is expected


def test_frame_to_reference_direction():
    frames, H, placement, real = scene()
    ref = frames[0].copy()
    ref[placement], ref[real] = 230, 120
    for i in range(len(frames)):
        forward = np.array([
            [1, 0, 2 * i], [0, 1, 0], [0, 0, 1]
        ], float)
        frames[i] = cv2.warpPerspective(ref, forward, (160, 128))
        H[i] = np.linalg.inv(forward)
    assert stability(frames, H, placement, real)["pass"] is True
    assert stability(frames, np.linalg.inv(H), placement, real)["pass"] is False


@pytest.mark.parametrize("fault", [
    "nan_H", "singular", "short_H", "unsupported", "empty",
    "small", "overlap", "short_video", "nan_pixels",
])
def test_stability_missing_evidence_fails(fault):
    frames, H, placement, real = scene()
    if fault == "nan_H":
        H[1, 0, 0] = math.nan
    elif fault == "singular":
        H[1] = 0
    elif fault == "short_H":
        H = H[:-1]
    elif fault == "unsupported":
        H[1, 0, 2] = 1000
    elif fault == "empty":
        placement[:] = False
    elif fault == "small":
        placement[:] = False
        placement[30:40, 20:30] = True
    elif fault == "overlap":
        real = placement.copy()
    elif fault == "short_video":
        frames, H = frames[:1], H[:1]
    else:
        frames = frames.astype(float)
        frames[0, 0, 0, 0] = math.nan
    assert stability(frames, H, placement, real) == {
        "value": None, "limit": 1.5, "pass": False
    }


def test_player_match_inside_only_and_worst_frame():
    frames, _, placement, _ = scene()
    player = frames.copy()
    player[:, ~placement] = 255
    assert player_match(player, frames, placement)["value"] == 99

    player[2, placement] = 255
    result = player_match(player, frames, placement)
    assert result["value"] == 0 and result["pass"] is False


@pytest.mark.parametrize("delta,expected", [(8, True), (9, False)])
def test_player_psnr_near_threshold(delta, expected):
    frames, _, placement, _ = scene()
    player = frames.copy()
    player[:, placement] = delta
    result = player_match(player, frames, placement)
    assert result["value"] == pytest.approx(
        10 * math.log10(255 ** 2 / delta ** 2)
    )
    assert result["pass"] is expected


def test_player_alignment_empty_masks_and_nan_fail():
    frames, _, placement, _ = scene()
    assert not player_match(frames[:-1], frames, placement)["pass"]

    masks = np.repeat(placement[None], len(frames), axis=0)
    assert player_match(frames, frames, masks)["pass"]
    masks[2] = False
    assert not player_match(frames, frames, masks)["pass"]

    bad = frames.astype(float)
    bad[1, 0, 0, 0] = math.nan
    assert not player_match(bad, frames, placement)["pass"]


def replies(values):
    values = iter(values)
    calls = []

    def chat(*args, **kwargs):
        calls.append((args, kwargs))
        return next(values)

    return chat, calls


def verdict(inserted=False, where=""):
    return json.dumps({
        "inserted": inserted, "confidence": 0.8,
        "where": where, "tell": "",
    })


@pytest.mark.parametrize("hits,expected", [(2, True), (3, False)])
def test_judge_spotted_fraction_boundary_and_blind_prompt(hits, expected):
    frames, _, _, _ = scene()
    answers = []
    for i in range(5):
        answers += (
            [verdict(True, "wall poster"), "yes"]
            if i < hits else [verdict()]
        )
    chat, calls = replies(answers)
    result = judge(frames, "SECRET_GROUND_TRUTH", chat=chat)
    assert result == {
        "value": hits / 5, "limit": 0.4, "pass": expected
    }
    for args, _ in calls:
        if isinstance(args[4], list):
            assert "SECRET_GROUND_TRUTH" not in json.dumps(args[4])


def test_judge_positive_on_wrong_object_is_not_a_detection():
    frames, _, _, _ = scene(n=1)
    chat, _ = replies([verdict(True, "unrelated window"), "no"])
    assert judge(frames, "poster", chat=chat)["value"] == 0


@pytest.mark.parametrize("answers", [
    ["not json"],
    [json.dumps({"inserted": None})],
    [json.dumps({
        "inserted": "false", "confidence": 0.8, "where": "", "tell": ""
    })],
    [json.dumps({
        "inserted": False, "confidence": math.nan, "where": "", "tell": ""
    })],
    [verdict(True, "")],
    [verdict(True, "poster"), "yesterday"],
    [verdict(True, "poster"), ""],
    [],
])
def test_malformed_or_failed_judge_is_unavailable(answers):
    frames, _, _, _ = scene(n=1)
    chat, _ = replies(answers)
    assert judge(frames, "poster", chat=chat) == {
        "value": None, "limit": 0.4, "pass": False
    }


def test_judge_no_key_empty_samples_and_missing_truth_fail_without_network(
    monkeypatch,
):
    frames, _, _, _ = scene()
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    assert not judge(frames, "poster")["pass"]

    for images, truth in (
        ([], "poster"), (frames, ""), (frames, ["poster"]),
    ):
        chat, calls = replies([])
        assert not judge(images, truth, chat=chat)["pass"]
        assert calls == []


def passing_checks():
    return {
        "stability": {"value": 1.5, "limit": 1.5, "pass": True},
        "player_match": {"value": 30.0, "limit": 30.0, "pass": True},
        "judge": {"value": 0.4, "limit": 0.4, "pass": True},
    }


def test_qa_summary_requires_all_checks_and_is_strict_json():
    row = qa_placement("c03", passing_checks())
    assert row["pass"] is True and row["reasons"] == []
    json.dumps(row, allow_nan=False)

    for bad in ({}, None, {"judge": passing_checks()["judge"]}):
        row = qa_placement("c03", bad)
        assert row["pass"] is False and row["reasons"]
        json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("field,value", [
    ("value", math.nan), ("value", math.inf), ("value", 10 ** 400),
    ("value", 2.0), ("pass", "true"), ("pass", False), ("limit", 999),
])
def test_qa_does_not_trust_a_forged_pass_flag(field, value):
    checks = passing_checks()
    checks["stability"][field] = value
    row = qa_placement("c03", checks)
    assert row["pass"] is False
    assert row["checks"]["stability"]["pass"] is False
    assert row["reasons"]
    json.dumps(row, allow_nan=False)
