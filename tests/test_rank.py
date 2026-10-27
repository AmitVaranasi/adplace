"""Tests for pipeline/rank.py with a fake chat function (no network)."""

import json

from pipeline import rank

INVENTORY = {"campaigns": [
    {"brand_id": "brewlark", "campaign_token": "brewlark-2026-q4", "category": "coffee",
     "preference_tags": ["coffee", "morning"]},
    {"brand_id": "kestrel", "campaign_token": "kestrel-2026-q4", "category": "sportswear",
     "preference_tags": ["running", "fitness"]},
]}
SLOTS = [{"slot_id": "clip-1", "surface": "billboard", "metrics": {"on_screen_s": 5.0, "mean_area_frac": 0.1}}]
ELIGIBLE = [{"slot_id": "clip-1", "brand_id": "brewlark", "ratio_error": 0.01},
            {"slot_id": "clip-1", "brand_id": "kestrel", "ratio_error": 0.01}]
RUNNER = {"profile_id": "viewer-b", "preferences": ["running"]}
P1 = [("muse", "http://muse", "k", "muse-x")]
P2 = [("muse", "http://muse", "k", "muse-x"), ("llama", "http://llama", "k", "llama-x")]


def fake(replies):
    """A chat() stand-in returning replies per base_url; an Exception value is raised."""
    calls = []

    def call(base_url, key, model, system, user):
        calls.append((base_url, user))
        r = replies[base_url]
        if isinstance(r, Exception):
            raise r
        return r
    return call, calls


def run(replies, providers=P1):
    call, calls = fake(replies)
    return rank.choose(SLOTS, ELIGIBLE, INVENTORY, RUNNER, ["night"], provider_list=providers, call=call,
                       use_cache=False), calls


def test_valid_model_choice_is_used():
    out, _ = run({"http://muse": '{"slot_id": "clip-1", "brand_id": "kestrel", "rationale": "For runners."}'})
    assert out["choice"]["status"] == "ok" and out["choice"]["brand_id"] == "kestrel"
    assert out["choice"]["rationale"] == "For runners."
    assert out["chooser"].startswith("muse")


def test_json_inside_prose_is_parsed():
    out, _ = run({"http://muse": 'Sure!\n```json\n{"slot_id": "clip-1", "brand_id": "brewlark", "rationale": "x"}\n```'})
    assert out["choice"]["brand_id"] == "brewlark"


def test_invented_id_is_rejected_and_next_provider_used():
    out, _ = run({"http://muse": '{"slot_id": "clip-1", "brand_id": "acme", "rationale": "x"}',
                  "http://llama": '{"slot_id": "clip-1", "brand_id": "kestrel", "rationale": "y"}'}, P2)
    assert out["choice"]["brand_id"] == "kestrel" and out["chooser"].startswith("llama")
    assert out["attempts"][0]["result"].startswith("rejected")


def test_errors_and_garbage_fall_back_to_rules():
    out, _ = run({"http://muse": TimeoutError(), "http://llama": "no json here"}, P2)
    assert out["chooser"].startswith("rules")
    assert out["choice"]["brand_id"] == "kestrel"  # runner preference wins in the fallback too
    assert [a["result"].split(":")[0] for a in out["attempts"]] == ["error", "rejected"]


def test_model_may_choose_no_placement():
    out, _ = run({"http://muse": '{"none": true, "rationale": "Nothing fits."}'})
    assert out["choice"]["status"] == "no_placement" and out["choice"]["reason"] == "Nothing fits."


def test_no_eligible_pairs_asks_no_model():
    call, calls = fake({})
    out = rank.choose(SLOTS, [], INVENTORY, RUNNER, provider_list=P1, call=call, use_cache=False)
    assert out["choice"]["status"] == "no_placement" and calls == []


def test_prompt_has_only_eligible_campaigns_and_no_tokens():
    prompt = rank.build_prompt(SLOTS, ELIGIBLE[:1], INVENTORY, RUNNER, ["night"])
    data = json.loads(prompt)
    assert [c["brand_id"] for c in data["campaigns"]] == ["brewlark"]
    assert "2026-q4" not in prompt


def test_long_rationale_is_trimmed():
    out, _ = run({"http://muse": json.dumps({"slot_id": "clip-1", "brand_id": "kestrel", "rationale": "a " * 200})})
    assert len(out["choice"]["rationale"]) <= rank.MAX_RATIONALE
