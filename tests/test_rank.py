"""Tests for pipeline/rank.py with a fake chat function (no network), including scene-image support."""

import base64
import hashlib
import json

import cv2
import numpy as np

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
PROMPT = rank.build_prompt(SLOTS, ELIGIBLE, INVENTORY, RUNNER, ["night"])
REPLY = '{"slot_id": "clip-1", "brand_id": "kestrel", "rationale": "For runners."}'
REJECTED = '{"slot_id": "clip-1", "brand_id": "acme", "rationale": "x"}'
IMG = b"\xff\xd8\xe0\x00scene-bytes-that-are-only-base64d-here"
P_VISION = [("muse", "http://muse", "k", "muse-x", None, True)]
P_MIXED = [("muse", "http://muse", "k", "muse-x", None, True),
           ("llama", "http://llama", "k", "llama-x", {"temperature": 0}, False)]


def fake(replies):
    """A chat() stand-in returning replies per base_url; an Exception value is raised."""
    calls = []

    def call(base_url, key, model, system, user, extra=None):
        calls.append((base_url, system, user))
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


def test_image_goes_to_vision_provider_as_data_url():
    call, calls = fake({"http://muse": REPLY})
    out = rank.choose(SLOTS, ELIGIBLE, INVENTORY, RUNNER, ["night"], provider_list=P_VISION, call=call,
                      use_cache=False, image=IMG)
    assert out["choice"]["status"] == "ok" and len(calls) == 1
    system, user = calls[0][1], calls[0][2]
    url = "data:image/jpeg;base64," + base64.b64encode(IMG).decode("ascii")
    assert user == [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": url}}]
    assert system.startswith(rank.SYSTEM) and "green" in system[len(rank.SYSTEM):]


def test_text_only_provider_gets_plain_string_even_with_image():
    call, calls = fake({"http://muse": TimeoutError(), "http://llama": REPLY})
    out = rank.choose(SLOTS, ELIGIBLE, INVENTORY, RUNNER, ["night"], provider_list=P_MIXED, call=call,
                      use_cache=False, image=IMG)
    assert out["chooser"].startswith("llama") and len(calls) == 2
    assert isinstance(calls[0][2], list)  # the vision provider got the image parts
    assert calls[1][1:] == (rank.SYSTEM, PROMPT)  # the text-only provider got the plain prompt, unmodified system


def test_four_tuple_providers_stay_text_only():
    call, calls = fake({"http://muse": REPLY})
    out = rank.choose(SLOTS, ELIGIBLE, INVENTORY, RUNNER, ["night"], provider_list=P1, call=call, use_cache=False,
                      image=IMG)
    assert out["choice"]["status"] == "ok" and len(calls) == 1
    assert calls[0][1:] == (rank.SYSTEM, PROMPT)


def test_chat_passes_list_content_through(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "fine"}}]}).encode()

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return FakeResponse()

    monkeypatch.setattr(rank.urllib.request, "urlopen", fake_urlopen)
    url = "data:image/jpeg;base64," + base64.b64encode(IMG).decode("ascii")
    content = [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": url}}]
    assert rank.chat("http://muse", "k", "muse-x", rank.SYSTEM, content, extra={"temperature": 0}) == "fine"
    assert captured["body"]["messages"][1]["content"] == content
    assert captured["body"]["temperature"] == 0


def _has_green_near(img, x, y, radius=2):
    """A green-dominant pixel within `radius` of (x, y); JPEG and downscaling soften the outline a little."""
    h, w = img.shape[:2]
    win = img[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)].astype(int)
    return bool(((win[:, :, 1] > 100) & (win[:, :, 1] > win[:, :, 0] + 30)
                 & (win[:, :, 1] > win[:, :, 2] + 30)).any())


def test_scene_image_is_jpeg_capped_and_outlines_the_quad():
    frame = np.full((240, 320, 3), 60, np.uint8)
    quad = [[60.4, 50.2], [259.6, 50.7], [260.1, 189.8], [59.9, 190.3]]
    data = rank.scene_image(frame, quad, max_side=160)
    assert isinstance(data, bytes) and data[:2] == b"\xff\xd8"
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    h, w = img.shape[:2]
    assert max(h, w) <= 160 and (h, w) == (120, 160)
    assert (frame == 60).all()  # the caller's frame is not modified
    for (x1, y1), (x2, y2) in zip(quad, quad[1:] + quad[:1]):
        for t in (0.2, 0.5, 0.8):
            assert _has_green_near(img, round((x1 + (x2 - x1) * t) * w / 320.0),
                                       round((y1 + (y2 - y1) * t) * h / 240.0))


def test_scene_image_keeps_small_frames_unscaled():
    frame = np.full((120, 160, 3), (40, 40, 200), np.uint8)
    quad = np.array([[20.0, 10.0], [140.0, 10.0], [140.0, 110.0], [20.0, 110.0]])
    data = rank.scene_image(frame, quad)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[:2] == (120, 160)
    for x, y in [(80, 10), (140, 60), (80, 110), (20, 60)]:
        assert _has_green_near(img, x, y)


def test_cache_key_without_image_is_the_legacy_one(tmp_path, monkeypatch):
    monkeypatch.setattr(rank, "CACHE", tmp_path)
    call, _ = fake({"http://muse": REPLY})
    out = rank.choose(SLOTS, ELIGIBLE, INVENTORY, RUNNER, ["night"], provider_list=P1, call=call, use_cache=True)
    assert out["attempts"][0]["result"] == "used"
    legacy = "muse-" + hashlib.sha256(
        json.dumps(["muse", "muse-x", rank.SYSTEM, PROMPT]).encode()).hexdigest()[:16] + ".txt"
    assert {p.name for p in tmp_path.iterdir()} == {legacy}


def test_cache_key_tracks_the_image(tmp_path, monkeypatch):
    monkeypatch.setattr(rank, "CACHE", tmp_path)
    call, _ = fake({"http://muse": REJECTED, "http://llama": REPLY})

    def go(image):
        before = {p.name for p in tmp_path.iterdir()}
        out = rank.choose(SLOTS, ELIGIBLE, INVENTORY, RUNNER, ["night"], provider_list=P_MIXED, call=call,
                          use_cache=True, image=image)
        return out, {p.name for p in tmp_path.iterdir()} - before

    out0, new0 = go(None)
    assert out0["chooser"].startswith("llama") and len(new0) == 2
    legacy_llama = "llama-" + hashlib.sha256(
        json.dumps(["llama", "llama-x", rank.SYSTEM, PROMPT]).encode()).hexdigest()[:16] + ".txt"
    assert legacy_llama in new0
    out1, new1 = go(b"image-one")
    out2, new2 = go(b"image-two")
    assert out1["choice"]["status"] == "ok" and out2["choice"]["status"] == "ok"
    assert len(new1) == 1 and len(new2) == 1 and not (new1 & new2)  # image <-> no image and image <-> image
    assert out1["attempts"][1]["result"] == "used (cached)"  # llama still hits its old image-free entry
    out_again, new_again = go(b"image-one")
    assert not new_again and out_again["attempts"][1]["result"] == "used (cached)"  # same image -> same key


# --- Joint assignment tests ---

A_SLOTS = [
    {"slot_id": "s1", "surface": "screen", "area_frac": 0.20, "neighbours": ["s2"]},
    {"slot_id": "s2", "surface": "screen", "area_frac": 0.15, "neighbours": ["s1", "s3"]},
    {"slot_id": "s3", "surface": "screen", "area_frac": 0.05, "neighbours": ["s2"]},
]
A_INV = {"campaigns": [
    {"brand_id": "brewlark", "category": "coffee", "eligible_surfaces": ["screen", "billboard"],
     "preference_tags": ["coffee"]},
    {"brand_id": "cafeluna", "category": "coffee", "eligible_surfaces": ["screen", "billboard"],
     "preference_tags": ["coffee"]},
    {"brand_id": "kestrel", "category": "sportswear", "eligible_surfaces": ["screen", "billboard"],
     "preference_tags": ["running"]},
    {"brand_id": "wallco", "category": "tools", "eligible_surfaces": ["billboard"],
     "preference_tags": ["tools"]},
]}
A_PROFILE = {"profile_id": "viewer-b", "preferences": ["coffee", "running"]}
A_VALID = {"assignment": {"s1": "brewlark", "s2": "kestrel"}, "rationale": "Coffee plus running."}


def fake_seq(map_url_to_replies):
    calls = []
    counters = {k: 0 for k in map_url_to_replies}

    def call(base_url, key, model, system, user, extra=None):
        calls.append((base_url, system, user))
        seq = map_url_to_replies[base_url]
        if isinstance(seq, list):
            r = seq[min(counters[base_url], len(seq) - 1)]
            counters[base_url] += 1
        else:
            r = seq
        if isinstance(r, Exception):
            raise r
        return r
    return call, calls


def arun(seq, max_ads=3, providers=P1):
    call, calls = fake_seq(seq)
    out = rank.choose_assignment(A_SLOTS, A_INV, A_PROFILE, ("night",), max_ads=max_ads,
                                 provider_list=providers, call=call, use_cache=False)
    return out, calls


def test_assignment_valid_model_answer_is_accepted():
    out, calls = arun({"http://muse": json.dumps(A_VALID)})
    assert out["assignment"] == A_VALID["assignment"] and out["chooser"].startswith("muse")
    assert out["attempts"][0]["result"].startswith("used") and len(calls) == 1


def test_assignment_duplicate_brand_rejected_and_retry_used():
    dup = json.dumps({"assignment": {"s1": "brewlark", "s2": "brewlark"}, "rationale": "dup"})
    out, calls = arun({"http://muse": [dup, json.dumps(A_VALID)]})
    assert len(calls) == 2 and "violations" in str(calls[1][2]).lower()
    assert out["assignment"] == A_VALID["assignment"]
    assert out["attempts"][0]["result"].startswith("rejected") and out["attempts"][1]["result"].startswith("used")


def test_assignment_ineligible_surface_rejected_and_retry_used():
    bad = json.dumps({"assignment": {"s1": "wallco"}, "rationale": "bad surface"})
    out, calls = arun({"http://muse": [bad, json.dumps(A_VALID)]})
    assert len(calls) == 2 and out["assignment"] == A_VALID["assignment"]
    assert out["attempts"][0]["result"].startswith("rejected")


def test_assignment_adjacent_same_category_rejected_and_retry_used():
    bad = json.dumps({"assignment": {"s1": "brewlark", "s2": "cafeluna"}, "rationale": "two coffees"})
    out, calls = arun({"http://muse": [bad, json.dumps(A_VALID)]})
    assert len(calls) == 2 and out["assignment"] == A_VALID["assignment"]
    assert "share category" in out["attempts"][0]["result"]


def test_assignment_twice_invalid_falls_back_to_rule():
    bad1 = json.dumps({"assignment": {"s1": "brewlark", "s2": "brewlark"}, "rationale": "dup"})
    bad2 = json.dumps({"assignment": {"s1": "wallco"}, "rationale": "bad"})
    out, calls = arun({"http://muse": [bad1, bad2]})
    assert len(calls) == 2 and out["chooser"] == "rule"
    assert rank.validate_assignment(out["assignment"], A_SLOTS, A_INV, 3) == []
    assert len(out["assignment"]) <= 3 and len(set(out["assignment"].values())) == len(out["assignment"])


def test_assignment_max_ads_respected():
    too_many = json.dumps({"assignment": {"s1": "brewlark", "s2": "kestrel"}, "rationale": "two"})
    ok_one = json.dumps({"assignment": {"s1": "brewlark"}, "rationale": "one"})
    out, _ = arun({"http://muse": [too_many, ok_one]}, max_ads=1)
    assert len(out["assignment"]) <= 1
    call, _ = fake_seq({})
    out2 = rank.choose_assignment(A_SLOTS, A_INV, A_PROFILE, max_ads=1, provider_list=[], call=call,
                                  use_cache=False)
    assert out2["chooser"] == "rule" and len(out2["assignment"]) <= 1
    assert rank.validate_assignment(out2["assignment"], A_SLOTS, A_INV, 1) == []


def test_assignment_no_provider_uses_rule():
    call, calls = fake_seq({})
    out = rank.choose_assignment(A_SLOTS, A_INV, A_PROFILE, provider_list=[], call=call, use_cache=False)
    assert out["chooser"] == "rule" and calls == []
    assert rank.validate_assignment(out["assignment"], A_SLOTS, A_INV, 3) == []
