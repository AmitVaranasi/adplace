from pipeline import auction


def inv(*campaigns):
    return {"campaigns": [{"brand_id": b, "bid_cpm": bid, "preference_tags": tags, "eligible_surfaces": ["poster", "glass"],
                           "contextual_exclusions": excl} for b, bid, tags, excl in campaigns]}


INV = inv(("a", 6.0, ["coffee"], []), ("b", 8.0, ["running"], []), ("c", 4.0, ["reading"], ["bar"]))


def test_second_price_and_relevance():
    r = auction.run([{"id": "p", "surface": "poster", "brands": ["a", "b", "c"], "area": 1}], INV, ["coffee"])
    lot = r["placements"]["p"]
    # a: 6 x 2 = 12 beats b: 8 x 1 = 8; a pays the least that still wins: 8 / 2 = 4
    assert lot["winner"] == "a" and lot["price_cpm"] == 4.0
    assert [x["brand"] for x in lot["bidders"]] == ["a", "b", "c"]


def test_one_placement_per_brand_biggest_first():
    ps = [{"id": "small", "surface": "poster", "brands": ["a", "b"], "area": 0.1},
          {"id": "big", "surface": "poster", "brands": ["a", "b"], "area": 0.5}]
    r = auction.run(ps, INV, [])
    assert r["assignment"] == {"big": "b", "small": "a"}


def test_drinkware_near_the_camera_is_sold_first():
    # a glass on the table is in focus and legible; the bigger poster behind it is often soft
    ps = [{"id": "glass", "surface": "glass", "brands": ["a", "b"], "area": 0.1},
          {"id": "big", "surface": "poster", "brands": ["a", "b"], "area": 0.5}]
    r = auction.run(ps, INV, [])
    assert r["assignment"] == {"glass": "b", "big": "a"}


def test_exclusions_artwork_and_reserve():
    r = auction.run([{"id": "p", "surface": "poster", "brands": ["c"], "area": 1}], INV, ["reading"], ["bar"])
    assert r["assignment"] == {} and r["placements"]["p"]["winner"] is None
    r = auction.run([{"id": "p", "surface": "poster", "brands": ["c"], "area": 1}], INV, [])
    assert r["assignment"] == {"p": "c"} and r["placements"]["p"]["price_cpm"] == auction.RESERVE_CPM
    r = auction.run([{"id": "p", "surface": "poster", "brands": ["zzz"], "area": 1}], INV, [])
    assert r["assignment"] == {}


def test_surface_the_brand_does_not_run_on():
    i = inv(("a", 6.0, [], []), ("b", 5.0, [], []))   # b names glass, so glass is a surface campaigns choose
    i["campaigns"][0]["eligible_surfaces"] = ["poster"]
    assert auction.run([{"id": "p", "surface": "glass", "brands": ["a"], "area": 1}], i, [])["assignment"] == {}
    # a surface no campaign names (an A-frame sign, say) is open to every brand with artwork for it
    assert auction.run([{"id": "p", "surface": "aframe", "brands": ["a"], "area": 1}], i, [])["assignment"] == {"p": "a"}


def test_price_never_above_bid_with_jitter():
    for seed in range(50):
        r = auction.run([{"id": "p", "surface": "poster", "brands": ["a", "b", "c"], "area": 1}], INV, ["coffee"],
                        jitter=0.3, seed=seed)
        lot = r["placements"]["p"]
        w = lot["bidders"][0]
        assert auction.RESERVE_CPM <= lot["price_cpm"] <= w["bid_cpm"]


def test_placements_of_track():
    track = {"width": 100, "height": 100,
             "slots": [{"id": "s", "surface": "poster", "brands": {"a": {}},
                        "quads": [None, [[0, 0], [10, 0], [10, 10], [0, 10]]]}],
             "objects": [{"id": "g", "surface": "glass", "brands": {"b": {}}, "boxes": [[0, 0, 10, 20, 0, 0], None]}]}
    ps = {p["id"]: p for p in auction.placements_of(track)}
    assert ps["s"]["area"] == 0.01 and ps["s"]["brands"] == ["a"]
    assert ps["g"]["area"] == 0.01


def test_left_over_placement_goes_to_a_brand_already_shown():
    ps = [{"id": "big", "surface": "poster", "brands": ["a", "b"], "area": 0.5},
          {"id": "small", "surface": "poster", "brands": ["b"], "area": 0.1}]
    r = auction.run(ps, INV, [])
    assert r["assignment"] == {"big": "b", "small": "b"}


def test_a_brand_skips_a_drink_it_does_not_fit():
    inv = {"campaigns": [{"brand_id": "coffee", "bid_cpm": 9, "preference_tags": [], "eligible_surfaces": ["glass"],
                          "unsuitable_contents": ["juice"]},
                         {"brand_id": "fizz", "bid_cpm": 2, "preference_tags": [], "eligible_surfaces": ["glass"]}]}
    juice = [{"id": "g", "surface": "glass", "contents": "juice", "brands": ["coffee", "fizz"], "area": 0.1}]
    assert auction.run(juice, inv, [])["assignment"] == {"g": "fizz"}
    latte = [{**juice[0], "contents": "latte"}]
    assert auction.run(latte, inv, [])["assignment"] == {"g": "coffee"}


def test_the_winner_says_which_interests_it_matched():
    r = auction.run([{"id": "p", "surface": "poster", "brands": ["a", "b", "c"], "area": 1}], INV, ["coffee"])
    lot = r["placements"]["p"]
    assert lot["matched"] == sorted({"coffee"} & set(next(c for c in INV["campaigns"]
                                                         if c["brand_id"] == lot["winner"])["preference_tags"]))
