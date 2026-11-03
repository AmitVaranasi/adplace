from pipeline import creator

PS = [{"id": "c01", "surface": "poster", "brands": ["a", "b"], "area": 0.5},
      {"id": "c03", "surface": "glass", "brands": ["a", "c"], "area": 0.1}]


def test_off_until_the_creator_turns_it_on(tmp_path):
    creator.start_off(tmp_path)
    s = creator.load(tmp_path)
    assert s["enabled"] is False and creator.allowed(PS, s) == []
    creator.save(tmp_path, {"enabled": True})
    creator.start_off(tmp_path)   # a re-run keeps the creator's choice
    assert creator.load(tmp_path)["enabled"] is True


def test_no_file_is_off(tmp_path):
    assert creator.load(tmp_path)["enabled"] is False and creator.allowed(PS, creator.load(tmp_path)) == []


def test_places_and_brands_the_creator_turned_down(tmp_path):
    creator.save(tmp_path, {"enabled": True, "places": {"c03": False}, "brands": {"b": False}})
    assert creator.allowed(PS, creator.load(tmp_path)) == [{**PS[0], "brands": ["a"]}]
