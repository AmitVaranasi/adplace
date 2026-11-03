from scripts.glass_decal import print_sigma
from scripts.ingest import LOGO_PX_MIN, decide_glass

OK = {"on_screen_s": 10.0, "longest_run_s": 10.0, "radius_px_median": 50.0, "shape_fit_fraction": 1.0,
      "print_on_glass_fraction": 1.0, "uncovered_visible_s": 0.0, "blank_patch": True, "travel_r_max": 0.0,
      "muse_plausible": True, "scene_safe": True, "logo_px": 43.0, "print_sigma": 0.5}


def decide(**m):
    return decide_glass({"id": "c01", "surface": "glass", "metrics": {**OK, **m}})


def test_a_sharp_readable_logo_passes():
    assert decide()["decision"] == "accept"


def test_a_logo_too_small_on_screen_is_declined():
    d = decide(logo_px=LOGO_PX_MIN - 10)
    assert d["decision"] == "decline" and d["reasons"][0].startswith("logo too small")


def test_a_logo_on_an_out_of_focus_bowl_is_declined():
    d = decide(logo_px=40.0, print_sigma=2.1)   # the m43999 ramekin
    assert d["decision"] == "decline" and d["reasons"][0].startswith("logo out of focus")


def test_print_blur_follows_focus():
    assert print_sigma(1.0) == 0.5 and abs(print_sigma(2.2) - 2.2) < 1e-9   # half when sharp, all when soft
