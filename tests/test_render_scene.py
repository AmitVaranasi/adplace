import numpy as np

from scripts.render_scene import manifest_id, reference_mask, static_occluder
from pipeline.maskstore import MaskStore, MaskWriter


class _Frames:
    w, h, n = 64, 48, 6


class _Occ:
    def __init__(self, masks):
        self.masks = masks

    def frame(self, i):
        return self.masks[i]


def _store(tmp_path, masks):
    w = MaskWriter(len(masks), *masks[0].shape)
    for i, m in enumerate(masks):
        w.add(i, mask=m)
    w.save(tmp_path / "s.npz")
    return MaskStore(tmp_path / "s.npz")


def test_votes_fill_a_moving_occluder_and_ignore_a_person_who_never_leaves(tmp_path):
    screen = np.zeros((48, 64), bool)
    screen[10:30, 10:50] = True
    masks, occ = [], []
    for i in range(6):
        m, o = screen.copy(), np.zeros_like(screen)
        o[15:40, 12 + 5 * i:18 + 5 * i] = True  # someone walking past
        o[20:40, 40:46] = True                   # someone standing still the whole time
        masks.append(m & ~o)
        occ.append(o)
    H = np.repeat(np.eye(3)[None], 6, axis=0)
    ref = reference_mask(_store(tmp_path, masks), H, _Frames(), occ=_Occ(occ), step=1)
    pad_y, pad_x = int(0.5 * 48), int(0.5 * 64)
    inner = ref[pad_y:pad_y + 48, pad_x:pad_x + 64]
    assert inner[10:30, 10:40].all()          # the walker's path is filled in
    assert not inner[20:30, 40:46].any()      # never seen unoccluded: no vote, not claimed as screen


def test_static_occluder_is_the_never_seen_part_of_the_fitted_quad():
    ref = np.zeros((60, 80), bool)
    ref[10:50, 10:70] = True
    ref[30:50, 35:41] = False  # a lamp post
    hole = static_occluder(ref, {"type": "quad", "quad": [[10, 10], [69, 10], [69, 49], [10, 49]]})
    assert hole[32:48, 36:40].all() and hole.sum() < 6 * 20 + 20


def test_manifest_id_depends_on_the_whole_assignment():
    inv = {"campaigns": [{"brand_id": b, "campaign_token": b + "-t"} for b in ("a", "b", "c")]}
    one = manifest_id("x.mp4", {"s1": "a", "s2": "b"}, inv)[0]
    assert one == manifest_id("x.mp4", {"s2": "b", "s1": "a"}, inv)[0]
    assert one != manifest_id("x.mp4", {"s1": "b", "s2": "a"}, inv)[0]
    assert one != manifest_id("x.mp4", {"s1": "a", "s2": "c"}, inv)[0]
