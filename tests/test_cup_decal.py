import cv2
import numpy as np

from scripts.glass_decal import HandleFree, glass_geometry


class Store:
    """Minimal MaskStore stand-in: one frame."""
    def __init__(self, m):
        self.m, self.n = m, 1

    def any(self, i):
        return bool(self.m.any())

    def box(self, i):
        ys, xs = np.nonzero(self.m)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def __getitem__(self, i):
        return self.m


def mug(handle=True):
    m = np.zeros((240, 320), np.uint8)
    cv2.rectangle(m, (120, 60), (200, 180), 1, -1)                       # body: 80 wide, 120 tall
    if handle:
        cv2.ellipse(m, (205, 120), (28, 34), 0, -90, 90, 1, 7)          # a loop on the right, 7 px thick
    return m.astype(bool)


def test_handle_removed_body_kept():
    plain, handled = mug(False), mug(True)
    out = HandleFree(Store(handled))[0]
    assert out.sum() >= 0.97 * plain.sum() and (out & ~plain).sum() < 0.02 * plain.sum()
    g, g0 = glass_geometry(out, 0.8), glass_geometry(plain, 0.8)
    assert abs(g[2] - g0[2]) < 2 and abs(g[0] - g0[0]) < 2 and abs(g[3]) < 0.02   # radius, centre, upright
    x0, y0, x1, y1 = HandleFree(Store(handled)).box(0)
    assert x1 <= 202


def test_squat_cup_accepted_only_with_cup_ratio():
    m = np.zeros((200, 300), np.uint8)
    cv2.rectangle(m, (100, 80), (200, 140), 1, -1)                       # 100 wide, 60 tall: a teacup
    assert glass_geometry(m.astype(bool)) is None
    assert glass_geometry(m.astype(bool), 0.8) is not None


def test_handle_cut_out_where_sam_outlined_it():
    plain = mug(False)
    lobe = np.zeros_like(plain, np.uint8)
    cv2.ellipse(lobe, (215, 120), (22, 34), 0, 0, 360, 1, -1)        # a handle whose hole SAM filled in
    lobe[:, :201] = 0
    handled = plain | lobe.astype(bool)
    assert HandleFree(Store(handled))[0].sum() > 1.1 * plain.sum()    # too solid for the opening to remove
    out = HandleFree(Store(handled), [Store(lobe.astype(bool))])[0]
    assert (out & ~plain).sum() == 0 and out.sum() >= 0.97 * plain.sum()
