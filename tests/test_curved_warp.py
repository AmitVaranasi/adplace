"""Synthetic inverse-cylinder mapping and compositing acceptance tests."""
import cv2
import numpy as np
from pipeline.curved_warp import composite_curved, warp_to_cylinder

def _cyl():
    rows = np.arange(40, 161, dtype=np.int32)
    return dict(a=0.0, b=100.0, y0=40, y1=160, rows=rows, half_width=np.full(len(rows), 50.0))

def _cyl_curved(e=12.0):
    c = _cyl()
    c["e0"] = float(e)
    c["e1"] = float(e)
    c["sag_top"] = float(e)
    c["sag_bottom"] = float(e)
    return c

def _creative(rgb=(180, 100, 40), height=31, width=201):
    image = np.empty((height, width, 4), np.uint8)
    image[..., :3] = rgb
    image[..., 3] = 255
    return image

def test_centre_column_and_vertical_endpoints_sample_correctly():
    creative = _creative()
    creative[..., 0] = np.arange(201, dtype=np.uint8)
    creative[..., 1] = np.arange(31, dtype=np.uint8)[:, None] * 8
    warped = warp_to_cylinder(creative, _cyl(), (200, 200, 3))
    assert warped.shape == (200, 200, 4)
    assert warped.dtype == np.uint8
    np.testing.assert_array_equal(warped[40:161, 100, 0], np.full(121, 100))
    np.testing.assert_array_equal(warped[41:160, 100, 3], np.full(119, 255))
    assert 0 < int(warped[40, 100, 3]) < 255
    assert 0 < int(warped[160, 100, 3]) < 255
    for y, source_y in ((40, 0), (100, 15), (160, 30)):
        np.testing.assert_array_equal(warped[y, 100, :3], creative[source_y, 100, :3])

def test_equal_angular_stripes_are_narrower_near_limbs():
    creative = _creative(height=20, width=1000)
    stripe = np.repeat(np.arange(10, dtype=np.uint8) * 25, 100)
    creative[..., :3] = stripe[None, :, None]
    row = warp_to_cylinder(creative, _cyl(), (200, 200))[100]
    visible = row[:, 3] > 0
    ids = np.rint(row[visible, 0].astype(float) / 25).astype(int)
    widths = np.bincount(ids, minlength=10)
    assert np.all(widths > 0)
    assert widths[0] < widths[4]
    assert widths[9] < widths[5]
    assert np.mean(widths[[0, 9]]) < 0.8 * np.mean(widths[4:6])

def test_alpha_support_and_eight_degree_side_fade():
    warped = warp_to_cylinder(_creative(), _cyl(), (200, 200))
    yy, xx = np.indices((200, 200))
    phi = np.arcsin(np.clip((xx - 100) / 50, -1, 1))
    outside = ((yy < 40) | (yy > 160) | (np.abs(xx - 100) >= 50) | (np.abs(phi) >= np.deg2rad(55)))
    assert np.all(warped[..., 3][outside] == 0)
    for x in (60, 61, 62, 63, 64):
        t = np.clip((np.deg2rad(55) - abs(phi[100, x])) / np.deg2rad(8), 0, 1)
        expected = round(255 * t * t * (3 - 2 * t))
        assert abs(int(warped[100, x, 3]) - expected) <= 1
    assert np.all(np.diff(warped[100, 60:65, 3].astype(int)) > 0)
    assert warped[100, 65, 3] == 255
    np.testing.assert_array_equal(warped[100, 60:101, 3], warped[100, 140:99:-1, 3])

def test_composite_preserves_outside_and_occluded_pixels_exactly():
    frame = np.random.default_rng(13).integers(0, 256, (200, 200, 3), dtype=np.uint8)
    creative = _creative()
    occluder = np.zeros((200, 200), bool)
    occluder[65:145, 90:111] = True
    originals = [array.copy() for array in (frame, creative, occluder)]
    for array in (frame, creative, occluder):
        array.setflags(write=False)
    result = composite_curved(frame, creative, _cyl(), occluder_mask=occluder)
    alpha = warp_to_cylinder(creative, _cyl(), frame.shape)[..., 3]
    np.testing.assert_array_equal(result[alpha == 0], frame[alpha == 0])
    np.testing.assert_array_equal(result[:40], frame[:40])
    np.testing.assert_array_equal(result[161:], frame[161:])
    np.testing.assert_array_equal(result[occluder], frame[occluder])
    assert np.any(result[(alpha > 0) & ~occluder] != frame[(alpha > 0) & ~occluder])
    assert result.shape == frame.shape and result.dtype == np.uint8
    for array, original in zip((frame, creative, occluder), originals):
        np.testing.assert_array_equal(array, original)

def test_dark_left_half_darkens_the_label():
    frame = np.full((200, 200, 3), 160, np.uint8)
    frame[:, :100] = 40
    result = composite_curved(frame, _creative(rgb=(160, 160, 160)), _cyl())
    left = float(result[70:131, 75:90].mean())
    right = float(result[70:131, 111:126].mean())
    assert left < 0.7 * right

def test_rgb_to_bgr_and_partial_alpha_blending():
    frame = np.full((200, 200, 3), 100, np.uint8)
    creative = _creative(rgb=(200, 100, 20))
    creative[..., 3] = 128
    result = composite_curved(frame, creative, _cyl(), shade=0)
    weight = 128 / 255
    expected = np.rint((1 - weight) * frame[100, 100] + weight * np.array([20, 100, 200])).astype(np.uint8)
    np.testing.assert_array_equal(result[100, 100], expected)

def test_transparent_or_offscreen_labels_return_an_independent_copy():
    frame = np.full((200, 200, 3), 80, np.uint8)
    creative = _creative()
    creative[..., 3] = 0
    result = composite_curved(frame, creative, _cyl())
    np.testing.assert_array_equal(result, frame)
    assert not np.shares_memory(result, frame)
    creative[..., 3] = 255
    cyl = _cyl()
    cyl["b"] = 1000
    result = composite_curved(frame, creative, cyl)
    np.testing.assert_array_equal(result, frame)

def test_label_edges_are_soft_and_corners_rounded():
    warped = warp_to_cylinder(_creative(), _cyl(), (200, 200, 3))
    assert 0 < int(warped[40, 100, 3]) < 255
    assert warped[41, 100, 3] == 255
    assert 0 < int(warped[160, 100, 3]) < 255
    assert warped[159, 100, 3] == 255
    assert int(warped[40, 60, 3]) == 0

def test_aliasing_reduced_vs_naive_remap():
    Wc = 1600
    creative = np.zeros((31, Wc, 4), np.uint8)
    creative[:, ::2, :3] = 255
    creative[..., 3] = 255
    ours = warp_to_cylinder(creative, _cyl(), (200, 200, 3))
    naive = warp_to_cylinder(creative, _cyl(), (200, 200, 3), antialias=False, feather_px=0, corner_radius=0)
    ro = ours[80:121, 75:125, 0].astype(np.float32)
    rn = naive[80:121, 75:125, 0].astype(np.float32)
    assert float(np.var(rn)) > 2000.0
    assert float(np.var(ro)) < 0.5 * float(np.var(rn))
    lo = cv2.Laplacian(ro, cv2.CV_32F, ksize=3)
    ln = cv2.Laplacian(rn, cv2.CV_32F, ksize=3)
    assert float(np.mean(lo * lo)) < float(np.mean(ln * ln))

def test_print_pattern_does_not_leak_into_label():
    frame = np.full((200, 200, 3), 150, np.uint8)
    for y in range(40, 161):
        for x in range(60, 141, 8):
            if ((x // 8) + (y // 8)) % 2 == 0:
                frame[y, x:x + 8] = 100
            else:
                frame[y, x:x + 8] = 200
    result = composite_curved(frame, _creative(rgb=(180, 100, 40)), _cyl())
    region = result[70:131, 80:121]
    per_chan = [float(np.var(region[..., c].astype(np.float32))) for c in range(3)]
    assert max(per_chan) < 500.0

def test_specular_highlight_brightens_label():
    frame = np.full((200, 200, 3), 120, np.uint8)
    frame[80:121, 97:104] = 200
    creative = _creative(rgb=(160, 160, 160))
    spec = composite_curved(frame, creative, _cyl(), specular=0.5)
    base = composite_curved(frame, creative, _cyl(), specular=0)
    hl = np.zeros((200, 200), bool); hl[80:121, 97:104] = True
    surr = np.zeros((200, 200), bool); surr[80:121, 75:90] = True
    assert float(spec[hl].mean()) > float(spec[surr].mean()) + 8
    assert float(spec[hl].mean()) > float(base[hl].mean()) + 5


def test_bowed_bottom_edge_follows_ellipse():
    cyl = _cyl_curved(e=12.0)
    warped = warp_to_cylinder(_creative(), cyl, (200, 200, 3), arc_deg=140, feather_px=0, corner_radius=0)
    alpha = warped[..., 3]
    rows_c = np.where(alpha[:, 100] > 0)[0]
    assert rows_c.size > 0
    assert abs(int(rows_c.max()) - (160 + 12)) <= 2
    col_e = 100 + int(round(50.0 * np.sin(np.deg2rad(60.0))))
    rows_e = np.where(alpha[:, col_e] > 0)[0]
    assert rows_e.size > 0
    assert abs(int(rows_e.max()) - (160 + 6)) <= 2


def test_zero_sag_matches_flat_behaviour():
    base = _cyl()
    zero = _cyl_curved(e=0.0)
    creative = _creative()
    w1 = warp_to_cylinder(creative, base, (200, 200, 3))
    w2 = warp_to_cylinder(creative, zero, (200, 200, 3))
    np.testing.assert_array_equal(w1, w2)
