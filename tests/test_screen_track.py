import cv2
import numpy as np
import pytest

from pipeline.planar_track import _fit, _project
from pipeline.screen_track import track_screen


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_projective_pole_is_rejected_during_any_fit_stage(monkeypatch, stage):
    import pipeline.planar_track as module

    x = np.array([
        [0., 0], [100, 0], [100, 80], [0, 80],
        [20, 30], [70, 20], [30, 60], [80, 65],
    ])
    quad = x[:4]
    pole = np.array([[1., 0, 0], [0, 1, 0], [.01, 0, 0]])
    calls = 0

    def candidate(*args, **kwargs):
        nonlocal calls
        h = pole if calls == stage else np.eye(3)
        calls += 1
        return h, np.ones((len(args[0]), 1), np.uint8)

    monkeypatch.setattr(module.cv2, "findHomography", candidate)
    assert module._fit(x, x.copy(), quad, 1.5, 4) is None


def test_valid_fit_is_unchanged():
    rng = np.random.default_rng(9)
    x = rng.uniform([0, 0], [150, 90], (60, 2))
    h = np.array([[1.1, .02, 30], [-.01, .95, 10], [.0002, -.0001, 1.]])
    q = np.array([[0., 0], [150, 0], [150, 90], [0, 90]])
    y = _project(x, h)
    fit = _fit(x, y, q, 1.5, 8)
    assert fit is not None
    np.testing.assert_allclose(_project(q, fit[0]), _project(q, h), atol=1e-4)


class BlankWall:
    n, w, h, ref = 96, 384, 256, 10

    def __init__(self, variant=0):
        self.variant = variant
        rng = np.random.default_rng(18)
        self.boxes = cv2.GaussianBlur(
            rng.integers(20, 240, (65, 110), np.uint8), (0, 0), .6
        )
        self.local = np.array([
            [0., 0], [160, 0], [160, 100], [0, 100]
        ])

    def __len__(self):
        return self.n

    def pose(self, i):
        x = 100+5*i if i <= 60 else 400-7*(i-60)
        return np.array([
            [1., .015*np.sin(.1*i), x],
            [.025*np.sin(.07*i), 1., 65+1.1*np.sin(1.8*i)],
            [.00012*np.sin(.05*i), .0001*np.cos(.1*i), 1.],
        ])

    @property
    def truth(self):
        return np.array([
            _project(self.local, self.pose(i)) for i in range(self.n)
        ])

    def occ(self, i):
        m = np.zeros((self.h, self.w), np.uint8)
        if 16 <= i <= 23 or 75 <= i <= 85:
            x = 235+10*(i-16) if i < 30 else 260-4*(i-75)
            cv2.ellipse(m, (x, 130), (20, 64), 0, 0, 360, 1, -1)
        if 38 <= i <= 40:
            m[:] = 1
        return m != 0

    def surface(self, i):
        if i in (6, 7, 90):
            return None
        q = self.truth[i]+np.random.default_rng(70+i).normal(0, 1.2, (4, 2))
        m = np.zeros((self.h, self.w), np.uint8)
        cv2.fillConvexPoly(m, np.rint(q).astype(np.int32), 1)
        return (m != 0) & ~self.occ(i)

    def __getitem__(self, i):
        if not 0 <= i < self.n:
            raise IndexError(i)
        gray = np.full((self.h, self.w), 195, np.uint8)
        # Texture at another depth gives a deliberately wrong camera prior.
        gray[185:250, :110] = np.roll(self.boxes, -3*i, axis=1)
        screen = np.full((101, 161), 18, np.uint8)
        screen[6:-6, 6:-6] = np.random.default_rng(
            10000*self.variant+i
        ).integers(30, 240, (89, 149), np.uint8)
        image = cv2.warpPerspective(screen, self.pose(i), (self.w, self.h))
        alpha = cv2.warpPerspective(
            np.ones(screen.shape, np.uint8), self.pose(i),
            (self.w, self.h), flags=cv2.INTER_NEAREST,
        ) != 0
        gray[alpha] = image[alpha]
        gray[self.occ(i)] = 45
        gray = cv2.GaussianBlur(gray, (0, 0), .55)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def __iter__(self):
        return (self[i] for i in range(self.n))


def _run(scene, **options):
    return track_screen(
        scene, scene.ref, scene.truth[scene.ref],
        surface_masks=scene.surface, occluder_masks=scene.occ,
        **options,
    )


def test_blank_wall_dynamic_screen_clipping_parallax_occlusion_and_reentry():
    scene = BlankWall()
    result = _run(scene)
    q, valid = result["quads"], result["valid"]
    truth = scene.truth
    visible = np.array([
        scene.surface(i) is not None and scene.surface(i).sum() > 300
        for i in range(len(scene))
    ])
    assert (valid & visible).sum() >= .9*visible.sum()

    clear = np.zeros((len(scene), 4), bool)
    for i in np.flatnonzero(valid):
        p = truth[i]
        inside = (
            (p[:, 0] > 6) & (p[:, 0] < scene.w-7)
            & (p[:, 1] > 6) & (p[:, 1] < scene.h-7)
        )
        ij = np.clip(
            np.rint(p).astype(int), [0, 0], [scene.w-1, scene.h-1]
        )
        clear[i] = inside & ~scene.occ(i)[ij[:, 1], ij[:, 0]]

    error = np.linalg.norm(q-truth, axis=2)[clear]
    print("blank wall corner median/p95", np.percentile(error, [50, 95]))
    assert np.median(error) < 1.
    assert np.percentile(error, 95) < 3.
    assert not valid[38:41].any()
    assert not valid[59:63].any()
    assert valid[70:].sum() > 20

    clipped = valid & (truth[:, 1, 0] > scene.w+10)
    assert clipped.any()
    assert (q[clipped, 1, 0] > scene.w).all()
    assert not result["diagnostics"]["observed_edges"][clipped, 1].any()

    np.testing.assert_array_equal(q[scene.ref], truth[scene.ref])
    for i in np.flatnonzero(valid):
        np.testing.assert_allclose(
            _project(q[i], result["H"][i]), truth[scene.ref], atol=1e-3
        )
    assert np.isnan(q[~valid]).all()


def test_screen_content_does_not_establish_motion():
    first, second = _run(BlankWall(1)), _run(BlankWall(2))
    valid = first["valid"] & second["valid"]
    delta = np.linalg.norm(
        first["quads"][valid]-second["quads"][valid], axis=2
    )
    assert np.percentile(delta, 95) < .5


def test_blank_image_is_not_a_bezel():
    frames = [np.full((120, 180, 3), 180, np.uint8)]*5
    q = np.array([[30., 20], [140, 20], [140, 90], [30, 90]])
    mask = np.zeros((120, 180), bool)
    mask[20:91, 30:141] = True
    with pytest.raises(ValueError, match="four unoccluded bezel edges"):
        track_screen(frames, 2, q, surface_masks=lambda i: mask)


def test_screen_input_validation():
    scene = BlankWall()
    with pytest.raises(ValueError, match="max_gap"):
        _run(scene, max_gap=-1)
    with pytest.raises(ValueError, match="masks"):
        track_screen(
            list(scene)[:3], 1, scene.truth[1],
            surface_masks=lambda i: np.zeros((2, 2), bool),
        )
