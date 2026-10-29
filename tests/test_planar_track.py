import cv2
import numpy as np
import pytest

from pipeline.planar_track import track_plane, _project


class Handheld:
    """Lazy, deterministic two-depth scene with independent printed bezel."""
    n, w, h = 90, 384, 256
    ref = 28

    def __init__(self, dynamic=False):
        self.dynamic = dynamic
        rng = np.random.default_rng(911)
        self.pad, self.pw, self.ph = 36, 116, 84
        self.corners = np.array([
            [36, 36], [152, 36], [152, 120], [36, 120]
        ], float)
        self.texture = cv2.GaussianBlur(
            rng.integers(15, 241, (157, 189), np.uint8),
            (0, 0), .65,
        )
        cv2.putText(
            self.texture, "PIZZA", (41, 78),
            cv2.FONT_HERSHEY_SIMPLEX, .48, 250, 2, cv2.LINE_AA,
        )
        self.background = cv2.GaussianBlur(
            rng.integers(0, 256, (self.h, self.w), np.uint8),
            (0, 0), .7,
        )

    def __len__(self):
        return self.n

    def offset(self, t):
        if t <= 30:
            return 1.5*(t-30)
        if t < 37:
            return 45*(t-30)
        if t <= 51:
            return 315.
        if t <= 58:
            return 45*(58-t)
        return 1.2*(t-58)

    def matrix(self, t):
        angle = .035*np.sin(.13*t)
        scale = 1 + .10*np.sin(.055*t)
        c, s = scale*np.cos(angle), scale*np.sin(angle)
        local = np.array([
            [c, -s, 0], [s, c, 0],
            [1e-4*np.sin(.09*t), 1.4e-4*np.cos(.08*t), 1.],
        ])
        center = np.array([
            [1., 0, -94], [0, 1., -78], [0, 0, 1.]
        ])
        place = np.array([
            [1., 0, 164+self.offset(t)+1.0*np.sin(1.8*t)],
            [0, 1., 126+1.2*np.cos(1.4*t)],
            [0, 0, 1.],
        ])
        return place @ local @ center

    @property
    def truth(self):
        return np.array([
            _project(self.corners, self.matrix(t))
            for t in range(self.n)
        ])

    def occ(self, t):
        m = np.zeros((self.h, self.w), np.uint8)
        if t == 66:
            cv2.ellipse(m, (178, 126), (130, 90), 0, 0, 360, 1, -1)
        if 12 <= t <= 25 or 69 <= t <= 81:
            k = t-12 if t <= 25 else t-69
            cv2.ellipse(
                m, (95+13*k, 126), (25, 55), -15, 0, 360, 1, -1
            )
        return m != 0

    def surface(self, t):
        if t in (6, 7, 8, 63, 64):
            return None
        rng = np.random.default_rng(1000+t)
        q = self.truth[t] + rng.normal(0, 1.4, (4, 2))
        m = np.zeros((self.h, self.w), np.uint8)
        cv2.fillConvexPoly(m, np.rint(q).astype(np.int32), 1)
        # SAM sometimes swallows the foreground.
        return (m != 0) & (~self.occ(t) if t % 3 else True)

    def __getitem__(self, t):
        if not 0 <= t < self.n:
            raise IndexError(t)
        bg = cv2.warpAffine(
            self.background,
            np.array([[1., 0, -2.3*t], [0, 1, .2*t]]),
            (self.w, self.h), borderMode=cv2.BORDER_WRAP,
        ).astype(np.float32)
        texture = self.texture.copy()
        if self.dynamic:
            texture[36:121, 36:153] = np.random.default_rng(
                4000+t
            ).integers(0, 256, (85, 117), np.uint8)
        if t in (6, 7, 8):
            texture[:] = 128

        v = (
            self.offset(min(t+1, self.n-1))
            - self.offset(max(0, t-1))
        ) / 2
        samples = 13 if abs(v) > 20 else 1
        premult, alpha = np.zeros_like(bg), np.zeros_like(bg)
        for dt in ((np.arange(samples)+.5)/samples-.5)*.5:
            pose = self.matrix(t).copy()
            pose[0] += dt*v*pose[2]
            premult += cv2.warpPerspective(
                texture.astype(np.float32), pose, (self.w, self.h)
            ) / samples
            alpha += cv2.warpPerspective(
                np.ones(texture.shape, np.float32),
                pose, (self.w, self.h),
            ) / samples
        gray = premult + bg*(1-alpha)
        gray[self.occ(t)] = 38
        return cv2.cvtColor(
            np.clip(np.rint(gray), 0, 255).astype(np.uint8),
            cv2.COLOR_GRAY2BGR,
        )

    def __iter__(self):
        return (self[t] for t in range(self.n))


def _run(dynamic):
    scene = Handheld(dynamic)
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        out = track_plane(
            scene, scene.ref, scene.truth[scene.ref],
            surface_masks=scene.surface, occluder_masks=scene.occ,
            dynamic_content=dynamic, context_px=30, max_width=384,
        )
    finally:
        cv2.setNumThreads(previous_threads)

    valid, q = out["valid"], out["quads"]
    errors = np.linalg.norm(q[valid]-scene.truth[valid], axis=2)
    median, p95 = np.median(errors), np.percentile(errors, 95)
    print(
        f"dynamic={dynamic}: median={median:.3f}px p95={p95:.3f}px "
        f"valid={valid.sum()}/{len(scene)} "
        f"max_motion="
        f"{np.linalg.norm(np.diff(scene.truth, axis=0), axis=2).max():.1f}px"
    )
    assert median < 1
    assert p95 < 3
    assert not valid[66]
    assert not valid[37:52].any()
    assert np.isnan(q[37:52]).all()
    assert valid[:26].sum() >= 22
    assert valid[52:57].any() and valid[57:].sum() >= 28
    assert valid.sum() >= 60
    assert any(s == "reacquired" for s in out["status"][52:58])
    np.testing.assert_array_equal(out["H"][scene.ref], np.eye(3))
    np.testing.assert_array_equal(q[scene.ref], scene.truth[scene.ref])
    for i in np.flatnonzero(valid):
        np.testing.assert_allclose(
            _project(q[i], out["H"][i]),
            scene.truth[scene.ref], atol=1e-3,
        )
    assert out["visible"][valid].min() > 0
    assert np.nanmedian(out["diagnostics"]["residual_px"]) < 1
    assert np.median(errors) < .7
    return out


def test_printed_plane_parallax_blur_occlusion_and_reentry():
    _run(False)


def test_dynamic_interior_is_never_a_tracking_feature():
    _run(True)


def test_blank_plane_and_background_are_not_valid_tracks():
    scene = Handheld()
    blank = [np.full((scene.h, scene.w, 3), 100, np.uint8)]*12
    result = track_plane(
        blank, 5, scene.truth[5], mask_fallback=False
    )
    assert result["valid"].sum() == 1
    assert result["status"][5] == "reference"


def test_masks_are_optional_and_bad_inputs_raise():
    scene = Handheld()
    frames = [scene[i] for i in range(5)]
    result = track_plane(frames, 2, scene.truth[2], context_px=25)
    assert result["valid"].all()
    assert np.median(
        np.linalg.norm(result["quads"]-scene.truth[:5], axis=2)
    ) < 1

    big = [cv2.resize(f, (768, 512)) for f in frames]
    expected = 2*scene.truth[:5]+.5
    reduced = track_plane(big, 2, expected[2], max_width=384)
    assert reduced["quads"].dtype == np.float64
    assert reduced["valid"].all()
    assert np.median(
        np.linalg.norm(reduced["quads"]-expected, axis=2)
    ) < 1
    with pytest.raises(TypeError):
        track_plane(frames, 2, scene.truth[2], magic=True)
    with pytest.raises(ValueError):
        track_plane(
            frames, 2, scene.truth[2],
            occluder_masks=lambda i: np.zeros((2, 2), bool),
        )


def test_scale_change_adds_templates_without_accumulating_drift():
    class Zoom(Handheld):
        n, ref = 24, 3

        def matrix(self, t):
            scale = .72 + .035*t
            return np.array([
                [scale, .015*np.sin(t), 164-94*scale],
                [0, scale, 126-78*scale], [0, 0, 1.],
            ])

        def occ(self, t):
            return np.zeros((self.h, self.w), bool)

    scene = Zoom()
    result = track_plane(
        scene, scene.ref, scene.truth[scene.ref],
        surface_masks=scene.surface, context_px=30,
    )
    assert result["valid"].all()
    assert result["diagnostics"]["keyframes"] >= 2
    errors = np.linalg.norm(result["quads"]-scene.truth, axis=2)
    assert np.median(errors) < 1
    assert np.percentile(errors, 95) < 3


def test_batch_smoothing_reduces_noise_without_flattening_shake():
    from pipeline.planar_track import _smooth
    rng = np.random.default_rng(44)
    t = np.arange(80)
    motion = 3*np.sin(1.9*t) + 45*np.maximum(0, t-38)
    truth = np.array([
        [10., 10], [110, 10], [110, 80], [10, 80]
    ])[None]
    truth = np.tile(truth, (len(t), 1, 1))
    truth[:, :, 0] += motion[:, None]
    observed = truth + rng.normal(0, .2, truth.shape)
    observed[0] = truth[0]
    smoothed, _, _ = _smooth(observed, np.full(len(t), .2), 0)
    assert np.mean((smoothed-truth)**2) < np.mean((observed-truth)**2)
    assert np.linalg.norm(smoothed-observed, axis=2).max() <= .200001
