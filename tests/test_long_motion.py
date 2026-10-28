import cv2
import numpy as np
import pytest

from pipeline.long_motion import camera_motion_long, motion_residuals


def project(p, h):
    q = np.c_[p, np.ones(len(p))] @ h.T
    return q[:, :2] / q[:, 2:]


class Plane:
    """Reiterable lazy frames/masks; render one frame at a time."""
    def __init__(self, n, masks=False, width=640):
        self.n, self.masks, self.width = n, masks, width
        self.passes = 0
        rng = np.random.default_rng(123)
        self.base = cv2.GaussianBlur(
            rng.integers(
                0, 256, (width * 9 // 16, width), np.uint8
            ),
            (0, 0), .8,
        )
        h, w = self.base.shape
        self.screen = np.zeros((h, w), np.uint8)
        self.screen[
            int(.24*h):int(.74*h), int(.34*w):int(.68*w)
        ] = 1

    def matrix(self, t):
        h, w = self.base.shape
        u = t / max(1, self.n - 1)
        scale = 1 + .12 * u
        return np.array([
            [scale, -.006*u, .035*w*u + (1-scale)*w/2],
            [.004*u, scale, -.018*h*u + (1-scale)*h/2],
            [1e-5*u * 640/w, -7e-6*u * 640/w, 1.],
        ])

    def __iter__(self):
        self.passes += 1
        assert self.passes <= 2, "source iterated more than twice"
        h, w = self.base.shape
        yy, xx = np.indices((h, w))
        for t in range(self.n):
            matrix = self.matrix(t)
            mask = cv2.warpPerspective(
                self.screen, matrix, (w, h),
                flags=cv2.INTER_NEAREST,
            ) != 0
            if self.masks:
                yield mask
            else:
                frame = cv2.warpPerspective(
                    self.base, matrix, (w, h),
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                animation = (
                    (xx*7 + yy*11 + 31*t) % 256
                ).astype(np.uint8)
                frame[mask] = animation[mask]
                yield frame


def run_plane(n, width=640):
    frames = Plane(n, width=width)
    masks = Plane(n, masks=True, width=width)
    H = camera_motion_long(
        frames, masks, keyframe_every=25, max_features=700
    )
    assert frames.passes == masks.passes == 2
    assert H.shape == (n, 3, 3) and H.dtype == np.float64
    np.testing.assert_array_equal(H[n//2], np.eye(3))
    h, w = frames.base.shape
    corners = np.array([
        [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]
    ], float)
    reference = frames.matrix(n//2)
    errors = []
    for t in range(n):
        truth = reference @ np.linalg.inv(frames.matrix(t))
        errors.append(np.linalg.norm(
            project(corners, H[t]) - project(corners, truth), axis=1
        ) * (640 / width))
    return np.asarray(errors)


def test_zoom_pan_and_no_growth_with_length():
    short, long = run_plane(300), run_plane(600)
    assert short.max() < .5
    assert long.max() < .5
    # Same geometric path, double duration: error must not accumulate.
    assert long.max() <= short.max() + .03


def test_downsample_and_lazy_two_passes(monkeypatch):
    original = cv2.goodFeaturesToTrack

    def checked(gray, *args, **kwargs):
        assert gray.shape[1] <= 640
        return original(gray, *args, **kwargs)

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", checked)
    assert run_plane(75, width=960).max() < .5


def test_blank_evidence_raises():
    frames = [np.zeros((180, 320), np.uint8)] * 30
    with pytest.raises(RuntimeError, match="keyframe"):
        camera_motion_long(frames)


def test_diagnostic_detects_drift_and_reports_missing_support():
    frames = Plane(60)
    truth = np.array([
        frames.matrix(30) @ np.linalg.inv(frames.matrix(t))
        for t in range(60)
    ])
    bad = truth.copy()
    bad[:, 0, :] += (
        np.linspace(-3., 3., len(bad))[:, None] * bad[:, 2, :]
    )
    residual = motion_residuals(
        Plane(60), bad, Plane(60, masks=True)
    )
    assert residual["count"].min() >= 10
    assert residual["median_px"][[0, -1]].min() > 2

    black = [np.zeros((180, 320), np.uint8)] * 5
    missing = motion_residuals(
        black, np.repeat(np.eye(3)[None], 5, axis=0)
    )
    assert not missing["count"].any()
    assert np.isnan(missing["p95_px"]).all()
