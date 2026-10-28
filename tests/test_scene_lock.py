"""Synthetic acceptance tests; no assets or file writes are needed."""
import cv2
import numpy as np
import pytest

import pipeline.scene_lock as scene
from pipeline.geometry import quad_from_mask


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = cv2.getNumThreads()
    cv2.setNumThreads(2)
    yield
    cv2.setNumThreads(previous)


def polygon(points, shape):
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(points).astype(np.int32), 1)
    return mask.astype(bool)


def texture(shape=(240, 360), seed=13):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, shape, dtype=np.uint8)
    return cv2.GaussianBlur(image, (0, 0), 0.7)


def transform(points, matrix):
    return cv2.perspectiveTransform(
        np.asarray(points, np.float64)[None],
        np.asarray(matrix, np.float64),
    )[0]


def test_static_noisy_masks_are_constant():
    rng = np.random.default_rng(11)
    image = texture()
    base = np.array([
        [95., 65.], [245., 65.], [245., 175.], [95., 175.]
    ])
    masks = [
        polygon(base + rng.normal(0, 0.7, base.shape), image.shape)
        for _ in range(41)
    ]
    quads = np.array([quad_from_mask(m) for m in masks])
    original = quads.copy()
    mask_copy = np.array(masks)

    H = scene.camera_motion([image] * len(masks), masks)
    np.testing.assert_array_equal(
        H, np.repeat(np.eye(3)[None], len(masks), axis=0)
    )
    result = scene.lock_quads(
        [image] * len(masks), quads, masks, H=H
    )
    np.testing.assert_allclose(
        result, np.repeat(result[:1], len(masks), axis=0),
        atol=1e-10, rtol=0,
    )
    assert np.mean(np.linalg.norm(result[0] - base, axis=1)) < 1.1
    np.testing.assert_array_equal(quads, original)
    np.testing.assert_array_equal(masks, mask_copy)


def test_known_homographies_follow_to_tenth_pixel():
    rng = np.random.default_rng(22)
    base = np.array([
        [70., 50.], [270., 60.], [250., 180.], [60., 170.]
    ])
    forward = []
    for t in np.linspace(-1, 1, 101):
        forward.append(np.array([
            [1 + .04*t, -.02*t, 18*t],
            [.01*t, 1 - .025*t, -9*t],
            [2e-5*t, -3e-5*t, 1.],
        ]))
    truth = np.array([transform(base, m) for m in forward])
    observed = truth + rng.normal(0, .2, truth.shape)
    observed[12:19] += 50
    observed[40] = np.nan
    weights = np.ones(len(observed))
    weights[12:19] = .0001
    H = np.linalg.inv(forward)

    result = scene.lock_points(observed, H, weights)
    assert np.linalg.norm(result - truth, axis=2).max() < .1
    canonical = np.array([
        transform(p, m) for p, m in zip(result, H)
    ])
    np.testing.assert_allclose(
        canonical, np.repeat(canonical[:1], len(H), axis=0),
        atol=1e-9, rtol=0,
    )


def test_estimated_camera_follows_global_homographies():
    rng = np.random.default_rng(33)
    image = texture((360, 480), 33)
    base = np.array([
        [145., 110.], [325., 110.], [325., 250.], [145., 250.]
    ])
    frames, masks, truth = [], [], []
    for t in np.linspace(-1, 1, 25):
        m = np.array([
            [1 + .012*t, -.006*t, 6*t],
            [.004*t, 1 + .008*t, -3*t],
            [6e-6*t, -4e-6*t, 1.],
        ])
        frames.append(cv2.warpPerspective(
            image, m, (480, 360),
            borderMode=cv2.BORDER_REFLECT_101,
        ))
        q = transform(base, m)
        truth.append(q)
        masks.append(polygon(q, image.shape))
    truth = np.array(truth)
    observed = truth + rng.normal(0, .08, truth.shape)

    H = scene.camera_motion(
        frames, masks, smoothness=0, static_threshold=0
    )
    np.testing.assert_array_equal(H[len(H)//2], np.eye(3))
    result = scene.lock_points(observed, H)
    assert np.linalg.norm(result - truth, axis=2).max() < .1

    # A different reference changes coordinates, not physical output.
    H0 = scene.camera_motion(
        frames, masks, ref_idx=0,
        smoothness=0, static_threshold=0,
    )
    other = scene.lock_points(observed, H0)
    assert np.linalg.norm(other - truth, axis=2).max() < .1


def test_occluders_excluded_at_detection_and_both_match_endpoints(
    monkeypatch,
):
    background = texture((240, 360), 44)
    frames, excluded = [], []
    patch = texture((140, 150), 45)
    for x in (25, 55, 85, 115, 145):
        image = background.copy()
        image[50:190, x:x+150] = patch
        mask = np.zeros(background.shape, bool)
        mask[50:190, x:x+150] = True
        frames.append(image)
        excluded.append(mask)

    real_detect = cv2.goodFeaturesToTrack
    real_flow, real_fit = scene._flow_pair, scene._fit
    context, counts = {}, {"detect": 0, "fit": 0}

    def detect(image, *args, **kwargs):
        assert not kwargs["mask"][excluded[2]].any()
        counts["detect"] += 1
        return real_detect(image, *args, **kwargs)

    def flow(ref, frame, points, ref_allowed, allowed, *args, **kwargs):
        context["ref"], context["target"] = ref_allowed, allowed
        return real_flow(
            ref, frame, points, ref_allowed, allowed, *args, **kwargs
        )

    def fit(src, dst, *args, **kwargs):
        assert scene._inside(src, context["ref"]).all()
        assert scene._inside(dst, context["target"]).all()
        counts["fit"] += 1
        return real_fit(src, dst, *args, **kwargs)

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", detect)
    monkeypatch.setattr(scene, "_flow_pair", flow)
    monkeypatch.setattr(scene, "_fit", fit)

    H = scene.camera_motion(frames, excluded)
    assert counts["detect"] == 1
    assert counts["fit"] == len(frames) - 1
    np.testing.assert_array_equal(
        H, np.repeat(np.eye(3)[None], len(frames), axis=0)
    )


def test_ribbons_preserve_format_and_follow_translation():
    count, samples = 31, 24
    x = np.linspace(40, 180, samples)
    top = 45 + .001 * (x - 110)**2
    bottom = top + 60
    ribbons, H, truth = [], [], []

    for i in range(count):
        delta = np.array([.5*i, -.2*i])
        # Symmetric noise has an exactly zero rigid estimate.
        noise = .05 * np.sin(2*np.pi*i/count)
        ribbons.append(dict(
            x=x+delta[0]+noise,
            top=top+delta[1]+noise,
            bottom=bottom+delta[1]+noise,
            tag="preserved",
        ))
        m = np.eye(3)
        m[:2, 2] = -delta
        H.append(m)
        truth.append((
            x+delta[0], top+delta[1], bottom+delta[1]
        ))

    frames = [np.zeros((180, 260), np.uint8)] * count
    originals = [r["x"].copy() for r in ribbons]
    result = scene.lock_ribbons(frames, ribbons, H=H)
    assert len(result) == count
    for r, expected, source, original in zip(
        result, truth, ribbons, originals
    ):
        assert r["tag"] == "preserved"
        for key, value in zip(("x", "top", "bottom"), expected):
            np.testing.assert_allclose(
                r[key], value, atol=1e-6, rtol=0
            )
            assert not np.shares_memory(r[key], source[key])
        np.testing.assert_array_equal(source["x"], original)


def test_missing_points_and_rejected_observations():
    p = np.array([
        [[10., 20.], [30., 40.]],
        [[np.nan, np.nan], [32., 42.]],
        [[100., 100.], [30., 40.]],
    ])
    weights = np.array([
        [1., 1.], [0., 0.], [0., 1.]
    ])
    result = scene.lock_points(p, [np.eye(3)] * 3, weights)
    np.testing.assert_allclose(
        result, np.repeat(p[:1], 3, axis=0)
    )
    with pytest.raises(ValueError, match="positive-weight"):
        scene.lock_points(
            p, [np.eye(3)] * 3, np.zeros((3, 2))
        )
    with pytest.raises(ValueError):
        scene.lock_points(p, [np.zeros((3, 3))] * 3)
    with pytest.raises(ValueError):
        scene.lock_points(p, [np.eye(3)] * 3, [-1, 1, 1])


def test_feature_failure_is_not_reported_as_static():
    frames = [np.zeros((100, 140), np.uint8)] * 5
    with pytest.raises(RuntimeError, match="background"):
        scene.camera_motion(frames)


def test_small_per_frame_motion_is_not_mistaken_for_static(monkeypatch):
    image = texture((180, 260), 66)
    frames = []
    for dx in np.linspace(-.6, .6, 21):
        m = np.array([[1., 0., dx], [0., 1., 0.]])
        frames.append(cv2.warpAffine(
            image, m, (260, 180),
            borderMode=cv2.BORDER_REFLECT_101,
        ))
    original = scene._flow_pair
    references = []

    def flow(ref, *args, **kwargs):
        references.append(ref)
        return original(ref, *args, **kwargs)

    monkeypatch.setattr(scene, "_flow_pair", flow)
    H = scene.camera_motion(frames)
    assert references
    assert all(ref is frames[10] for ref in references)
    assert max(abs(m[0, 2]) for m in H) > .4


def test_slow_deformation_is_explicit_and_strongly_smoothed():
    t = np.linspace(-1, 1, 61)
    p = np.zeros((61, 2, 2))
    p[:, :, 0] = 10 + t[:, None]
    p[:, :, 1] = [20, 40]
    H = [np.eye(3)] * len(t)

    rigid = scene.lock_points(p, H)
    assert np.ptp(rigid[:, :, 0], axis=0).max() < 1e-10
    deforming = scene.lock_points(
        p, H, deformation_smoothness=1000
    )
    assert np.ptp(deforming[:, :, 0], axis=0).min() > 1.5
    assert (
        np.linalg.norm(
            np.diff(deforming, n=2, axis=0), axis=2
        ).max() < .01
    )
    with pytest.raises(ValueError):
        scene.lock_points(p, H, deformation_smoothness=1)
