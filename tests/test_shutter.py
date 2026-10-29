import hashlib

import cv2
import numpy as np
import pytest

from pipeline import composite
from pipeline.shutter import shutter_quads


def _fixture():
    rng = np.random.default_rng(71)
    frame = rng.integers(0, 256, (83, 119, 3), np.uint8)
    art = rng.integers(0, 256, (29, 37, 4), np.uint8)
    q = np.array([
        [-3.2, 9.7], [89.4, 4.3], [103.2, 70.4], [2.1, 77.3]
    ])
    occ = np.zeros(frame.shape[:2], bool)
    occ[18:60, 43:52] = True
    illumination = np.linspace(
        .5, 1.4, 37, dtype=np.float32
    )[None].repeat(29, 0)
    return frame, art, q, occ, illumination


def test_none_is_byte_identical_to_original_single_warp():
    frame, art, q, occ, illumination = _fixture()
    digests = []
    for emissive in (False, True):
        options = dict(
            occluder_mask=occ, illum=illumination,
            emissive=emissive, blur_sigma=1.2, grain_sigma=1.7,
        )
        old_path = composite.composite_frame(frame, art, q, **options)
        explicit_none = composite.composite_frame(
            frame, art, q, motion_quads=None, **options
        )
        stationary = composite.composite_frame(
            frame, art, q, motion_quads=[q], **options
        )
        np.testing.assert_array_equal(old_path, explicit_none)
        np.testing.assert_array_equal(old_path, stationary)
        digests.append(hashlib.sha256(old_path.tobytes()).hexdigest())
    # Original-file golden outputs, measured with OpenCV 4.11.
    assert digests == GOLDENS


def test_translation_is_a_box_filter_including_transparent_edges():
    rng = np.random.default_rng(12)
    frame = np.full((96, 160, 3), 43, np.uint8)
    art = rng.integers(0, 256, (37, 53, 4), np.uint8)
    q = np.array([[45., 28], [98, 28], [98, 65], [45, 65]])
    quads = np.array([q+[-18, 0], q, q+[18, 0]])
    positions = shutter_quads(quads, 1, shutter=.5)
    assert len(positions) == 9
    np.testing.assert_allclose(
        np.asarray(positions)-q,
        np.array([[[x, 0]]*4 for x in range(-4, 5)]),
        atol=1e-5,
    )
    kw = dict(
        bleed_px=0, blur_sigma=0, grain_sigma=0, match_energy=False
    )
    sharp = composite.composite_frame(frame, art, q, **kw)
    blurred = composite.composite_frame(
        frame, art, q, motion_quads=positions, **kw
    )
    expected = cv2.blur(
        sharp, (9, 1), borderType=cv2.BORDER_REPLICATE
    )
    error = np.abs(blurred.astype(float)-expected.astype(float))
    assert error.max() <= 1
    assert error.mean() < .07

    mask = np.zeros(frame.shape[:2], bool)
    mask[:, 74:82] = True
    occluded = composite.composite_frame(
        frame, art, q, motion_quads=positions,
        occluder_mask=mask, **kw,
    )
    np.testing.assert_array_equal(occluded[mask], frame[mask])


def test_projective_interpolation_endpoints_spacing_and_gaps():
    q = np.array([[20., 20], [100, 20], [100, 90], [20, 90]])
    h = np.array([
        [1.08, .02, 13], [-.03, .98, -1], [.003, -.001, 1.]
    ])
    p = np.c_[q, np.ones(4)] @ h.T
    b = p[:, :2]/p[:, 2:]
    positions = shutter_quads([q, b], 0, shutter=1)
    assert np.linalg.norm(
        np.diff(positions, axis=0), axis=2
    ).max() <= 1.001
    assert np.all(np.isfinite(positions))

    pair = shutter_quads([q, b], 0, shutter=1, samples=2)
    assert np.max(np.abs(pair[1]-(.75*q+.25*b))) > .03
    gap = np.full((4, 2), np.nan)
    assert shutter_quads([q, gap, b], 1) == []
    np.testing.assert_array_equal(
        shutter_quads([gap, q, gap], 1)[0], q
    )
    np.testing.assert_array_equal(
        shutter_quads([q, b], 0, shutter=0)[0], q
    )
    with pytest.raises(ValueError):
        shutter_quads([q], 0, samples=0)
    with pytest.raises(ValueError):
        composite.composite_frame(
            *_fixture()[:3], motion_quads=[],
            blur_sigma=0, grain_sigma=0,
        )


GOLDENS = [
    "3758ccae0deefc24e980b0afe5987c165a608b41198ee23f168617a92b1ba26e",
    "c4a9b68bff98d8dbdc4d9b54f7b7ff291583d25a76785738125440cd6279f0de",
]
