"""Synthetic regression checks; pytest or unittest."""
import unittest

import cv2
import numpy as np

from pipeline.photometric import (
    channel_exposure, defocus_track, noise_model, add_noise,
    match_compression, estimate_compression,
)


def track(frames):
    h, w = frames[0].shape[:2]
    q = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    return np.repeat(q[None], len(frames), axis=0), np.ones(len(frames), bool)


class PhotometricTests(unittest.TestCase):
    def test_rgb_gain_veil_and_occlusion(self):
        rng = np.random.default_rng(4)
        base = cv2.GaussianBlur(
            rng.uniform(20, 220, (140, 200, 3)).astype(np.float32),
            (0, 0), .6,
        )
        gain = np.array([1.12, .92, 1.05])
        veil = np.array([4., 7., -3.])
        changed = base*gain[::-1]+veil[::-1]
        frames = [base]*7+[changed.copy() for _ in range(15)]
        mask = np.zeros(base.shape[:2], bool)
        mask[25:75, 30:85] = True
        for f in frames[7:]:
            f[mask] = [220, 30, 80]
            f[90:100, 100:120] = 245  # Unmasked glare.
        q, valid = track(frames)
        g, b = channel_exposure(
            frames, q, valid, 0, lambda i: mask if i >= 7 else None
        )
        np.testing.assert_allclose(g[-3], gain, atol=.025)
        np.testing.assert_allclose(b[-3], veil, atol=1.5)
        np.testing.assert_array_equal(g[0], np.ones(3))

    def test_blank_invalid_and_black_are_safe(self):
        frames = [
            np.full((140, 200, 3), [80., 100., 120.], np.float32)
        ]
        frames += [
            frames[0]*np.array([.9, 1.1, 1.2]) for _ in range(10)
        ]
        q, valid = track(frames)
        g, b = channel_exposure(frames, q, valid, 0, None)
        np.testing.assert_allclose(g[-1], [1.2, 1.1, .9], atol=.003)
        self.assertEqual(np.abs(b).max(), 0.)
        q[3] = np.nan
        valid[3] = False
        g, b = channel_exposure(
            [np.zeros_like(f) for f in frames], q, valid, 0, None
        )
        self.assertTrue(np.isfinite(g).all() and np.isfinite(b).all())
        np.testing.assert_array_equal(g, np.ones_like(g))

    def test_defocus_with_shutter_motion(self):
        rng = np.random.default_rng(9)
        image = np.full((256, 320), 65., np.float32)
        image[128:] = 190
        image[:, 160:] += 25
        image += cv2.GaussianBlur(
            rng.normal(0, 18, image.shape).astype(np.float32),
            (0, 0), .45,
        )
        reference_sigma, target_sigma = .7, 1.9
        ref = cv2.GaussianBlur(image, (0, 0), reference_sigma)
        target = cv2.GaussianBlur(image, (0, 0), target_sigma)
        target = cv2.filter2D(
            target, -1, np.ones((1, 9), np.float32)/9
        )
        frames = [np.repeat(ref[..., None], 3, axis=2)]*7
        frames += [np.repeat(target[..., None], 3, axis=2)]*15
        q, valid = track(frames)
        motion = np.zeros((len(frames), 2))
        motion[7:, 0] = 9.
        sigma = defocus_track(frames, q, valid, 0, None, motion)
        self.assertLess(abs(float(sigma[-3])-target_sigma), .3)
        self.assertTrue(np.all(sigma >= sigma[0]))
        self.assertLess(abs(float(sigma[0])-reference_sigma), .3)

    def test_noise_variance_luma_and_spatial_correlation(self):
        rng = np.random.default_rng(17)
        levels = np.repeat(np.linspace(30, 220, 8), 48)
        image = np.broadcast_to(
            levels[None, :, None], (240, 384, 3)
        ).copy().astype(np.float32)
        params = dict(
            shot=np.array([.025, .04, .06]),
            read=np.array([.8, 1.2, 1.8]),
            corr_sigma=.55,
        )
        frames = [
            add_noise(image, np.ones(image.shape[:2]), params, rng)
            for _ in range(7)
        ]
        q, valid = track(frames)
        fitted = noise_model(frames, q, valid, None)[3]
        np.testing.assert_allclose(
            fitted["shot"], params["shot"], rtol=.3, atol=.005
        )
        for L in (40, 120, 210):
            np.testing.assert_allclose(
                fitted["shot"]*L+fitted["read"],
                params["shot"]*L+params["read"], rtol=.25,
            )
        self.assertLess(abs(fitted["corr_sigma"]-.55), .15)
        out = add_noise(
            image, np.ones(image.shape[:2]), fitted,
            np.random.default_rng(8),
        )
        np.testing.assert_array_equal(out, add_noise(
            image, np.ones(image.shape[:2]), fitted,
            np.random.default_rng(8),
        ))
        outside = np.zeros(image.shape[:2], bool)
        outside[40:120, 40:120] = True
        out = add_noise(
            image, outside, fitted, np.random.default_rng(8)
        )
        np.testing.assert_array_equal(out[~outside], image[~outside])

    def test_clean_plate_channel_gain(self):
        from pipeline.clean_plate import CleanPlate

        yy, xx = np.mgrid[:80, :100]
        base = np.repeat((60+xx*.5+yy*.2)[..., None], 3, axis=2)
        rgb_gain = np.array(
            [[1., 1., 1.]]*3+[[1.15, .9, 1.1]]*6
        )
        rgb_veil = np.array(
            [[0., 0., 0.]]*3+[[3., 5., -2.]]*6
        )
        frames = [
            np.rint(base*g[::-1]+v[::-1]).astype(np.uint8)
            for g, v in zip(rgb_gain, rgb_veil)
        ]
        q, valid = track(frames)
        plate = CleanPlate(
            frames, q, valid, rgb_gain[:, ::-1],
            rgb_veil[:, ::-1], width=100,
        )
        self.assertLess(
            float(np.nanmean(abs(plate.plate(5)-plate.rect[0]))), .6
        )

    def test_print_detail_is_not_shot_noise(self):
        rng = np.random.default_rng(22)
        image = np.full((180, 288, 3), 30., np.float32)
        image[:, 144:] = (
            140+65*np.sin(np.arange(144)[None, :, None]*.45)
        )
        frames = [
            (image+rng.normal(0, 1, image.shape)).astype(np.float32)
            for _ in range(5)
        ]
        q, valid = track(frames)
        params = noise_model(frames, q, valid, None)[2]
        self.assertLess(float(np.max(params["shot"])), .03)
        self.assertLess(float(np.max(params["read"])), 2.)

    def test_compression_quality_max_and_odd_crop(self):
        rng = np.random.default_rng(21)
        image = rng.integers(
            0, 256, (79, 103, 3), dtype=np.uint8
        )
        np.testing.assert_array_equal(image, match_compression(
            image,
            dict(quality=100, qstep=8, subsample=True, chroma_sigma=1.2),
        ))
        p = dict(
            quality=99, qstep=3., chroma_sigma=.6,
            subsample=True, origin=(13, 9),
        )
        out = match_compression(image, p)
        self.assertEqual(out.shape, image.shape)
        np.testing.assert_array_equal(
            out, match_compression(image, p)
        )
        q, valid = track([image])
        model = estimate_compression([image], q, valid, None)
        self.assertTrue(np.isfinite(model[0]["qstep"]))


if __name__ == "__main__":
    unittest.main()
