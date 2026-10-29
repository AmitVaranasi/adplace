"""What lies on top of a surface, separated from the surface itself, so it can be laid back over the ad.

Work in the surface's own rectified frame (the track maps every frame onto it). The print, stains and creases stay put
there; reflections, glare, a glass salt shaker passing over it and people move. So the median of nearby frames,
after undoing each frame's exposure (observed = gain * reference + veil), is the bare surface: the clean plate. What
a frame shows beyond that plate is the layer in front of the surface:

  residual_i = rectified_i - (gain_i * plate_i + veil_i)

Three uses in the renderer:
  - reflections and sheen: the positive part of the residual is added over the ad (light reflected off the glass or
    gloss in front of a sign does not care what is printed behind it);
  - transparent occluders (a glass shaker): inside their mask the whole signed residual is added, so the ad shows
    through the glass with the glass's own highlights and dark cap on top;
  - occluder fringes: along a hand's soft edge the original pixel mixes hand and old surface; knowing the old
    surface (the plate) swaps it for the ad exactly: out = original + (1 - alpha) * (ad - plate).
"""
import cv2
import numpy as np


def rect_size(quad, width=320):
    q = np.asarray(quad, float)
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return width, int(np.clip(round(width * h / max(w, 1)), 80, 480))


def _to_rect(quad, size):
    W, H = size
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    return cv2.getPerspectiveTransform(np.asarray(quad, np.float32), dst)


class CleanPlate:
    """Holds the rectified colour frames of one tracked surface and serves its plate and residual per frame.

    frames: re-iterable BGR frames; quads (N,4,2), valid (N,), gain/veil (N,) from the exposure fit;
    solid(i) -> bool mask of opaque occluders (people, hands) or None; glass(i) -> bool mask of transparent ones.
    """

    def __init__(self, frames, quads, valid, gain, veil, solid=None, glass=None, window=15, width=320):
        self.quads, self.valid, self.gain, self.veil = quads, valid, gain, veil
        self.window = window
        ref = int(np.flatnonzero(valid)[0]) if valid.any() else 0
        self.size = rect_size(quads[ref], width)
        W, H = self.size
        n = len(valid)
        self.rect = np.zeros((n, H, W, 3), np.uint8)
        self.seen = np.zeros((n, H, W), bool)   # pixel inside the image and not behind an opaque occluder
        self.glass = np.zeros((n, H, W), bool)
        ones = None
        for i, f in enumerate(frames):
            if not valid[i]:
                continue
            if ones is None:
                ones = np.ones(f.shape[:2], np.uint8)
            M = _to_rect(quads[i], self.size)
            self.rect[i] = cv2.warpPerspective(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            ok = ones.copy()
            if solid is not None:
                s = solid(i)
                if s is not None:
                    ok &= ~(cv2.dilate(s.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
            gl = glass(i) if glass is not None else None
            if gl is not None:
                g = cv2.dilate(gl.astype(np.uint8), np.ones((5, 5), np.uint8))
                self.glass[i] = cv2.warpPerspective(g, M, (W, H), flags=cv2.INTER_NEAREST) > 0
                ok &= ~(g > 0)
            self.seen[i] = cv2.warpPerspective(ok, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0) > 0

    def plate(self, i):
        """Bare surface at frame i in the reference exposure (NaN where no nearby frame saw it), plus i's exposure."""
        lo, hi = max(0, i - self.window), min(len(self.valid), i + self.window + 1)
        js = [j for j in range(lo, hi) if self.valid[j]]
        stack = np.stack([(self.rect[j].astype(np.float32) - self.veil[j]) / max(self.gain[j], 1e-3) for j in js])
        stack[~np.stack([self.seen[j] for j in js])] = np.nan
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                p = np.nanmedian(stack, axis=0)
        return p

    def layers(self, i, edge_scale=40.0, floor=3.0):
        """(add, plate_now) in rectified space for frame i.

        add: what to lay over the ad: the positive residual (reflections, glare) away from the old print's edges,
        the full signed residual inside transparent occluders. plate_now: the bare surface at frame i's exposure
        (for fringe decontamination), NaN where unknown."""
        p = self.plate(i)
        now = self.gain[i] * p + self.veil[i]
        r = self.rect[i].astype(np.float32) - now
        known = np.isfinite(r).all(-1)
        r[~known] = 0
        base = np.median(r[known & self.seen[i]], axis=0) if (known & self.seen[i]).any() else np.zeros(3)
        r -= base  # only local variations: uniform offsets belong to the exposure fit
        luma = cv2.cvtColor(np.nan_to_num(p).clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        grad = cv2.magnitude(cv2.Sobel(luma, cv2.CV_32F, 1, 0), cv2.Sobel(luma, cv2.CV_32F, 0, 1))
        calm = np.clip(1 - cv2.GaussianBlur(grad, (0, 0), 1.5) / edge_scale, 0, 1)[..., None]
        reflect = np.clip(r - floor, 0, None) * calm  # light added on top; small differences are noise
        add = np.where(self.glass[i][..., None], 0.0, reflect)
        add[~(self.seen[i] | self.glass[i])] = 0
        return add, now

    def glass_alpha(self, i, now, scale=45.0):
        """Opacity of a transparent occluder per rectified pixel: 0 where it looks like the bare surface behind it
        (clear glass: the ad should show through), 1 where it does not (cheese inside the shaker, its metal cap,
        highlights). A plain signed residual added to the ad turned a clear shaker brown over dark lettering."""
        g = self.glass[i]
        if not g.any():
            return None
        diff = np.abs(self.rect[i].astype(np.float32) - np.nan_to_num(now, nan=-999.0)).max(-1)
        a = np.clip((diff - 8.0) / scale, 0, 1)
        a = cv2.GaussianBlur(a, (0, 0), 1.0)
        a[~g] = np.nan
        return a

    def to_frame(self, img, i, shape):
        """Warp a rectified float image into frame i; returns (image, coverage mask)."""
        h, w = shape[:2]
        Minv = np.linalg.inv(_to_rect(self.quads[i], self.size))
        valid = np.isfinite(img).all(-1).astype(np.uint8) if img.ndim == 3 else np.isfinite(img).astype(np.uint8)
        out = cv2.warpPerspective(np.nan_to_num(img).astype(np.float32), Minv, (w, h), flags=cv2.INTER_LINEAR)
        # Unknown pixels are zeros before warping; linear interpolation bleeds them into their neighbours, which
        # drew a dark rim along hands. Trust only pixels well inside the known region.
        valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
        cover = cv2.warpPerspective(valid, Minv, (w, h), flags=cv2.INTER_NEAREST) > 0
        return out, cover
