"""Photometric tracks in display-referred 0..255 units.

Frames are re-iterable BGR arrays; quads are TL,TR,BR,BL in screen pixels.
Exposure outputs are RGB (artwork order). Noise parameters use BGR.
Defocus is uncapped screen-pixel sigma, with a reference floor. motion_px
is an (N,2) shutter displacement, not inter-frame velocity; scalar (N,)
speeds permit only near-still measurements. Missing evidence is interpolated
within valid runs; a wholly unsupported run uses a conservative fallback.
No source codec QP or physical sensor-noise claim is made from decoded pixels.
"""
from functools import lru_cache

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import nnls


def _smooth(values, valid, default, width=5):
    x = np.asarray(values, float).copy()
    one = x.ndim == 1
    if one:
        x = x[:, None]
    default = np.broadcast_to(default, x.shape[1:])
    out = np.broadcast_to(default, x.shape).copy()
    ends = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(int)))
    for a, b in zip(ends[::2], ends[1::2]):
        t = np.arange(a, b)
        for c in range(x.shape[1]):
            good = np.isfinite(x[a:b, c])
            if good.any():
                y = np.interp(t, t[good], x[a:b, c][good])
                y = median_filter(
                    y, size=min(width, 2 * ((b-a-1)//2)+1), mode="nearest"
                )
                out[a:b, c] = gaussian_filter1d(y, 1.0, mode="nearest")
    return out[:, 0] if one else out


def _quad(q):
    q = np.asarray(q, np.float32)
    return (
        q.shape == (4, 2)
        and np.isfinite(q).all()
        and cv2.isContourConvex(q)
        and cv2.contourArea(q) > 16
    )


def _rect(frame, q, size, occ=None, ring=False):
    w, h = size
    q = np.asarray(q, np.float32)
    if ring:
        q = q.mean(0) + 1.4 * (q - q.mean(0))
    dst = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    H = cv2.getPerspectiveTransform(q, dst)
    support = np.ones(frame.shape[:2], np.uint8)
    if occ is not None:
        support &= ~cv2.dilate(
            np.asarray(occ, np.uint8), np.ones((7, 7), np.uint8)
        )
    r = cv2.warpPerspective(frame.astype(np.float32), H, size)
    m = cv2.warpPerspective(support, H, size, flags=cv2.INTER_NEAREST)
    m = cv2.erode(
        m, np.ones((5, 5), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0
    ) > 0
    if ring:
        # Exact interior of the unexpanded quad, with a small safety margin.
        x, y = int(np.ceil(w/7)), int(np.ceil(h/7))
        m[y-2:h-y+2, x-2:w-x+2] = False
    return r, m, H


def _fit(x, y):
    """Robust affine fit; a blank/dark reference supports only a ratio."""
    g = float(np.median((y + 1) / (x + 1)))
    if np.ptp(np.percentile(x, [10, 90])) < 25:
        return np.clip(g, .35, 3), 0.
    keep = np.ones(len(x), bool)
    for _ in range(5):
        A = np.column_stack((x[keep], np.ones(keep.sum())))
        g, b = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        e = y - g*x - b
        mad = 1.4826 * np.median(np.abs(e - np.median(e)))
        keep = np.abs(e - np.median(e)) <= max(2., 2.8*mad)
        if keep.sum() < 80:
            break
    if not (.1 <= g <= 4 and -60 <= b <= 120):
        # A failed textured affine fit is missing evidence, not a blank surface.
        return np.nan, np.nan
    return g, b


def channel_exposure(
    frames, quads, valid, ref, occ_fn, size=(200, 140), ring=False
):
    """Return gain, veil (N,3), RGB; y = gain*x + veil on matched pixels."""
    n = len(valid)
    gain, veil = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    if not 0 <= ref < n or not valid[ref] or not _quad(quads[ref]):
        return np.ones((n, 3)), np.zeros((n, 3))
    r0, m0, _ = _rect(
        frames[ref], quads[ref], size, occ_fn(ref) if occ_fn else None, ring
    )
    r0 = cv2.GaussianBlur(r0, (0, 0), .8)
    for i, f in enumerate(frames):
        if not valid[i] or not _quad(quads[i]):
            continue
        r, m, _ = _rect(
            f, quads[i], size, occ_fn(i) if occ_fn else None, ring
        )
        r = cv2.GaussianBlur(r, (0, 0), .8)
        for c in range(3):
            k = m & m0 & (r0[..., c] > 3) & (r0[..., c] < 252)
            k &= (r[..., c] > 3) & (r[..., c] < 252)
            if k.sum() >= 100:
                gain[i, 2-c], veil[i, 2-c] = _fit(
                    r0[..., c][k], r[..., c][k]
                )
    gain = np.clip(_smooth(gain, valid, 1.), .1, 4.)
    veil = np.clip(_smooth(veil, valid, 0.), -60, 120)
    gain[ref], veil[ref] = 1, 0
    return gain, veil


def _reference_sigma(frame, q, occ):
    """Fit isolated step edges, rejecting texture and occluded profiles."""
    from .composite import _edge_blur

    roi, m, _ = _native(frame, q, occ)
    if m.sum() < 100:
        return 0.
    y = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    s = cv2.GaussianBlur(y, (0, 0), .8)
    gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, scale=1/8)
    gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, scale=1/8)
    return _edge_blur(y, m, gx, gy, cv2.magnitude(gx, gy), 0.)


def defocus_track(frames, quads, valid, ref, occ_fn, motion_px=None):
    """Absolute sigma, >= reference sigma, in output-frame pixels.

    Fit attenuation of the same rectified texture. Only Fourier frequencies
    with <.08 cycles of shutter travel are used, so line-motion attenuation
    is <2.1%. A scalar motion track rejects speeds >.6 px. At least 60 usable
    frequencies and source texture are required; no evidence means no extra blur.
    """
    n = len(valid)
    raw = np.full(n, np.nan)
    if not 0 <= ref < n or not valid[ref] or not _quad(quads[ref]):
        return np.zeros(n)
    f0 = frames[ref]
    base = _reference_sigma(
        f0, quads[ref], occ_fn(ref) if occ_fn else None
    )
    q0 = np.asarray(quads[ref], float)
    w = int(np.clip(
        (np.linalg.norm(q0[1]-q0[0]) + np.linalg.norm(q0[2]-q0[3]))/2,
        64, 384
    ))
    h = int(np.clip(
        w * np.linalg.norm(q0[3]-q0[0]) /
        max(np.linalg.norm(q0[1]-q0[0]), 1),
        64, 384
    ))
    r0, m0, H0 = _rect(
        f0, q0, (w, h), occ_fn(ref) if occ_fn else None
    )
    y0 = cv2.cvtColor(r0, cv2.COLOR_BGR2GRAY)
    motion = (
        np.zeros((n, 2)) if motion_px is None
        else np.asarray(motion_px, float)
    )
    if motion.shape not in ((n,), (n, 2)):
        raise ValueError("motion_px must be (N,) or (N,2)")
    if motion_px is None:
        centres = np.nanmean(quads, axis=1)
        # Without an explicit shutter vector, only trust near-still frames.
        motion = (
            np.linalg.norm(np.gradient(centres, axis=0), axis=1)
            if n > 1 else np.zeros(n)
        )
    c0 = q0.mean(0)
    d0 = H0[2, :2]@c0 + H0[2, 2]
    p0 = (H0[:2, :2]@c0 + H0[:2, 2])/d0
    J0 = (H0[:2, :2] - p0[:, None]*H0[2, :2])/d0
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    u0 = J0[0, 0]*fx + J0[1, 0]*fy
    v0 = J0[0, 1]*fx + J0[1, 1]*fy
    freq0 = u0*u0 + v0*v0
    window = np.outer(np.hanning(h), np.hanning(w))

    for i, f in enumerate(frames):
        if not valid[i] or not _quad(quads[i]):
            continue
        if motion.ndim == 1 and (
            not np.isfinite(motion[i]) or motion[i] > .6
            or motion[ref] > .6
        ):
            continue
        r, m, H = _rect(
            f, quads[i], (w, h), occ_fn(i) if occ_fn else None
        )
        k = m & m0
        if k.mean() < .7:
            continue
        y = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        x, z = y0[k], y[k]
        if np.std(x) < 3:
            continue
        g, b = _fit(x, z)
        # Reject local glare/occluders missed by masks, before the shared window.
        e = cv2.GaussianBlur(y-g*y0-b, (0, 0), 3)
        k &= np.abs(e-np.median(e[k])) < max(
            10., 4*np.median(np.abs(e[k]-np.median(e[k])))
        )
        if k.mean() < .7:
            continue
        win = window*k
        A = np.fft.fft2((y0-np.mean(y0[k]))*win)
        B = np.fft.fft2((y-np.mean(y[k]))*win)

        # Native-screen displacement -> rectified displacement at quad centre.
        c = np.asarray(quads[i], float).mean(0)
        den = H[2, :2]@c + H[2, 2]
        p = (H[:2, :2]@c + H[:2, 2])/den
        J = (H[:2, :2] - p[:, None]*H[2, :2])/den
        ux = J[0, 0]*fx + J[1, 0]*fy
        uy = J[0, 1]*fx + J[1, 1]*fy
        freq2 = ux*ux + uy*uy
        take = (freq2 > .025**2) & (freq2 < .28**2)
        if motion.ndim == 2:
            if not np.isfinite(motion[[i, ref]]).all():
                continue
            take &= np.abs(ux*motion[i, 0] + uy*motion[i, 1]) < .08
            take &= np.abs(u0*motion[ref, 0] + v0*motion[ref, 1]) < .08

        power = np.abs(A)**2
        take &= (
            power > max(1., np.percentile(power[take], 40))
            if take.any() else False
        )
        if take.sum() < 60:
            continue
        xx = freq2[take]
        coherence = abs(np.vdot(A[take], B[take])) / max(
            np.linalg.norm(A[take])*np.linalg.norm(B[take]), 1e-8
        )
        if coherence < .5:
            continue
        yy = np.log(np.maximum(np.abs(B[take])**2, 1e-6)/power[take])
        yy -= 4*np.pi**2*base**2*(freq0[take]-freq2[take])
        keep = np.ones(len(xx), bool)
        for _ in range(4):
            slope, intercept = np.polyfit(xx[keep], yy[keep], 1)
            e = yy-(slope*xx+intercept)
            keep = np.abs(e-np.median(e)) < max(
                .15, 3*1.4826*np.median(np.abs(e-np.median(e)))
            )
        raw[i] = np.clip(-slope/(4*np.pi**2), 0., 16.)

    raw[ref] = 0.
    variance = _smooth(raw, valid, 0.)
    sigma = np.sqrt(base*base + np.maximum(variance, 0))
    sigma[ref] = base
    return sigma.astype(np.float32)


def _native(frame, q, occ=None):
    """Native-grid surface + 16px ring; never resample sensor/codec statistics."""
    h, w = frame.shape[:2]
    q = np.asarray(q, float)
    x0, y0 = np.maximum(np.floor(q.min(0)).astype(int)-18, 0)
    x1, y1 = np.minimum(np.ceil(q.max(0)).astype(int)+19, [w, h])
    if x1 <= x0 or y1 <= y0:
        return (
            np.zeros((1, 1, 3), np.float32),
            np.zeros((1, 1), bool),
            (0, 0),
        )
    image = frame[y0:y1, x0:x1].astype(np.float32)
    m = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.rint(q-[x0, y0]).astype(np.int32), 1)
    m = cv2.dilate(m, np.ones((33, 33), np.uint8))
    if occ is not None:
        o = np.asarray(occ[y0:y1, x0:x1], np.uint8)
        m &= ~cv2.dilate(o, np.ones((9, 9), np.uint8))
    m = cv2.erode(
        m, np.ones((5, 5), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    return image, m > 0, (int(x0), int(y0))


@lru_cache(maxsize=8)
def _plane(size):
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    A = np.column_stack((np.ones(size*size), xx.ravel(), yy.ravel()))
    return A, np.linalg.pinv(A)


def _patches(image, mask, size=12):
    A, inv = _plane(size)
    records = []
    smooth = cv2.GaussianBlur(image, (0, 0), 1.4)
    stride = size * max(1, int(np.sqrt(mask.size/384)/size))
    for y in range(0, image.shape[0]-size+1, stride):
        for x in range(0, image.shape[1]-size+1, stride):
            if not mask[y:y+size, x:x+size].all():
                continue
            p = image[y:y+size, x:x+size]
            if p.min() <= 2 or p.max() >= 253:
                continue
            v = p.reshape(-1, 3)
            residual = (v-A@(inv@v)).reshape(size, size, 3)
            z = smooth[y:y+size, x:x+size].reshape(-1, 3)
            curvature = float(np.mean((z-A@(inv@z))**2))
            var = np.var(residual, axis=(0, 1), ddof=3)
            records.append((
                float(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).mean()),
                var, curvature, residual
            ))

    # A bright bin containing only print edges is not shot-noise evidence.
    if not records:
        return []
    quiet = np.percentile(
        [[r[2], np.mean(r[1])] for r in records], 10, axis=0
    )
    records = [
        r for r in records
        if r[2] <= max(.2, 4*quiet[0])
        and np.mean(r[1]) <= max(2., 16*quiet[1])
    ]
    selected = []
    for lo in range(0, 256, 32):
        group = [r for r in records if lo <= r[0] < lo+32]
        group.sort(key=lambda r: r[2])
        selected.extend(group[:max(1, int(np.ceil(.4*len(group))))])
    return selected


def noise_model(frames, quads, valid, occ_fn):
    """Per-frame dicts: BGR shot/read variances, correlation blur/covariance.

    Var(channel | encoded luma L) = shot*L + read. This is a decoded-image
    approximation including denoising/compression, fitted only on flat patches.
    Insufficient luma span yields a constant variance.
    """
    n = len(valid)
    raw = np.full((n, 16), np.nan)
    supports = np.zeros(n, int)
    for i, f in enumerate(frames):
        if not valid[i] or not _quad(quads[i]):
            continue
        image, m, _ = _native(
            f, quads[i], occ_fn(i) if occ_fn else None
        )
        patches = _patches(image, m)
        supports[i] = len(patches)
        if len(patches) < 4:
            continue
        L = np.array([p[0] for p in patches])
        V = np.array([p[1] for p in patches])
        coef = np.zeros((2, 3))
        for c in range(3):
            if np.ptp(np.percentile(L, [10, 90])) >= 35:
                A = np.column_stack((L, np.ones(len(L))))
                keep = np.ones(len(L), bool)
                for _ in range(3):
                    coef[:, c] = nnls(A[keep], V[keep, c])[0]
                    e = V[:, c]-A@coef[:, c]
                    keep = np.abs(e-np.median(e)) <= max(
                        .2, 3*1.4826*np.median(np.abs(e-np.median(e)))
                    )
            else:
                coef[1, c] = np.median(V[:, c])

        R = np.stack([p[3] for p in patches])
        # Lag-one covariance, pooling both axes/channels.
        var = max(float(np.mean(R*R)), 1e-8)
        rho = .5 * (
            np.mean(R[:, :, 1:]*R[:, :, :-1])
            + np.mean(R[:, 1:]*R[:, :-1])
        ) / var
        candidates = np.linspace(0, 1.2, 49)
        expected = []
        for s in candidates:
            g = _kernel(float(s))
            expected.append(float(
                np.sum(g[1:]*g[:-1])/np.sum(g*g)
            ))
        sigma = float(candidates[np.argmin(
            np.abs(np.array(expected)-max(0., rho))
        )])
        samples = R.reshape(-1, 3)
        sd = np.maximum(np.std(samples, axis=0), 1e-5)
        C = (
            np.corrcoef((samples/sd).T)
            if np.all(sd > 1e-5) else np.eye(3)
        )
        C = np.nan_to_num(C)
        np.fill_diagonal(C, 1.)
        C = .95*C + .05*np.eye(3)
        raw[i] = np.r_[coef.ravel(), sigma, C.ravel()]

    defaults = np.r_[np.zeros(7), np.eye(3).ravel()]
    values = _smooth(raw, valid, defaults)
    return [
        dict(
            shot=np.clip(v[:3], 0, 2),
            read=np.clip(v[3:6], 0, 100),
            corr_sigma=float(np.clip(v[6], 0, 1.2)),
            channel_corr=v[7:].reshape(3, 3),
            support=int(supports[i]),
        )
        for i, v in enumerate(values)
    ]


def _kernel(sigma):
    if sigma < .15:
        return np.array([0., 1., 0.])
    radius = max(1, int(np.ceil(3*sigma)))
    x = np.arange(-radius, radius+1)
    g = np.exp(-.5*(x/sigma)**2)
    return g/g.sum()


def add_noise(img, region, params, rng):
    """Add target noise to clean BGR; region is bool or alpha in [0,1].

    Existing variance, if known, may be supplied as params['existing_var'].
    Caller owns the seeded Generator; no global RNG or frozen per-frame seed.
    """
    alpha = np.asarray(region, np.float32)
    if alpha.shape != img.shape[:2] or not np.isfinite(alpha).all():
        raise ValueError("region must match image height/width")
    alpha = np.clip(alpha, 0, 1)
    ys, xs = np.nonzero(alpha > 0)
    if len(xs) == 0:
        return img.copy()
    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    g = _kernel(float(params.get("corr_sigma", 0.)))
    pad = len(g)//2
    shape = (y1-y0+2*pad, x1-x0+2*pad, 3)
    C = np.asarray(params.get("channel_corr", np.eye(3)), float)
    eigen, vectors = np.linalg.eigh((C+C.T)/2)
    root = (
        vectors @ np.diag(np.sqrt(np.maximum(eigen, 1e-6))) @ vectors.T
    )
    z = (rng.standard_normal(shape)@root.T).astype(np.float32)
    z = cv2.sepFilter2D(z, -1, g, g)/(g@g)
    z = z[pad:pad+y1-y0, pad:pad+x1-x0]
    p = img[y0:y1, x0:x1].astype(np.float32)
    L = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    existing = np.asarray(params.get("existing_var", 0.), np.float32)
    if existing.ndim >= 2 and existing.shape[:2] == img.shape[:2]:  # given for the whole image: crop like p
        existing = existing[y0:y1, x0:x1]
    variance = np.maximum(
        0,
        L[..., None]*params["shot"] + params["read"]
        - existing,
    )
    p += z*np.sqrt(variance)*alpha[y0:y1, x0:x1, None]
    out = img.copy()
    if img.dtype == np.uint8:
        p = np.rint(np.clip(p, 0, 255)).astype(np.uint8)
    out[y0:y1, x0:x1] = p
    return out


def estimate_compression(frames, quads, valid, occ_fn):
    """Decoded-image proxies, not a codec-QP estimator. Native grid only.

    Measure excess differences at each possible 8px block phase on flat pixels.
    Chroma softening requires co-located Y/chroma edges; otherwise leave it alone.
    quality=100 is the explicit byte-exact bypass.
    """
    n = len(valid)
    raw = np.full((n, 4), np.nan)
    phases = np.zeros((n, 2), int)
    for i, f in enumerate(frames):
        if not valid[i] or not _quad(quads[i]):
            continue
        p, m, (x0, y0) = _native(
            f, quads[i], occ_fn(i) if occ_fn else None
        )
        if m.sum() < 200:
            continue
        ycc = cv2.cvtColor(p, cv2.COLOR_BGR2YCrCb)
        Y = ycc[..., 0]
        low = cv2.GaussianBlur(Y, (0, 0), 1.5)
        gx, gy = np.gradient(low)
        flat = m & (np.hypot(gx, gy) < 2)
        excesses = []
        for axis, offset in ((1, x0), (0, y0)):
            d = np.abs(np.diff(Y, axis=axis))
            ok = (
                flat[:, 1:] & flat[:, :-1]
                if axis else flat[1:] & flat[:-1]
            )
            coord = (np.arange(d.shape[axis])+offset+1)%8
            means = []
            for phase in range(8):
                k = np.broadcast_to(
                    (coord == phase)[None, :]
                    if axis else (coord == phase)[:, None],
                    d.shape,
                ) & ok
                means.append(
                    float(np.mean(d[k])) if k.sum() >= 20 else np.nan
                )
            if np.isfinite(means).all():
                phase = int(np.argmax(means))
                excesses.append(max(
                    0., means[phase]-np.median(means)
                ))
                phases[i, 0 if axis else 1] = phase
        excess = float(np.mean(excesses)) if excesses else 0.
        # Subtract the positive sampling bias from choosing the strongest phase.
        qstep = np.clip(4*max(0., excess-.15), 0, 6)

        ratios = []
        dy, dx = np.gradient(Y)
        ey = np.hypot(dx, dy)
        for c in (1, 2):
            ch = ycc[..., c]
            cy, cx = np.gradient(ch)
            ec = np.hypot(cx, cy)
            aligned = m & (ey > 4) & (ec > 1)
            aligned &= np.abs(dx*cx+dy*cy) > .8*ey*ec
            if aligned.sum() < 80:
                continue
            cy2, cx2 = np.gradient(
                cv2.GaussianBlur(ch, (0, 0), 1.)
            )
            yy2, yx2 = np.gradient(
                cv2.GaussianBlur(Y, (0, 0), 1.)
            )
            rc = np.mean(ec[aligned]**2) / max(
                np.mean((cx2*cx2+cy2*cy2)[aligned]), 1e-6
            )
            ry = np.mean(ey[aligned]**2) / max(
                np.mean((yx2*yx2+yy2*yy2)[aligned]), 1e-6
            )
            ratios.append(np.sqrt(max(
                0., ry/max(rc, 1e-6)-1
            )))
        chroma = (
            float(np.clip(np.median(ratios), 0, 1.2))
            if ratios else 0.
        )
        raw[i] = [qstep, chroma, excess, len(ratios)]

    values = _smooth(raw, valid, 0.)
    # Pool stream-grid evidence to prevent frame-to-frame grid shimmer.
    weights = np.maximum(
        0., np.nan_to_num(raw[:, 2])-.15
    )**2 * valid
    phase = tuple(
        int(np.argmax(np.bincount(
            phases[:, c], weights=weights, minlength=8
        )))
        for c in range(2)
    )
    subsample = (
        bool(np.median(values[valid, 1]) > .45)
        if np.any(valid) else False
    )
    return [
        dict(
            quality=99 if v[0] > 0 or v[1] > .05 else 100,
            qstep=float(v[0]),
            chroma_sigma=float(v[1]),
            subsample=subsample,
            strength=.35,
            phase=phase,
            block_excess=float(v[2]),
            chroma_support=float(v[3]),
        )
        for i, v in enumerate(values)
    ]


@lru_cache(maxsize=1)
def _dct_matrix():
    k = np.arange(8)[:, None]
    C = np.cos(np.pi*(2*np.arange(8)+1)*k/16)*.5
    C[0] /= np.sqrt(2)
    return C.astype(np.float32)


def match_compression(ad_region_bgr, params):
    """Mild chroma loss + 8x8 DCT quantisation, preserving block DC.

    origin=(x,y) locates this crop on the output grid; phase is the measured
    source-grid phase. Process a padded ROI and blend only through ad alpha.
    quality is a bypass switch, not a JPEG quality/QP conversion.
    """
    if params.get("quality", 100) >= 100:
        return ad_region_bgr.copy()
    image = np.asarray(ad_region_bgr)
    x0, y0 = params.get("origin", (0, 0))
    px, py = params.get("phase", (0, 0))
    left, top = (x0-px)%8, (y0-py)%8
    h, w = image.shape[:2]
    bottom, right = (-(h+top))%8, (-(w+left))%8
    p = cv2.copyMakeBorder(
        image.astype(np.float32), top, bottom, left, right,
        cv2.BORDER_REFLECT_101,
    )
    ycc = cv2.cvtColor(p, cv2.COLOR_BGR2YCrCb)
    sigma = float(params.get("chroma_sigma", 0))
    for c in (1, 2):
        ch = ycc[..., c]
        if sigma > .05:
            ch = cv2.GaussianBlur(ch, (0, 0), sigma)
        if params.get("subsample", False):
            ch = cv2.resize(
                cv2.resize(
                    ch, (p.shape[1]//2, p.shape[0]//2),
                    interpolation=cv2.INTER_AREA,
                ),
                p.shape[1::-1], interpolation=cv2.INTER_LINEAR,
            )
        ycc[..., c] = ch
    step = float(params.get("qstep", 0))
    if step > 0:
        C = _dct_matrix()
        q = step*(1+np.add.outer(np.arange(8), np.arange(8))/8)
        for c in range(3):
            ch = ycc[..., c]
            blocks = ch.reshape(
                ch.shape[0]//8, 8, ch.shape[1]//8, 8
            ).transpose(0, 2, 1, 3)
            d = C@blocks@C.T
            dq = np.rint(d/q)*q
            dq[..., 0, 0] = d[..., 0, 0]
            restored = C.T@dq@C
            ch[:] = restored.transpose(0, 2, 1, 3).reshape(ch.shape)
    out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)[top:top+h, left:left+w]
    s = float(np.clip(params.get("strength", .35), 0, 1))
    out = (1-s)*image.astype(np.float32)+s*out
    return (
        np.rint(np.clip(out, 0, 255)).astype(np.uint8)
        if image.dtype == np.uint8 else out
    )
