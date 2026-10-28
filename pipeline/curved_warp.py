"""Inverse cylindrical RGBA warping and illumination-aware BGR compositing."""
import cv2
import numpy as np
try:
    from pipeline.composite import estimate_blur_and_grain
except Exception:
    def estimate_blur_and_grain(frame, quad):
        return 0.0, 0.0

def _e0_e1(cyl):
    try:
        e0 = float(cyl.get("e0", 0.0))
    except Exception:
        e0 = 0.0
    try:
        e1 = float(cyl.get("e1", 0.0))
    except Exception:
        e1 = 0.0
    if not np.isfinite(e0):
        e0 = 0.0
    if not np.isfinite(e1):
        e1 = 0.0
    return e0, e1

def _label_yl(cyl, yy, phi):
    e0, e1 = _e0_e1(cyl)
    if e0 == 0.0 and e1 == 0.0:
        return yy
    y0f = float(cyl["y0"])
    y1f = float(cyl["y1"])
    if y1f == y0f:
        return yy - e0 * np.cos(phi)
    # Signed and linear in height: rings above the camera bow up, rings below it bow down.
    e = e0 + (e1 - e0) * (yy - y0f) / (y1f - y0f)
    return yy - e * np.cos(phi)

def _geometry(cyl, frame_shape, arc_deg):
    if not np.isfinite(arc_deg) or not 0 < arc_deg <= 180:
        raise ValueError("arc_deg must be finite and in (0, 180]")
    height, width = frame_shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    radii = np.interp(np.arange(height), cyl["rows"], cyl["half_width"]).astype(np.float32)[:, None]
    distance = xx - (float(cyl["a"]) * yy + float(cyl["b"]))
    phi = np.arcsin(np.clip(distance / np.maximum(radii, 1e-6), -1, 1))
    half_arc = np.deg2rad(arc_deg) * 0.5
    yl = _label_yl(cyl, yy, phi)
    y0f = float(cyl["y0"])
    y1f = float(cyl["y1"])
    valid = ((yl >= y0f) & (yl <= y1f) & (radii > 0) & (np.abs(distance) < radii) & (np.abs(phi) <= half_arc))
    return yy, phi, valid, half_arc

def _smooth01(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def warp_to_cylinder(creative_rgba, cyl, frame_shape, arc_deg=110, antialias=True, feather_px=1.5, corner_radius=None) -> np.ndarray:
    """Map angular/vertical coordinates to the creative; feather angular edges."""
    creative = np.asarray(creative_rgba)
    if (creative.ndim != 3 or creative.shape[2] != 4 or min(creative.shape[:2]) == 0 or creative.dtype != np.uint8):
        raise ValueError("creative must be a nonempty RGBA uint8 image")
    yy, phi, valid, half_arc = _geometry(cyl, frame_shape, arc_deg)
    yl = _label_yl(cyl, yy, phi)
    Hf, Wf = yy.shape
    work = creative
    if antialias:
        mid_y = int(np.clip((float(cyl["y0"]) + float(cyl["y1"])) // 2, 0, Hf - 1))
        cnt = int(np.count_nonzero(valid[mid_y])) if Hf > 0 else 0
        if cnt == 0:
            cnt = int(np.max(np.count_nonzero(valid, axis=1))) if np.any(valid) else 0
        screen_w = max(cnt, 1)
        screen_h = max(int(float(cyl["y1"]) - float(cyl["y0"]) + 1), 1)
        target_w = max(8, int(round(screen_w * 2)))
        target_h = max(8, int(round(screen_h * 2)))
        ch, cw = creative.shape[:2]
        new_w, new_h = cw, ch
        if cw > target_w * 1.25:
            new_w = target_w
        if ch > target_h * 1.25:
            new_h = target_h
        if new_w != cw or new_h != ch:
            rgb = cv2.resize(creative[..., :3], (new_w, new_h), interpolation=cv2.INTER_AREA)
            alp = cv2.resize(creative[..., 3], (new_w, new_h), interpolation=cv2.INTER_AREA)
            work = np.dstack([rgb, alp]).astype(np.uint8)
    height, width = work.shape[:2]
    y0f = float(cyl["y0"])
    y1f = float(cyl["y1"])
    denom = max(y1f - y0f, 1e-6)
    map_x = ((phi + half_arc) / (2 * half_arc) * (width - 1)).astype(np.float32)
    map_y = ((yl - y0f) / denom * (height - 1)).astype(np.float32)
    map_x[~valid] = -1
    map_y[~valid] = -1
    warped = cv2.remap(work, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fade = np.clip((half_arc - np.abs(phi)) / np.deg2rad(8), 0, 1)
    fade = fade * fade * (3 - 2 * fade)
    warped[..., 3] = np.rint(warped[..., 3].astype(np.float32) * fade).astype(np.uint8)
    warped[~valid] = 0
    if feather_px and float(feather_px) > 0 and np.any(valid):
        feather = float(feather_px)
        y_fac = _smooth01((yl - y0f + 0.5) / feather) * _smooth01((y1f - yl + 0.5) / feather)
        label_h = float(y1f - y0f + 1)
        r = float(corner_radius) if corner_radius is not None else 0.04 * label_h
        corner_fac = np.ones_like(y_fac, dtype=np.float32)
        if r > 0.5:
            xx = np.broadcast_to(np.arange(Wf, dtype=np.float32)[None, :], (Hf, Wf))
            xmin_row = np.full(Hf, np.nan, dtype=np.float64)
            xmax_row = np.full(Hf, np.nan, dtype=np.float64)
            for y in range(Hf):
                xs = np.where(valid[y])[0]
                if xs.size:
                    xmin_row[y] = float(xs[0]); xmax_row[y] = float(xs[-1])
            y_idx = np.clip(yl.astype(np.int32), 0, Hf - 1)
            xmin_pp = xmin_row[y_idx]; xmax_pp = xmax_row[y_idx]
            yt_top, yt_bot = y0f + r, y1f - r
            xcl = xmin_pp + r; xcr = xmax_pp - r
            d = np.full_like(y_fac, np.inf, dtype=np.float64)
            m_tl = valid & (yl < yt_top) & (xx < xmin_pp + r)
            d[m_tl] = np.sqrt((xx[m_tl] - xcl[m_tl]) ** 2 + (yl[m_tl] - yt_top) ** 2)
            m_tr = valid & (yl < yt_top) & (xx > xmax_pp - r)
            d[m_tr] = np.sqrt((xx[m_tr] - xcr[m_tr]) ** 2 + (yl[m_tr] - yt_top) ** 2)
            m_bl = valid & (yl > yt_bot) & (xx < xmin_pp + r)
            d[m_bl] = np.sqrt((xx[m_bl] - xcl[m_bl]) ** 2 + (yl[m_bl] - yt_bot) ** 2)
            m_br = valid & (yl > yt_bot) & (xx > xmax_pp - r)
            d[m_br] = np.sqrt((xx[m_br] - xcr[m_br]) ** 2 + (yl[m_br] - yt_bot) ** 2)
            is_corner = m_tl | m_tr | m_bl | m_br
            corner_fac[is_corner] = _smooth01((r - d[is_corner] + 0.5) / feather).astype(np.float32)
        combined = (y_fac.astype(np.float32) * corner_fac).astype(np.float32)
        warped[..., 3] = np.rint(warped[..., 3].astype(np.float32) * combined).astype(np.uint8)
        warped[~valid] = 0
    return warped

def composite_curved(frame_bgr, creative_rgba, cyl, occluder_mask=None, arc_deg=110, shade=0.7, specular=0.5, antialias=True, feather_px=1.5, corner_radius=None) -> np.ndarray:
    """Shade the wrapped label, preserving every unpainted pixel exactly."""
    frame = np.asarray(frame_bgr)
    if (frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) == 0 or frame.dtype != np.uint8):
        raise ValueError("frame must be a nonempty BGR uint8 image")
    if not np.isfinite(shade) or shade < 0:
        raise ValueError("shade must be finite and nonnegative")
    if not np.isfinite(specular) or specular < 0:
        raise ValueError("specular must be finite and nonnegative")
    warped = warp_to_cylinder(creative_rgba, cyl, frame.shape, arc_deg, antialias=antialias, feather_px=feather_px, corner_radius=corner_radius)
    alpha = warped[..., 3].astype(np.float32) / 255.0
    visible = alpha > 0
    active = visible.copy()
    if occluder_mask is not None:
        occluder = np.asarray(occluder_mask, dtype=bool)
        if occluder.shape != frame.shape[:2]:
            raise ValueError("occluder_mask must match the frame dimensions")
        active &= ~occluder
    result = frame.copy()
    if not np.any(active):
        return result
    H, W = frame.shape[:2]
    luminance = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2GRAY)
    yy, phi, valid, _ = _geometry(cyl, frame.shape, arc_deg)
    xx = np.broadcast_to(np.arange(W, dtype=np.float32)[None, :], (H, W))
    radii = np.interp(np.arange(H), cyl["rows"], cyl["half_width"]).astype(np.float32)[:, None]
    distance = xx - (float(cyl["a"]) * yy + float(cyl["b"]))
    y0f, y1f = float(cyl["y0"]), float(cyl["y1"])
    mug_mask = (yy >= y0f) & (yy <= y1f) & (radii > 0) & (np.abs(distance) < radii)
    yc = (y0f + y1f) * 0.5
    half_h = max((y1f - y0f) * 0.5, 1.0)
    label_w = int(np.max(np.count_nonzero(valid, axis=1))) if np.any(valid) else W
    label_h = max(float(y1f - y0f + 1), 1.0)
    # Robust parametric diffuse fit: a+b*cos(phi)+c*sin(phi)+d*cos(2phi) x low-order vertical.
    phi64 = phi.astype(np.float64); yy64 = yy.astype(np.float64); lum64 = luminance.astype(np.float64)
    fit_mask = mug_mask
    C_map = np.full((H, W), float(np.median(luminance[fit_mask])) if np.any(fit_mask) else 150.0, dtype=np.float64)
    R_map = np.ones((H, W), dtype=np.float64)
    if np.count_nonzero(fit_mask) >= 20:
        ph_m = phi64[fit_mask]; L_m = lum64[fit_mask]; y_m = yy64[fit_mask]
        A = np.column_stack([np.ones_like(ph_m), np.cos(ph_m), np.sin(ph_m), np.cos(2.0 * ph_m)])
        N = A.shape[0]
        inl = np.ones(N, dtype=bool)
        coeff = np.linalg.lstsq(A[inl], L_m[inl], rcond=None)[0]
        for _ in range(6):
            pred = A @ coeff
            res = L_m - pred
            med = float(np.median(res[inl]))
            mad = float(np.median(np.abs(res[inl] - med)))
            srob = 1.4826 * mad
            thr = max(2.5 * srob, 7.0)
            new_inl = np.abs(res - med) < thr
            if int(np.count_nonzero(new_inl)) < max(20, int(0.3 * N)):
                break
            if np.array_equal(new_inl, inl):
                break
            inl = new_inl
            coeff = np.linalg.lstsq(A[inl], L_m[inl], rcond=None)[0]
        C_map = (coeff[0] + coeff[1] * np.cos(phi64) + coeff[2] * np.sin(phi64) + coeff[3] * np.cos(2.0 * phi64))
        C_map = np.maximum(C_map, 1.0)
        Cm_inl = (A[inl] @ coeff)
        Cm_inl = np.maximum(Cm_inl, 1.0)
        r_inl = L_m[inl] / Cm_inl
        yn_inl = (y_m[inl] - yc) / half_h
        B = np.column_stack([np.ones_like(yn_inl), yn_inl, yn_inl * yn_inl])
        p, _, _, _ = np.linalg.lstsq(B, r_inl, rcond=None)
        yn_full = (yy64 - yc) / half_h
        R_map = p[0] + p[1] * yn_full + p[2] * yn_full * yn_full
        R_map = np.maximum(R_map, 0.2)
    ref = valid if np.any(valid) else fit_mask
    medC = float(np.median(C_map[ref])) if np.any(ref) else max(float(np.median(C_map)), 1e-6)
    medR = float(np.median(R_map[ref])) if np.any(ref) else 1.0
    medC = max(medC, 1e-6); medR = max(abs(medR), 1e-6)
    if shade == 0:
        shading = np.ones((H, W), dtype=np.float32)
    else:
        col_f = np.clip((C_map / medC) ** float(shade), 0.3, 1.6)
        row_f = np.clip((R_map / medR) ** float(shade), 0.6, 1.4)
        shading = (col_f * row_f).astype(np.float32)
    expected = (C_map * R_map).astype(np.float32)
    resid = (luminance - expected).astype(np.float32)
    sigX = max(4.0, float(label_w) * 0.04)
    sigY = max(16.0, float(label_h) * 0.20)
    smooth_res = cv2.GaussianBlur(resid, (0, 0), sigX, sigY, borderType=cv2.BORDER_REPLICATE)
    highlight = np.clip(smooth_res, 0, None).astype(np.float32)
    label = warped[..., :3][..., ::-1].astype(np.float32) * shading[..., None]
    if specular > 0:
        label = label + float(specular) * highlight[..., None]
    try:
        ys, xs = np.where(valid)
        if ys.size:
            x0, x1 = int(xs.min()), int(xs.max())
            quad = np.array([[x0, float(cyl["y0"])], [x1, float(cyl["y0"])], [x1, float(cyl["y1"])], [x0, float(cyl["y1"])]], dtype=np.float32)
            b_sig, g_std = estimate_blur_and_grain(frame, quad)
        else:
            b_sig, g_std = 0.0, 0.0
    except Exception:
        b_sig, g_std = 0.0, 0.0
    # Single screen-space blur: remap already softens, so use 0.8x measured sigma.
    if np.isfinite(b_sig) and b_sig > 0.3:
        bs = float(b_sig) * 0.8
        label = cv2.GaussianBlur(label, (0, 0), bs, borderType=cv2.BORDER_REPLICATE)
    if np.isfinite(g_std) and g_std > 1e-9:
        rng = np.random.default_rng()  # fresh grain every frame, as in real footage (a fixed seed freezes it)
        label = label + rng.normal(0.0, float(g_std), size=label.shape).astype(np.float32)
    weight = alpha[active, None]
    mixed = frame[active].astype(np.float32) * (1 - weight) + label[active] * weight
    result[active] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    return result
