import sys, json, numpy as np, cv2
sys.path.insert(0, '.')
from scipy.optimize import minimize
from pipeline.maskstore import MaskStore
W, H, cx, cy = 1280, 720, 640., 360.
S = 'cache/patio_waiter-scene/'
plates = {'pizza': MaskStore(S + 'obj_plate_0.npz'), 'far_plate': MaskStore(S + 'obj_plate_1.npz'),
          'big_plate': MaskStore(S + 'obj_plate_2.npz')}

def rim(store, i):
    """Outer contour points of the mask, minus image-border points; RANSAC-trimmed to an ellipse."""
    if not store.any(i):
        return None
    m = store[i].astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    p = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    p = p[(p[:, 0] > 3) & (p[:, 0] < W - 4) & (p[:, 1] > 3) & (p[:, 1] < H - 4)]
    if len(p) < 40:
        return None
    for _ in range(3):  # drop points far from the fitted ellipse (other objects touching the outline)
        (ex, ey), (a, b), ang = cv2.fitEllipse(p.astype(np.float32))
        t = np.radians(ang); c, s = np.cos(t), np.sin(t)
        x, y = p[:, 0] - ex, p[:, 1] - ey
        u, v = (c * x + s * y) / (a / 2), (-s * x + c * y) / (b / 2)
        r = np.abs(np.hypot(u, v) - 1) * min(a, b) / 2
        keep = r < max(1.5, np.percentile(r, 80))
        if keep.all():
            break
        p = p[keep]
    return p

def rect(f, n):
    """Homography that turns the camera to look straight along -n (plane fronto-parallel)."""
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    n = n / np.linalg.norm(n)
    z = np.array([0, 0, 1.])
    v = np.cross(n, z); s = np.linalg.norm(v); c = n @ z
    R = np.eye(3) if s < 1e-12 else cv2.Rodrigues(v / s * np.arctan2(s, c))[0]
    return K @ R @ np.linalg.inv(K)

def roundness(points, Hr):
    q = cv2.perspectiveTransform(points.reshape(-1, 1, 2), Hr).reshape(-1, 2)
    (_, _), (a, b), _ = cv2.fitEllipse(q.astype(np.float32))
    return min(a, b) / max(a, b)

def nvec(th):  # table normal from (tilt, roll-ish) angles; camera looks down at the table => n points up (-y)
    a, b = th
    return np.array([np.sin(b), -np.cos(a) * np.cos(b), np.sin(a) * np.cos(b)])

def fit_frame(f, rims, start=(0.3, 0.)):
    cost = lambda th: sum((1 - roundness(p, rect(f, nvec(th)))) ** 2 for p in rims)
    best = min((minimize(cost, x0, method='Nelder-Mead', options=dict(xatol=1e-5, fatol=1e-10)) for x0 in
                [start, (0.6, 0.), (1.0, 0.), (0.3, 0.2), (0.3, -0.2)]), key=lambda r: r.fun)
    return best.fun, best.x

if __name__ == '__main__':
    frames = [114, 122, 130, 138, 146, 156]
    rims = {i: [r for r in (rim(s, i) for s in plates.values()) if r is not None] for i in frames}
    print({i: len(r) for i, r in rims.items()})
    for f in [600, 800, 950, 1100, 1250, 1400, 1600, 1900, 2300, 3000]:
        res = [fit_frame(f, rims[i]) for i in frames if len(rims[i]) >= 2]
        tot = sum(r[0] for r in res)
        tilt = [np.degrees(r[1][0]) for r in res]
        print(f"f={f:5d} total non-roundness {tot:.2e}  per-frame tilt {np.round(tilt, 1)}")
