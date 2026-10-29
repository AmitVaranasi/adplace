"""Flicker of the inserted can vs the real dip bowl beside it: stabilise on the table track (real objects stand
still), then measure each region's frame-to-frame change.  python eval/can_flicker.py outputs/<render>.mp4"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.maskstore import MaskStore  # noqa: E402
from scripts.place3d import CanPlacement  # noqa: E402

c = CanPlacement()
H = np.load(ROOT / "cache/patio_waiter-scene/table_track_v3.npz")["H"]
cap = cv2.VideoCapture(sys.argv[1])
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
(x0, y0, x1, y1), _, cov, _ = c.layer(156, c.frames[156], "fizzwell")
can = np.zeros((720, 1280), np.uint8)
can[y0:y1, x0:x1] = cov > 0.9
bowl = MaskStore(str(ROOT / "cache/patio_waiter-scene/obj_bowl_1.npz"))[156].astype(np.uint8)
for lo, hi in ((92, 112), (112, 157)):
    stab = [cv2.warpPerspective(frames[i], H[i], (1280, 720)).astype(np.float32) for i in range(lo, hi)]
    vis = [cv2.warpPerspective(np.ones((720, 1280), np.float32), H[i], (1280, 720)) > 0.99 for i in range(lo, hi)]
    for name, m in (("can", can), ("dip bowl (real)", bowl)):
        m = cv2.erode(m, np.ones((5, 5), np.uint8)) > 0
        d = [np.abs(stab[k + 1] - stab[k])[m & vis[k] & vis[k + 1]].mean() for k in range(len(stab) - 1)
             if (m & vis[k] & vis[k + 1]).sum() > 200]
        if d:
            print(f"frames {lo}-{hi - 1} {name:16s} frame-to-frame change: median {np.median(d):5.2f}  "
                  f"p90 {np.percentile(d, 90):5.2f}  max {max(d):5.2f}")
