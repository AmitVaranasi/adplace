"""Why does the inserted can look out of place? Measured against the real objects beside it, plus a Muse critique.

  python eval/can_diagnose.py outputs/patio_waiter_can_test3.mp4 138
"""
import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.maskstore import MaskStore  # noqa: E402
from pipeline.rank import chat  # noqa: E402
from scripts.place3d import CanPlacement  # noqa: E402

video, i = sys.argv[1], int(sys.argv[2])
cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, i)
frame = cap.read()[1]
can = CanPlacement()
(x0, y0, x1, y1), _, cov, _ = can.layer(i, can.frames[i], "fizzwell")
regions = {"can": np.zeros(frame.shape[:2], bool)}
regions["can"][y0:y1, x0:x1] = cov > 0.95
S = ROOT / "cache/patio_waiter-scene"
for name, f in (("juice glass", "obj_glass_0"), ("cola glass", "obj_glass_1"), ("dip bowl", "obj_bowl_1"),
                ("big bowl", "obj_bowl_0"), ("big plate", "obj_plate_2"), ("pizza", "obj_plate_0")):
    st = MaskStore(str(S / f"{f}.npz"))
    if st.any(i):
        regions[name] = st[i]
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
lum = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
lap = np.abs(cv2.Laplacian(lum, cv2.CV_32F))
hp = lum - cv2.GaussianBlur(lum, (0, 0), 1.5)
print(f"frame {i}: {'region':12s} {'sat':>5s} {'lum':>5s} {'p2':>5s} {'p98':>5s} {'contrast':>8s} {'edge':>6s} {'fine':>6s} {'warm':>6s}")
for name, m in regions.items():
    m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if m.sum() < 100:
        continue
    b, g, r = (frame[..., k][m].astype(float).mean() for k in range(3))
    print(f"  {name:12s} {hsv[..., 1][m].mean():5.0f} {lum[m].mean():5.0f} {np.percentile(lum[m], 2):5.0f} "
          f"{np.percentile(lum[m], 98):5.0f} {lum[m].std():8.1f} {lap[m].mean():6.2f} {hp[m].std():6.2f} {r / b:6.2f}")
print("  sat = saturation, p2/p98 = darkest/brightest levels, edge = mean |Laplacian|, fine = high-pass std, warm = R/B")

pad = 140
crop = frame[max(y0 - pad, 0):y1 + pad // 2, max(x0 - 2 * pad, 0):x1 + 2 * pad]
cv2.imwrite(str(ROOT / "outputs/place3d/diag_crop.jpg"), crop)
url = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1]).decode()
q = ("One object in this video frame was rendered with CG and composited in: the drink can on the table. Every other "
     "object is real footage. As a senior VFX compositing supervisor, list concretely what gives the can away compared "
     "with the real objects around it (glasses, bowls, plates, pizza). Consider lighting direction and softness, "
     "shading across the cylinder, specular highlights and reflections, the top and rim, contact shadow and grounding, "
     "colour, saturation, black level and contrast versus the footage's grade, sharpness and depth of field, noise and "
     "compression texture, edges, scale and perspective, and the label design. Rank the issues by how much each "
     "contributes to the can looking fake, most important first, each with a specific fix. Reply as JSON: "
     "{\"issues\": [{\"issue\": ..., \"evidence\": ..., \"fix\": ...}]}")
text = chat("https://api.meta.ai/v1", os.environ["MODEL_API_KEY"], "muse-spark-1.3", "You are a senior VFX compositing supervisor.",
            [{"type": "text", "text": q}, {"type": "image_url", "image_url": {"url": url}}],
            extra={"max_completion_tokens": 8000}, timeout=300)
try:
    for k, it in enumerate(json.loads(text[text.index("{"):text.rindex("}") + 1])["issues"], 1):
        print(f"{k}. {it['issue']}\n   evidence: {it['evidence']}\n   fix: {it['fix']}")
except (ValueError, KeyError, TypeError):
    print(text[:3000])
