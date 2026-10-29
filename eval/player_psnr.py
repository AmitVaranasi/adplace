"""PSNR of the live player (window.renderFrameForTest dump) against the server render of the same assignment,
inside each slot's quad. Usage: player_psnr.py <clip stem> <frames.json> <server.mp4> [crop_frame]"""
import base64
import json
import sys

import cv2
import numpy as np

stem, dump, server = sys.argv[1:4]
raw = open(dump).read()
d = json.loads(raw[raw.index('{'):raw.rindex('}') + 1]) if not raw.lstrip().startswith(('{', '"')) else json.loads(raw)
if isinstance(d, str):
    d = json.loads(d)
tr = json.load(open(f"app/static/player/{stem}/track.json"))


def frames(path, top=None):
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(f[:top] if top else f)


srv, orig = frames(server), frames(f"app/static/player/{stem}/stack.mp4", tr["height"])


def psnr(a, b, m):
    e = ((a.astype(float) - b.astype(float)) ** 2)[m].mean()
    return 10 * np.log10(255 ** 2 / e) if e else 99.


for k, u in sorted(d.items(), key=lambda kv: int(kv[0])):
    i = int(k)
    b = cv2.imdecode(np.frombuffer(base64.b64decode(u.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
    for slot in tr["slots"]:
        q = slot["quads"][i]
        if q is None:
            continue
        m = np.zeros(b.shape[:2], np.uint8)
        cv2.fillConvexPoly(m, np.round(np.array(q)).astype(np.int32), 1)
        m = m > 0
        print(f"frame {i:3d} {slot['id']:16s} browser-vs-server {psnr(b, srv[i], m):5.1f} dB"
              f"   original-vs-server {psnr(orig[i], srv[i], m):5.1f} dB")
    if len(sys.argv) > 4 and i == int(sys.argv[4]):
        cv2.imwrite(f"outputs/{stem}_cmp_{i}.png", np.hstack([b, srv[i]]))
