"""Blind "spot the inserted ad" test with Muse Spark as the judge, against Mirriad's own showreel as the bar.

  python scripts/blind_test.py --n 8

Three pools of single frames, shuffled together, judged one at a time with no context about which pool they came
from:
  ours      frames from our renders, at frames where a placement is visible
  original  the same frames from the untouched source clips (how often the judge cries wolf)
  mirriad   frames from Mirriad's showreel (every shot there carries an inserted placement)
The disclosure label (bottom-left) and Mirriad's corner logo are cropped off every frame alike. A detection only
counts when the judge also points at the right object, so a lucky "yes" on a cluttered frame does not count.
Results go to eval/blind_<date>.json.
"""
import argparse
import base64
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.rank import chat  # noqa: E402

API = "https://api.meta.ai/v1"
MIRRIAD = Path.home() / "Claude Cowork" / "adplace-notes" / "reference_videos" / "mirriad" / "anKMWpjXTS4.mp4"
# Showreel times (s) with a clearly inserted placement, and what was inserted (for scoring the judge's "where").
MIRRIAD_SHOTS = [(7, "Swisse poster"), (12, "Swisse poster"), (16, "cola cans on table"), (25, "FUN inflatable"),
                 (39, "magazines/products on table"), (57, "cereal box on counter"), (61, "bus-stop poster"),
                 (66, "phone ad poster"), (75, "FUN display"), (79, "soft-drink can on TV screen/shelf")]
CLIPS = [  # clip, render, render offset (start frame), {slot: what it is}
    ("pizzeria_counter", "pizzeria_counter_v19", 80, {"counter_sign": "black wall sign", "counter_freezer": "freezer side panel"}),
    ("pizzeria_box", "pizzeria_box_v9", 0, {"box_lid": "pizza box lid print"}),
    ("patio_waiter", "patio_waiter_v9", 0, {"patio_wall": "poster on the grey wall", "patio_napkin": "menu card on table"}),
]
Q = ("Look carefully at this video frame. Has an advertisement, sign, poster, label or product been DIGITALLY "
     "INSERTED into it after filming (virtual product placement / VFX compositing), as opposed to being "
     "physically present when it was shot? Judge only from visual evidence: edges, lighting, focus, noise, "
     "perspective, occlusion, how it sits in the scene. Answer JSON only: "
     '{"inserted": true/false, "confidence": 0-1, "where": "the object you think was inserted, or empty", '
     '"tell": "the visual evidence, or empty"}')


def crop(img, mirriad=False):
    h, w = img.shape[:2]
    if mirriad:  # 16:9 picture inside the reel, logo bottom-right
        return img[: int(h * 0.88), : int(w * 0.86)]
    return img[: int(h * 0.88)]  # disclosure label bottom-left


def frame_at(path, i):
    c = cv2.VideoCapture(str(path))
    c.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = c.read()
    return f if ok else None


def pools(n, seed):
    rng = random.Random(seed)
    items = []
    for clip, render, off, slots in CLIPS:
        cand = []
        for sid, what in slots.items():
            z = np.load(ROOT / "cache" / f"{clip}-scene" / f"track_{sid}.npz")
            cand += [(int(i), what) for i in np.flatnonzero(z["valid"]) if i >= off]
        for i, what in rng.sample(cand, min(n, len(cand))):
            ours, orig = frame_at(ROOT / "outputs" / f"{render}.mp4", i - off), frame_at(ROOT / "assets/clips" / f"{clip}.mp4", i)
            if ours is not None and orig is not None:
                items.append({"pool": "ours", "src": f"{render}#{i}", "truth": what, "img": crop(ours)})
                items.append({"pool": "original", "src": f"{clip}#{i}", "truth": "", "img": crop(orig)})
    cap = cv2.VideoCapture(str(MIRRIAD))
    fps = cap.get(cv2.CAP_PROP_FPS)
    for t, what in MIRRIAD_SHOTS:
        f = frame_at(MIRRIAD, int(t * fps))
        if f is not None:
            items.append({"pool": "mirriad", "src": f"mirriad#{t}s", "truth": what, "img": crop(f, True)})
    rng.shuffle(items)
    return items


def judge(img):
    small = cv2.resize(img, (1024, int(1024 * img.shape[0] / img.shape[1])))
    url = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).decode()
    text = chat(API, os.environ["MODEL_API_KEY"], "muse-spark-1.3", "You are a forensic VFX reviewer.",
                [{"type": "text", "text": Q}, {"type": "image_url", "image_url": {"url": url}}],
                extra={"max_completion_tokens": 6000}, timeout=300)
    try:
        return json.loads(text[text.index("{"):text.rindex("}") + 1])
    except (ValueError, TypeError, AttributeError):
        return {"inserted": None, "raw": text[:300]}


def located(verdict, truth):
    """Did the judge point at the inserted object? Muse Spark decides from the two descriptions (cheap call)."""
    where = (verdict.get("where") or "").strip()
    if not where or not truth:
        return False
    text = chat(API, os.environ["MODEL_API_KEY"], "muse-spark-1.3", "Answer yes or no only.",
                f"Inserted object: '{truth}'. A reviewer said the inserted object is: '{where}'. "
                "Do these refer to the same object?", extra={"max_completion_tokens": 3000}, timeout=120)
    return bool(text) and text.strip().lower().startswith("y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="frames per clip")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    items = pools(args.n, args.seed)
    rows = []
    for k, it in enumerate(items):
        v = judge(it["img"])
        hit = bool(v.get("inserted")) and (it["pool"] == "original" or located(v, it["truth"]))
        rows.append({k2: it[k2] for k2 in ("pool", "src", "truth")} | {"verdict": v, "flagged": hit})
        print(f"{k + 1}/{len(items)} {it['pool']:8s} {it['src']:28s} inserted={v.get('inserted')} "
              f"conf={v.get('confidence')} where={str(v.get('where'))[:40]!r} -> {'DETECTED' if hit else '-'}",
              flush=True)
    summary = {}
    for pool in ("ours", "original", "mirriad"):
        r = [x for x in rows if x["pool"] == pool]
        summary[pool] = {"frames": len(r), "flagged": sum(x["flagged"] for x in r),
                         "rate": round(sum(x["flagged"] for x in r) / max(len(r), 1), 3)}
    by_clip = {}
    for x in rows:
        if x["pool"] == "ours":
            c = x["src"].split("#")[0]
            by_clip.setdefault(c, [0, 0])
            by_clip[c][0] += x["flagged"]
            by_clip[c][1] += 1
    summary["ours_by_clip"] = {c: f"{a}/{b}" for c, (a, b) in by_clip.items()}
    out = ROOT / "eval" / f"blind_{time.strftime('%Y%m%d_%H%M')}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2), "\n->", out)


if __name__ == "__main__":
    main()
