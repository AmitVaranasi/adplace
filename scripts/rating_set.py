"""Blinded realism rating (spec S8): build the stimulus set, then score the collected ratings.

  python scripts/rating_set.py build     # renders unlabelled placements + originals into eval/rating/s01.mp4 ...
  python scripts/rating_set.py score eval/ratings.json   # or a text file of results codes

Stimuli get opaque ids in a shuffled order so file names give nothing away; the key stays in eval/rating_key.json
(never published). Originals are included as controls: a placement "passes" when raters score it about as natural as
the untouched footage. Unlabelled renders are for this test only and are never distributed.
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.render import render_campaign  # noqa: E402

EVAL = ROOT / "eval"
OUT = EVAL / "rating"
KEY = EVAL / "rating_key.json"
# (clip id in configs/clips.json, brand or None for the untouched original)
STIMULI = [
    ("assets/clips/times_square.mp4", None), ("assets/clips/times_square.mp4", "kestrel"),
    ("assets/clips/times_square.mp4", "paperwhale"),
    ("assets/clips/billboard_night.mp4", None), ("assets/clips/billboard_night.mp4", "brewlark"),
    ("assets/clips/billboard_night_occluded.mp4", None), ("assets/clips/billboard_night_occluded.mp4", "kestrel"),
    ("assets/clips/billboard_night_occluded.mp4", "paperwhale"),
]


def transcode(src, dst):
    """Small web-playable copy: H.264 CRF 23, no audio, no metadata that could name the source."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-map_metadata", "-1", "-an",
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
                   check=True)


def build(seed):
    clips = json.loads((ROOT / "configs" / "clips.json").read_text())
    stage = EVAL / "stage"
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    order = list(STIMULI)
    random.Random(seed).shuffle(order)
    key = {}
    for i, (clip, brand) in enumerate(order, 1):
        sid = f"s{i:02d}"
        if brand is None:
            src = ROOT / clip
        else:
            entry = clips[clip]
            src = render_campaign(clip, str(ROOT / entry["analysis"]), brand, entry["emissive"], label=False,
                                  out_dir=stage)["output"]
        transcode(src, OUT / f"{sid}.mp4")
        key[sid] = {"clip": Path(clip).stem, "brand": brand}
        print(sid, Path(clip).stem, brand or "original")
    KEY.write_text(json.dumps(key, indent=2))
    shutil.rmtree(stage, ignore_errors=True)


def score(path):
    """A JSON list of {"initials": ..., "scores": {"s01": 1-5, ...}} (the page's db rows), or a text file of the
    results codes raters send back (RC1-<initials>-<id><score>..., e.g. RC1-JK-014023...)."""
    key = json.loads(KEY.read_text())
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        raters = json.loads(text)
    else:
        raters = []
        for line in text.split():
            _, initials, digits = line.strip().split("-", 2)
            raters.append({"initials": initials,
                           "scores": {f"s{digits[i:i + 2]}": int(digits[i + 2]) for i in range(0, len(digits), 3)}})
    by = {}
    for r in raters:
        for sid, v in r["scores"].items():
            k = key[sid]
            by.setdefault((k["clip"], "original" if k["brand"] is None else "placement"), []).append(float(v))
    print(f"{len(raters)} raters")
    for (clip, cond), vals in sorted(by.items()):
        print(f"{clip:28s} {cond:9s} mean {np.mean(vals):.2f}  sd {np.std(vals):.2f}  n {len(vals)}")
    placed = [v for (c, cond), vals in by.items() if cond == "placement" for v in vals]
    orig = [v for (c, cond), vals in by.items() if cond == "original" for v in vals]
    print(f"all placements mean {np.mean(placed):.2f} (target >= 4) vs originals {np.mean(orig):.2f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score")
    s.add_argument("ratings")
    args = ap.parse_args()
    build(args.seed) if args.cmd == "build" else score(args.ratings)


if __name__ == "__main__":
    main()
