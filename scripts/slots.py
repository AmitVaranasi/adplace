"""Export every surface SAM 3.1 found in a clip as its own slot (wide scenes have several), from the recorded stream.

  python scripts/slots.py assets/clips/piccadilly.mp4 --prompt screen [--min-frames 0.8] [--min-area 0.004] [--masks]

Needs a prior `analyse.py --backend api` run with the same prompt (it records the raw stream next to the cache).
Writes cache/<clip>-<hash>-slot<object id>/{surface_masks.npz, meta.json, quads.npy} per kept object and prints the
slot list to paste into configs/clips.json. --masks tracks each slot with the mask-anchored tracker (screens and other
surfaces whose masks are reliable); without it only masks are written.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.sam_api import paste, parse_masks  # noqa: E402
from pipeline.sam_track import read_frames  # noqa: E402
from scripts.analyse import cache_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--min-frames", type=float, default=0.8, help="fraction of frames the object must be seen in")
    ap.add_argument("--min-area", type=float, default=0.004, help="mean mask area as a fraction of the frame")
    ap.add_argument("--masks", action="store_true", help="also track each slot with the mask-anchored tracker")
    args = ap.parse_args()

    base = cache_dir(args.clip, {"model": "meta/sam-3.1", "prompt": args.prompt})
    events_file = base.with_name(base.name + "-events.jsonl")
    if not events_file.exists():
        raise SystemExit(f"no recorded stream at {events_file}; run analyse.py --backend api --prompt {args.prompt!r}")
    events = [json.loads(line) for line in events_file.read_text().splitlines()]
    frames, fps = read_frames(args.clip)
    n, (h, w) = len(frames), frames[0].shape[:2]

    objects = {}
    for (oid, fi), rec in parse_masks(events).items():
        if 0 <= fi < n:
            objects.setdefault(oid, {})[fi] = rec
    stem = Path(args.clip).stem
    slots = []
    for oid, recs in sorted(objects.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        masks = np.zeros((n, h, w), bool)
        scores = np.full(n, np.nan)
        for fi, rec in recs.items():
            masks[fi] = paste(rec, h, w)
            if masks[fi].any():
                scores[fi] = rec.confidence if rec.confidence is not None else 1.0
        seen = np.isfinite(scores)
        area = float(masks[seen].mean()) if seen.any() else 0.0
        keep = seen.mean() >= args.min_frames and area >= args.min_area
        print(f"object {oid}: seen {seen.mean():.0%} of frames, area {100 * area:.2f}% of frame",
              "-> slot" if keep else "-> skipped")
        if not keep:
            continue
        out = base.with_name(f"{base.name}-slot{oid}")
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "surface_masks.npz", masks=masks, scores=scores)
        (out / "meta.json").write_text(json.dumps(
            {"model": "meta/sam-3.1", "prompt": args.prompt, "object": oid, "clip": str(args.clip), "frames": n,
             "fps": fps, "size": [w, h], "coverage": float(seen.mean()), "mean_area_frac": area}, indent=2))
        tracked = None
        if args.masks:
            from pipeline.mask_track import track_from_masks
            try:
                r = track_from_masks([m if m.any() else None for m in masks])
                np.save(out / "quads.npy", np.asarray(r["quads"]))
                tracked = True
            except ValueError as e:  # not quadrilateral (curved, heavily occluded): needs another surface model
                print(f"  object {oid}: not tracked as a flat quad ({e})")
                tracked = False
        slots.append({"slot_id": f"{stem}-{oid}", "analysis": str(out.relative_to(ROOT)),
                      **({} if tracked is None else {"quad_tracked": tracked})})
    print(json.dumps(slots, indent=2))


if __name__ == "__main__":
    main()
