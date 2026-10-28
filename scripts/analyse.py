"""Analysis pass for one clip: SAM 3 surface masks, cached under cache/<clip>-<hash>/.

  python scripts/analyse.py assets/clips/billboard_night.mp4 --prompt "billboard"
  python scripts/analyse.py assets/clips/times_square.mp4 --prompt "screen" --backend api   # SAM 3.1, Meta Model API
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.sam_track import read_frames, segment_video  # noqa: E402

CACHE_VERSION = 1
SEGMENT = 48  # frames per SAM process; one long process accumulates memory until an 18 GB Mac swaps


def cache_dir(clip, settings):
    h = hashlib.sha256(Path(clip).read_bytes()).hexdigest()[:12]
    key = hashlib.sha256(json.dumps({**settings, "v": CACHE_VERSION, "src": h}, sort_keys=True).encode()).hexdigest()[:8]
    return ROOT / "cache" / f"{Path(clip).stem}-{h}-{key}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clip")
    p.add_argument("--prompt", default="billboard")
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--backend", choices=["local", "api"], default="local",
                   help="local: SAM 3 on this Mac; api: SAM 3.1 on the Meta Model API (needs MODEL_API_KEY)")
    p.add_argument("--prefer", choices=["confidence", "area"], default="confidence",
                   help="api backend: among equally persistent objects prefer higher confidence or larger area")
    p.add_argument("--frames", help="only segment this range 'start:end' (others stay empty), e.g. to fill gaps")
    args = p.parse_args()

    if args.backend == "api":
        settings = {"model": "meta/sam-3.1", "prompt": args.prompt}
        if args.prefer != "confidence":
            settings["prefer"] = args.prefer
    else:
        settings = {"model": "facebook/sam3", "prompt": args.prompt, "chunk": args.chunk, "proc_side": 640,
                    "frames": args.frames}
    out = cache_dir(args.clip, settings)
    if (out / "surface_masks.npz").exists():
        print("cached:", out)
        return
    frames, fps = read_frames(args.clip)
    t0 = time.time()
    a, b = (int(v) for v in args.frames.split(":")) if args.frames else (0, len(frames))
    masks = np.zeros((len(frames),) + frames[0].shape[:2], bool)
    scores = np.full(len(frames), np.nan)
    api_info = None
    if args.backend == "api":
        from pipeline.sam_api import segment_video_api, stream_events, upload
        # The raw stream is kept so a different object choice never needs (or pays for) a second API call.
        rec = cache_dir(args.clip, {"model": "meta/sam-3.1", "prompt": args.prompt}).with_name(
            cache_dir(args.clip, {"model": "meta/sam-3.1", "prompt": args.prompt}).name + "-events.jsonl")
        if rec.exists():
            events = [json.loads(line) for line in rec.read_text().splitlines()]
        else:
            events = list(stream_events(upload(args.clip), args.prompt))
            rec.write_text("\n".join(json.dumps(e) for e in events))
        masks, scores, api_info = segment_video_api(args.clip, args.prompt, len(frames), frames[0].shape[1::-1],
                                                    events=events, prefer=args.prefer)
    elif b - a > SEGMENT:
        # Run each segment in a fresh process (cached on its own), then merge.
        for s0 in range(a, b, SEGMENT):
            seg = f"{s0}:{min(s0 + SEGMENT, b)}"
            subprocess.run([sys.executable, __file__, args.clip, "--prompt", args.prompt, "--chunk", str(args.chunk),
                            "--frames", seg], check=True)
            z = np.load(cache_dir(args.clip, {**settings, "frames": seg}) / "surface_masks.npz")
            masks[s0:s0 + SEGMENT], scores[s0:s0 + SEGMENT] = z["masks"][s0:s0 + SEGMENT], z["scores"][s0:s0 + SEGMENT]
    else:
        masks[a:b], scores[a:b] = segment_video(frames[a:b], args.prompt, chunk=args.chunk)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "surface_masks.npz", masks=masks, scores=scores)
    meta = {**settings, "clip": str(args.clip), "frames": len(frames), "fps": fps, "size": frames[0].shape[1::-1],
            "coverage": float(np.isfinite(scores).mean()), "sam_seconds": round(time.time() - t0, 1)}
    if api_info:
        meta["api"] = api_info
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("cached:", out)


if __name__ == "__main__":
    main()
