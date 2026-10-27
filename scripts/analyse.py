"""Analysis pass for one clip: SAM 3 surface masks, cached under cache/<clip>-<hash>/.

  python scripts/analyse.py assets/clips/billboard_night.mp4 --prompt "billboard"
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.sam_track import read_frames, segment_video  # noqa: E402

CACHE_VERSION = 1


def cache_dir(clip, settings):
    h = hashlib.sha256(Path(clip).read_bytes()).hexdigest()[:12]
    key = hashlib.sha256(json.dumps({**settings, "v": CACHE_VERSION, "src": h}, sort_keys=True).encode()).hexdigest()[:8]
    return ROOT / "cache" / f"{Path(clip).stem}-{h}-{key}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clip")
    p.add_argument("--prompt", default="billboard")
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--frames", help="only segment this range 'start:end' (others stay empty), e.g. to fill gaps")
    args = p.parse_args()

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
    masks[a:b], scores[a:b] = segment_video(frames[a:b], args.prompt, chunk=args.chunk)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "surface_masks.npz", masks=masks, scores=scores)
    meta = {**settings, "clip": str(args.clip), "frames": len(frames), "fps": fps, "size": frames[0].shape[1::-1],
            "coverage": float(np.isfinite(scores).mean()), "sam_seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("cached:", out)


if __name__ == "__main__":
    main()
