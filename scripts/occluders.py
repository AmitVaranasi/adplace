"""Build each slot's occluders.npz from every moving foreground listed for the clip (people, buses, cars, ...).

  python scripts/occluders.py assets/clips/piccadilly.mp4

configs/clips.json lists, per clip, "occluders": [<analysis dir>, ...]. Each dir's all_objects.npz (every object SAM 3.1
found for that prompt) is written from the recorded stream when missing. The
union is written to every slot's analysis dir: the renderer draws it over the ad (a bus crossing in front of a
billboard covers it) and scripts/lock.py keeps it out of the camera-motion estimate (a passing bus is not the camera).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.sam_api import parse_masks, paste  # noqa: E402
from pipeline.sam_track import read_frames  # noqa: E402
from scripts.analyse import cache_dir  # noqa: E402


def all_objects(d, n, h, w):
    """Union of every object in a recorded SAM 3.1 stream, cached as all_objects.npz."""
    out = d / "all_objects.npz"
    if out.exists():
        return np.load(out)["masks"][:n]
    meta = json.loads((d / "meta.json").read_text())
    base = cache_dir(meta["clip"], {"model": "meta/sam-3.1", "prompt": meta["prompt"]})
    events_file = base.with_name(base.name + "-events.jsonl")
    if not events_file.exists():
        raise SystemExit(f"{d}: no recorded stream; run analyse.py --backend api --prompt {meta['prompt']!r}")
    masks = np.zeros((n, h, w), bool)
    for (_, fi), rec in parse_masks([json.loads(l) for l in events_file.read_text().splitlines()]).items():
        if 0 <= fi < n:
            masks[fi] |= paste(rec, h, w)
    np.savez_compressed(out, masks=masks)
    return masks


def main(clip):
    entry = json.loads((ROOT / "configs" / "clips.json").read_text())[clip]
    frames, _ = read_frames(ROOT / clip)
    n, (h, w) = len(frames), frames[0].shape[:2]
    occ = np.zeros((n, h, w), bool)
    for src in entry.get("occluders", []):
        d = ROOT / src
        m = all_objects(d, n, h, w)
        occ[:len(m)] |= m
        print(f"{src}: {json.loads((d / 'meta.json').read_text()).get('prompt')!r}, mean area {m.mean():.3f}")
    for sl in entry["slots"]:
        np.savez_compressed(ROOT / sl["analysis"] / "occluders.npz", masks=occ)
    print(f"{Path(clip).stem}: occluders written to {len(entry['slots'])} slots (mean area {occ.mean():.3f})")


if __name__ == "__main__":
    main(sys.argv[1])
