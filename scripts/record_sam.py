"""Record SAM 3.1 Responses streams for a clip without building masks (long clips: masks are built per slot later).

  python scripts/record_sam.py assets/clips/piccadilly_long.mp4 screen person bus

Uploads the clip once and writes each prompt's raw event stream where analyse.py/slots.py/occluders.py look for it.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.sam_api import stream_events, upload  # noqa: E402
from scripts.analyse import cache_dir  # noqa: E402


def main(clip, prompts):
    file_id = upload(clip)
    print(f"uploaded {clip}", flush=True)
    for prompt in prompts:
        base = cache_dir(clip, {"model": "meta/sam-3.1", "prompt": prompt})
        out = base.with_name(base.name + "-events.jsonl")
        t0, n = time.time(), 0
        with open(out, "w") as f:
            for event in stream_events(file_id, prompt):
                f.write(json.dumps(event) + "\n")
                n += 1
        print(f"{prompt!r}: {n} events in {time.time() - t0:.0f} s -> {out.name}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
