"""Run the final read check (scripts/ingest.read_check) on a video that was processed before it existed: render the
first viewer's version, let Muse Spark read each glass or cup logo off it, and record the result in the run's report.

  .venv/bin/python scripts/read_check.py upload_m44007 runs/r_eval_m44007
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(stem, run_dir):
    from pipeline.clipcfg import all_clips
    from scripts.ingest import read_check
    from scripts.render_dynamic import render
    run_dir = ROOT / run_dir
    report = json.loads((run_dir / "report.json").read_text())
    clip = report["clip"]
    entry = all_clips()[clip]
    viewers = json.loads((ROOT / "app" / "static" / "player" / stem / "assignments.json").read_text())
    assignment = next(iter(viewers.values()))["assignment"]
    out = run_dir / "qa_read.mp4"
    render(clip, entry, assignment, out, label=False)
    cap, rendered = cv2.VideoCapture(str(out)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        rendered.append(f)
    for p in report["placements"]:
        if p.get("decision") == "placed" and p.get("surface") in ("glass", "cup"):
            q = read_check({"id": p["id"]}, rendered, stem, assignment)
            p["qa"] = q
            print(stem, p["id"], assignment.get(p["id"]), q["pass"], q["checks"])
    (run_dir / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
