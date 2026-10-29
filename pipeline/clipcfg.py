"""Clip configuration: the hand-made clips (configs/clips.json, in git) and uploaded ones (configs/uploads.json,
written by scripts/ingest.py, not in git). Everything that renders or exports looks clips up here."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS, UPLOADS = ROOT / "configs" / "clips.json", ROOT / "configs" / "uploads.json"


def all_clips():
    out = json.loads(CLIPS.read_text())
    if UPLOADS.exists():
        out.update(json.loads(UPLOADS.read_text()))
    return out


def entry(clip):
    return all_clips()[clip]


def save_upload(clip, value):
    data = json.loads(UPLOADS.read_text()) if UPLOADS.exists() else {}
    data[clip] = value
    UPLOADS.write_text(json.dumps(data, indent=1))
