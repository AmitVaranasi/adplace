"""Clip configuration: the hand-made clips (configs/clips.json, in git) and uploaded ones (configs/uploads.json,
written by scripts/ingest.py, not in git). Everything that renders or exports looks clips up here."""
import fcntl
import json
import os
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
    """Read-modify-write under an exclusive lock (two uploads can finish at once), replaced atomically."""
    with open(UPLOADS.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(UPLOADS.read_text()) if UPLOADS.exists() else {}
        data[clip] = value
        tmp = UPLOADS.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, UPLOADS)
