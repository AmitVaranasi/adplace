"""The creator's say over their own video: placement is opt-in, and they approve which places and which brands.

Stored next to the player export as creator.json:

  {"enabled": false, "places": {"c01": true, "c03": false}, "brands": {"kestrel": false}}

A place or brand the file doesn't name is approved (so a re-export that finds a new place doesn't silently drop it
from an opted-in video), but nothing runs until `enabled` is true. No file means off: uploads start that way (ingest
writes the file), and a video only carries ads once a creator.json says so.
"""
from __future__ import annotations

import json
from pathlib import Path

NAME = "creator.json"


def load(export_dir: Path) -> dict:
    p = Path(export_dir) / NAME
    s = json.loads(p.read_text()) if p.exists() else {"enabled": False}
    return {"enabled": bool(s.get("enabled")), "places": dict(s.get("places", {})), "brands": dict(s.get("brands", {}))}


def save(export_dir: Path, settings: dict) -> dict:
    s = {"enabled": bool(settings.get("enabled")),
         "places": {k: bool(v) for k, v in settings.get("places", {}).items()},
         "brands": {k: bool(v) for k, v in settings.get("brands", {}).items()}}
    (Path(export_dir) / NAME).write_text(json.dumps(s, indent=2) + "\n")
    return s


def start_off(export_dir: Path) -> None:
    """A fresh upload: placement stays off until the creator turns it on. Keeps an earlier choice."""
    if not (Path(export_dir) / NAME).exists():
        save(export_dir, {"enabled": False})


def allowed(placements: list[dict], settings: dict) -> list[dict]:
    """The auction's placements cut to what the creator approved: none while placement is off."""
    if not settings["enabled"]:
        return []
    out = []
    for p in placements:
        if not settings["places"].get(p["id"], True):
            continue
        out.append({**p, "brands": [b for b in p["brands"] if settings["brands"].get(b, True)]})
    return out
