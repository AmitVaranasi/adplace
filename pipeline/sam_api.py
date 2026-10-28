"""SAM 3.1 on the Meta Model API: text-prompted video segmentation without running the model locally.

Same result shape as sam_track.segment_video: masks (N, H, W) bool at frame resolution for the single object that
best matches the prompt, and scores (N,) with NaN where it was not found. The clip is uploaded once through the
Files API (MP4/MOV with H.264/H.265 only) and the streamed text output is parsed with Meta's meta-sam-parser.
Needs MODEL_API_KEY.
"""
import asyncio
import json
import os

import cv2
import numpy as np
import requests
from meta_sam_parser import (SegmentationMaskRecord, decode_mask_to_raster, parse_responses_stream,
                             video_segmentation_format)

BASE = "https://api.meta.ai/v1"


def _headers():
    return {"Authorization": f"Bearer {os.environ['MODEL_API_KEY']}"}


def upload(path):
    """Files API upload; returns the file id."""
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/files", headers=_headers(), data={"purpose": "user_data"},
                          files={"file": (os.path.basename(path), f, "video/mp4")}, timeout=300)
    r.raise_for_status()
    return r.json()["id"]


def stream_events(file_id, prompt):
    """Raw Responses API events (dicts) for one text-prompted video request."""
    body = {"model": "sam-3.1", "stream": True,
            "metadata": {"include_confidence": "true"},
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt}, {"type": "input_video", "file_id": file_id}]}]}
    with requests.post(f"{BASE}/responses", headers={**_headers(), "Accept": "text/event-stream"}, json=body,
                       stream=True, timeout=900) as r:
        if r.status_code != 200:
            raise RuntimeError(f"SAM 3.1 request failed: HTTP {r.status_code}: {r.text[:300]}")
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                data = line[5:].strip()
                if data and data != "[DONE]":
                    yield json.loads(data)


def parse_masks(events):
    """Latest-revision mask records keyed by (object_id, frame_index)."""
    async def source():
        for e in events:
            yield e

    async def run():
        latest = {}
        async for snap in parse_responses_stream(source(), video_segmentation_format()):
            for rec in snap.records:
                if isinstance(rec, SegmentationMaskRecord) and rec.frame is not None:
                    k = (rec.object_id, rec.frame.frame_index)
                    if k not in latest or rec.revision >= latest[k].revision:
                        latest[k] = rec
        return latest

    return asyncio.run(run())


def paste(rec, h, w):
    """Full-frame bool mask from a box-local mask record."""
    m = rec.mask
    raster = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width)
    b = rec.bounds
    x0, y0 = int(np.floor(b.left)), int(np.floor(b.top))
    x1, y1 = int(np.ceil(b.right)), int(np.ceil(b.bottom))
    out = np.zeros((h, w), bool)
    if x1 <= x0 or y1 <= y0:
        return out
    local = cv2.resize(raster, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST).astype(bool)
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    out[cy0:cy1, cx0:cx1] = local[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    return out


def segment_video_api(clip, prompt, n_frames, size, file_id=None):
    """(masks, scores, info) for the object seen in the most frames (ties: higher mean confidence, then area)."""
    w, h = size
    file_id = file_id or upload(clip)
    recs = parse_masks(stream_events(file_id, prompt))
    objects = {}
    for (oid, fi), rec in recs.items():
        objects.setdefault(oid, {})[fi] = rec
    info = {"file_id": file_id, "objects": {}}
    if not objects:
        return np.zeros((n_frames, h, w), bool), np.full(n_frames, np.nan), info

    def rank(oid):
        frames = objects[oid]
        conf = [r.confidence for r in frames.values() if r.confidence is not None]
        area = np.mean([(r.bounds.right - r.bounds.left) * (r.bounds.bottom - r.bounds.top) for r in frames.values()])
        return (len(frames), float(np.mean(conf)) if conf else 0.0, area)

    for oid in objects:
        n, c, a = rank(oid)
        info["objects"][oid] = {"frames": n, "mean_conf": round(c, 3), "mean_box_area": round(float(a))}
    best = max(objects, key=rank)
    info["chosen"] = best
    masks = np.zeros((n_frames, h, w), bool)
    scores = np.full(n_frames, np.nan)
    for fi, rec in objects[best].items():
        if 0 <= fi < n_frames:
            masks[fi] = paste(rec, h, w)
            scores[fi] = rec.confidence if rec.confidence is not None else 1.0
    return masks, scores, info
