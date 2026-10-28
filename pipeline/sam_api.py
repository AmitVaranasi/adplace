"""SAM 3.1 video segmentation through the Meta Model API.

Returns masks (N, H, W) bool in source-frame coordinates, scores (N,) with
NaN where the selected object has no visible mask, and API metadata.
Needs MODEL_API_KEY. The size must match the displayed source-video size.
"""
import asyncio
import json
import os

import cv2
import numpy as np
import requests
from meta_sam_parser import (
    SegmentationMaskRecord,
    decode_mask_to_raster,
    parse_responses_stream,
    video_segmentation_format,
)

BASE = "https://api.meta.ai/v1"


def _headers():
    return {"Authorization": f"Bearer {os.environ['MODEL_API_KEY']}"}


def upload(path):
    """Files API upload; returns the file id."""
    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE}/files",
            headers=_headers(),
            data={"purpose": "user_data"},
            files={"file": (os.path.basename(path), f, "video/mp4")},
            timeout=300,
        )
    r.raise_for_status()
    return r.json()["id"]


def _sse_event(data_lines, event_type):
    if not data_lines:
        return None
    data = "\n".join(data_lines)
    if data == "[DONE]":
        return None
    event = json.loads(data)
    if not isinstance(event, dict):
        raise ValueError("Expected a Responses event JSON object.")
    # Official JSON has type; the SSE event field can supply it if omitted.
    if event_type:
        event.setdefault("type", event_type)
    return event


def stream_events(file_id, prompt):
    """Yield Responses event dicts, respecting SSE message boundaries."""
    body = {
        "model": "sam-3.1",
        "stream": True,
        "metadata": {"include_confidence": "true"},
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_video", "file_id": file_id},
            ],
        }],
    }
    with requests.post(
        f"{BASE}/responses",
        headers={**_headers(), "Accept": "text/event-stream"},
        json=body,
        stream=True,
        timeout=900,
    ) as r:
        if r.status_code != 200:
            raise RuntimeError(
                f"SAM 3.1 request failed: HTTP {r.status_code}: {r.text[:300]}"
            )
        data_lines, event_type = [], None
        for line in r.iter_lines(decode_unicode=True):
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if not line:
                event = _sse_event(data_lines, event_type)
                data_lines, event_type = [], None
                if event is not None:
                    yield event
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data_lines.append(value)
                if '"video_frames_processed"' in value:  # the final usage block: log spend against the budget
                    try:
                        frames = json.loads(value)["response"]["usage"]["video_frames_processed"]
                        from pipeline.usage import record_sam
                        record_sam(frames, purpose=prompt)
                    except (ValueError, KeyError, TypeError):
                        pass
            elif field == "event":
                event_type = value
        event = _sse_event(data_lines, event_type)
        if event is not None:
            yield event


def parse_masks(events):
    """Latest valid mask per (object_id, source frame), from a complete response.

    Snapshots are cumulative append-only histories, including prior revisions.
    Reading the final result avoids repeatedly scanning those histories.
    """
    async def source():
        try:
            for event in events:
                yield event
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()

    async def run():
        parsed = parse_responses_stream(source(), video_segmentation_format())
        async with parsed:
            result = await parsed.final_result()
        if result.outcome.status != "completed":
            raise RuntimeError(f"SAM 3.1 stream incomplete: {result.outcome!r}")
        latest = {}
        for rec in result.records:
            if isinstance(rec, SegmentationMaskRecord) and rec.frame is not None:
                key = (rec.object_id, rec.frame.frame_index)
                if key not in latest or rec.revision > latest[key].revision:
                    latest[key] = rec
        return latest

    return asyncio.run(run())


def _mask_region(rec, h, w):
    """Return (x, y, clipped bool raster), or None for an offscreen box."""
    b = rec.bounds
    # Parsed wire corners are integers, already converted to half-open bounds.
    x0, y0 = int(np.floor(b.left)), int(np.floor(b.top))
    x1, y1 = int(np.ceil(b.right)), int(np.ceil(b.bottom))
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0 or cx1 <= cx0 or cy1 <= cy0:
        return None
    m = rec.mask
    raster = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(
        m.height, m.width
    )
    if raster.shape != (y1 - y0, x1 - x0):
        raster = cv2.resize(
            raster, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST
        )
    local = raster[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0].astype(bool)
    return cx0, cy0, local


def paste(rec, h, w):
    """Resize the box-local raster to its full box, then clip and paste."""
    out = np.zeros((h, w), bool)
    region = _mask_region(rec, h, w)
    if region is not None:
        x, y, local = region
        out[y:y + local.shape[0], x:x + local.shape[1]] = local
    return out


def segment_video_api(clip, prompt, n_frames, size, file_id=None, events=None, prefer="confidence"):
    """Choose the object visible in most requested frames.

    Ties use mean available mask confidence, then mean box area (prefer="area" swaps the two: the most prominent of
    several equally persistent surfaces, e.g. the main screen among many in Times Square). `events` replays a
    recorded Responses stream instead of calling the API (see analyse.py's events cache). Missing
    confidence on a visible mask gets a presence score of 1.0, as before;
    this fallback is not a measured model confidence.
    """
    w, h = size
    if events is None:
        file_id = file_id or upload(clip)
        events = stream_events(file_id, prompt)
    recs = parse_masks(events)
    objects = {}
    for (oid, fi), rec in recs.items():
        if not 0 <= fi < n_frames:
            continue
        region = _mask_region(rec, h, w)
        if region is not None and region[2].any():
            objects.setdefault(oid, {})[fi] = rec
    masks = np.zeros((n_frames, h, w), bool)
    scores = np.full(n_frames, np.nan)
    info = {"file_id": file_id, "objects": {}}
    if not objects:
        return masks, scores, info

    def rank(oid):
        frames = objects[oid]
        conf = [r.confidence for r in frames.values() if r.confidence is not None]
        area = np.mean([
            (r.bounds.right - r.bounds.left) * (r.bounds.bottom - r.bounds.top)
            for r in frames.values()
        ])
        c = float(np.mean(conf)) if conf else 0.0
        return (len(frames), float(area), c) if prefer == "area" else (len(frames), c, float(area))

    for oid in objects:
        n, *rest = rank(oid)
        a, c = rest if prefer == "area" else rest[::-1]
        info["objects"][oid] = {
            "frames": n, "mean_conf": round(c, 3), "mean_box_area": round(a)
        }
    best = max(objects, key=rank)
    info["chosen"] = best
    for fi, rec in objects[best].items():
        masks[fi] = paste(rec, h, w)
        scores[fi] = rec.confidence if rec.confidence is not None else 1.0
    return masks, scores, info
