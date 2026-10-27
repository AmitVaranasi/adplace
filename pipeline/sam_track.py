"""SAM 3 open-vocabulary video segmentation, run on a laptop.

The video session keeps per-frame memory for every frame it has seen, which exhausts an 18 GB Mac on long clips, so
the clip is segmented in short chunks with a fresh session each, and the target object is linked across chunks by
mask overlap. On an 18 GB Mac the MPS pool peaks near 9 GB with 4-frame chunks (about 4 s/frame). Unload other
local models first (e.g. `ollama stop gemma4:12b`).
"""
import time

import cv2
import numpy as np


def read_frames(video_path, max_side=None):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if max_side and max(f.shape[:2]) > max_side:
            s = max_side / max(f.shape[:2])
            f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"Could not read any frames from {video_path}")
    return frames, fps


def _iou(a, b):
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else 0.0


def segment_video(frames, prompt, checkpoint="facebook/sam3", proc_side=640, chunk=4, device=None, verbose=True):
    """Masks of the single object best matching `prompt` in every frame.

    Returns (masks (N, H, W) bool at frame resolution, scores (N,) float with NaN where the object was not found).
    In the first chunk the target is the object seen in the most frames (ties: higher mean score); later chunks pick
    the object whose first mask overlaps the last known mask most.
    """
    import torch
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16
    h, w = frames[0].shape[:2]
    s = min(1.0, proc_side / max(h, w))
    small = [cv2.cvtColor(cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
             if s < 1 else cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

    model = Sam3VideoModel.from_pretrained(checkpoint, torch_dtype=dtype).to(device).eval()
    processor = Sam3VideoProcessor.from_pretrained(checkpoint)

    masks = np.zeros((len(frames), h, w), bool)
    scores = np.full(len(frames), np.nan)
    last = None
    for start in range(0, len(frames), chunk):
        t0 = time.time()
        session = processor.init_video_session(video=small[start:start + chunk], inference_device=device,
                                               video_storage_device="cpu", dtype=dtype)
        processor.add_text_prompt(session, prompt)
        per_obj = {}  # obj_id -> {frame: (mask, score)}
        with torch.inference_mode():
            for out in model.propagate_in_video_iterator(session):
                res = processor.postprocess_outputs(session, out)
                for oid, score, m in zip(res["object_ids"].tolist(), res["scores"].tolist(), res["masks"]):
                    m = m.cpu().numpy().astype(bool)
                    if m.any():
                        per_obj.setdefault(oid, {})[start + out.frame_idx] = (m, score)
        del session
        if device == "mps":
            torch.mps.empty_cache()
        if per_obj:
            if last is None:
                best = max(per_obj.values(), key=lambda d: (len(d), np.mean([v[1] for v in d.values()])))
            else:
                best = max(per_obj.values(), key=lambda d: _iou(d[min(d)][0], last))
            for i, (m, score) in best.items():
                masks[i] = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                scores[i] = score
            last = best[max(best)][0]
        if verbose:
            print(f"  sam3 '{prompt}' frames {start}-{min(start + chunk, len(frames)) - 1}: {len(per_obj)} objects, "
                  f"{time.time() - t0:.0f}s", flush=True)
    return masks, scores
