"""Turn a video into a per-frame bob position. Every backend returns (times, xy) with NaN where the bob wasn't found."""
import cv2
import numpy as np

COLOR_RANGES = {  # HSV ranges; red wraps around hue 0
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "blue": [((100, 120, 70), (130, 255, 255))],
    "green": [((40, 80, 60), (85, 255, 255))],
    "yellow": [((20, 120, 100), (35, 255, 255))],
}


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


def track_color(video_path, color="red", min_area=40):
    """Baseline: HSV threshold + largest blob centroid. This is what SAM has to beat."""
    frames, fps = read_frames(video_path)
    xy = np.full((len(frames), 2), np.nan)
    for i, f in enumerate(frames):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        mask = sum(cv2.inRange(hsv, lo, hi) for lo, hi in COLOR_RANGES[color])
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            if stats[k, cv2.CC_STAT_AREA] >= min_area:
                xy[i] = cents[k]
    return np.arange(len(frames)) / fps, xy


def track_sam3(video_path, prompt="pendulum bob", checkpoint="facebook/sam3", max_side=640, device=None, chunk=4,
               verbose=True):
    """SAM 3 open-vocabulary video tracking via transformers. Needs Hugging Face access to the gated checkpoint.

    The video session keeps per-frame memory for every frame it has seen, which exhausts a laptop's RAM on long
    clips, so the video is tracked in chunks with a fresh session each, and the bob is linked across chunks by
    position. On an 18 GB Mac the MPS driver pool peaks near 9 GB with 4-frame chunks (about 4 s/frame);
    24-frame chunks pass 13 GB.
    """
    import time

    import torch
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16
    frames, fps = read_frames(video_path, max_side=max_side)
    scale = 1.0
    cap = cv2.VideoCapture(str(video_path)); full_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH); cap.release()
    if full_w:
        scale = full_w / frames[0].shape[1]  # report positions in original-video pixels

    model = Sam3VideoModel.from_pretrained(checkpoint, torch_dtype=dtype).to(device).eval()
    processor = Sam3VideoProcessor.from_pretrained(checkpoint)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

    xy, last = np.full((len(frames), 2), np.nan), None
    for start in range(0, len(frames), chunk):
        t0 = time.time()
        session = processor.init_video_session(video=rgb[start:start + chunk], inference_device=device,
                                               video_storage_device="cpu", dtype=dtype)
        processor.add_text_prompt(session, prompt)
        per_obj = {}  # obj_id -> {frame: (x, y, score)}
        with torch.inference_mode():
            for out in model.propagate_in_video_iterator(session):
                res = processor.postprocess_outputs(session, out)
                for oid, score, mask in zip(res["object_ids"].tolist(), res["scores"].tolist(), res["masks"]):
                    ys, xs = np.nonzero(mask.cpu().numpy())
                    if len(xs):
                        per_obj.setdefault(oid, {})[start + out.frame_idx] = (xs.mean() * scale, ys.mean() * scale,
                                                                             score)
        del session
        if device == "mps":
            torch.mps.empty_cache()
        if per_obj:
            if last is None:
                # The bob is the object SAM keeps for the most frames; ties go to higher mean score.
                best = max(per_obj.values(), key=lambda d: (len(d), np.mean([v[2] for v in d.values()])))
            else:
                # Later chunks: the object whose first sighting is nearest where the bob was last seen.
                best = min(per_obj.values(), key=lambda d: np.hypot(*(np.array(d[min(d)][:2]) - last)))
            for i, (x, y, _) in best.items():
                xy[i] = (x, y)
            last = np.array(best[max(best)][:2])
        if verbose:
            print(f"  sam3 frames {start}-{min(start + chunk, len(frames)) - 1}: {len(per_obj)} objects, "
                  f"{time.time() - t0:.0f}s", flush=True)
    return np.arange(len(frames)) / fps, xy


def track_meta_api(video_path, prompt="pendulum bob"):
    """Hosted SAM 3.1 through the Meta Model API. Fill in once the official endpoint and schema are confirmed."""
    raise NotImplementedError(
        "Meta Model API backend not wired yet: add the official SAM 3.1 endpoint, auth and response schema "
        "from the participant docs, then return (times, xy) like the other backends."
    )


BACKENDS = {"color": track_color, "sam3": track_sam3, "meta": track_meta_api}
