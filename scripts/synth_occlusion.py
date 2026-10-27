"""Add a synthetic foreground occluder (a dark walking silhouette) to a clip, with its ground-truth mask.

Tests the occlusion path (SAM "person" masks, tracker ignoring occluder features, ad composited behind) on footage
we are allowed to use. The occluder is synthetic and must be disclosed as such wherever it is shown.

  python scripts/synth_occlusion.py assets/clips/billboard_night.mp4 --start 30 --end 90
Writes assets/clips/<clip>_occluded.mp4 and cache/<clip>_occluded_truth.npz (masks (N,H,W) bool).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.sam_track import read_frames  # noqa: E402


def silhouette(h, w, cx, base_y, height, phase):
    """Soft alpha (float32 0..1) of a person seen from behind, walking (slight bob and arm swing)."""
    a = np.zeros((h, w), np.float32)
    s = height / 7.5  # head-heights
    bob = 0.05 * s * np.sin(2 * phase)
    top = base_y - height + bob
    head_c = (int(cx), int(top + 0.5 * s))
    cv2.ellipse(a, head_c, (int(0.42 * s), int(0.55 * s)), 0, 0, 360, 1, -1, cv2.LINE_AA)
    cv2.rectangle(a, (int(cx - 0.18 * s), int(top + 0.9 * s)), (int(cx + 0.18 * s), int(top + 1.3 * s)), 1, -1)
    torso = np.array([[cx - 1.05 * s, top + 1.35 * s], [cx + 1.05 * s, top + 1.35 * s],
                      [cx + 0.85 * s, top + 4.2 * s], [cx - 0.85 * s, top + 4.2 * s]], np.int32)
    cv2.fillConvexPoly(a, torso, 1, cv2.LINE_AA)
    cv2.ellipse(a, (int(cx), int(top + 1.5 * s)), (int(1.05 * s), int(0.4 * s)), 0, 180, 360, 1, -1, cv2.LINE_AA)
    for side in (-1, 1):  # arms swing opposite to each other
        sw = 0.35 * s * np.sin(phase) * side
        cv2.line(a, (int(cx + side * 0.95 * s), int(top + 1.6 * s)),
                 (int(cx + side * 1.05 * s + sw), int(top + 4.0 * s)), 1, int(0.42 * s), cv2.LINE_AA)
    for side in (-1, 1):  # legs
        sw = 0.4 * s * np.sin(phase) * side
        cv2.line(a, (int(cx + side * 0.4 * s), int(top + 4.1 * s)), (int(cx + side * 0.45 * s + sw), int(base_y)),
                 1, int(0.55 * s), cv2.LINE_AA)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--start", type=int, default=30)
    ap.add_argument("--end", type=int, default=90)
    args = ap.parse_args()

    frames, fps = read_frames(args.clip)
    h, w = frames[0].shape[:2]
    masks = np.zeros((len(frames), h, w), bool)
    out_frames = []
    rng = np.random.default_rng(0)
    for i, f in enumerate(frames):
        f = f.astype(np.float32)
        if args.start <= i < args.end:
            t = (i - args.start) / (args.end - args.start)
            cx = -0.15 * w + t * 1.3 * w  # walks left to right across the frame, close to camera
            a = silhouette(h, w, cx, base_y=h * 1.25, height=h * 1.3, phase=t * 6 * np.pi)
            # Motion blur along the walking direction, then a soft edge.
            k = np.zeros((1, 9), np.float32); k[0, :] = 1 / 9
            a = cv2.GaussianBlur(cv2.filter2D(a, -1, k), (0, 0), 1.2)
            # Nearly black figure with a faint cool rim where it is lit by the screen.
            body = np.full_like(f, 6.0) + rng.normal(0, 2.0, f.shape).astype(np.float32)
            rim = (a > 0.05) & (a < 0.6)
            body[rim] += np.array([28, 22, 18], np.float32)
            f = f * (1 - a[..., None]) + body * a[..., None]
            masks[i] = a > 0.5
        out_frames.append(np.clip(f, 0, 255).astype(np.uint8))

    out = ROOT / "assets" / "clips" / f"{Path(args.clip).stem}_occluded.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(out)],
                         stdin=subprocess.PIPE)
    for f in out_frames:
        p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
    truth = ROOT / "cache" / f"{Path(args.clip).stem}_occluded_truth.npz"
    np.savez_compressed(truth, masks=masks)
    print("wrote", out, "and", truth, f"(occluder in frames {args.start}-{args.end - 1})")


if __name__ == "__main__":
    main()
