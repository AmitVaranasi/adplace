"""Day-1 go/no-go: paste a flat creative onto the tracked quad and export, to judge sliding at 1x playback.

No realism here on purpose (that is pipeline/composite.py); this only tests whether the geometry holds.

  python scripts/killtest.py assets/clips/billboard_night.mp4 cache/<dir> assets/creatives/brewlark_16x9.png
Writes outputs/killtest_<clip>.mp4 (side by side: original | placed, with the track status) and prints jitter stats.
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.sam_track import read_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def paste(frame, creative_bgra, quad):
    h, w = creative_bgra.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    H = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    size = frame.shape[1::-1]
    warped = cv2.warpPerspective(creative_bgra[:, :, :3], H, size, flags=cv2.INTER_LINEAR)
    alpha = cv2.warpPerspective(creative_bgra[:, :, 3], H, size, flags=cv2.INTER_LINEAR)[..., None] / 255.0
    return (frame * (1 - alpha) + warped * alpha).astype(np.uint8)


def jitter(quads):
    """Per-frame corner acceleration (px): high values mean visible shake; smooth camera motion stays low."""
    ok = np.isfinite(quads).all(axis=(1, 2))
    acc = [np.linalg.norm(quads[i + 1] - 2 * quads[i] + quads[i - 1], axis=1).mean()
           for i in range(1, len(quads) - 1) if ok[i - 1] and ok[i] and ok[i + 1]]
    return np.array(acc)


def main(clip, cache, creative_path):
    frames, fps = read_frames(clip)
    quads = np.load(Path(cache) / "quads.npy")
    creative = cv2.imread(creative_path, cv2.IMREAD_UNCHANGED)  # BGRA for an RGBA PNG
    if creative.shape[2] == 3:
        creative = np.dstack([creative, np.full(creative.shape[:2], 255, np.uint8)])

    out = ROOT / "outputs" / f"killtest_{Path(clip).stem}.mp4"
    h, w = frames[0].shape[:2]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                           f"{2 * w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt",
                           "yuv420p", str(out)], stdin=subprocess.PIPE)
    for i, f in enumerate(frames):
        placed = paste(f, creative, quads[i]) if np.isfinite(quads[i]).all() else f.copy()
        tag = "tracked" if np.isfinite(quads[i]).all() else "NO TRACK"
        cv2.putText(placed, f"{i:3d} {tag}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        ff.stdin.write(np.hstack([f, placed]).tobytes())
    ff.stdin.close(); ff.wait()

    acc = jitter(quads)
    print(f"tracked {np.isfinite(quads).all(axis=(1, 2)).mean():.0%} of frames; corner jitter (accel) "
          f"median {np.median(acc):.2f} px, p95 {np.percentile(acc, 95):.2f} px, max {acc.max():.2f} px")
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
