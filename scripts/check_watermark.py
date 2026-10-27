"""Video Seal round trip on a real clip: embed, re-encode with H.264 at CRF 23, verify; plus an unwatermarked control.

  python scripts/check_watermark.py assets/clips/billboard_night.mp4
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import watermark  # noqa: E402


def read(path):
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames, cap.get(cv2.CAP_PROP_FPS)


def write_h264(frames, fps, path, crf=23):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
                          str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()


def main(clip):
    frames, fps = read(clip)
    campaigns = ["brand-a-001", "brand-b-001", "brand-c-001"]
    t0 = time.time()
    marked = watermark.embed(frames, campaigns[1])
    t_embed = time.time() - t0
    psnr = np.mean([cv2.PSNR(a, b) for a, b in zip(frames, marked)])
    with tempfile.TemporaryDirectory() as d:
        write_h264(marked, fps, Path(d) / "marked.mp4")
        write_h264(frames, fps, Path(d) / "clean.mp4")
        rmarked, _ = read(Path(d) / "marked.mp4")
        rclean, _ = read(Path(d) / "clean.mp4")
    t0 = time.time()
    pos = watermark.verify(rmarked, campaigns)
    t_verify = time.time() - t0
    neg = watermark.verify(rclean, campaigns)
    print(f"{len(frames)} frames {frames[0].shape[1]}x{frames[0].shape[0]}  embed {t_embed:.1f}s  verify {t_verify:.1f}s  "
          f"PSNR {psnr:.1f} dB")
    print(f"watermarked, CRF 23 re-encode: decoded {pos['campaign']}  bit accuracy {pos['bit_accuracy']:.3f}")
    print(f"clean control:                 decoded {neg['campaign']}  bit accuracy {neg['bit_accuracy']:.3f}")
    ok = pos["campaign"] == campaigns[1] and neg["campaign"] is None
    print("ROUND TRIP:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1])
