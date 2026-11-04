"""Assemble the 3-minute demo draft from scripts/record_demo.py's raw footage: each segment's clip cut to its line of
voice-over (a neural voice through edge-tts, or macOS `say`), then an end card. Writes outputs/demo/adplace_demo.mp4.

  python scripts/assemble_demo.py [--engine edge|say] [--voice en-US-AndrewMultilingualNeural] [--rate +0%]
Replace the narration with a human voice-over in any editor: outputs/demo/vo_<segment>.m4a are the lines alone.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "demo"

# (raw clip, seconds into it to start, voice-over) -- the lines are docs/DEMO.md's recording plan
SEGMENTS = [
    ("phones", 2.5, "This is AdPlace, an opt-in ad format for short videos. One video, two viewers, and each sees a "
                    "different sponsor printed on the same glass, picked by an auction when their video loads. Nothing "
                    "was re-rendered to switch brands, and every ad is labelled as AI product placement."),
    ("optin", 0.5, "Creators decide. Every upload starts with placement off. Here the creator turns it on, approves each "
                   "place, and turns a brand down: the commuter's glass re-bids to another sponsor straight away. And "
                   "it's strict: on twenty-three stock videos it had never seen, it measured sixty-five places and "
                   "kept one, with a reason for every no."),
    ("report", 1.0, "When a creator uploads, Muse Spark reads the scene and proposes places an ad could go. SAM 3.1 "
                    "outlines each one through the shot. We track it against the room and keep only what stays still "
                    "and in view. The glass she picks up is declined: someone moves it. Muse also turned down a second "
                    "poster as an unnatural place for an ad."),
    ("player", 1.0, "The logo takes the glass's shape, focus and grain, and hands stay in front of it. As a final check, "
                    "Muse Spark reads the brand back off the rendered video; if it can't read it, the placement is "
                    "dropped. The poster artwork is Muse Image."),
    ("verify", 0.0, "The copy a viewer keeps carries an invisible Meta Video Seal watermark. Drop it on Check a file and "
                    "it names the campaign in each place. It survives re-compression; a heavy crop removes it."),
    ("auction", 1.0, "Because the heavy work happens once, at upload, the auction at playback only picks names: under a "
                     "millisecond on the server. These are test bids from a demo inventory."),
    ("close", 2.5, "Next: objects people pick up, more kinds of surface, and an opt-in setting inside a short-video "
                   "feed."),
]
CARD = ["AdPlace", "1 of 23 unseen videos placed  ·  10-15 min per clip, once  ·  test bids",
        "Meta SAM 3.1  ·  Muse Spark  ·  Muse Image  ·  Video Seal"]


def run(*cmd):
    subprocess.run([str(c) for c in cmd], check=True)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="edge", choices=["edge", "say"])
    ap.add_argument("--voice", default=None)
    ap.add_argument("--rate", default=None)
    ap.add_argument("--out", default="adplace_demo.mp4")
    a = ap.parse_args()
    # Check a file takes a minute to answer: keep the download (its first seconds) and the result (its last ones)
    v = OUT / "verify.mp4"
    tail = max(duration(v) - 9, 6)
    run("ffmpeg", "-y", "-loglevel", "error", "-i", v, "-filter_complex",
        f"[0:v]trim=1:6,setpts=PTS-STARTPTS[a];[0:v]trim={tail},setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1[v]",
        "-map", "[v]", "-c:v", "libx264", "-crf", "18", OUT / "verify_cut.mp4")
    parts = []
    for name, start, line in SEGMENTS:
        name = "verify_cut" if name == "verify" else name
        aiff, vo = OUT / f"vo_{name}.raw", OUT / f"vo_{name}.m4a"
        if a.engine == "edge":   # Microsoft's neural voices (free, no key): only the narration text is sent
            for attempt in range(4):   # the service sometimes drops a request
                r = subprocess.run([str(Path(sys.executable).parent / "edge-tts"), "--voice",
                                    a.voice or "en-US-AndrewMultilingualNeural", f"--rate={a.rate or '+0%'}",
                                    "--text", line, "--write-media", str(aiff)], capture_output=True, text=True)
                if r.returncode == 0:
                    break
                time.sleep(3 * (attempt + 1))
            else:
                raise SystemExit(f"edge-tts failed for {name}: {r.stderr[-300:]}")
        else:
            run("say", "-v", a.voice or "Samantha", "-r", a.rate or "165", "-o", aiff.with_suffix(".aiff"), line)
            aiff = aiff.with_suffix(".aiff")
        run("ffmpeg", "-y", "-loglevel", "error", "-i", aiff, "-ar", "44100", "-ac", "2", "-c:a", "aac", "-b:a", "160k", vo)
        aiff.unlink()
        length = duration(vo) + 0.8   # a breath after each line
        clip = OUT / f"{name}.mp4"
        # the clip from `start`, held on its last frame if the line runs longer than the footage
        part = OUT / f"_part_{name}.mp4"
        run("ffmpeg", "-y", "-loglevel", "error", "-ss", start, "-i", clip, "-i", vo,
            "-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration=60,trim=duration={length},setpts=PTS-STARTPTS,"
                               f"fps=30,format=yuv420p[v];[1:a]apad,atrim=duration={length}[a]",
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-b:a", "160k", "-ar", "44100",
            "-ac", "2", part)
        parts.append(part)
        print(f"{name}: {length:.1f} s", flush=True)
    # end card: the last phones frame, darkened, with the figures
    card = OUT / "_part_card.mp4"
    text = ",".join(f"drawtext=fontfile=/System/Library/Fonts/Supplemental/Arial.ttf:text='{t}':fontcolor=white:fontsize={s}:x=(w-text_w)/2:y={y}"
                    for t, s, y in zip(CARD, (64, 30, 26), (330, 430, 490)))
    run("ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.1", "-i", OUT / "close.mp4", "-f", "lavfi", "-t", "4",
        "-i", "anullsrc=r=44100:cl=stereo", "-filter_complex",
        f"[0:v]loop=loop=120:size=1:start=0,trim=duration=4,setpts=PTS-STARTPTS,fps=30,eq=brightness=-0.35,{text},"
        f"format=yuv420p[v]", "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-crf", "18", "-c:a", "aac",
        "-shortest", card)
    parts.append(card)
    lst = OUT / "_parts.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    final = OUT / a.out
    run("ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", final)
    for p in parts + [lst]:
        p.unlink()
    print(f"wrote {final}: {duration(final):.1f} s")


if __name__ == "__main__":
    main()
