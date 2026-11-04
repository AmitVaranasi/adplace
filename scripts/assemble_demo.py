"""Assemble the 3-minute demo draft from scripts/record_demo.py's raw footage: each segment's clip cut to its line of
voice-over (a neural voice through edge-tts, or macOS `say`), then an end card. Writes outputs/demo/adplace_demo.mp4.

  python scripts/assemble_demo.py [--engine edge|say] [--voice en-US-AvaMultilingualNeural] [--rate +0%]
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

# (raw clip, seconds into it to start, chapter caption, voice-over) -- the lines are docs/DEMO.md's recording plan
SEGMENTS = [
    ("phones", 2.5, "The idea",
     "Short-video creators earn from brands only when they film a sponsored post. AdPlace is a prototype of a different "
     "ad format: the creator opts in, and a sponsor appears inside videos they've already made. Here, one video plays "
     "to two viewers. The commuter sees Brewlark on the glass; the runner sees Fizzwell. Same footage, nothing "
     "re-rendered, and every ad is labelled."),
    ("report", 1.0, "1  Upload",
     "It starts when a creator uploads. Meta's Muse Spark reads the scene, proposes places an ad could go, and checks "
     "that the scene is safe for brands. SAM 3.1 outlines each place through the whole shot, along with the people in "
     "front of it. We track every place against the room and measure it: how long it's on screen, how steady it is, "
     "how often it's covered. Only still, visible places pass. Everything else is declined, with a plain reason, like "
     "the glass she picks up."),
    ("optin", 2.0, "2  The creator decides",
     "Nothing runs until the creator says yes. Placement starts off. They turn it on, approve each place, and choose "
     "the brands they're happy to carry. Untick a brand, and that viewer's glass re-bids to another sponsor "
     "straight away."),
    ("auction", 2.0, "3  Each viewer, their own sponsor",
     "When a viewer's video loads, a second-price auction picks a brand for each place, matched to their interests, "
     "in under a millisecond. The heavy work was done once, at upload, so playback only picks names, and the browser "
     "draws the winner in live."),
    ("player", 1.0, "4  It should look filmed",
     "The logo takes the glass's shape, focus and grain, and the writing hand stays in front of it. Switch to the "
     "original to compare. As a final check, Muse Spark reads the brand back off the rendered video. If it can't "
     "read it, the placement is dropped. The poster artwork is made with Muse Image."),
    ("verify", 0.0, "5  Disclosed and traceable",
     "Every ad carries a visible label. And the copy a viewer keeps carries an invisible Meta Video Seal watermark. "
     "Drop it on Check a file, and it names the campaign in each place. It survives re-compression; a heavy crop "
     "removes it."),
    ("funnel", 1.0, "Where it stands",
     "It's deliberately strict. On twenty-three stock videos it had never seen, it measured sixty-five places and "
     "kept one. It would rather place nothing than place badly, and today it works best in cafés and restaurants, "
     "on still glasses and posters."),
    ("close", 2.5, "Next",
     "Next: objects people pick up, more kinds of surfaces, and an opt-in setting inside a short-video feed. "
     "AdPlace: a creator's own videos, sponsored with their consent."),
]
TITLE = ["AdPlace", "Sponsors inside videos creators already made, with their consent"]
CARD = ["AdPlace", "1 of 23 unseen videos placed  ·  10-15 min per clip, once  ·  test bids",
        "Meta SAM 3.1  ·  Muse Spark  ·  Muse Image  ·  Video Seal"]


def run(*cmd):
    subprocess.run([str(c) for c in cmd], check=True)


FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def caption(text):
    """The chapter, bottom left, on a dark band: where a viewer is in the story."""
    return (f"drawbox=x=0:y=ih-84:w=iw:h=84:color=black@0.62:t=fill,"
            f"drawtext=fontfile={FONT}:text='{text}':fontcolor=white:fontsize=34:x=40:y=h-62")


def title_card(src, lines, out, seconds=3.5):
    """A still of `src`'s first frame, darkened, with the title (silent)."""
    text = ",".join(f"drawtext=fontfile={FONT}:text='{t}':fontcolor=white:fontsize={sz}:x=(w-text_w)/2:y={y}"
                    for t, sz, y in zip(lines, (88, 34, 30), (350, 470, 525)))
    run("ffmpeg", "-y", "-loglevel", "error", "-ss", "3", "-i", src, "-f", "lavfi", "-t", seconds,
        "-i", "anullsrc=r=44100:cl=stereo", "-filter_complex",
        f"[0:v]loop=loop=200:size=1:start=0,trim=duration={seconds},setpts=PTS-STARTPTS,fps=30,eq=brightness=-0.4,"
        f"{text},format=yuv420p[v]", "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-crf", "18", "-c:a", "aac",
        "-ar", "44100", "-ac", "2", "-shortest", out)


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
    opening = OUT / "_part_title.mp4"
    title_card(OUT / "phones.mp4", TITLE, opening)
    parts = [opening]
    for name, start, chapter, line in SEGMENTS:
        name = "verify_cut" if name == "verify" else name
        aiff, vo = OUT / f"vo_{name}.raw", OUT / f"vo_{name}.m4a"
        if a.engine == "edge":   # Microsoft's neural voices (free, no key): only the narration text is sent
            for attempt in range(4):   # the service sometimes drops a request
                r = subprocess.run([str(Path(sys.executable).parent / "edge-tts"), "--voice",
                                    a.voice or "en-US-AvaMultilingualNeural", f"--rate={a.rate or '+0%'}",
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
                               f"fps=30,{caption(chapter)},format=yuv420p[v];[1:a]apad,atrim=duration={length}[a]",
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
