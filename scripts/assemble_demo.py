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
def run(*cmd):
    subprocess.run([str(c) for c in cmd], check=True)


FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def caption(text):
    """The chapter, bottom left, on a dark band: where a viewer is in the story."""
    return (f"drawbox=x=0:y=ih-84:w=iw:h=84:color=black@0.62:t=fill,"
            f"drawtext=fontfile={FONT}:text='{text}':fontcolor=white:fontsize=34:x=40:y=h-62")


CARDS = ROOT / "scripts" / "demo_cards"   # title.html, end.html: rendered to outputs/demo/cards/*.png


def render_cards():
    """The title and end cards: HTML in the app's look, screenshotted at the video's size."""
    import os
    from playwright.sync_api import sync_playwright
    chrome = os.path.expanduser("~/Library/Caches/ms-playwright/chromium-1243/chrome-mac-arm64/"
                                "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing")
    (OUT / "cards").mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=chrome, headless=True,
                              args=["--headless=new", "--allow-file-access-from-files"])
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        for name in ("title", "end"):
            pg.goto((CARDS / f"{name}.html").as_uri())
            pg.wait_for_timeout(700)
            pg.screenshot(path=str(OUT / "cards" / f"{name}.png"))
        b.close()


def image_card(png, out, seconds):
    """A designed card held for `seconds`: a slow push-in, faded in and out (silent)."""
    frames = int(seconds * 30)
    run("ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", png, "-f", "lavfi", "-t", seconds,
        "-i", "anullsrc=r=44100:cl=stereo", "-filter_complex",
        f"[0:v]scale=2880:1800,zoompan=z='1+0.035*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={frames}:s=1440x900:fps=30,fade=t=in:st=0:d=0.5,fade=t=out:st={seconds - 0.4}:d=0.4,format=yuv420p[v]",
        "-map", "[v]", "-map", "1:a", "-t", seconds, "-c:v", "libx264", "-crf", "16", "-c:a", "aac", "-ar", "44100",
        "-ac", "2", out)


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
    render_cards()
    opening = OUT / "_part_title.mp4"
    image_card(OUT / "cards" / "title.png", opening, 4.5)
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
    card = OUT / "_part_card.mp4"
    image_card(OUT / "cards" / "end.png", card, 6)
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
