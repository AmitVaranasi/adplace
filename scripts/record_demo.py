"""Record raw screen footage of the running app for the demo video (docs/DEMO.md), one file per part, into
outputs/demo/<part>.mp4. Real app, real data: the player, the auction and a finished upload's report (?run=).

  python scripts/record_demo.py [--base http://127.0.0.1:8411] [--run r_still_m43248] [--parts viewers,report,...]

Headless Chrome on the Mac's GPU (Metal), so the player composites in real time. Needs the app running.
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "demo"
CHROME = Path(os.path.expanduser("~/Library/Caches/ms-playwright/chromium-1243/chrome-mac-arm64/"
                                 "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"))
SIZE = {"width": 1440, "height": 900}


def creator(base, enabled, clip="upload_m43248"):
    """A video's opt-in (pipeline/creator.py): off for the opt-in part to turn on, on for the rest."""
    import json
    import urllib.request
    req = urllib.request.Request(f"{base}/api/creator/{clip}", method="PUT",
                                 data=json.dumps({"enabled": enabled}).encode(),
                                 headers={"content-type": "application/json"})
    urllib.request.urlopen(req).read()


HERO = "upload_m1737"   # the iced coffee: an automatic placement, read by Muse in every sampled frame


def reels_ready(pg):
    pg.wait_for_function('!document.getElementById("play").disabled', timeout=120000)
    pg.wait_for_timeout(2000)


def phones(pg, base, run):
    """0:00-0:20: two phones, one video, two sponsors on the same glass."""
    pg.goto(f"{base}/reels?clip={HERO}&steady=1")
    reels_ready(pg)
    pg.click("#play")
    pg.wait_for_timeout(18000)


def optin(pg, base, run):
    """0:20-0:50: the library, the test-set funnel, then the creator turns placement on and turns a brand down."""
    pg.goto(f"{base}/")
    pg.wait_for_timeout(3500)
    pg.click('[data-filter="test"]')
    pg.wait_for_timeout(4500)
    creator(base, False, HERO)
    pg.goto(f"{base}/reels?clip={HERO}&steady=1")
    reels_ready(pg)
    pg.click("#play")
    pg.wait_for_timeout(3500)
    pg.click(".switch")                                   # on: both phones' ads appear
    pg.wait_for_timeout(4500)
    pg.click("#brands input[value=brewlark]")             # the commuter's glass re-bids
    pg.wait_for_timeout(4500)
    pg.click("#brands input[value=brewlark]")
    pg.wait_for_timeout(2500)
    creator(base, True, HERO)


def report(pg, base, run):
    """0:50-1:30: what Meta's models found in the bar clip, what was placed and why the rest was declined."""
    pg.goto(f"{base}/upload?run={run}")
    pg.wait_for_selector("#report-title", state="visible", timeout=60000)
    pg.wait_for_timeout(3000)
    for _ in range(5):
        pg.mouse.wheel(0, 320)
        pg.wait_for_timeout(1500)
    for d in pg.query_selector_all("details")[:2]:
        d.click()
        pg.wait_for_timeout(1500)
    pg.wait_for_timeout(1500)


def player_ready(pg):
    pg.wait_for_function('!document.getElementById("play").disabled', timeout=120000)
    pg.wait_for_function('document.getElementById("auction").textContent.includes("auction")', timeout=30000)


def player(pg, base, run):
    """1:30-2:15: the full player: the logo behind the writing hand, Original vs With ads; then the bar clip."""
    pg.goto(f"{base}/player?clip={HERO}&viewer=viewer-a&steady=1")
    player_ready(pg)
    pg.click("#play")
    pg.wait_for_timeout(6000)
    for mode in ("original", "ads", "original", "ads"):
        pg.click(f'[data-mode="{mode}"]')
        pg.wait_for_timeout(1800)
    pg.wait_for_timeout(3000)
    pg.goto(f"{base}/player?clip=upload_m43248&viewer=viewer-a&steady=1")
    player_ready(pg)
    pg.click("#play")
    pg.wait_for_timeout(10500)


def verify(pg, base, run):
    """The viewer's download (pre-rendered, so instant), then Check a file names its campaigns."""
    pg.goto(f"{base}/player?clip={HERO}&viewer=viewer-a&steady=1")
    player_ready(pg)
    pg.wait_for_timeout(1500)
    with pg.expect_download(timeout=900000) as dl:
        pg.click("#download")
    path = OUT / "viewer_download.mp4"
    dl.value.save_as(path)
    pg.wait_for_timeout(2500)
    pg.goto(f"{base}/verify")
    pg.wait_for_timeout(1500)
    pg.set_input_files("#file", str(path))
    pg.wait_for_timeout(1000)
    pg.click("#check")
    pg.wait_for_function('!document.getElementById("result").hidden', timeout=300000)
    pg.wait_for_timeout(5000)


def auction(pg, base, run):
    """2:15-2:40: both bids again on the two-place bar clip: prices, runner-up, time."""
    pg.goto(f"{base}/reels?clip=upload_m43248&steady=1")
    reels_ready(pg)
    pg.click("#play")
    pg.wait_for_timeout(3000)
    for _ in range(2):
        pg.click("#rebid")
        pg.wait_for_timeout(3000)


def close(pg, base, run):
    """End card under the voice-over: the two phones playing."""
    phones(pg, base, run)


PARTS = {"phones": phones, "optin": optin, "report": report, "player": player, "verify": verify, "auction": auction,
         "close": close}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8411")
    ap.add_argument("--run", default="r_eval_m43248")
    ap.add_argument("--parts", default=",".join(PARTS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for clip in (HERO, "upload_m43248"):   # every part but the opt-in starts with ads on
        creator(a.base, True, clip)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=str(CHROME), headless=True,
                                    args=["--headless=new", "--use-angle=metal", "--ignore-gpu-blocklist"])
        for name in a.parts.split(","):
            tmp = OUT / f"_{name}"
            shutil.rmtree(tmp, ignore_errors=True)
            ctx = browser.new_context(viewport=SIZE, device_scale_factor=1, record_video_dir=str(tmp),
                                      record_video_size=SIZE, accept_downloads=True)
            pg = ctx.new_page()
            PARTS[name](pg, a.base, a.run)
            video = pg.video.path()
            ctx.close()
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-c:v", "libx264", "-crf", "18",
                            "-pix_fmt", "yuv420p", "-r", "30", str(OUT / f"{name}.mp4")], check=True)
            shutil.rmtree(tmp)
            print("recorded", OUT / f"{name}.mp4", flush=True)
        browser.close()


if __name__ == "__main__":
    main()
