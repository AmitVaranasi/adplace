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


def creator(base, enabled):
    """The bar clip's opt-in (pipeline/creator.py): off for the reels part to turn on, on for the rest."""
    import json
    import urllib.request
    req = urllib.request.Request(f"{base}/api/creator/upload_m43248", method="PUT",
                                 data=json.dumps({"enabled": enabled}).encode(),
                                 headers={"content-type": "application/json"})
    urllib.request.urlopen(req).read()


def reels(pg, base, run):
    """Two phones, one video: placement off, the creator turns it on, both play in sync, a brand is turned down."""
    creator(base, False)
    pg.goto(f"{base}/reels?clip=upload_m43248&a=viewer-a&b=viewer-b")
    pg.wait_for_function('!document.getElementById("play").disabled', timeout=120000)
    pg.wait_for_timeout(2500)
    pg.click(".switch")
    pg.wait_for_timeout(2500)
    pg.click("#play")
    pg.wait_for_timeout(5000)
    pg.click("#brands input[value=brewlark]")
    pg.wait_for_timeout(4000)
    pg.click("#brands input[value=brewlark]")
    pg.wait_for_timeout(1500)
    pg.select_option('[data-side="b"] select', "viewer-c")
    pg.wait_for_timeout(4000)
    creator(base, True)


def player_ready(pg):
    pg.wait_for_function('!document.getElementById("play").disabled', timeout=120000)
    pg.wait_for_function('document.getElementById("auction").textContent.includes("auction")', timeout=30000)


def viewers(pg, base, run):
    pg.goto(f"{base}/player?clip=upload_m43248")
    player_ready(pg)
    pg.wait_for_timeout(1500)
    for v in ("viewer-a", "viewer-c", "viewer-b", "viewer-d"):
        pg.click(f'[data-profile="{v}"]')
        pg.wait_for_timeout(2200)


def report(pg, base, run):
    pg.goto(f"{base}/upload?run={run}")
    pg.wait_for_selector("#report-title", state="visible", timeout=60000)
    pg.wait_for_timeout(2500)
    for _ in range(6):
        pg.mouse.wheel(0, 300)
        pg.wait_for_timeout(1200)
    pg.wait_for_timeout(1500)


def auction(pg, base, run):
    pg.goto(f"{base}/player?clip=upload_m43248")
    player_ready(pg)
    pg.click('[data-profile="viewer-c"]')
    pg.wait_for_timeout(2500)
    for _ in range(2):
        pg.click("#rebid")
        pg.wait_for_timeout(2000)
    pg.click('[data-mode="original"]')
    pg.wait_for_timeout(1500)
    pg.click('[data-mode="ads"]')
    pg.wait_for_timeout(1000)
    pg.click("#play")
    pg.wait_for_timeout(10500)   # the whole 10 s clip at full speed


def patio(pg, base, run):
    pg.goto(f"{base}/player?clip=patio_waiter")
    player_ready(pg)
    pg.click("#play")
    for v in ("viewer-b", "viewer-e", "viewer-f"):
        pg.wait_for_timeout(3500)
        pg.click(f'[data-profile="{v}"]')
    pg.wait_for_timeout(4000)


def download(pg, base, run):
    pg.goto(f"{base}/player?clip=upload_m43248")
    player_ready(pg)
    pg.click("#download")
    pg.wait_for_function('document.querySelector("#download-status a") || '
                         'document.getElementById("download-status").textContent.includes("failed")', timeout=900000)
    pg.wait_for_timeout(3000)


PARTS = {"reels": reels, "viewers": viewers, "report": report, "auction": auction, "patio": patio, "download": download}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8411")
    ap.add_argument("--run", default="r_still_m43248")
    ap.add_argument("--parts", default=",".join(PARTS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    creator(a.base, True)   # the player parts need the bar clip's ads on
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=str(CHROME), headless=True,
                                    args=["--headless=new", "--use-angle=metal", "--ignore-gpu-blocklist"])
        for name in a.parts.split(","):
            tmp = OUT / f"_{name}"
            shutil.rmtree(tmp, ignore_errors=True)
            ctx = browser.new_context(viewport=SIZE, device_scale_factor=1, record_video_dir=str(tmp),
                                      record_video_size=SIZE)
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
