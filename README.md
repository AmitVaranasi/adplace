# AdPlace

Same scene, a different brand for each viewer, always disclosed.

A prototype of an opt-in ad format for short videos: creators approve placements in videos they've already filmed,
and different viewers see different labelled sponsors, with nothing re-rendered to switch brands. Built as a standalone prototype; a
short-video feed such as Reels is where it would ship.

Upload a video once. AdPlace looks for places in its longest shot that could carry an ad (it measures posters,
pictures, signs and screens, and glasses, cups and bowls that stay still; on real uploads so far, wall posters and
still glasses have passed), tracks each one against the room, and declines anything it can't place
well, saying why. When a viewer presses play, a second-price auction picks a brand for every placement in well under a
millisecond, and the browser composites the winners into the footage live: matched to its perspective, light, focus
and grain, behind the people and hands in front, wrapped round glasses and cups, and on a blank patch of a cup that has
a design of its own. Every frame carries a visible "AI product placement" label; the file a viewer downloads carries
an invisible Meta Video Seal watermark naming the campaign.

Meta tooling: SAM 3.1 and Muse Spark 1.3 through the Meta Model API, Muse Image 1.0 for the brand artwork, Video Seal.
Built for the Meta AI Developer Hackathon (Oct 26 – Nov 4, 2026). Spec: `SPEC.md`. Upload contract: `INGEST.md`.
Demo script: `docs/DEMO.md`. Write-up: `docs/SUBMISSION.md`.

## Quick start

```bash
git clone https://github.com/AmitVaranasi/adplace && cd adplace
python3.12 -m venv .venv
.venv/bin/pip install --no-deps videoseal==1.0.1 && .venv/bin/pip install -r requirements.txt
export MODEL_API_KEY=...            # Meta Model API key: needed to process uploads, not to watch the demo
.venv/bin/python -m uvicorn app.main:app --port 8411
```

Then open:

| Page | What it is |
|---|---|
| `http://127.0.0.1:8411/` | My videos: the creator's library (uploads in progress, places for ads, placement on/off, remove) |
| `/upload` | upload a clip; live progress, then a report of what was placed and why the rest was declined |
| `/reels?clip=upload_m43248` | creator settings (opt in, approve places and brands) beside two viewers' phones, each with its own auction |
| `/player?clip=upload_m43248` | the full live player: any viewer, the auction's prices, Original / With ads, download |
| `/verify` | Check a file: drop any video to see whether it carries an AdPlace watermark and which campaigns it names |

The repo carries the demo data so a fresh clone plays straight away: the live players (`app/static/player/*`, with
their `stack.mp4`), and the latest report, thumbnails and poster for each video in the library (`runs/*`). Without a
key you can browse the library, reports, players and `/reels`, and run the playback auction; uploading a new video
calls SAM 3.1 and Muse Spark and needs `MODEL_API_KEY`. Rendering a watermarked download needs the source clip and
the upload's mask cache, which stay local (`assets/clips/`, `cache/`); on a fresh clone, upload a video first.

It is a local prototype with no accounts: anyone who can reach the server can change a video's settings or start
an upload (which spends Meta API credit), so don't expose it to the internet as is.

```bash
.venv/bin/python scripts/ingest.py <video>   # the upload pipeline from the command line
.venv/bin/python -m pytest -q                # tests
```

## Meta models: what runs where

| Model | How | Where in the code | What for |
|---|---|---|---|
| SAM 3.1 | Meta Model API (`sam-3.1`), responses decoded with `meta-sam-parser` | `pipeline/sam_api.py`, `pipeline/fast_mask_codec.py`, `scripts/ingest.py` | text-prompted video segmentation of every place Muse lists (posters, frames, signs, menus, screens), of cups, glasses and bowls, of cup handles (to keep logos off them) and of people in front |
| Muse Spark 1.3 | Meta Model API (`muse-spark-1.3`) | `scripts/ingest.py`, `pipeline/rank.py` | reads sampled frames of each scene (the places an ad could go and their kind, brand safety, context tags); classifies each vessel (glass, cup, bowl; plain or branded); judges whether each place is natural for an ad; brand choice with a stated reason on the hand-made clips |
| Muse Image 1.0 | Meta Model API (`muse-image-1.0`) | `scripts/gen_creatives.py` | the fictional brands' artwork (posters, prints, labels) |
| Video Seal 1.0 | open weights, run locally | `pipeline/watermark.py` | invisible watermark naming the campaign in every downloaded file; found after re-encoding to 854 px wide at CRF 28 or 640 px at CRF 23, lost below that or after cropping |

The upload and playback path uses only these. The early hand-made-clip scripts (`scripts/analyse.py`) can also run SAM 3 open
weights locally through `transformers` (`--backend local`, not in requirements.txt); everything an upload does goes
through the API.

## Built during the hackathon

| Day | Added |
|---|---|
| 11 (Nov 3) | Sound (none of the stock demo clips has an audio track, so it shows only on uploads that have one): an upload keeps its chosen shot's audio (`assets/clips/uploads/<stem>.m4a`), the player plays it in step with the video (muted until turned on), `/reels` lets one phone at a time be heard, and downloads carry it with the watermark still verifying (patch drafted by GPT-6.1 Sol). Readable logos: a logo must be at least 28 px tall and sharp enough to read, its blur following the vessel's focus; m44025 and m43999 now place nothing and their players were retired. Vessels never fully in view are declined with reasons instead of crashing the tracker |
| 10 (Nov 3) | Two critical reviews (GPT-6.1 Sol and Cursor, `outputs/judge/`) and fixes: the phone view keeps every ad in frame (a 4:5 window, else the whole frame letterboxed) with a readable label; three more vessel logos (Muse Image) so cups and glasses carry six brands and two viewers differ; consent fails closed (no file or no auction answer: no ads); "Check a file" now recognises a viewer's download (scene manifests) after re-encoding; reports count measured places apart from kinds we can't place on yet; honest QA wording for glasses and cups; path, size and cleanup fixes in the server; locked upload config; a fresh clone runs (requirements.txt, demo data, MIT license) |
| 9 (Nov 2–3) | The application: an opt-in placement setting for short videos (`/reels`). Uploads start with placement off (`pipeline/creator.py`, `creator.json` beside the export); the creator turns it on and approves each place and brand, and every auction and download respects it. Two phones play the same video in sync for two viewers, each with its own auction (prices, runner-up, time). My videos (`/`): the creator's library with uploads in progress, places for ads, the placement switch and Remove; shared navigation across pages. Open-ended discovery merged: Muse lists every place with its kind, bowls carry logos, logos at least a fifth of the vessel's height |
| 8 (Nov 2) | Still glasses behind bowls pinned by their outline, upright; logos moved onto the part of a glass that stays in view; each vessel measured once per upload (cached), player export 2.8 min -> 25 s. 23 unseen videos: 3 ads placed |
| 7 (Nov 1–2) | Glass and cup logos composited live in the browser from a brand-independent atlas, so any brand can win at playback with nothing rendered (matches the server within ~1.5–2 grey levels). Second-price playback auction (`pipeline/auction.py`, `/api/auction`, < 0.2 ms) shown in the player, also driving the watermarked download. Cups and mugs: Muse Spark classifies every SAM "cup" object (glass, cup, other; branded or not), SAM outlines handles to cut out, ink dark on light cups and white on dark, and the logo goes on a blank patch of the cup's own design with clear space round it. Only still objects carry ads; a still object keeps its logo while someone walks in front (camera-carried pose). 13 unseen stock videos: 36 candidates, 1 placed, every decline checked by eye. Demo kit: `docs/DEMO.md`, `docs/SUBMISSION.md`, `scripts/record_demo.py` |
| 6 (Oct 31) | Logos printed on drinking glasses in any uploaded clip (`scripts/glass_decal.py`): wrapped round the glass, its light, focus and grain, behind the fingers holding it; rigid shape fit per frame, resting glasses pinned to the camera. Gates for glasses; glasses occlude posters behind them. The blind Muse judge removed from the final checks (a person judges the look) |
| 5 (Oct 29–30) | Upload pipeline (`scripts/ingest.py`, `/upload`): shot detection, Muse Spark scene reading and brand safety, SAM 3.1 segmentation, tracking, conservative gates with plain-language reasons, live player export, final checks on a rendered copy; 67x faster SAM mask decoding |
| 4 (Oct 29) | Nine placements per viewer in one patio shot: wall poster, menu print, a 3D can, bottle and table-tent card on the table, a logo printed on a real glass, a print replacing the waiter's T-shirt logo, and an A-frame sign on the floor (`scripts/place3d.py`, `scripts/shirt_print.py`). No depth model: table and floor tilt from plate and planter rims SAM 3.1 segmented (a circle images as an ellipse), scale from their sizes, tracks by planar tracking and optical flow. Objects are ray-traced, lit like the footage, cast soft shadows, and only people/plants nearer than an object's foot cover it. New artwork from Muse Image (tent cards, bottle labels, glass and shirt prints). Live player composites any number of objects; pizzeria TV added to it (28–36 dB vs server) |
| 3 (Oct 28) | Live per-viewer WebGL player (GPT Sol), 30–38 dB vs the server render, real time; "Download my version" with Video Seal; a 3D drink can on the patio table; matte fixes (no wall holes before the waiter, no light ring round people) |
| 2 (Oct 27, late night) | After round-1 rating (night footage itself rated 1/5): night clips dropped; 4 new daylight clips (Piccadilly, Shibuya, Yonge-Dundas, two mug clips). Mask-anchored tracking for changing screens (GPT): Times Square max error 8.4 -> 3.5 px, no original-screen sliver. New surface shapes: cylinders/mugs and curved facade 'ribbons' (GPT; GLM's W&B quota ran out). Multiple slots per clip; Muse picks slot x brand from a labelled scene image. Rating page v2 (13 daylight clips) |
| 2 (Oct 27, night) | Hero clip: Times Square screen in daylight (vertical, handheld pan). SAM 3.1 through the Meta Model API (no local GPU/memory needed; raw stream recorded so re-selection is free); API client reviewed and tested by GPT. Muse Spark is the live ranker and sees the scene frame with the surface outlined (GLM). Blinded realism-rating page published (8 clips, originals as controls) |
| 2 (Oct 27, late) | Model ranking (`pipeline/rank.py`): Muse Spark (Meta Model API) when `MODEL_API_KEY` is set, then Llama 3.3 70B (W&B Inference), then the deterministic rules; every reply validated against the eligible set, invented IDs rejected, replies cached by prompt hash; app shows which provider was used |
| 2 (Oct 27, evening) | Second surface (daylight roadside sign; road-safety signs excluded from placement); synthetic occluder test clip with ground truth; occlusion-aware tracking that fits the board's visible edges (GPT); occluder-mask gap filling; slot discovery and per-viewer choice (GLM); exposure matching for printed surfaces (GLM); segmented SAM runs; demo web app v0 |
| 2 (Oct 27) | Spec agreed over 3 review rounds (Claude, GPT-6.1 Sol, GLM-5.3). SAM 3 surface masks with on-disk cache; quad tracking with re-acquisition (GPT); compositing with blur/grain matching, bleed and emissive mode (GPT); Video Seal embed/verify; three fictional brands; render pipeline. Kill test on a handheld night billboard: 95% of frames tracked, 1.6 px median corner error, watermark verified on every exported variant. |

## Early pipeline for the hand-made clips (developer notes)

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install --no-deps videoseal==1.0.1 \
  && .venv/bin/pip install av calflops einops lpips omegaconf pycocotools pytorch_msssim "timm==0.9.16"
ollama stop gemma4:12b   # free RAM for SAM 3 on an 18 GB Mac

# 1. Analyse once per clip (SAM 3 masks, cached)
.venv/bin/python scripts/analyse.py assets/clips/billboard_night.mp4 --prompt billboard
.venv/bin/python scripts/analyse.py assets/clips/billboard_night.mp4 --prompt screen --frames 76:120   # fill gaps

# 2. Track the surface quad (merges mask caches)
python3 scripts/track_eval.py assets/clips/billboard_night.mp4 cache/<billboard dir> cache/<screen dirs...>

# 3. Render a campaign: composite + label + watermark + export + verify
.venv/bin/python scripts/render.py assets/clips/billboard_night.mp4 cache/<billboard dir> brewlark --emissive
```

Video Seal's pip package has two packaging issues on macOS: `decord` has no Apple Silicon wheel (not needed here), and
its model card and attenuation config are resolved relative to the working directory. `pipeline/videoseal_cfg/`
vendors the config and `pipeline/watermark.py` loads it by absolute path.

## Results on uploads (Nov 2)

| Check | Result |
|---|---|
| 23 unseen stock videos (`runs/testset_all23.txt`) | 56 candidates, 3 placed on the Nov 2 code; 1 placed on today's code (the Nov 3 readability gates decline a logo under 28 px tall on screen, and the glass with no clear space for one); every decline checked by eye: nothing to place on (8 videos), a fast blurred sweep, a parking sign, branded or embossed glasses, sauce ramekins, objects picked up or carried in |
| Bar clip (10 s) upload, end to end | 379 s with SAM results cached (595 s before the per-object cache; add 1–4 min for SAM on a fresh upload); wall poster and the still matcha glass placed, the glass she drinks from declined (travels 4.8 radii in a second, limit 0.5) |
| Still glass vs camera | logo moves with the camera within 0.000 px median per frame |
| Live player vs server, glass and cup logos | 1.3–1.9 grey levels mean difference on the logo |
| Playback auction | 0.06–0.2 ms on the server, 5–17 ms round trip |
| Live player frame rate (M3 Pro, Chrome) | 23–24 fps on 24 fps clips |
| Watermarked download from an auction result | verified, 156 s for a 10 s clip |

## Earlier results (billboard_night, 120 frames, 1416x720)

| Check | Result | Target |
|---|---|---|
| Frames tracked | 100% (deterministic) | — |
| Corner error vs SAM mask (median / p90 / max) | 1.5 / 3.0 / 3.7 px | ≤ 3 px median |
| Corner jitter (median / p95) | 0.46 / 2.0 px | no visible slide |
| Watermark after export (3 brands) | verified, 97–99.6% bits | token recovered |
| Clean control | 52% bits, not detected | not detected |
| Compositing | 28–31 ms/frame | — |
| Occluded span (synthetic passer-by covering up to 82% of the board) vs clean-clip truth | 2.5 px median, 5.9 px max, 0 frames lost | ≤ 3 / ≤ 6 px |
| SAM 3 "person" mask vs ground-truth occluder | IoU 0.977 median (found in 87% of frames; gaps filled) | — |
| Daylight roadside sign | 100% tracked, 1.6 px median | — |
| Times Square hero (SAM 3.1 API, 120 frames in 2.7 min) | 100% tracked, 2.7 px median, 6.4 px max; watermark 99.2–100% bits | ≤ 3 px median |
| Muse Spark choice with scene image | matching brand for each viewer, 1–3 s per choice | matches interests |
| Llama 3.3 70B choice, 3 viewers on the night billboard | matching brand for each (brewlark / kestrel / paperwhale) | matches interests |
| Road-safety sign | no placement, no model call | refused |

Footage is openly licensed; see `assets/ATTRIBUTION.md`. Brands are fictional. Code: MIT (`LICENSE`).
