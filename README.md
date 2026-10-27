# AdPlace

Same scene, a different brand for each viewer, always disclosed.

AdPlace finds flat surfaces in a video (billboards, posters, screens, signs), tracks them, and places a brand's ad on
them after filming. Every placement carries a visible "AI product placement" label and an invisible Meta Video Seal
watermark holding the campaign token, so placements can always be detected.

Built for the Meta AI Developer Hackathon (Oct 26 – Nov 4, 2026). Full spec: `SPEC.md`. Module contract: `CONTRACT.md`.

## Built during the hackathon

| Day | Added |
|---|---|
| 2 (Oct 27) | Spec agreed over 3 review rounds (Claude, GPT-6.1 Sol, GLM-5.3). SAM 3 surface masks with on-disk cache; quad tracking with re-acquisition (GPT); compositing with blur/grain matching, bleed and emissive mode (GPT); Video Seal embed/verify; three fictional brands; render pipeline. Kill test on a handheld night billboard: 95% of frames tracked, 1.6 px median corner error, watermark verified on every exported variant. |

## Pipeline

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

## Results so far (billboard_night, 120 frames, 1416x720)

| Check | Result | Target |
|---|---|---|
| Frames tracked | 100% (deterministic) | — |
| Corner error vs SAM mask (median / p90 / max) | 1.5 / 3.0 / 3.7 px | ≤ 3 px median |
| Corner jitter (median / p95) | 0.46 / 2.0 px | no visible slide |
| Watermark after export (3 brands) | verified, 97–99.6% bits | token recovered |
| Clean control | 52% bits, not detected | not detected |
| Compositing | 28 ms/frame | — |

Footage is openly licensed; see `assets/ATTRIBUTION.md`. Brands are fictional.
