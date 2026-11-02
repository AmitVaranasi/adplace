# Ingest: any uploaded video -> per-viewer placements, conservatively

Goal: a clip nobody tuned gets only placements the pipeline can show will hold up. Fewer ads that look right beat
more ads that might not. Every candidate the pipeline finds is either placed or declined with a reason, and the
viewer sees that report (the "always disclosed" story extends to "and here is what we chose not to do").

## Stages (scripts/ingest.py, owner: Claude)

1. **upload** - the file is saved as `runs/<run_id>/input.mp4` (re-encoded to <= 1280 px wide, constant frame rate).
2. **shots** - hard cuts and dissolves split the clip; each shot is processed on its own.
3. **segment** - Muse Spark reads three keyframes (twice, independently) and lists, open-ended, every place that
   could carry an ad, each with a short name SAM can find and a kind: `display` (an existing poster, picture, sign,
   menu board, screen: the ad covers it), `print` (napkin, menu, placemat, table card, box: the logo is printed on
   it), `vessel` (glass, cup, mug, bowl, ramekin: printed round it) or `other` (walls, tables, plates, clothing...).
   It also judges brand safety (children, hospital, accident -> no ads at all) and gives context tags. SAM 3.1 (Meta
   Model API) then outlines every display and print place by its name (up to 10), vessels through `cup` and `bowl`,
   cups' handles, and `person` for what's in front. `other` places are reported as found, with no placement method
   yet. Nothing is excluded by a fixed list: the gates below do the filtering.
4. **track** - each candidate surface is tracked through its shot (planar_track / mask tracking, as now).
5. **score** - `pipeline/gates.py` (owner: GPT Sol) accepts or declines each candidate from measured facts.
6. **render / export** - accepted candidates become `dynamic_slots`; `render_dynamic` and `export_player` run as for
   the hand-made clips. 3D table objects only when the table calibrates itself (>= 2 round objects agreeing on its
   tilt, the plate-rim method); otherwise flat placements only.
   Drinking glasses, coffee cups and mugs (Muse names "drinking glass", "coffee cup" or "mug"; SAM prompt `cup` finds
   all three) are each classified by Muse Spark first ("glass", "cup" or "other", and plain or not); anything else or
   already branded is declined without tracking. Glasses become `kind: "glass_decal"` candidates: a white logo
   wrapped round the glass, placed from its own outline each frame (`scripts/glass_decal.py`), as soft and grainy
   as the glass measures. Cups and mugs become `kind: "cup_decal"` (`CupDecal`): the same wrap and tracking with the
   handle opened out of the mask, squat shapes allowed, and an opaque print, dark on a light cup and white on a dark
   one. The gates below apply to both. Their gates (`ingest.decide_glass`): on screen >= 2 s, longest run >= 1.5 s, glass >= 44 px
   wide, the glass's shape (a tapered glass measured where it is seen whole) fitted to its visible edges in >= 80% of
   frames, the logo's centre at least 0.3 radii inside the glass in >= 90% of whole-glass frames, no frames where the
   glass is seen without the logo, the glass stays still (it travels at most 0.5 radii relative to the room
   in any one second; glasses people pick up are declined for now), Muse says it is a plain unbranded glass. QA: none automatic for now (the blind
   judge is off; a person reviews the look). Flat ads: stability only.
7. **qa** - `eval/qa.py` (owner: GPT Sol) checks every rendered placement; a failing one is removed and re-exported.
8. **done** - `report.json` written; the upload page shows it (owner: Cursor).

## Run folder: `runs/<run_id>/`

| File | Written by | Content |
|---|---|---|
| `input.mp4` | upload | the normalised clip |
| `progress.jsonl` | every stage | one JSON object per line: `{"t": iso8601, "stage": str, "pct": 0-100, "msg": str}`; stages in order `upload, shots, segment, track, score, render, export, qa, done`, or `error` (then `msg` says why) |
| `candidates.json` | track | list of candidates (below) |
| `decisions.json` | score | `[{"id", "decision": "accept" or "decline", "score": 0-1, "reasons": [str]}]` |
| `qa.json` | qa | `[{"id", "pass": bool, "checks": {name: {"value", "limit", "pass"}}, "reasons": [str]}]` |
| `report.json` | done | what the upload page shows (below) |
| `thumbs/<id>.jpg` | track | a keyframe with the candidate outlined |

### Candidate

```json
{
  "id": "c03", "shot": 0, "prompt": "poster", "object": 2, "kind": "flat", "surface": "poster",
  "frames": [12, 118], "fps": 29.97, "frame_size": [1280, 720], "thumb": "thumbs/c03.jpg",
  "metrics": {
    "frames": 107, "visible_frames": 101, "on_screen_s": 3.37, "visible_fraction": 0.94,
    "mean_area_frac": 0.031, "aspect_ratio": 0.74, "edge_clipped_fraction": 0.0, "max_occlusion_frac": 0.22,
    "longest_run_s": 3.1,
    "track_residual_px_median": 1.4, "track_residual_px_p90": 2.9,
    "corner_jitter_px_median": 0.5, "corner_jitter_px_p95": 1.8,
    "motion_blur_px_median": 2.0, "min_side_px": 96, "mean_luma": 141, "luma_std_over_time": 6.0,
    "mean_occlusion_frac": 0.08, "dynamic_content": false,
    "muse_plausible": true, "muse_reason": "printed poster on a cafe wall", "scene_safe": true
  }
}
```

The first nine metric keys are `pipeline/discover.slot_metrics`; the rest are added by the track stage:
`track_residual_px_*` = surface-mask boundary vs tracked quad; `corner_jitter_px_*` = |second difference| of
corner positions; `motion_blur_px_median` = median corner speed x shutter (0.5 frame); `min_side_px` = shortest
quad edge, median over visible frames; `luma_std_over_time` = std of the surface's mean luma across frames.

### report.json

```json
{
  "run_id": "r_20261030_1412_ab12", "clip_stem": "upload_ab12", "source_name": "cafe.mp4",
  "duration_s": 12.4, "status": "done", "elapsed_s": 412,
  "summary": {"candidates": 9, "placed": 3, "declined": 6},
  "placements": [
    {"id": "c03", "surface": "poster", "decision": "placed", "thumb": "/runs/r_.../thumbs/c03.jpg",
     "on_screen_s": 3.37, "reasons": [], "qa": {"pass": true, "reasons": []}},
    {"id": "c05", "surface": "sign", "decision": "declined", "thumb": "/runs/r_.../thumbs/c05.jpg",
     "on_screen_s": 0.9, "reasons": ["on screen 0.9 s, needs 2.0 s"], "qa": null}
  ],
  "player_url": "/player?clip=upload_ab12",
  "viewers": [{"profile_id": "viewer-a", "label": "Morning commuter"}]
}
```

## API (app/main.py, owner: Claude)

| Method | Path | Returns |
|---|---|---|
| POST | `/api/ingest` (multipart `file`) | `{"run_id"}`; processing continues in the background |
| GET | `/api/runs/{run_id}/progress` | `{"events": [last 50 progress lines], "latest": {...}, "done": bool, "error": str or null}` |
| GET | `/api/runs/{run_id}/report` | `report.json` (404 until done) |
| GET | `/runs/{run_id}/...` | static files of the run folder (thumbs) |
| GET | `/upload` | the upload page (owner: Cursor) |
| POST | `/api/viewer_render` | existing: `{"clip": clip_stem, "profile_id"}` -> watermarked file for one viewer |

The live player is the existing `/player?clip=<clip_stem>`.
