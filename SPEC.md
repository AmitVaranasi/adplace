# Spec FINAL (converged round 3: Claude, GPT-6.1 Sol, GLM-5.3): AI virtual product placement (working name "AdPlace")

Solo, Meta AI Developer Hackathon, Oct 27 – Nov 4 2026. Judging: impact 30, Meta tooling 25, execution 20, demo 25.
18 GB M-series Mac, no big GPU. Meta tools: SAM 3/3.1, Muse (Meta Model API, access unconfirmed), Video Seal.

## S1 Story
"Same scene, different brand for each viewer, always disclosed."
Personalisation + disclosure is the headline; automatic slot discovery is the mechanism.
Disclosure has two layers: a visible "AI product placement" label in the player and the exported video (for people), and a Video Seal watermark carrying a campaign token (for machines). The watermark is described as machine-readable identification, not as viewer disclosure.

## S2 Scope
In: planar surfaces only (poster, wall sign, framed picture, TV/laptop/phone screen; billboard optional). Offline render, never claimed as real-time.
Out: mugs/3D objects, curved screens, strong perspective change, shots that cross a scene cut.
Unsupported shots are rejected explicitly, with a reason. "No suitable placement" is a valid result for a scene or a viewer.

## S3 Tracking
1. Pick a reference frame where the surface is fully visible.
2. From the SAM 3 mask, extract the boundary lines and intersect them into one canonical convex quad; validate area, aspect ratio and corner order. If extraction fails, allow a disclosed 4-click correction.
3. Propagate the canonical quad frame to frame with RANSAC homography from feature points on the same plane (the surface and coplanar wall around it; exclude background that causes parallax, e.g. behind a laptop). Feature points on foreground occluders are excluded. SAM's mask is a cross-check evaluated only on the unoccluded, in-frame part of the surface, so expected occlusion and frame clipping never count as tracking failure; the quad is never refit to the visible part of the mask.
4. Conservative smoothing (no visible lag). Occlusion and frame-edge clipping are handled only at compositing, via SAM masks of occluders ("person", "hand").
5. If confidence collapses (homography inliers, mask/quad disagreement), drop that interval or shot rather than show a bad frame.
6. Verify SAM object identity stays consistent across the 4-frame chunks.

## S4 Architecture: analyse once, render per campaign
The analysis pass (SAM, quads, occluder masks, illumination maps) runs once per clip and is cached; cache keys include source hash, resolution, timebase, model and settings.
Rendering runs once per unique (slot, creative) combination, and outputs are reused across all viewers mapped to that campaign. Per-frame compositing cost is measured and reported honestly, separately from segmentation, watermarking and encoding.

## S5 Realism
Preserve the artwork's own colours and contrast. Apply normalised illumination estimated from a blank reference of the surface (shading falloff only), then match blur and grain to the surround, tuned visually. Screens are emissive: no shading; reflections only if a clean reference exists, never from arbitrary original screen content.

## S6 Muse
Deterministic code computes area, on-screen duration, visibility and occlusion for each slot. Before any model is asked, code filters slot–creative pairs by aspect ratio, eligible surfaces and contextual exclusions. Muse ranks the eligible slots, picks a brand per viewer profile from the inventory, and writes a one-line rationale. The rationale explains a choice; it is not presented as brand-safety proof. Returned IDs are validated against the eligible set ("no suitable placement" allowed); the same filter and validation apply to every fallback model. Fallback if Muse is unavailable: Llama (still Meta), then Claude.
Muse access and its real input capabilities (image input?) are tested on Oct 27.

## S7 Watermark
Video Seal embeds a campaign token; watermarking happens before the final encode, and the exact submitted export is verified. The verifier reports clip/segment-level detection and the decoded token, with an unwatermarked negative control. No promise of spatial localisation or of robustness beyond the re-encode tested. Round-trip tested on the Mac on Oct 27.

## S8 Acceptance
- Geometry: annotated corner error on ~10 hand-labelled frames per clip, target ≤ 3 px median at 720p; no visible slide at 1x playback (reviewer inspection).
- Realism: blinded artefact rating by 5+ viewers who rate each clip on "How visually natural is this placement? 1 = obvious artefacts, 5 = no visible artefacts"; exploratory, target mean ≥ 4. Plus frame-by-frame playback inspection by the developer.
- Watermark: token recovered from the exact submitted export after one H.264 re-encode at CRF 23; negative control not detected.
- Demo app: full flow runs end to end on the hero clip from a cold start using cached analysis.

## S9 Content
Primary: self-filmed 4–5 s clips at 720p/24 fps: a poster on a wall (gentle pan), a TV or laptop screen (gentle pan), one take with a person walking in front. For every surface, also film a blank/off reference from the same setup (plain paper where the poster goes, TV/laptop off): the illumination estimate in S5 needs it and it cannot be recreated later. Street billboard optional. Licensed footage (e.g. Blender's Tears of Steel) is backup only, after checking licence and attribution.
Brands are 3 fictional brands with artwork made per surface aspect ratio. No real trademarks.

## S10 Inventory and profiles
Inventory fields: brand id, campaign token, category, creative path, aspect ratio, eligible surfaces, contextual exclusions, preference tags. 3 synthetic viewer profiles with category preferences only; no personal data collected.

## S11 Demo app
Local web app: FastAPI + plain HTML/JS, OpenCV/PyTorch workers, ffmpeg. Flow: upload or pick a clip → SAM candidates and ranked slots → choose slot → pick viewer profile → rendered variant with label → verify an exported file. Cached analysis is labelled as cached; real processing times are shown. A CLI uses the same pipeline as a fallback.

## S12 Repo
app/ (API, UI, jobs) · pipeline/ (discover, track, occlude, composite, match, watermark, export) · configs/ (inventory, profiles, thresholds) · assets/ (clips, creatives, attribution) · cache/ · outputs/ · scripts/ (analyse, render, verify, smoke test). The chunked SAM 3 tracker is ported from video-to-physics.

## S13 Plan
- Oct 27: Muse access check; Video Seal round-trip on the Mac; film the poster clip; SAM 3 track; canonical quad + homography; paste a flat logo; export and play. Go/no-go tonight: the flat logo shows no visible slide at 1x on the poster clip after smoothing. If not met, the morning of Oct 28 switches the hero clip to the S15 fallback instead of open-ended debugging.
- Oct 28: realism (illumination, blur, grain); screen clip.
- Oct 29: occlusion and smoothing; hero clip frozen.
- Oct 30: slot discovery + Muse ranking.
- Oct 31: profiles, brand choice, per-campaign rendering, visible label.
- Nov 1: watermark verifier, web app; pipeline frozen.
- Nov 2: end-to-end on all clips, artefact-rating test, fixes.
- Nov 3: record and edit the 3-minute video.
- Nov 4: submission only.

## S14 Demo (3 min)
0:00–0:20 same scene, three fictional brands, visible label. 0:20–0:50 upload → SAM candidates → ranked slots → selected plane (cached marked). 0:50–1:20 profiles → brand choices + rationales → variants. 1:20–1:50 full-speed hero playback with camera motion and a person passing in front, then a brief tracking-overlay view. 1:50–2:20 verify a re-encoded export: token recovered; unwatermarked file negative. 2:20–2:45 second surface + measured timings. 2:45–3:00 reusable geometry, creator-controlled placements, Meta tooling, honest limits. No unsourced market figures.

## S15 Top risk and fallback
Sliding geometry under motion/occlusion. Fallback: locked-off poster + gently panning screen with precomputed variants; keep personalisation, visible label and verified watermark; cut automatic discovery before sacrificing visual quality.
