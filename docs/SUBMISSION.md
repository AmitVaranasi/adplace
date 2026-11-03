# AdPlace: submission form answers

Paste-ready answers for the Meta AI Developer Hackathon form (cerebralvalley.ai, due Wed Nov 4, 9:00 PM ET). Every
figure was measured in this repo; the source is in brackets and is not part of the answer.

**Track:** Indie developer (solo).
**Repo:** https://github.com/AmitVaranasi/adplace
**Demo video:** (link once uploaded)

## Project name

AdPlace

## Tagline (one line)

An opt-in ad format for short videos: creators approve placements in videos they've already filmed, and each viewer
sees different labelled sponsors in the scene itself, with nothing re-rendered to switch brands.

## What it does

A creator uploads a video once. Muse Spark reads the scene: every place an ad could go (a poster, a picture frame, a
menu, a glass on the counter, a bowl on the table), whether the scene is safe for ads, and what it's about. SAM 3.1
outlines each place through the whole shot, along with the people in front of it and the handles of cups. AdPlace
tracks every place against the room and measures it: how long it's on screen, how steady it is, how much people cover
it, whether someone picks it up. Only places that stay still and in view carry ads; every other one is declined with a
plain reason ("Someone picks it up or moves it", "No blank patch on the mug's own design").

Placement is off until the creator turns it on. They approve each place and each brand from their library, and every
auction respects that. When a viewer presses play, a second-price auction picks a brand for each place in about a
tenth of a millisecond, and the browser composites the winner into the footage live: in the scene's perspective,
light, focus and grain, wrapped round glasses and mugs, behind the hands and people in front. Two phones side by side
show the same video to two viewers with two different sponsors.

Every frame with an ad carries a visible "AI product placement" label, and the file a viewer downloads carries an
invisible Meta Video Seal watermark naming the campaign, still found after re-encoding.

## How we used Meta's AI tools

- SAM 3.1 (Meta Model API): text-prompted video segmentation of every place Muse lists, of cups, glasses and bowls,
  of cup handles (so a logo never lands on one) and of the people in front of a placement.
- Muse Spark 1.3 (Meta Model API): reads each scene (an open-ended list of places and their kinds, brand safety,
  context tags); classifies each vessel (glass, cup, bowl; plain or branded); judges whether each place is a natural
  spot for an ad.
- Muse Image 1.0 (Meta Model API): the fictional brands' artwork.
- Video Seal 1.0 (Meta open weights): the invisible watermark in every downloaded copy.

No other models run in the product.

## How we built it

Python (FastAPI, OpenCV, NumPy) for the upload pipeline and server; WebGL 2 for the live player, which draws the same
result as the server's renderer (within about 1.5 grey levels on the logo, a consistency check, not a realism score
[7 frames of the bar clip]) at the video's frame rate [23–24 fps on 24 fps clips, M3 Pro, Chrome]. Whether a
placement looks real was judged by eye. Everything that depends on the video (geometry, light, mattes, the
glass's own shading) is worked out once at upload and is the same for every brand, so playback only has to pick
names. About 500 automated tests, passing on a fresh clone.

## Challenges

- Logos sliding on moving glasses. A logo re-fitted to a glass every frame drifted visibly while someone drank from
  it. Rigid shape fitting and pinning a resting glass to the camera's motion stopped the slide on still glasses, but
  a glass swung to the mouth still slid, so objects that move are declined for now.
- Knowing what not to place on. Most real footage has nothing suitable, and a bad placement is worse than none. The
  gates decline rather than guess, and say why.
- Cups with designs of their own: the logo goes on a blank patch, measured on the unwrapped cup with the handle cut
  out, never over the cup's own drawing.

## Honest limits

- Yield is low. On 23 stock videos it had never seen, it placed 3 ads from 56 candidates, and every decline was
  checked by eye [`runs/testset_all23.txt`]. With open-ended discovery on 4 clips Muse named 43 places, 19 of them kinds we can measure, and 6 were placed
  [`runs/r_eval_*`]; the two sets ran on different code and aren't comparable.
- Only still objects carry ads. Objects people pick up are declined.
- Processing a 10-second upload takes a few minutes, once, offline [5.7 s clip: 8.6 min; 5 s clip with nothing to
  place: 57 s]. Picking brands at playback takes about 0.1 ms.
- The bids are test bids from a demo inventory of fictional brands. It's a standalone prototype, not connected to a
  real ad auction, with no accounts: it runs locally.
- The 3D products and the nine placements on the patio clip were set up by hand; uploads get flat surfaces and
  vessels only. Uploads use one shot (the longest) and play without sound.

## What's next

Ship it as an opt-in setting in a short-video feed: the creator's switch at upload, placements filled by the
platform's ad auction. On the technical side: objects that move, products inserted in 3D on any upload (working now
on a hand-calibrated clip), and faster processing.

## Built during the hackathon

First commit Oct 27, 2026; the README's "Built during the hackathon" table lists what was added each day.
