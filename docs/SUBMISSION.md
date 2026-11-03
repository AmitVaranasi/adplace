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

A creator uploads a video once. Muse Spark reads sampled frames of its longest shot and proposes the places an ad
could go (a poster, a picture frame, a menu, a glass on the counter, a bowl on the table), whether the scene is safe for ads, and what it's about. SAM 3.1
outlines each place through the whole shot, along with the people in front of it and the handles of cups. AdPlace
tracks each candidate against the room and measures it: how long it's on screen, how steady it is, how much people cover
it, whether someone picks it up. Only places that stay still and in view carry ads; every other one is declined with a
plain reason ("Someone picks it up or moves it", "No blank patch on the mug's own design").

Placement is off until the creator turns it on. They approve each place and each brand from their library, and every
auction respects that. When a viewer's video loads, a second-price auction picks a brand for each place in about a
tenth of a millisecond on the server, and the browser composites the winner into the footage live: in the scene's perspective,
light, focus and grain, printed on glasses (and cups, on test clips), behind the hands and people in front. Two phones side by side
show the same video to two viewers, each with the sponsors their own auction picked.

Every frame with an ad carries a visible "AI product placement" label, and the file a viewer downloads carries an
invisible Meta Video Seal watermark carrying an id this server resolves to the campaigns: found after re-encoding to 854 px wide at CRF 28 or 640 px at CRF 23, lost below that [a bar download, re-encoded with libx264]. A slight crop can survive (a 90% crop was still found); a 9:16 re-crop of a landscape download, as a repost to a
vertical feed would do, is not.

## How we used Meta's AI tools

- SAM 3.1 (Meta Model API): text-prompted video segmentation of each candidate place Muse proposes, of cups, glasses and bowls,
  of cup handles (so a logo never lands on one) and of the people in front of a placement.
- Muse Spark 1.3 (Meta Model API): reads each scene (an open-ended list of places and their kinds, brand safety,
  context tags); classifies each vessel (glass, cup, bowl; plain or branded); judges whether each place is a natural
  spot for an ad.
- Muse Image 1.0 (Meta Model API): the fictional brands' artwork.
- Video Seal 1.0 (Meta open weights): the invisible watermark in every downloaded copy.

The upload and playback path uses only these. (The early hand-made clips could also rank brands with Llama 3.3
70B, another Meta model, when a W&B key is set; uploads don't use it.)

## How we built it

Python (FastAPI, OpenCV, NumPy) for the upload pipeline and server; WebGL 2 for the live player, which draws the same
result as the server's renderer (within about 1.5 grey levels on the logo, a consistency check, not a realism score
[7 frames of the bar clip]) at the video's frame rate [23–24 fps on 24 fps clips, M3 Pro, Chrome]. Whether a
placement looks real was judged by eye. Everything that depends on the video (geometry, light, mattes, the
glass's own shading) is worked out once at upload and is the same for every brand, so playback only has to pick
names. 280 test functions (about 500 cases with their parameters), passing on a fresh clone.

## Challenges

- Logos sliding on moving glasses. A logo re-fitted to a glass every frame drifted visibly while someone drank from
  it. Rigid shape fitting and pinning a resting glass to the camera's motion stopped the slide on still glasses, but
  a glass swung to the mouth still slid, so objects that move are declined for now.
- Knowing what not to place on. Most real footage has nothing suitable, and a bad placement is worse than none. The
  gates decline rather than guess, and say why.
- Cups with designs of their own: the logo goes on a blank patch, measured on the unwrapped cup with the handle cut
  out, never over the cup's own drawing.

## How it would make money (hypothetical)

Each placement is sold per viewer, at the second price the auction clears: per 1,000 views, a video earns the sum of
its placements' clearing CPMs, shared between the creator and the platform. With the demo inventory's test bids the
auction clears at a median of $4.00 CPM per placement ($2.02–$8.70) [`pipeline/auction.py`, 20 bids per viewer on
the bar and m44007 players], so a video with one or two places would earn about $4–8 per 1,000 views on top
of whatever else it earns, without the creator making a sponsored video. These are test bids, not market prices.
Against that, yield is low today: 1 of the 23 unseen videos has a place that passes on today's code (3 did
before the Nov 3 gates), all in cafés and restaurants, so the first market is footage of tables, counters and walls: food, drink and lifestyle creators.

## Honest limits

- Yield is low. On 23 stock videos it had never seen, it placed 3 ads from 56 candidates, and every decline was
  checked by eye [`runs/testset_all23.txt`]. Re-run on today's code, which adds readability gates (a logo at least
  28 px tall on screen, and sharp enough to read), two of those three are declined (too small; no clear space for
  the logo): 1 placed. With open-ended discovery on 4 clips Muse named
  43 places, 19 of them kinds we can measure; 6 were placed before the gates and 3 after (a bar's wall poster and
  two glasses) [`runs/r_eval_*`, `runs/r_v2_m44025`, `runs/r_v4_m43999`].
- Only still objects carry ads. Objects people pick up are declined.
- Processing takes 10 to 15 minutes for a 10-second upload with places to measure, once, offline (under a minute when
  nothing in it can carry an ad) [10.2 s bar clip: 12.1 min alone, 15.7 min with two other uploads running; 5.7 s
  clip: 8.2 min; 5 s clip with nothing to place: 40–57 s]. Picking brands at playback takes about 0.1 ms on the server (5–15 ms round trip).
- The bids are test bids from a demo inventory of fictional brands. It's a standalone prototype, not connected to a
  real ad auction, with no accounts: it runs locally.
- The five placements on the patio clip were set up by hand (its rendered 3D can and bottle were dropped on Nov 3:
  they read as computer-made); uploads get flat surfaces and vessels only. Uploads use one shot (the longest); its sound plays muted until a viewer turns it on (the stock
  test footage has none).

## What's next

Ship it as an opt-in setting in a short-video feed: the creator's switch at upload, placements filled by the
platform's ad auction. On the technical side: objects that move, products inserted in 3D (tried on a hand-calibrated clip,
not yet convincing enough to ship), and faster processing.

## Built during the hackathon

First commit Oct 27, 2026; the README's "Built during the hackathon" table lists what was added each day.
