# Demo video (3 minutes)

Judging: impact 30, Meta tooling 25, execution 20, demo 25. Every number said aloud below was measured in this repo;
where it came from is in brackets. Raw screen footage for each part is in `outputs/demo/` (recorded by
`scripts/record_demo.py` from the running app; re-record with `python scripts/record_demo.py`).

## 0:00–0:20 Same scene, a different brand for each viewer

Screen: the live player on the bar clip (`/player?clip=upload_m43248`). Click Morning commuter, Book lover, Weekend
runner: the poster on the wall and the logo on the matcha glass change with each click. "AI product placement" label
bottom left.

Say: "This is one video. Every viewer sees different brands in it, placed after filming, always labelled. Nothing was
re-rendered when I clicked: the brands were chosen by an auction the moment each viewer started watching."

## 0:20–1:00 Upload: one pass, no hand work

Screen: `/upload`, drop `assets/clips/raw3/m43248.mp4`, the stages tick through (Muse Spark reads the scene, SAM 3.1
segments, tracking, deciding, building the live player, final checks). Cut the waiting; show the report.

Say: "A creator uploads once. Muse Spark reads the scene and checks it's safe for ads. SAM 3.1, through the Meta Model
API, outlines every surface and glass through the shot. We track each one against the room and keep only what stays
still: the poster, and the matcha glass on the counter. We decline the rest and say why, in plain words: the menu
board drifts, and her Fizzwell glass gets picked up, so it can't carry an ad yet."

Point at: the Fizzwell card ("Someone picks it up or moves it…"), the menu board card.

## 1:00–1:40 The bid at playback

Screen: back in the player. The panel "This viewer's ads": winner, clearing price, runner-up per placement; "decided in
0.07 ms". Click "Run the bid again" a couple of times; prices move. Toggle Original / With ads. Play at full speed.

Say: "Everything that depends on the video, the geometry, light, focus, grain and what's in front, was worked out at
upload and doesn't depend on the brand. So the auction only picks names: a second-price auction weighted by how well
each brand fits the viewer and the scene, in under a tenth of a millisecond. The browser then lays the winning artwork
on live, wrapped round the glass, as soft and grainy as the footage, behind her fingers."

## 1:40–2:10 How far it goes: the patio

Screen: `/player?clip=patio_waiter`, switch viewers. Nine placements: wall poster, menu print, a 3D can, bottle,
table-tent card, a logo on a real glass, a print on the waiter's T-shirt, an A-frame sign.

Say: "This is how far the placements go on a clip we set up by hand: 3D products ray-traced onto the table, lit like
the footage, and people walking in front of them."

## 2:10–2:35 Always disclosed, always detectable

Screen: "Download my version" in the player; the file comes back with "watermark verified" and a manifest id. Optional:
`/` → verify an exported file, token recovered; an unwatermarked file is not detected.

Say: "The file you take away carries a visible label and an invisible Meta Video Seal watermark naming the campaign, so a
placement can always be found, even after re-encoding."

## 2:35–3:00 Honest limits and what's next

Say, with the numbers from the test set (below):

- "On 23 videos it had never seen, it placed three ads and declined 53 candidates, explaining why each time. Most
  declines are scenes with nothing still to print on."
- "A 10-second upload takes a few minutes, once. Picking brands at playback takes a tenth of a millisecond."
- "Meta tooling: SAM 3.1 and Muse Spark through the Meta Model API, Muse Image for the brand artwork, Video Seal for
  the watermark."
- "Next: objects people pick up, and 3D products on any upload."

## Test-set figures

23 stock videos never used in development, final code (`runs/testset_all23.txt`): 56 candidates found, 3 placed,
all on still glasses on restaurant tables (m44025: kept on while a shoulder passes in front; m43999: moved onto the
part a bowl doesn't hide; m44007: an orange juice). Declines, all checked by eye: nothing to place on (8 videos), a
fast blurred bar sweep, a night terrace's signs (one a parking sign), branded glasses, sauce ramekins, glasses and
teacups picked up or carried in. Say: "On 23 videos it had never seen, it placed three ads and declined 53
candidates, each with a reason we checked by hand. It would rather place nothing than place badly."

Optional extra shot: `/player?clip=upload_pair` (two mugs with drawings on them; each logo goes on a blank patch,
dark ink on the light mug, white on the black one).
