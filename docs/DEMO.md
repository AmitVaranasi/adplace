# Demo video (3 minutes)

Judging: impact 30, Meta tooling 25, execution 20, demo 25. Every number said aloud below was measured in this repo;
where it came from is in brackets. Raw screen footage for each part is in `outputs/demo/` (recorded by
`scripts/record_demo.py` from the running app; re-record with `python scripts/record_demo.py`).

## 0:00–0:20 Two phones, one video, different sponsors

Screen: `/reels?clip=upload_m43248`, placement already on, both phones playing in sync. Left: Morning commuter
(Brewlark on the wall poster), right: Weekend runner (Kestrel). Raw footage: `outputs/demo/reels.mp4`.

Say: "This is a prototype of an opt-in ad format for short videos. One video, two viewers, two different sponsors in
the scene itself, labelled as AI product placement. Switching brands re-rendered nothing for either of them."

## 0:20–0:50 The creator opts in

Screen: `/` (My videos): the creator's library, each video with its places for ads and its placement switch. Click
"Upload a video", drop `assets/clips/raw3/m43248.mp4`; back in the library it shows as processing; skip past the wait
("a few minutes, once"). Open "Settings & preview" on `/reels`: start with the switch off (both phones show the
original, "The creator hasn't turned on product placement"), turn on "Allow product placement"; both phones' ads
appear. Untick Brewlark: the commuter's poster re-bids to another brand.

Say: "Placement is off until the creator turns it on. They approve each place we found and each brand they're happy
to carry, and every viewer's auction respects that."

## 0:50–1:30 What Meta's models found, and what they turned down

Screen: the upload report (`/upload?run=r_eval_m43248`) or the "Places we didn't use" list on `/reels`. Show the poster
and the matcha glass placed; read two declines aloud (the Fizzwell glass she picks up; the dark wall gap her head
covers).

Say: "Muse Spark reads the scene, lists every place an ad could go and checks it's safe for ads. SAM 3.1 outlines each
one through the shot. We track them against the room and keep only what stays still and in view, and we say in plain
words why we turned the rest down."

## 1:30–2:15 Why it looks filmed, and why it's always findable

Screen: full player `/player?clip=upload_m43248`, play at full speed; pause on the hand passing the glass, the logo
wrapped round the matcha glass, the poster's perspective and grain. Toggle Original / With ads. Then "Download my
version": "watermark verified".

Say: "The ad takes the scene's perspective, light, focus and grain, and hands stay in front of it. Every ad carries a
visible label, and the file you take away carries an invisible Meta Video Seal watermark naming the campaign, still
there after re-encoding."

## 2:15–2:40 The bid at playback

Screen: back on `/reels`, "Run both bids again": prices, runner-up and "decided in 0.1 ms" under each phone.

Say: "Everything that depends on the video was worked out once, at upload, and is the same for every brand. So when
someone presses play, a second-price auction only picks names, in about a tenth of a millisecond, and the browser
draws the winner in live. That's choosing the brand, not processing the video. These are test bids."

## 2:40–3:00 Honest numbers and what's next

Say, with the figures below, the two test sets kept separate:

- "On 23 videos it had never seen, it placed 3 ads out of 56 candidates. With open-ended discovery on four clips, it
  measured 19 places and placed 6. It turns down what it can't place well, and says why."
- "Processing takes a few minutes per 10-second clip, once, offline. Only still objects carry ads so far, which limits
  how much inventory a video has today."
- "Next: objects people pick up, and shipping this as an opt-in setting in a short-video feed."

End on the two phones playing.

## Test-set figures

Open-ended discovery (Nov 2, 4 clips, `runs/r_eval_*`): Muse named 43 places; 24 were kinds we have no way to place on yet (plates, tables, shirts, walls) and were not measured; of the 19 measured, 6 were placed (m43999: a ramekin and a bread
bowl; m44025 and m44007: orange-juice glasses; m43248: the wall poster and the matcha glass). Reported separately from
the 23-video set below, which ran on the code before discovery.

23 stock videos never used in development, final code (`runs/testset_all23.txt`): 56 candidates found, 3 placed,
all on still glasses on restaurant tables (m44025: kept on while a shoulder passes in front; m43999: moved onto the
part a bowl doesn't hide; m44007: an orange juice). Declines, all checked by eye: nothing to place on (8 videos), a
fast blurred bar sweep, a night terrace's signs (one a parking sign), branded glasses, sauce ramekins, glasses and
teacups picked up or carried in. Say: "On 23 videos it had never seen, it placed three ads and declined 53
candidates, each with a reason we checked by hand. It would rather place nothing than place badly."
