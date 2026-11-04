# Demo video (3 minutes)

Judging: impact 30, Meta tooling 25, execution 20, demo 25. Every number said aloud below was measured in this repo;
where it came from is in brackets. Raw screen footage for each part is in `outputs/demo/` (recorded by
`scripts/record_demo.py` from the running app; re-record with `python scripts/record_demo.py`).

## Recording plan

Raw footage: `python scripts/record_demo.py` (app running) writes one silent 1440x900 file per segment to
`outputs/demo/`. Narrate over them in any editor (iMovie, CapCut, DaVinci), or use them as the reference for a live
screen recording of the same steps. Voice-over below: about 320 words, 3:00 at a calm pace (judges found the earlier script too dense). Record it in one take per
segment; trim each clip to its line.

| Time | File | On screen | Voice-over |
|---|---|---|---|
| 0:00-0:20 | `phones.mp4` | Two phones, the iced coffee: Brewlark on the left, Fizzwell on the right | "This is AdPlace, an opt-in ad format for short videos. One video, two viewers, and each sees a different sponsor printed on the same glass, picked by an auction when their video loads. Nothing was re-rendered to switch brands, and every ad is labelled as AI product placement." |
| 0:20-0:50 | `optin.mp4` | My videos; the Test set funnel; the switch off, then on; Brewlark unticked | "Creators decide. Every upload starts with placement off. Here the creator turns it on, approves each place, and turns a brand down: the commuter's glass re-bids to another sponsor straight away. And it's strict: on twenty-three stock videos it had never seen, it measured sixty-five places and kept one, with a reason for every no." |
| 0:50-1:30 | `report.mp4` | The bar clip's report: the poster and glass placed, the declines | "When a creator uploads, Muse Spark reads the scene and proposes places an ad could go. SAM 3.1 outlines each one through the shot. We track it against the room and keep only what stays still and in view. The glass she picks up is declined: someone moves it. Muse also turned down a second poster as an unnatural place for an ad." |
| 1:30-2:05 | `player.mp4` | The iced coffee with the hand writing; Original / With ads; the bar clip | "The logo takes the glass's shape, focus and grain, and hands stay in front of it. As a final check, Muse Spark reads the brand back off the rendered video; if it can't read it, the placement is dropped. The poster artwork is Muse Image." |
| 2:05-2:30 | `verify.mp4` | Download (watermark verified); Check a file names the campaign | "The copy a viewer keeps carries an invisible Meta Video Seal watermark. Drop it on Check a file and it names the campaign in each place. It survives re-compression; a heavy crop removes it." |
| 2:30-2:45 | `auction.mp4` | Both bids again: prices, runner-up, time | "Because the heavy work happens once, at upload, the auction at playback only picks names: under a millisecond on the server. These are test bids from a demo inventory." |
| 2:45-3:00 | `close.mp4` + text card | Card: "1 of 23 unseen videos placed · 10-15 min per clip, once · test bids · Meta SAM 3.1, Muse Spark, Muse Image, Video Seal" | "Next: objects people pick up, more kinds of surface, and an opt-in setting inside a short-video feed." |

Before recording live (if not using the raw files):
- App running on the same machine; open each URL once first so the video is cached.
- Browser window 1440x900, zoom 100%, no other tabs or bookmarks bar; dark mode.
- Use the `&steady=1` URLs so the brands match the voice-over on every take.
- Placement on for `upload_m1737` and `upload_m43248` (My videos switch), except the moment you turn it on.
- Downloads are pre-rendered for each viewer (`outputs/viewers/`), so Download returns at once.
- Don't upload an unseen clip on camera, don't show the patio clip as pipeline output (it was set up by hand), and
  don't say it survives a repost to a vertical feed (a 9:16 crop removes the watermark).

## 0:00–0:20 Two phones, one video, different sponsors

Screen: `/reels?clip=upload_m1737&steady=1` (bids without jitter, so the brands match this script on every take),
placement already on, both phones playing in sync: an iced coffee on a tray while someone writes. Left: Morning
commuter, Brewlark on the glass ("Matches: coffee, morning"); right: Weekend runner, Fizzwell on the same glass.
This placement came out of the upload pipeline untouched (Mixkit 1737, `runs/r_v5_m1737`); Muse Spark read the brand
off the rendered video in every sampled frame and rated it 4 of 5 for looking printed. Then switch to
`/reels?clip=upload_m43248&steady=1` for a scene with two places (a wall poster and a glass).

Say: "This is a prototype of an opt-in ad format for short videos. One video, two viewers, each with sponsors picked for them in
the scene itself, labelled as AI product placement. Switching brands re-rendered nothing for either of them."

## 0:20–0:50 The creator opts in

Screen: `/` (My videos): the creator's library, each video with its places for ads and its placement switch. Click
"Upload a video", drop `assets/clips/raw3/m43248.mp4`; back in the library it shows as processing; skip past the wait
("10 to 15 minutes, once") and open the report of the same clip processed earlier, saying so: "this is that clip,
processed earlier". Don't upload an unseen clip on camera: most find nothing to place on. Open "Settings & preview" on `/reels`: start with the switch off (both phones show the
original, "Product placement is off for this video"), turn on "Allow product placement"; both phones' ads
appear. Untick Brewlark: the commuter's glass re-bids to another brand.

Say: "Placement is off until the creator turns it on. They approve each place we found and each brand they're happy
to carry, and every viewer's auction respects that."

## 0:50–1:30 What Meta's models found, and what they turned down

Screen: the upload report (`/upload?run=r_eval_m43248`) or the "Places we didn't use" list on `/reels`. Show the poster
and the matcha glass placed; read two declines aloud (the glass she picks up: "someone picks it up or moves it"; the
second poster: "Muse Spark judged it not a natural place for an ad").

Say: "Muse Spark reads the scene, proposes the places an ad could go and checks it's safe for ads. SAM 3.1 outlines each
one through the shot. We track them against the room and keep only what stays still and in view, and we say in plain
words why we turned the rest down."

## 1:30–2:15 Why it looks filmed, and how it stays traceable

Screen: full player `/player?clip=upload_m43248`, play at full speed; pause on the hand passing the glass, the logo
on the matcha glass with her hand in front of it, the poster's perspective and blur. Toggle Original / With ads. Then "Download my
version" for a viewer whose copy was rendered before recording (a new one takes one to three minutes):
"watermark verified"; drop that file on Check a file (`/verify`) to show the campaigns it names.

Say: "The ad takes the scene's perspective, light, focus and grain, and hands stay in front of it. Every ad carries a
visible label, and the file you take away carries an invisible Meta Video Seal watermark carrying a campaign id, still
there after the file is compressed again, though not after it's cropped."

## 2:15–2:40 The bid at playback

Screen: back on `/reels`, "Run both bids again": prices, runner-up and "decided in" well under a millisecond under each phone.

Say: "Everything that depends on the video was worked out once, at upload, and is the same for every brand. So when
a viewer's video loads, a second-price auction only picks names, in under a millisecond on the server, and the browser
draws the winner in live. That's choosing the brand, not processing the video. These are test bids."

## 2:40–3:00 Honest numbers and what's next

Say (about 40 words; put the figures on screen as text instead of reading them):

"It's strict: on 23 videos it had never seen, it placed one ad and explained every decline. Next: objects people
pick up, more kinds of surface, and shipping it as an opt-in setting in a short-video feed."

On screen: "1 of 23 unseen videos placed · 3 of 19 measured places · 10–15 min per 10 s clip, once · test bids".

End on the two phones playing.

## Test-set figures

Open-ended discovery (Nov 2, 4 clips, `runs/r_eval_*`): Muse named 43 places; 24 were kinds we have no way to place on yet (plates, tables, shirts, walls) and were not measured; of the 19 measured, 6 were placed (m43999: a ramekin and a bread
bowl; m44025 and m44007: orange-juice glasses; m43248: the wall poster and the matcha glass). Since Nov 3 a logo must come out at least 28 px tall on screen and sharp enough to read: the m44025 glass (18 px) and m43999's ramekin and bread bowl (out of focus) are declined: 3 placed. Reported separately from
the 23-video set below, which ran on the code before discovery.

23 stock videos never used in development, final code (`runs/testset_all23.txt`): 56 candidates found, 3 placed,
all on still glasses on restaurant tables (m44025: kept on while a shoulder passes in front; m43999: moved onto the
part a bowl doesn't hide; m44007: an orange juice). Declines, all checked by eye: nothing to place on (8 videos), a
fast blurred bar sweep, a night terrace's signs (one a parking sign), branded glasses, sauce ramekins, glasses and
teacups picked up or carried in. With the readability gates added on Nov 3 (a logo at least 28 px tall on screen, and sharp enough to read), the
m44025 glass is declined (18 px), and on today's code m43999's glass is too (no clear space for the logo): 1 placed. Say: "On 23 videos it had never seen, it placed one ad and
declined the rest, each with a reason we checked by hand. It would rather place nothing than place badly."
