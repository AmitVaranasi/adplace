# Demo video (3 minutes)

Judging: impact 30, Meta tooling 25, execution 20, demo 25. Every number said aloud below was measured in this repo;
where it came from is in brackets. Raw screen footage for each part is in `outputs/demo/` (recorded by
`scripts/record_demo.py` from the running app; re-record with `python scripts/record_demo.py`).

## Recording plan

Raw footage: `python scripts/record_demo.py` (app running) writes one silent 1440x900 file per segment to
`outputs/demo/`. Narrate over them in any editor (iMovie, CapCut, DaVinci), or use them as the reference for a live
screen recording of the same steps. Voice-over below: about 420 words in eight chapters, 2:40 with the Ava voice. Record it in one take per
segment; trim each clip to its line.

| Chapter (on-screen caption) | Raw clip | Voice-over |
|---|---|---|
| Title card | `phones.mp4` (still) | (silent, 3.5 s): "AdPlace · Sponsors inside videos creators already made, with their consent" |
| The idea | `phones.mp4` | "Short-video creators earn from brands only when they film a sponsored post. AdPlace is a prototype of a different ad format: the creator opts in, and a sponsor appears inside videos they've already made. Here, one video plays to two viewers. The commuter sees Brewlark on the glass; the runner sees Fizzwell. Same footage, nothing re-rendered, and every ad is labelled." |
| 1  Upload | `report.mp4` | "It starts when a creator uploads. Meta's Muse Spark reads the scene, proposes places an ad could go, and checks that the scene is safe for brands. SAM 3.1 outlines each place through the whole shot, along with the people in front of it. We track every place against the room and measure it: how long it's on screen, how steady it is, how often it's covered. Only still, visible places pass. Everything else is declined, with a plain reason, like the glass she picks up." |
| 2  The creator decides | `optin.mp4` | "Nothing runs until the creator says yes. Placement starts off. They turn it on, approve each place, and choose the brands they're happy to carry. Untick a brand, and that viewer's glass re-bids to another sponsor straight away." |
| 3  Each viewer, their own sponsor | `auction.mp4` | "When a viewer's video loads, a second-price auction picks a brand for each place, matched to their interests, in under a millisecond. The heavy work was done once, at upload, so playback only picks names, and the browser draws the winner in live." |
| 4  It should look filmed | `player.mp4` | "The logo takes the glass's shape, focus and grain, and the writing hand stays in front of it. Switch to the original to compare. As a final check, Muse Spark reads the brand back off the rendered video. If it can't read it, the placement is dropped. The poster artwork is made with Muse Image." |
| 5  Disclosed and traceable | `verify.mp4` | "Every ad carries a visible label. And the copy a viewer keeps carries an invisible Meta Video Seal watermark. Drop it on Check a file, and it names the campaign in each place. It survives re-compression; a heavy crop removes it." |
| Where it stands | `funnel.mp4` | "It's deliberately strict. On twenty-three stock videos it had never seen, it measured sixty-five places and kept one. It would rather place nothing than place badly, and today it works best in cafés and restaurants, on still glasses and posters." |
| Next | `close.mp4` | "Next: objects people pick up, more kinds of surfaces, and an opt-in setting inside a short-video feed. AdPlace: a creator's own videos, sponsored with their consent." |
| End card | `close.mp4` (still) | "1 of 23 unseen videos placed · 10-15 min per clip, once · test bids · Meta SAM 3.1 · Muse Spark · Muse Image · Video Seal" |

The draft: `python scripts/assemble_demo.py --out adplace_demo_ava.mp4` (Ava, a Microsoft neural voice via edge-tts; about 2:40). The lines live in `SEGMENTS` there; edit them and re-run to change the video.

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
