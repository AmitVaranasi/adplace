# AdPlace: submission write-up

Draft for the Meta AI Developer Hackathon form. Every figure was measured in this repo; the source is in brackets.

## One line

Upload a video once; every viewer then sees brands placed into the scene itself (on the poster on the wall, printed on
the glass on the counter), picked by an auction when they press play, always labelled and watermarked.

## What it does

- **Upload once, no hand work.** Muse Spark reads the scene and checks it's safe for ads. SAM 3.1 outlines every
  poster, picture frame, sign, menu board, screen, glass, cup and mug through the shot. Each is tracked against the
  room and measured: how long it's visible, how steady its outline is, how much people cover it, whether it moves.
  Only surfaces that pass strict gates carry ads, and only objects that stay still. Everything else is declined with a
  plain reason ("Someone picks it up or moves it", "It's mostly cut off by the edge of the frame").
- **Placements that look filmed.** Flat ads take the surface's perspective, light, focus, motion blur and grain; people
  walking in front stay in front. Logos on glasses, cups and mugs wrap round the curve, take the vessel's light and
  highlights, sit behind the fingers holding it, and are white enamel on glass, dark ink on a light mug.
- **The bid at playback.** Everything that depends on the video is worked out at upload and is the same for every
  brand. When a viewer presses play, a second-price auction, weighted by how well each brand fits the viewer and the
  scene, picks a brand per placement in under a tenth of a millisecond [`/api/auction`, 0.06–0.09 ms]. The browser
  then composites the winner live in WebGL at the video's frame rate [23 fps on a 24 fps clip, M3 Pro]; nothing is
  rendered per viewer.
- **Always disclosed, always detectable.** Every frame carries a visible "AI product placement" label. The file a viewer
  downloads carries an invisible Meta Video Seal watermark naming the campaign, verified after re-encoding.

## How we used Meta's tools

- **SAM 3.1** (Meta Model API): text-prompted video segmentation of every candidate surface, glass, cup and the people
  in front of them, across the whole shot.
- **Muse Spark 1.3** (Meta Model API): reads each scene (surfaces present, brand safety, context tags), judges each
  candidate (is this a plain glass, cup or mug, or a natural place for an ad), and picks brands per viewer with a
  stated rationale in the hand-made demo clips.
- **Muse Image 1.0**: the fictional brands' artwork (posters, prints, labels).
- **Video Seal**: the invisible watermark carrying the campaign token.

## How it's built

Python (OpenCV, NumPy, FastAPI) for the pipeline and server; WebGL 2 for the live player. The player reproduces the
server's compositing: on the glass logo, within about 1.5 grey levels of the server render on average [checked on 7
frames of the bar clip]. No local models: all segmentation and vision go through the Meta Model API.

## Challenges

- **Logos sliding on moving glasses.** A logo re-fitted to a glass every frame drifted visibly while someone drank from
  it. Rigid shape fitting, camera-motion pinning and filtering got a still glass to move with the camera to within
  0.000 px median per frame, but a glass swung to the mouth still showed a slide. So objects that move are declined for
  now [Fizzwell glass on the bar clip: travels 4.8 radii within a second, limit 0.5].
- **Knowing what not to place on.** Most real footage has nothing suitable. The gates decline rather than place badly,
  and the report says why.

## Honest limits

- On 13 videos it had never seen [`runs/testset2_summary.txt`]: RESULTS. Most had no poster, sign, screen or still
  vessel big and steady enough; a fast blurred sweep through a bar and a parking sign (correctly) got nothing.
- Objects that people pick up don't carry ads yet.
- A 10-second upload takes about 13 minutes of processing, once [bar clip: 774 s].
- 3D products (a can placed on a table) work on hand-calibrated demo clips, not yet on uploads.

## What's next

Objects that move (held glasses, carried cups), 3D products on uploads, and faster upload processing.
