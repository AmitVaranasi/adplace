"""Export a clip for live, per-viewer ad compositing in the browser (app/static/player.html).

  python scripts/export_player.py assets/clips/patio_waiter.mp4

All the expensive, brand-independent work is done once here; swapping which brand a viewer sees is then a texture
change in the player. Writes app/static/player/<stem>/:

  stack.mp4         top half: the original video; bottom half: the foreground matte (white = something in front of
                    the ad: people, hands, a tray, glass), so both always decode in step
  track.json        per slot and frame: corners (null where the surface is not usable), per-channel exposure gain
                    and veil, defocus blur; per brand: artwork file and its extra look-match blur
  art/<slot>_<brand>.png   artwork graded and look-matched to that surface (for ink slots: per-channel transmission)
  art/<glass>_glass.png    a still glass's light per frame, the same for every brand; the player wraps the brand's logo
                           (../logos/<brand>_glass.png) round it live, so a brand chosen at bid time needs no render
  assignments.json  per viewer profile: Muse Spark's slot -> brand choice and its reason

The matte reproduces render_dynamic.render's occlusion: SAM occluders plus keyed objects on blank surfaces, refined to
soft edges, and glass opacity for ink slots. Not reproduced live: shutter-integrated motion blur is approximated by the
player from neighbouring frames' corners; edge decontamination and reflections are server-render only. The
downloadable, watermarked file for a viewer still comes from render_dynamic (Video Seal needs the server).
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.render_dynamic import (Frames, Union, drop_on_screen, inset, keep_attached, occluders,  # noqa: E402
                                    prepare_layer, quad_mask, tighten_to_plate, union_store)

OUT = ROOT / "app" / "static" / "player"


def eligible_brands(slot, inventory):
    fmt = slot.get("creative_format")
    out = []
    for c in inventory["campaigns"]:
        if slot.get("surface", "poster") not in c["eligible_surfaces"]:
            continue
        if fmt and not (ROOT / "assets" / "creatives" / "gen" / f"{c['brand_id']}_{fmt}.png").exists():
            continue
        out.append(c["brand_id"])
    return out


def save_art(path, art, multiply, long_side=1024):
    """Artwork for the browser, at most `long_side` px (no surface in these clips is larger on screen)."""
    h, w = art.shape[:2]
    k = min(1.0, long_side / max(h, w))
    if k < 1:
        art = cv2.resize(art, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    if multiply:  # transmission in [0,1], BGR float -> 8-bit PNG (BGR is what cv2 writes as an RGB file)
        cv2.imwrite(str(path), np.clip(np.rint(art * 255), 0, 255).astype(np.uint8))
    else:  # opaque RGBA -> JPEG
        cv2.imwrite(str(path), cv2.cvtColor(art, cv2.COLOR_RGBA2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])


def matte_frame(f, i, occ, layers, glass, frames):
    """Foreground alpha in [0,1] over the whole frame, as render_dynamic composites it."""
    from pipeline.matte import refine_occluder
    h, w = f.shape[:2]
    m = occ.frame(i) if occ else np.zeros((h, w), bool)
    sam = m.copy()
    alpha = np.zeros((h, w), np.float32)
    plates = []
    for slot, tr, art, blur, grain, plate in layers:
        if not tr["valid"][i]:
            continue
        if slot.get("dynamic_content"):
            m = drop_on_screen(m, quad_mask(tr["quads"][i], h, w))
        if plate is not None:
            add, now = plate.layers(i)
            P, pc = plate.to_frame(now, i, f.shape)
            q = inset(tr["quads"][i], float(slot.get("inset_px", 2)))
            known = pc & quad_mask(q, h, w)
            plates.append((P, known))
            if slot.get("difference_key"):
                d = np.abs(f.astype(np.float32) - P).max(-1)
                key = (d > float(slot.get("key_threshold", 28))) & known
                key = cv2.morphologyEx(key.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                m |= keep_attached(cv2.dilate(key, np.ones((3, 3), np.uint8)) > 0, sam)
        if glass is not None and slot.get("glass_prompts"):  # ink is on the lid, not on the shaker in front
            gm = cv2.GaussianBlur(glass.frame(i).astype(np.float32), (0, 0), 1.5)
            alpha = np.maximum(alpha, gm)
    if m.any():
        al = refine_occluder(f, m, radius=4, gain=2.0)
        for P, known in plates:
            al = tighten_to_plate(f, al, m, P, known)
        alpha = np.maximum(alpha, al)
    return np.clip(alpha, 0, 1)


def shade_track(frames, tr, occ, start):
    """What render_dynamic.carry_shading needs that a shader can't cheaply get: per frame, the median of the
    blurred grey surface (occluders excluded) inside the un-inset quad, or None where it skips the frame."""
    median, quads = [], []
    for i, f in enumerate(frames):
        if i < start:
            continue
        if not tr["valid"][i]:
            median.append(None), quads.append(None)
            continue
        m = quad_mask(tr["quads"][i], frames.h, frames.w)
        if occ:
            m &= ~occ.frame(i)
        gray = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
        median.append(round(max(float(np.median(gray[m])), 1.0), 3) if m.sum() >= 200 else None)
        quads.append(np.asarray(tr["quads"][i], float).round(2).tolist())
    return {"lo": 0.45, "hi": 1.35, "median": median, "quads": quads}


FORMAT = {"can": "can_wrap", "bottle": "bottle_label", "tent": "table_tent", "glass_print": "glass_print",
          "glass_decal": "glass_print", "cup_decal": "glass_print",
          "shirt_print": "shirt_print", "aframe": "poster"}


def export_object(clip, spec, frames, start, inventory, d):
    """A 3D product or a print on a real thing (scripts/place3d.py) as per-frame sprites: everything is ray-traced
    here, the player only blends.

    art/<id>_<brand>.png: RGBA, unpremultiplied colour and final coverage (people in front of the object already
    cut out). art/<id>_shadow.png: grey, the table shadow multiplier (1 where people stand in front of it). Atlases of
    fixed-size cells, one per frame, at `boxes[i]` in the frame. The player does
    rgb = mix(prev * shadow, colour, alpha), which equals the server's apply() for the same frame."""
    from pipeline.matte import refine_occluder
    from scripts.place3d import make
    obj = make(clip, spec)
    kind = spec.get("kind", "can")
    brands = eligible_brands({"surface": spec.get("surface", "can"), "creative_format": FORMAT[kind]}, inventory)
    layers = {}
    for i, f in enumerate(frames):
        if i >= start and i in obj.poses:
            ls = {b: obj.layer(i, f, b) for b in brands}
            if all(L is not None for L in ls.values()):  # None: outside the picture this frame
                layers[i] = (f, ls)
    if not layers:
        return None
    sizes = np.array([[L[0][2] - L[0][0], L[0][3] - L[0][1]] for _, ls in layers.values() for L in ls.values()])
    # big objects (the A-frame fills much of the frame) are stored smaller: eight brands at full size would be
    # hundreds of MB of GPU memory in the browser; the player upsamples them (the server render stays full size)
    scale = min(1.0, 360.0 / sizes.max())
    small = lambda x: x if scale == 1.0 else cv2.resize(x, (max(1, round(x.shape[1] * scale)), max(1, round(x.shape[0] * scale))),
                                                         interpolation=cv2.INTER_AREA)
    cw, ch = (np.ceil(sizes.max(0) * scale).astype(int) + 2)   # 2 px gap: no bleeding between cells
    cols = max(1, 4096 // cw)
    rows = -(-len(layers) // cols)
    shadow_atlas = np.full((rows * ch, cols * cw), 255, np.uint8)
    colour = {b: np.zeros((rows * ch, cols * cw, 4), np.uint8) for b in brands}
    boxes = [None] * (frames.n - start)
    for k, (i, (f, ls)) in enumerate(sorted(layers.items())):
        cx, cy = (k % cols) * cw, (k // cols) * ch
        (x0, y0, x1, y1) = ls[brands[0]][0]
        h, w = y1 - y0, x1 - x0
        front = obj.front(i)
        kf = refine_occluder(f, front, radius=4, gain=2.0)[y0:y1, x0:x1] if front is not None else np.zeros((h, w))
        for b in brands:
            _, col, cov, shadow = ls[b]
            a = cov * (1 - kf)
            if b == brands[0]:
                # server: k*f + (1-k)*(prev*sh*(1-cov) + c*cov), with prev = f under people -> prev*sh'*(1-a) + c*a
                sh = small((kf + (1 - kf) * shadow * (1 - cov)) / np.maximum(1 - a, 1e-4))
                shadow_atlas[cy:cy + sh.shape[0], cx:cx + sh.shape[1]] = np.rint(np.clip(sh, 0, 1) * 255)
            rgba = small(np.dstack([col, a * 255]).astype(np.float32))
            colour[b][cy:cy + rgba.shape[0], cx:cx + rgba.shape[1]] = np.rint(rgba).clip(0, 255)
        boxes[i - start] = [int(x0), int(y0), int(w), int(h), int(cx), int(cy)]
    cv2.imwrite(str(d / "art" / f"{spec['id']}_shadow.png"), shadow_atlas)
    for b in brands:
        cv2.imwrite(str(d / "art" / f"{spec['id']}_{b}.png"), colour[b])
    print(f"  {spec['id']}: {len(brands)} brands, {len(layers)} frames, atlas {shadow_atlas.shape[1]}x{shadow_atlas.shape[0]}",
          flush=True)
    return {"id": spec["id"], "surface": spec.get("surface", "can"), "boxes": boxes, "scale": scale,
            "shadow": f"art/{spec['id']}_shadow.png",
            "brands": {b: {"art": f"art/{spec['id']}_{b}.png"} for b in brands}}


def export_glass(clip, spec, frames, start, inventory, d):
    """A logo on a still glass (scripts/glass_decal.py), wrapped live in the player so any brand, even one that wins
    its bid after this export, needs no rendering: only the logo's own alpha image.

    art/<id>_glass.png: atlas of fixed-size cells, one per frame at `boxes[i]`; RGB = the print's colour there (the
    glass's light and highlights, the same for every brand), A = where the glass itself is seen. `place[i]`: rim
    centre, radius, axis angle, stretch and the logo's motion (px/frame) for the shutter smear. ../logos/<brand>_glass.png:
    the logo's alpha, shared by every clip. The player reproduces GlassDecal.apply: cylinder wrap, fade at the
    sides, defocus `sigma`, smear, x 0.88 x visibility, then over with luma grain."""
    from scripts.glass_decal import CURVE, MAX_HALF
    from scripts.place3d import make
    obj = make(clip, spec)
    brands = eligible_brands({"surface": spec.get("surface", "glass"), "creative_format": "glass_print"}, inventory)
    cells = {}
    for i, f in enumerate(frames):
        if i < start or i not in obj.place or not obj.store.any(i):
            continue
        L = obj.layer(i, f, brands[0])
        if L is None:
            continue
        (x0, y0, x1, y1), colour, _, _ = L
        vis = cv2.erode(obj.store[i][y0:y1, x0:x1].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        cells[i] = (x0, y0, np.dstack([np.clip(colour, 0, 255), cv2.GaussianBlur(vis, (0, 0), 0.8) * 255]))
    if not cells:
        return None
    sizes = np.array([c[2].shape[1::-1] for c in cells.values()])
    cw, ch = sizes.max(0) + 2
    cols = max(1, 4096 // cw)
    atlas = np.zeros((-(-len(cells) // cols) * ch, cols * cw, 4), np.uint8)
    boxes, place = [None] * (frames.n - start), [None] * (frames.n - start)
    for k, (i, (x0, y0, cell)) in enumerate(sorted(cells.items())):
        cx, cy = (k % cols) * cw, (k // cols) * ch
        atlas[cy:cy + cell.shape[0], cx:cx + cell.shape[1]] = np.rint(cell)
        boxes[i - start] = [int(x0), int(y0), int(cell.shape[1]), int(cell.shape[0]), int(cx), int(cy)]
        nb = [j for j in (i - 1, i + 1) if j in obj.place]   # as ShirtPrint.motion_blur
        v = ((obj.place[max(nb)][:2] - obj.place[min(nb + [i])][:2]) / max(max(nb) - min(nb + [i]), 1)
             if nb else np.zeros(2))
        place[i - start] = [round(float(x), 3) for x in (*obj.place[i][:5], *v)]
    for old in (d / "art").glob(f"{spec['id']}_*.png"):   # per-brand sprites of the older, pre-rendered format
        old.unlink()
    cv2.imwrite(str(d / "art" / f"{spec['id']}_glass.png"), atlas)
    logos = d.parent / "logos"
    logos.mkdir(exist_ok=True)
    for b in brands:
        cv2.imwrite(str(logos / f"{b}_glass.png"), np.rint(obj.art(b) * 255).astype(np.uint8))
    print(f"  {spec['id']}: glass, {len(brands)} brands live, {len(cells)} frames, atlas {atlas.shape[1]}x{atlas.shape[0]}",
          flush=True)
    return {"id": spec["id"], "kind": "glass", "surface": spec.get("surface", "glass"), "contents": spec.get("contents", ""),
            "boxes": boxes,
            "place": place, "atlas": f"art/{spec['id']}_glass.png", "sigma": round(float(obj.sigma), 3),
            "grain": round(float(obj.grain), 3), "half_angle": round(float(obj.half), 4),
            # per brand, the player widens the print so the logo is logo_height of the vessel's height (half_for)
            "logo_height": obj.logo_h, "ratio": round(float(obj.ratio), 4), "max_half": MAX_HALF,
            "lateral": round(float(obj.lateral), 4), "curve": CURVE, "ink": obj.ink_alpha,
            "shutter": 0.5, "brands": {b: {"art": f"../logos/{b}_glass.png"} for b in brands}}


def assignments(clip, entry, slots, inventory, brands):
    from pipeline.rank import choose_assignment
    profiles = json.loads((ROOT / "configs" / "profiles.json").read_text())["profiles"]
    chooser_slots = [{"slot_id": s["id"], "surface": s.get("surface", "poster"), "area_frac": s.get("area_frac", 0.05),
                      "neighbours": s.get("neighbours", [])} for s in slots]
    # Only brands the player has artwork for, per slot: restrict the inventory the chooser sees.
    allowed = {b for bs in brands.values() for b in bs}
    inv = {**inventory, "campaigns": [c for c in inventory["campaigns"] if c["brand_id"] in allowed]}
    out = {}
    for p in profiles:
        r = choose_assignment(chooser_slots, inv, p, entry.get("context_tags", ()), max_ads=len(slots))
        a = {sid: b for sid, b in r["assignment"].items() if b in brands.get(sid, [])}
        for s in slots:  # every slot carries an ad in the demo; fill gaps with the first eligible unused brand
            if s["id"] not in a:
                a[s["id"]] = next((b for b in brands[s["id"]] if b not in a.values()), brands[s["id"]][0])
        out[p["profile_id"]] = {"label": p["label"], "preferences": p.get("preferences", []), "assignment": a,
                                "chooser": r["chooser"], "rationale": r["rationale"]}
        print(f"  {p['profile_id']} ({p['label']}): {a}  [{r['chooser']}] {r['rationale'][:100]}", flush=True)
    return out


def main(clip, skip_video=False):
    t0 = time.time()
    from pipeline.clipcfg import entry as clip_entry
    entry = clip_entry(clip)
    inventory = json.loads((ROOT / "configs" / "inventory.json").read_text())
    frames = Frames(ROOT / clip)
    occ = occluders(clip, entry, frames)
    start = int(entry.get("player_start", 0))
    slots = [s for s in entry["dynamic_slots"] if s.get("player", True)]
    d = OUT / Path(clip).stem
    (d / "art").mkdir(parents=True, exist_ok=True)

    track = {"clip": clip, "fps": frames.fps, "frames": frames.n - start, "width": frames.w, "height": frames.h,
             "start": start, "slots": []}
    layers, brands, glass = [], {}, None
    for slot in slots:
        brands[slot["id"]] = eligible_brands(slot, inventory)
        multiply = slot.get("blend") == "multiply"
        base = None
        per_brand = {}
        for b in brands[slot["id"]]:
            layer = prepare_layer(clip, slot, b, frames, occ, with_plate=base is None)
            _, tr, art, blur, grain, plate = layer
            if base is None:
                base = layer
                base_sigma = np.asarray(tr["defocus_sigma"], float)
            name = f"art/{slot['id']}_{b}.{'png' if multiply else 'jpg'}"
            save_art(d / name, art, multiply)
            sig = np.asarray(tr["defocus_sigma"], float)
            extra = float(np.sqrt(np.maximum(np.median(sig ** 2 - base_sigma ** 2), 0)))  # look-match extra
            per_brand[b] = {"art": name, "extra_blur": round(extra, 3)}
        layers.append(base)
        _, tr, *_ = base
        if slot.get("glass_prompts"):
            glass = Union([union_store(clip, p, frames) for p in slot["glass_prompts"]])
        ins = 0.0 if multiply else float(slot.get("inset_px", 2))
        quads = [inset(tr["quads"][i], ins).round(2).tolist() if tr["valid"][i] else None
                 for i in range(start, frames.n)]
        gain = np.asarray(tr["gain"], float).reshape(frames.n, -1)
        veil = np.asarray(tr["veil"], float).reshape(frames.n, -1)
        if gain.shape[1] == 1:
            gain, veil = np.repeat(gain, 3, 1), np.repeat(veil, 3, 1)
        track["slots"].append({
            "id": slot["id"], "surface": slot.get("surface"), "blend": "multiply" if multiply else "over",
            "ink_strength": float(slot.get("ink_strength", 1.0)), "shutter": float(tr["shutter"]),
            "quads": quads, "gain": gain[start:].round(4).tolist(), "veil": veil[start:].round(2).tolist(),
            "sigma": base_sigma[start:].round(3).tolist(), "grain": round(float(base[4]), 3),
            "brands": per_brand})
        if slot.get("shading") and not multiply:
            track["slots"][-1]["shade"] = shade_track(frames, tr, occ, start)
        print(f"  {slot['id']}: {len(per_brand)} brands, valid {sum(q is not None for q in quads)}/{len(quads)}",
              flush=True)
    track["objects"] = []
    from scripts.place3d import active, objects_for
    order = [o.id for o in objects_for(clip, entry)]  # far to near: nearer objects draw on top
    for spec in sorted(active(entry), key=lambda s: order.index(s["id"])):
        o = (export_glass if spec.get("kind") in ("glass_decal", "cup_decal") else export_object)(clip, spec, frames, start,
                                                                                   inventory, d)
        if o:
            track["objects"].append(o)
            brands[spec["id"]] = list(o["brands"])
    for old in (d / "art").glob("*_mask.png"):  # superseded atlas format
        old.unlink()
    placed = {o["id"] for o in track["objects"]}
    for old in (d / "art").glob("*_glass.png"):  # a glass placed by an earlier export and declined since
        if old.name[:-len("_glass.png")] not in placed:
            old.unlink()
    audio = (ROOT / clip).with_suffix(".m4a")
    audio_out = d / "audio.m4a"
    if audio.is_file():
        if start:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio), "-ss", str(start / frames.fps),
                            "-t", str((frames.n - start) / frames.fps), "-vn", "-c:a", "aac",
                            "-movflags", "+faststart", str(audio_out)], check=True)
        else:
            shutil.copyfile(audio, audio_out)
        track["audio"] = "audio.m4a"
    else:
        audio_out.unlink(missing_ok=True)
    (d / "track.json").write_text(json.dumps(track))

    if not skip_video:
        h, w = frames.h, frames.w
        enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                "-s", f"{w}x{2 * h}", "-r", str(frames.fps), "-i", "-", "-c:v", "libx264",
                                "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                str(d / "stack.mp4")], stdin=subprocess.PIPE)
        for i, f in enumerate(frames):
            if i < start:
                continue
            a = matte_frame(f, i, occ, layers, glass, frames)
            m = np.repeat(np.rint(a * 255).astype(np.uint8)[..., None], 3, 2)
            enc.stdin.write(np.vstack([f, m]).tobytes())
            if i % 100 == 0:
                print(f"  stack {i}/{frames.n} {time.time() - t0:.0f} s", flush=True)
        enc.stdin.close()
        if enc.wait():
            raise SystemExit("ffmpeg failed")

    from scripts.place3d import active
    choice_slots = slots + active(entry)  # the chooser sees a can as one more surface
    (d / "assignments.json").write_text(json.dumps(assignments(clip, entry, choice_slots, inventory, brands),
                                                   indent=1))
    print(f"exported {d} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--skip-video", action="store_true")
    a = ap.parse_args()
    main(a.clip, a.skip_video)
