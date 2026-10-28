"""Place several ads in one long shot, streaming: a different brand per screen, per viewer, always disclosed.

  python scripts/render_scene.py assets/clips/piccadilly_long.mp4 prep            # camera + reference shapes (once)
  python scripts/render_scene.py assets/clips/piccadilly_long.mp4 render --profile commuter [--out outputs/]
  python scripts/render_scene.py assets/clips/piccadilly_long.mp4 render --assign piccadilly_long-0=nimbus ...

Long clips never hold a dense mask stack or the whole video in memory:
- masks live in pipeline.maskstore stores (box-local, bit-packed), expanded one frame at a time;
- camera motion comes from pipeline.long_motion (keyframes + joint refinement, frame -> reference homographies);
- every screen gets ONE reference-space shape: each frame's SAM mask is warped into the reference view and the frames
  vote, which fills holes left by people and buses (they move, the screen does not), cannot flinch (one rigid shape),
  and extends past the frame edge so a screen leaving the frame shows a cropped ad rather than a squeezed one;
- rendering, labelling, Video Seal and encoding run in chunks; the watermark carries one manifest id for the whole
  slot x campaign assignment of this variant.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import composite, watermark  # noqa: E402
from pipeline.maskstore import MaskStore, MaskWriter  # noqa: E402
from scripts.render import add_label, campaign  # noqa: E402

PAD = 0.5  # reference canvas margin, as a fraction of the frame, for parts of screens outside the reference frame


class Frames:
    """Lazy, re-iterable BGR frame reader (frames[i] seeks; iteration streams)."""

    def __init__(self, path):
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.w, self.h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def __len__(self):
        return self.n

    def __iter__(self):
        cap = cv2.VideoCapture(self.path)
        for _ in range(self.n):
            ok, f = cap.read()
            if not ok:
                break
            yield f
        cap.release()

    def __getitem__(self, i):
        cap = cv2.VideoCapture(self.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        cap.release()
        if not ok:
            raise IndexError(i)
        return f


class Union:
    """Re-iterable per-frame OR of several mask stores (camera-motion exclusion, occluders)."""

    def __init__(self, stores, dilate=0):
        self.stores, self.dilate = stores, dilate

    def __len__(self):
        return len(self.stores[0])

    def frame(self, i):
        m = np.zeros((self.stores[0].h, self.stores[0].w), bool)
        for s in self.stores:
            if s.any(i):
                m |= s[i]
        if self.dilate:
            m = cv2.dilate(m.astype(np.uint8), np.ones((2 * self.dilate + 1,) * 2, np.uint8)) > 0
        return m

    def __iter__(self):
        return (self.frame(i) for i in range(len(self)))


def scene_dir(clip):
    d = ROOT / "cache" / f"{Path(clip).stem}-scene"
    d.mkdir(parents=True, exist_ok=True)
    return d


def union_store(clip, prompt, frames):
    """Every object SAM 3.1 found for a prompt (people, buses) merged into one store, built from the recorded stream."""
    from pipeline.sam_api import _mask_region, parse_masks
    from scripts.analyse import cache_dir
    out = scene_dir(clip) / f"occ_{prompt}.npz"
    if not out.exists():
        base = cache_dir(clip, {"model": "meta/sam-3.1", "prompt": prompt})
        events = [json.loads(line) for line in base.with_name(base.name + "-events.jsonl").read_text().splitlines()]
        writer = MaskWriter(frames.n, frames.h, frames.w)
        for (_, fi), rec in parse_masks(events).items():
            region = _mask_region(rec, frames.h, frames.w) if 0 <= fi < frames.n else None
            if region is not None and region[2].any():
                writer.add(fi, box=region[:2], local=region[2])
        writer.save(out)
    return MaskStore(out)


def canvas(frames):
    ox, oy = int(PAD * frames.w), int(PAD * frames.h)
    return (frames.w + 2 * ox, frames.h + 2 * oy), np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], float)


def reference_mask(store, H, frames, occ=None, step=2, keep=0.5):
    """Vote each frame's mask into the reference canvas; a pixel belongs to the screen when it was inside the mask in
    at least `keep` of the frames that saw it unoccluded. Frames where a person or bus covers a pixel do not vote on
    it, so someone standing in front for most of the shot is not mistaken for part of the facade."""
    (cw, ch), offset = canvas(frames)
    hits, seen = np.zeros((ch, cw), np.float32), np.zeros((ch, cw), np.float32)
    full = np.ones((frames.h, frames.w), np.uint8)
    for i in range(0, len(store), step):
        T = offset @ H[i]
        clear = full if occ is None else (~occ.frame(i)).astype(np.uint8)
        seen += cv2.warpPerspective(clear, T, (cw, ch), flags=cv2.INTER_NEAREST)
        if store.any(i):
            hits += cv2.warpPerspective((store[i] & (clear > 0)).astype(np.uint8), T, (cw, ch),
                                        flags=cv2.INTER_LINEAR)
    ref = (hits >= keep * np.maximum(seen, 1)) & (seen > 0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ref.astype(np.uint8), connectivity=8)
    if n > 1:  # the screen is the largest component; drop stray specks
        ref = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ref


def prep(clip, entry):
    from pipeline.long_motion import camera_motion_long
    frames = Frames(ROOT / clip)
    d = scene_dir(clip)
    slots = {s["slot_id"]: MaskStore(ROOT / s["analysis"] / "masks.npz") for s in entry["slots"]}
    occ = [union_store(clip, p, frames) for p in entry.get("occluder_prompts", [])]
    if not (d / "camera_H.npy").exists():
        t0 = time.time()
        H = camera_motion_long(frames, exclude_masks=Union(list(slots.values()) + occ, dilate=4))
        np.save(d / "camera_H.npy", H)
        print(f"camera motion: {time.time() - t0:.0f} s")
    H = np.load(d / "camera_H.npy")
    shapes = {}
    for sid, store in slots.items():
        ref = reference_mask(store, H, frames, occ=Union(occ, dilate=3) if occ else None)
        np.save(d / f"ref_{sid}.npy", np.packbits(ref))
        shape = next(s for s in entry["slots"] if s["slot_id"] == sid).get("shape", {"type": "quad"})
        shapes[sid] = fit_shape(ref, shape)
        hole = static_occluder(ref, shapes[sid])
        np.save(d / f"hole_{sid}.npy", np.packbits(hole))
        print(f"{sid}: reference area {ref.sum()} px, {shapes[sid]['type']}, static occluder {hole.sum()} px")
    (d / "shapes.json").write_text(json.dumps(shapes, indent=1))


def fit_shape(ref, shape):
    if shape.get("type") == "ribbon":
        from pipeline.ribbon import ribbon_from_mask, smooth_edges
        r = smooth_edges(ribbon_from_mask(ref))
        return {"type": "ribbon", "x": list(map(float, r["x"])), "top": list(map(float, r["top"])),
                "bottom": list(map(float, r["bottom"])), "creative": shape.get("creative", "16x9")}
    from pipeline.mask_track import track_from_masks
    # Fit the convex hull: a lamp post in front notches the outline, which is not a quadrilateral but is still
    # the screen (the notch becomes a static occluder, see static_occluder).
    contours, _ = cv2.findContours(ref.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = np.zeros(ref.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.vstack(contours)), 1)
    try:
        quad = np.asarray(track_from_masks([hull > 0])["quads"][0], float)
    except ValueError:
        approx = cv2.approxPolyDP(cv2.convexHull(np.vstack(contours)), 0.02 * cv2.arcLength(
            cv2.convexHull(np.vstack(contours)), True), True).reshape(-1, 2)
        quad = approx if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(np.vstack(contours)))
        quad = np.asarray(quad, float)
    return {"type": "quad", "quad": quad.tolist()}


def static_occluder(ref, shape, open_px=5):
    """Parts of the fitted surface that were never inside the screen mask in any frame: something fixed stands in front
    (a lamp post, a traffic light). Thin slivers along the fitted edge are fit error, not objects, so open them away."""
    region = np.zeros(ref.shape, np.uint8)
    if shape["type"] == "quad":
        cv2.fillConvexPoly(region, np.round(np.asarray(shape["quad"])).astype(np.int32), 1)
    else:
        poly = np.r_[np.c_[shape["x"], shape["top"]], np.c_[shape["x"], shape["bottom"]][::-1]]
        cv2.fillPoly(region, [np.round(poly).astype(np.int32)], 1)
    hole = (region > 0) & ~ref
    hole = cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8)) > 0
    return hole


def frame_geometry(shape, T_inv):
    """Reference-canvas shape -> this frame (T_inv maps canvas to frame)."""
    def project(p):
        q = np.c_[p, np.ones(len(p))] @ T_inv.T
        return q[:, :2] / q[:, 2:]
    if shape["type"] == "quad":
        return project(np.asarray(shape["quad"]))
    x = np.asarray(shape["x"])
    top, bottom = project(np.c_[x, shape["top"]]), project(np.c_[x, shape["bottom"]])
    xm = 0.5 * (top[:, 0] + bottom[:, 0])
    return {"x": xm, "top": np.interp(xm, top[:, 0], top[:, 1]), "bottom": np.interp(xm, bottom[:, 0], bottom[:, 1])}


def grade_slot(creative, frames, store, occ, ring_px=41):
    """Grade into the light of this screen, measured once on the frame where it is largest (pristine source)."""
    from pipeline.grade import apply_grade, scene_grade
    areas = [(store.box(i)[2] - store.box(i)[0]) * (store.box(i)[3] - store.box(i)[1]) for i in range(len(store))]
    i = int(np.argmax(areas))
    f, m = frames[i], store[i]
    region = m & ~occ.frame(i) if occ else m
    ring = (cv2.dilate(m.astype(np.uint8), np.ones((ring_px, ring_px), np.uint8)) > 0) & ~m
    return apply_grade(creative, scene_grade(f[region], f[ring]))


def manifest_id(clip, assignment, inventory):
    """One Video Seal message for a whole multi-ad variant: hashes clip + every slot -> campaign token."""
    tokens = {c["brand_id"]: c["campaign_token"] for c in inventory["campaigns"]}
    body = Path(clip).stem + "|" + "|".join(f"{s}={tokens[b]}" for s, b in sorted(assignment.items()))
    return "scene-" + hashlib.sha256(body.encode()).hexdigest()[:16], body


def render(clip, entry, assignment, out, chunk=48, label=True):
    from scripts.render import soft_occlude
    frames = Frames(ROOT / clip)
    d = scene_dir(clip)
    H = np.load(d / "camera_H.npy")
    shapes = json.loads((d / "shapes.json").read_text())
    _, offset = canvas(frames)
    inventory = json.loads((ROOT / "configs" / "inventory.json").read_text())
    occ = Union([MaskStore(d / f"occ_{p}.npz") for p in entry.get("occluder_prompts", [])])
    slot_meta = {s["slot_id"]: s for s in entry["slots"]}
    layers = []
    for sid, brand in assignment.items():
        c, _ = campaign(brand)
        shape = shapes[sid]
        key = shape.get("creative") if shape["type"] == "ribbon" else None
        if key is None:  # quad: the creative whose aspect is closest to the reference quad's
            from scripts.render import pick_creative
            art = pick_creative(c, np.asarray(shape["quad"]))
        else:
            art = cv2.cvtColor(cv2.imread(str(ROOT / c["creatives"][key]), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        store = MaskStore(ROOT / slot_meta[sid]["analysis"] / "masks.npz")
        art = grade_slot(art, frames, store, occ if occ.stores else None)
        layers.append((sid, shape, art, slot_meta[sid].get("emissive", True)))
    mid, body = manifest_id(clip, assignment, inventory)
    (cw, ch), _ = canvas(frames)
    holes = np.zeros((ch, cw), np.uint8)
    for sid in assignment:
        if (d / f"hole_{sid}.npy").exists():
            holes |= np.unpackbits(np.load(d / f"hole_{sid}.npy"))[:ch * cw].reshape(ch, cw)

    def occluders_at(i):
        m = occ.frame(i) if occ.stores else np.zeros((frames.h, frames.w), bool)
        if holes.any():
            m |= cv2.warpPerspective(holes, np.linalg.inv(offset @ H[i]), (frames.w, frames.h),
                                     flags=cv2.INTER_NEAREST) > 0
        return m

    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{frames.w}x{frames.h}", "-r", str(frames.fps), "-i", "-", "-c:v", "libx264",
                            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                           stdin=subprocess.PIPE)
    t0, batch, start = time.time(), [], 0

    def flush(batch, start):
        originals, placed = batch, []
        for k, f in enumerate(batch):
            i = start + k
            T_inv = np.linalg.inv(offset @ H[i])
            g = f
            for sid, shape, art, emissive in layers:
                geo = frame_geometry(shape, T_inv)
                if shape["type"] == "quad":
                    g = composite.composite_frame(g, art, geo, emissive=emissive, match_energy=False)
                else:
                    from pipeline.ribbon import composite_ribbon
                    g = composite_ribbon(g, art, geo, emissive=emissive, match_energy=False)
            placed.append(g)
        placed = soft_occlude(originals, placed, [occluders_at(start + k) for k in range(len(batch))])
        if label:
            placed = [add_label(p) for p in placed]
        for p in watermark.embed(placed, mid):
            enc.stdin.write(p.tobytes())

    for i, f in enumerate(frames):
        batch.append(f)
        if len(batch) == chunk:
            flush(batch, i + 1 - chunk)
            batch = []
            print(f"  {i + 1}/{frames.n} frames, {time.time() - t0:.0f} s", flush=True)
    if batch:
        flush(batch, frames.n - len(batch))
    enc.stdin.close()
    if enc.wait():
        raise SystemExit(f"ffmpeg failed writing {out}")

    # Verify the exact exported file on samples from start, middle and end (bounded memory).
    exported = Frames(out)
    picks = [i for s in (0, exported.n // 2 - 24, exported.n - 48) for i in range(max(s, 0), min(s + 48, exported.n))]
    result = watermark.verify([exported[i] for i in picks[::2]], [mid])
    report = {"clip": clip, "assignment": assignment, "manifest_id": mid, "manifest": body, "output": str(out),
              "frames": frames.n, "label": label, "render_s": round(time.time() - t0, 1), "verify": result,
              "verified": result["campaign"] == mid}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("cmd", choices=["prep", "render"])
    ap.add_argument("--assign", nargs="*", default=[], help="slot_id=brand_id ...")
    ap.add_argument("--profile", help="viewer profile name from configs/profiles.json (Muse picks the assignment)")
    ap.add_argument("--max-ads", type=int, default=3)
    ap.add_argument("--out")
    ap.add_argument("--no-label", action="store_true", help="only for blinded realism rating, never distribution")
    args = ap.parse_args()
    entry = json.loads((ROOT / "configs" / "clips.json").read_text())[args.clip]
    if args.cmd == "prep":
        return prep(args.clip, entry)
    assignment = dict(a.split("=", 1) for a in args.assign)
    if args.profile:
        assignment = choose_for_profile(args.clip, entry, args.profile, args.max_ads)
    tag = args.profile or "custom"
    out = Path(args.out) if args.out else ROOT / "outputs" / f"{Path(args.clip).stem}_{tag}.mp4"
    report = render(args.clip, entry, assignment, out, label=not args.no_label)
    print(json.dumps({k: report[k] for k in ("output", "assignment", "manifest_id", "render_s", "verified")},
                     indent=2), f"\nbit accuracy {report['verify']['bit_accuracy']:.3f}")


def choose_for_profile(clip, entry, profile_name, max_ads):
    from pipeline.rank import choose_assignment
    profiles = {p["profile_id"]: p for p in json.loads((ROOT / "configs" / "profiles.json").read_text())["profiles"]}
    inventory = json.loads((ROOT / "configs" / "inventory.json").read_text())
    slots = [{"slot_id": s["slot_id"], "surface": s.get("surface", "screen"), "area_frac": s.get("area_frac", 0.0),
              "neighbours": s.get("neighbours", [])} for s in entry["slots"]]
    result = choose_assignment(slots, inventory, profiles[profile_name], entry.get("context_tags", ()),
                               max_ads=max_ads)
    print(f"{result['chooser']}: {result['assignment']} -- {result['rationale']}")
    return result["assignment"]


if __name__ == "__main__":
    main()
