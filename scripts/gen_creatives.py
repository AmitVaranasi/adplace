"""Photographic, print-real creatives for the fictional demo brands, generated with Meta's muse-image-1.0.

  python scripts/gen_creatives.py                       # every brand x format not yet generated
  python scripts/gen_creatives.py --brand slicedash --format box_print --force

The flat vector cards from make_creatives.py were the loudest tell next to Mirriad's work, which places real
campaign artwork. Each format matches what the surface physically is: a pizza box lid carries one- or two-colour
flexo ink on kraft board, a freezer side a vinyl wrap, a wall a photographic poster, a dark sign light lettering on
black. Images are full-bleed flat artwork (the scene supplies the physical poster), cropped to the format's aspect,
and written to assets/creatives/gen/<brand>_<format>.png. Muse Spark reads each one back to check the brand name is
spelled right and no real trademark crept in; failures are regenerated (up to 3 tries). Spend goes to the ledger.
"""
import argparse
import base64
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.usage import record_image, summary  # noqa: E402
from scripts.make_creatives import BRANDS  # noqa: E402

OUT = ROOT / "assets" / "creatives" / "gen"
API = "https://api.meta.ai/v1"

CATEGORY = {
    "brewlark": "specialty coffee: a latte with latte art or espresso in a ceramic cup, roasted beans",
    "kestrel": "running shoes: a runner's shoe mid-stride on a city street",
    "paperwhale": "books: a stack of hardback books and an open book in warm reading light",
    "nimbus": "low-cost airline: an airliner wing over clouds at golden hour",
    "fernleaf": "local groceries: fresh vegetables and fruit in a crate",
    "voltway": "e-bikes and scooters: a sleek electric scooter on a city street at dusk",
    "fizzwell": "cola soft drink: an ice-cold straight-sided cylindrical glass bottle (plain, NOT a contour or hourglass bottle, no ribbed glass) of dark cola with condensation and ice cubes",
    "slicedash": "pizza delivery app: a hot pepperoni pizza slice with stretching cheese",
}

FORMATS = {
    # aspect (w/h), generation size, art direction
    "poster": (3 / 4, "1024x1365", "a photographic street poster, product hero shot, bold headline at the top, "
                                   "tagline below it, clean modern layout"),
    "landscape": (4 / 3, "1365x1024", "a landscape print advertisement, product hero shot on one side, headline "
                                      "and tagline on the other"),
    "dark_sign": (4 / 3, "1365x1024", "a printed sign on a matte black background, brand name in large light "
                                      "lettering, small product photo, minimal; suits a dark restaurant wall"),
    "vinyl_tall": (9 / 16, "1024x1820", "a tall narrow vinyl wrap for the side panel of a shop freezer, product "
                                        "photo filling most of the height, brand name running large"),
    "box_print": (16 / 10, "1365x853", "artwork printed directly on brown kraft cardboard with only one or two "
                                       "flat ink colours (flexographic print on a pizza box lid): a simple line "
                                       "illustration and the brand name, no photograph, no gradients, the "
                                       "cardboard colour shows through as the background"),
    "menu_card": (16 / 10, "1365x853", "a printed paper table menu card / placemat header, cream paper, small food "
                                       "photographs, brand name and tagline, restaurant print style"),
}

RULES = ("Flat full-bleed artwork seen straight on, filling the whole image edge to edge. NOT a mockup: no wall, "
         "no frame, no border, no tape, no pins, no shadow, no perspective, no hands. No QR codes, no app store "
         "or Google Play badges, no real company logos or trademarks, no website URLs. Spell the brand name "
         "exactly as given, in capital letters. Keep all other text to the tagline only.")


def _post(path, body, timeout=600):
    req = urllib.request.Request(API + path, json.dumps(body).encode(),
                                 {"Authorization": "Bearer " + os.environ["MODEL_API_KEY"],
                                  "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def prompt(bid, fmt):
    b, (_, _, art) = BRANDS[bid], FORMATS[fmt]
    colours = f"brand colours RGB{b['bg']} and RGB{b['accent']}"
    if fmt == "dark_sign":
        colours = f"black background, lettering in RGB{b['bg'] if sum(b['bg']) > 180 else b['fg']}"
    if fmt == "box_print":
        colours = f"ink colours: RGB{b['bg']} and near-black"
    return (f"Advertisement for a fictional brand called {b['name']} ({CATEGORY[bid]}). Tagline: "
            f"'{b['tagline']}'. Design: {art}. {colours}. {RULES}")


def generate(bid, fmt):
    aspect, size, _ = FORMATS[fmt]
    r = _post("/images/generations", {"model": "muse-image-1.0", "prompt": prompt(bid, fmt), "n": 1, "size": size})
    record_image(len(r["data"]), f"creative {bid} {fmt}")
    d = r["data"][0]
    raw = base64.b64decode(d["b64_json"]) if "b64_json" in d else urllib.request.urlopen(d["url"]).read()
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size  # centre-crop to the exact aspect
    if w / h > aspect:
        nw = int(h * aspect)
        im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = int(w / aspect)
        im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return im


def check(bid, path):
    """Muse Spark reads the artwork back: brand spelled right, no real trademark, flat (not a mockup)."""
    from pipeline.rank import chat
    small = Image.open(path)
    small.thumbnail((768, 768))
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=85)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    q = (f"This is ad artwork for a fictional brand '{BRANDS[bid]['name']}'. Answer JSON only: "
         '{"brand_spelled_correctly": bool, "other_text": "any other words you can read", '
         '"real_trademarks": "names of any real companies/logos/app-store badges/QR codes visible, or any '
         'distinctive real trade dress such as the Coca-Cola contour bottle; else empty", '
         '"is_flat_artwork": bool (false if it shows a wall, frame, tape, perspective or a mockup)}')
    text = chat(API, os.environ["MODEL_API_KEY"], "muse-spark-1.3", "You check ad artwork. Reply with JSON only.",
                [{"type": "text", "text": q}, {"type": "image_url", "image_url": {"url": url}}],
                extra={"max_completion_tokens": 4000}, timeout=180)
    try:
        v = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except ValueError:
        return False, text
    ok = v.get("brand_spelled_correctly") and not v.get("real_trademarks") and v.get("is_flat_artwork")
    return bool(ok), v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", nargs="*", default=list(CATEGORY))
    ap.add_argument("--format", nargs="*", default=list(FORMATS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for bid in args.brand:
        for fmt in args.format:
            path = OUT / f"{bid}_{fmt}.png"
            if path.exists() and not args.force:
                continue
            for attempt in range(3):
                generate(bid, fmt).save(path)
                ok, verdict = check(bid, path)
                print(f"{bid} {fmt} try {attempt + 1}: {'ok' if ok else 'REJECT'} {verdict}", flush=True)
                if ok:
                    break
            else:
                path.rename(path.with_name(path.stem + "_rejected.png"))
    print(summary())


if __name__ == "__main__":
    main()
