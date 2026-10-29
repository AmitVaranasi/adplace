"""Generate artwork for the fictional demo brands at each supported aspect ratio.

  python scripts/make_creatives.py      # writes assets/creatives/<brand>_<ratio>.png
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "creatives"
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

BRANDS = {
    "brewlark": {"name": "BREWLARK", "tagline": "Coffee that sings back", "bg": (46, 28, 20), "fg": (245, 214, 160),
                 "accent": (214, 120, 52)},
    "kestrel": {"name": "KESTREL RUN", "tagline": "Every street is a track", "bg": (12, 40, 72), "fg": (255, 255, 255),
                "accent": (255, 92, 70)},
    "paperwhale": {"name": "PAPERWHALE", "tagline": "Books for deep dives", "bg": (232, 238, 240), "fg": (20, 50, 70),
                   "accent": (40, 140, 170)},
    "nimbus": {"name": "NIMBUS AIR", "tagline": "Fly further for less", "bg": (24, 92, 170), "fg": (255, 255, 255),
               "accent": (255, 196, 64)},
    "fernleaf": {"name": "FERNLEAF", "tagline": "Groceries grown nearby", "bg": (34, 92, 62), "fg": (242, 236, 214),
                 "accent": (178, 208, 92)},
    "voltway": {"name": "VOLTWAY", "tagline": "Ride electric across town", "bg": (22, 22, 28), "fg": (120, 250, 200),
                "accent": (240, 240, 240)},
    "fizzwell": {"name": "FIZZWELL", "tagline": "Crisp cola, real cane sugar", "bg": (176, 18, 36), "fg": (255, 246, 230),
                 "accent": (20, 20, 24)},
    "slicedash": {"name": "SLICEDASH", "tagline": "Hot pizza at your door in 20", "bg": (255, 132, 20), "fg": (40, 20, 10),
                  "accent": (255, 255, 255)},
}
RATIOS = {"16x9": (1600, 900), "4x3": (1600, 1200), "3x4": (1200, 1600), "9x16": (900, 1600), "2x1": (1800, 900)}


def font(size):
    return ImageFont.truetype(FONT, size)


def fit_text(draw, text, max_w, start):
    size = start
    while size > 12 and draw.textlength(text, font=font(size)) > max_w:
        size -= 4
    return font(size)


def render(brand, w, h):
    img = Image.new("RGBA", (w, h), brand["bg"] + (255,))
    d = ImageDraw.Draw(img)
    m = int(min(w, h) * 0.08)
    d.rectangle([0, h - m, w, h], fill=brand["accent"])  # accent band
    r = int(min(w, h) * 0.12)
    cx, cy = w // 2, int(h * 0.32)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=brand["accent"])  # logo mark
    d.ellipse([cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2], fill=brand["bg"])
    f = fit_text(d, brand["name"], w - 2 * m, int(h * 0.16))
    d.text((cx, int(h * 0.58)), brand["name"], font=f, fill=brand["fg"], anchor="mm")
    f = fit_text(d, brand["tagline"], w - 2 * m, int(h * 0.07))
    d.text((cx, int(h * 0.74)), brand["tagline"], font=f, fill=brand["fg"], anchor="mm")
    return img


def dark(brand):
    """The same brand on near-black, for dark surfaces (a black wall sign, a dark menu board): an orange panel where
    a black sign was reads as pasted on, the brand colour as lettering on black reads as the sign itself."""
    lum = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    ink = brand["bg"] if lum(brand["bg"]) > 60 else brand["fg"]
    # The accent band is drawn in the lettering colour: a white band on a black sign reads as a lit strip.
    return {**brand, "bg": (16, 16, 18), "fg": ink, "accent": ink}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for bid, brand in BRANDS.items():
        for ratio, (w, h) in RATIOS.items():
            render(brand, w, h).save(OUT / f"{bid}_{ratio}.png")
            render(dark(brand), w, h).save(OUT / f"{bid}_{ratio}_dark.png")
    print("wrote", 2 * len(BRANDS) * len(RATIOS), "creatives to", OUT)


if __name__ == "__main__":
    main()
