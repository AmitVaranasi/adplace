"""Generate artwork for the three fictional demo brands at each supported aspect ratio.

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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for bid, brand in BRANDS.items():
        for ratio, (w, h) in RATIOS.items():
            render(brand, w, h).save(OUT / f"{bid}_{ratio}.png")
    print("wrote", len(BRANDS) * len(RATIOS), "creatives to", OUT)


if __name__ == "__main__":
    main()
