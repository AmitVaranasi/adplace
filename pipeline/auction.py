"""The bid at playback: which brand each placement in a video carries for this viewer, decided when they press play.

Every placement's geometry, light and mattes were prepared at upload and do not depend on the brand (export_player),
so the auction only has to pick names. Per placement, nearest the camera first (glasses, cups and bowls on the
table: in focus and the most legible), then the rest in order of size (the biggest is worth most):

  eligible   the brand has artwork for this placement, can run on this surface, its exclusions don't match the
             scene, and it hasn't already won another placement in this video (one placement per brand). A
             placement no new brand can fill is then auctioned again among the brands already in the video
  score      bid_cpm x relevance; relevance = 1 + 1 per viewer interest the brand matches + 0.25 per scene tag it
             matches. A brand that fits pays less per point of score, as in quality-weighted ad auctions
  price      second price: the least the winner could have bid and still won (runner-up's score / winner's
             relevance), at least the reserve, never above its own bid

`jitter` (default 0) scales every bid by a random factor in [1 - jitter, 1 + jitter], standing in for the pacing and
budget changes that make real bids move from one impression to the next.
"""
from __future__ import annotations

import random
import time

RESERVE_CPM = 1.0
NEAR = {"glass", "cup", "mug", "bowl"}   # sold first: in focus, near the camera, where a viewer reads a brand


def relevance(campaign, preferences, context_tags):
    tags = set(campaign.get("preference_tags", []))
    return 1.0 + 1.0 * len(tags & set(preferences)) + 0.25 * len(tags & set(context_tags))


def run(placements, inventory, preferences, context_tags=(), jitter=0.0, seed=None):
    """placements: [{"id", "surface", "brands": [brand ids with artwork], "area": relative size}]. Returns
    {"assignment": {placement id: brand}, "placements": {id: {"winner", "price_cpm", "bidders": [...]}}, "ms"}."""
    t0 = time.perf_counter()
    rng = random.Random(seed)
    campaigns = {c["brand_id"]: c for c in inventory["campaigns"]}
    scene = set(context_tags)
    known = {x for c in campaigns.values() for x in c.get("eligible_surfaces", [])}   # surfaces campaigns name
    taken, assignment, detail = set(), {}, {}
    order = sorted(placements, key=lambda p: (p.get("surface") not in NEAR, -p.get("area", 0.0)))
    for p in order + order:   # second pass: placements left empty, brands already in the video may bid again
        if p["id"] in assignment:
            continue
        again = p["id"] in detail
        bidders = []
        for b in p["brands"]:
            c = campaigns.get(b)
            if c is None or (b in taken) != again or scene & set(c.get("contextual_exclusions", [])):
                continue
            if p.get("surface") in known and p["surface"] not in c.get("eligible_surfaces", []):
                continue
            if p.get("contents") and p["contents"] in c.get("unsuitable_contents", []):
                continue   # a coffee brand on a glass of orange juice reads as a mistake
            bid = float(c.get("bid_cpm", RESERVE_CPM)) * (1 + rng.uniform(-jitter, jitter))
            if bid < RESERVE_CPM:
                continue
            rel = relevance(c, preferences, context_tags)
            bidders.append({"brand": b, "bid_cpm": round(bid, 2), "relevance": rel, "score": round(bid * rel, 3)})
        bidders.sort(key=lambda x: (-x["score"], x["brand"]))
        if not bidders:
            detail[p["id"]] = {"winner": None, "price_cpm": None, "bidders": []}
            continue
        w = bidders[0]
        floor = bidders[1]["score"] / w["relevance"] if len(bidders) > 1 else RESERVE_CPM
        price = min(max(floor, RESERVE_CPM), w["bid_cpm"])
        taken.add(w["brand"])
        assignment[p["id"]] = w["brand"]
        detail[p["id"]] = {"winner": w["brand"], "price_cpm": round(price, 2), "bidders": bidders,
                           # why this viewer: the interests of theirs the winner targets
                           "matched": sorted(set(campaigns[w["brand"]].get("preference_tags", [])) & set(preferences))}
    return {"assignment": assignment, "placements": detail, "ms": round(1000 * (time.perf_counter() - t0), 2)}


def placements_of(track):
    """The auction's view of a player export (app/static/player/<clip>/track.json)."""
    out = []
    for s in track.get("slots", []):
        qs = [q for q in s["quads"] if q]
        area = sum(abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(q, q[1:] + q[:1]))) / 2 for q in qs) / max(len(qs), 1)
        out.append({"id": s["id"], "surface": s.get("surface"), "brands": list(s["brands"]),
                    "area": area / (track["width"] * track["height"])})
    for o in track.get("objects", []):
        bs = [b for b in o["boxes"] if b]
        area = sum(b[2] * b[3] for b in bs) / max(len(bs), 1) * 0.5   # a wrapped print covers about half its box
        out.append({"id": o["id"], "surface": o.get("surface"), "brands": list(o["brands"]),
                    "contents": o.get("contents", ""), "area": area / (track["width"] * track["height"])})
    return out
