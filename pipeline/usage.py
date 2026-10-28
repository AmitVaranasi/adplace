"""Ledger of Meta Model API spend (we have a fixed $150 credit; the API exposes no balance endpoint).

Every Muse Spark completion and SAM 3.1 video request appends one line to logs/meta_usage.jsonl with its cost at the
published pay-as-you-go prices (dev.meta.ai/docs/pricing-rate-limits, checked 2026-10-28). Reasoning tokens are
counted as output, the conservative reading. `python -m pipeline.usage` prints the running total and what is left.
"""
import json
import sys
import time
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[1] / "logs" / "meta_usage.jsonl"
BUDGET_USD = 150.0
MUSE_PER_M = {"input": 1.25, "cached": 0.15, "output": 4.25}  # muse-spark-1.x Standard, USD per 1M tokens
SAM_VIDEO_PER_FRAME, SAM_MIN_FRAMES = 0.20 / 1000, 50


def muse_cost(usage):
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    fresh = usage.get("prompt_tokens", 0) - cached
    return (fresh * MUSE_PER_M["input"] + cached * MUSE_PER_M["cached"]
            + usage.get("completion_tokens", 0) * MUSE_PER_M["output"]) / 1e6


def sam_cost(frames):
    return max(int(frames), SAM_MIN_FRAMES) * SAM_VIDEO_PER_FRAME


def record(kind, cost, **fields):
    LEDGER.parent.mkdir(exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "usd": round(cost, 6),
                            **fields}) + "\n")


def record_muse(usage, purpose=""):
    record("muse", muse_cost(usage), purpose=purpose, prompt_tokens=usage.get("prompt_tokens", 0),
           completion_tokens=usage.get("completion_tokens", 0))


def record_sam(frames, purpose=""):
    record("sam", sam_cost(frames), purpose=purpose, frames=int(frames))


def summary():
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines()] if LEDGER.exists() else []
    spent = {}
    for r in rows:
        spent[r["kind"]] = spent.get(r["kind"], 0.0) + r["usd"]
    total = sum(spent.values())
    return {"by_kind": {k: round(v, 3) for k, v in spent.items()}, "spent": round(total, 3),
            "left": round(BUDGET_USD - total, 2), "entries": len(rows)}


if __name__ == "__main__":
    json.dump(summary(), sys.stdout, indent=2)
    print()
