"""Machine-readable disclosure: embed a campaign token in a clip with Meta Video Seal, and recover it later.

Video Seal carries a fixed-length bit message. A campaign id is hashed into that message; verification decodes the
message from a clip and compares it with the tokens of known campaigns. A clip counts as watermarked for a campaign
when the bit accuracy clears THRESHOLD; unwatermarked video sits near 0.5 (chance).
"""
import hashlib

import numpy as np
import torch

THRESHOLD = 0.9
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        import tempfile
        from pathlib import Path

        import videoseal
        from omegaconf import OmegaConf
        # The pip package resolves its model card and attenuation config relative to the working directory and does
        # not ship the latter, so point a copy of the card at the vendored config by absolute path.
        cfg_dir = Path(__file__).parent / "videoseal_cfg"
        card = OmegaConf.load(cfg_dir / "videoseal_1.0.yaml")
        card.args.attenuation_config = str(cfg_dir / "attenuation.yaml")
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(card, f.name)
        _MODEL = videoseal.load(Path(f.name)).eval()
    return _MODEL


def nbits():
    return _model().embedder.msg_processor.nbits if hasattr(_model().embedder, "msg_processor") else 256


def campaign_bits(campaign_id, n=None):
    """Deterministic bit message for a campaign id."""
    n = n or nbits()
    digest = b"".join(hashlib.sha256(f"{campaign_id}:{i}".encode()).digest() for i in range((n + 255) // 256))
    return np.unpackbits(np.frombuffer(digest, np.uint8))[:n].astype(bool)


def _to_tensor(frames):
    arr = np.stack([f[:, :, ::-1] for f in frames]).astype(np.float32) / 255.0  # BGR -> RGB
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def _to_frames(tensor):
    arr = (tensor.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    return [np.ascontiguousarray(f[:, :, ::-1]) for f in arr]


def embed(frames, campaign_id):
    """Watermark BGR frames with the campaign's token. Returns new BGR frames."""
    msg = torch.from_numpy(campaign_bits(campaign_id)).float().unsqueeze(0)
    with torch.no_grad():
        out = _model().embed(_to_tensor(frames), msgs=msg, is_video=True)
    return _to_frames(out["imgs_w"])


def extract(frames):
    """Decoded bit message of a clip (aggregated over frames)."""
    with torch.no_grad():
        msg = _model().extract_message(_to_tensor(frames), aggregation="avg")
    return msg.squeeze(0).numpy().astype(bool)


def verify(frames, campaign_ids):
    """Best-matching campaign and its bit accuracy. campaign is None when nothing clears THRESHOLD."""
    bits = extract(frames)
    scores = {c: float((bits == campaign_bits(c, len(bits))).mean()) for c in campaign_ids}
    best = max(scores, key=scores.get)
    return {"campaign": best if scores[best] >= THRESHOLD else None, "bit_accuracy": scores[best], "scores": scores}
