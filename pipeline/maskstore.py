"""Compact per-frame mask storage for long clips: box-local, bit-packed, expanded one frame at a time.

A 588-frame 720p clip as a dense (N, H, W) bool stack is 540 MB per object; SAM masks are small relative to the
frame, so cropping to each frame's box and packing bits brings that to a few MB, and readers never hold more than
one full-size frame mask at once.
"""
import numpy as np


class MaskWriter:
    def __init__(self, n_frames, height, width):
        self.n, self.h, self.w = int(n_frames), int(height), int(width)
        self.boxes = np.zeros((self.n, 4), np.int32)  # x0, y0, x1, y1 (half-open); x1 == x0 means empty
        self.chunks = [b""] * self.n

    def add(self, index, mask=None, box=None, local=None):
        """Store frame `index` from a full (H, W) mask, or from a box-local raster `local` at box (x0, y0). ORs into
        whatever the frame already holds, so several objects can be merged into one store."""
        if mask is not None:
            mask = np.asarray(mask, bool)
            ys, xs = np.nonzero(mask)
            if not len(ys):
                return
            x0, y0 = int(xs.min()), int(ys.min())
            local = mask[y0:ys.max() + 1, x0:xs.max() + 1]
        else:
            local = np.asarray(local, bool)
            x0, y0 = box
            if not local.any():
                return
        x1, y1 = x0 + local.shape[1], y0 + local.shape[0]
        if self.boxes[index, 2] > self.boxes[index, 0]:  # merge with the existing box
            old = self.frame(index)
            old[y0:y1, x0:x1] |= local
            self.chunks[index] = b""
            self.boxes[index] = 0
            return self.add(index, mask=old)
        self.boxes[index] = (x0, y0, x1, y1)
        self.chunks[index] = np.packbits(local, axis=None).tobytes()

    def frame(self, index):
        return _unpack(self.boxes[index], self.chunks[index], self.h, self.w)

    def save(self, path):
        lengths = np.array([len(c) for c in self.chunks], np.int64)
        np.savez_compressed(path, shape=np.array([self.n, self.h, self.w]), boxes=self.boxes,
                            offsets=np.concatenate([[0], np.cumsum(lengths)]),
                            data=np.frombuffer(b"".join(self.chunks), np.uint8))


class MaskStore:
    """Read side: len(store), store[i] -> (H, W) bool, iteration, store.box(i), store.any(i)."""

    def __init__(self, path):
        z = np.load(path)
        self.n, self.h, self.w = (int(v) for v in z["shape"])
        self.boxes, self.offsets, self.data = z["boxes"], z["offsets"], z["data"]

    def __len__(self):
        return self.n

    def any(self, index):
        return bool(self.boxes[index, 2] > self.boxes[index, 0])

    def box(self, index):
        return tuple(int(v) for v in self.boxes[index])

    def __getitem__(self, index):
        if index < 0:
            index += self.n
        lo, hi = self.offsets[index], self.offsets[index + 1]
        return _unpack(self.boxes[index], self.data[lo:hi].tobytes(), self.h, self.w)

    def __iter__(self):
        return (self[i] for i in range(self.n))


def _unpack(box, chunk, h, w):
    out = np.zeros((h, w), bool)
    x0, y0, x1, y1 = (int(v) for v in box)
    if x1 > x0 and y1 > y0:
        bits = np.unpackbits(np.frombuffer(chunk, np.uint8), count=(x1 - x0) * (y1 - y0))
        out[y0:y1, x0:x1] = bits.reshape(y1 - y0, x1 - x0).astype(bool)
    return out
