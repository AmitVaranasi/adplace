# Module contract

Shared conventions for everyone working on this repo (Claude, GPT, GLM). The full spec is `SPEC.md`.

## Conventions
- Python 3, NumPy, OpenCV (cv2 4.11), SciPy. No new dependencies without asking.
- Frames are BGR `uint8` arrays of shape (H, W, 3), as returned by `cv2`.
- Masks are `bool` arrays of shape (H, W).
- A quad is a `float32` array of shape (4, 2) in pixel (x, y), ordered top-left, top-right, bottom-right, bottom-left
  **as seen on the surface** (so the creative is never mirrored). Missing quads are all-NaN.
- Creatives are RGBA `uint8` (H, W, 4) in RGB order (PIL convention); convert at the boundary.
- Functions are pure: no file I/O, no prints, no globals. Each module has tests under `tests/` that run with
  `python -m pytest tests/test_<module>.py -q` on synthetic data only (no downloads, no SAM, no network).
- Write code like a careful senior engineer: small functions, short docstrings, comments only where the why is
  not obvious.

## pipeline/geometry.py  (owner: GPT)
```python
def quad_from_mask(mask) -> np.ndarray | None
    """Fit one convex quad to a planar surface mask: boundary lines intersected into 4 corners, validated
    (area, convexity, corner order, aspect ratio). None if the mask is not quad-like."""

def propagate_quad(frames, ref_idx, ref_quad, surface_masks=None, occluder_masks=None) -> dict
    """Carry the canonical quad from frames[ref_idx] to every frame by frame-to-frame RANSAC homography on
    features from the surface and its coplanar surround (a band around the quad), excluding features on
    occluders. surface_masks (list of mask|None) are a cross-check, evaluated only on the unoccluded,
    in-frame part of the surface; the quad is never refit to the visible mask.
    Returns {"quads": (N,4,2) float32 with NaN where dropped,
             "confidence": (N,) float in [0,1],
             "inliers": (N,) int,
             "status": list[str] per frame: "ok" | "occluded" | "lost"}."""

def smooth_quads(quads, strength=0.5) -> np.ndarray
    """Conservative temporal smoothing of corner tracks without visible lag (e.g. zero-phase filtering over
    valid runs); NaN runs stay NaN."""
```

## pipeline/composite.py  (owner: GLM)
```python
def illumination_map(blank_frame, quad, size) -> np.ndarray
    """From a frame where the surface is blank (plain paper / screen off), rectify the quad to `size`=(w, h)
    and return a smooth, normalised (mean 1.0) float32 shading map of shape (h, w)."""

def composite_frame(frame, creative_rgba, quad, illum=None, occluder_mask=None, emissive=False,
                    blur_sigma=None, grain_sigma=None) -> np.ndarray
    """Warp the creative onto `quad` and blend into the BGR frame. Preserve the creative's own colours;
    multiply by `illum` (shading only) unless emissive. If blur_sigma/grain_sigma are None, estimate them from
    the frame around the quad. Pixels under occluder_mask keep the original frame. Anti-aliased edges.
    Returns a new frame; a NaN quad returns the frame unchanged."""

def estimate_blur_and_grain(frame, quad) -> tuple[float, float]
    """Estimate Gaussian blur sigma and noise sigma of the scene in a band around the quad."""

def render_clip(frames, quads, creative_rgba, illum=None, occluder_masks=None, emissive=False) -> list
    """composite_frame over a clip; blur/grain estimated once from the reference frame and reused."""
```

## pipeline/sam_track.py, pipeline/watermark.py, scripts/, app/  (owner: Claude)
SAM 3 analysis, Video Seal, footage, caching, integration and the demo app.
