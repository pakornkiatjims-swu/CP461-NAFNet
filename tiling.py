"""Tile-based inference helpers (pure numpy, no torch) so large images fit in limited RAM."""
import math
import numpy as np


def _ramp(n, lo, hi, ov):
    """1-D blend weights: fade in over `ov` px if the tile has a neighbour before it,
    fade out if it has one after it, flat 1.0 otherwise (image borders)."""
    r = np.ones(n, np.float32)
    if lo and ov > 0:
        r[:ov] = np.linspace(0, 1, ov + 2, dtype=np.float32)[1:-1]
    if hi and ov > 0:
        r[-ov:] = np.minimum(r[-ov:], np.linspace(1, 0, ov + 2, dtype=np.float32)[1:-1])
    return r


def _starts(length, tile, stride):
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, stride))
    if s[-1] + tile < length:
        s.append(length - tile)
    return s


def tile_restore(img, run, tile=384, overlap=32, progress=None):
    """img: float32 HxWx3 in [0,1]. run: fn(HxWx3 float32) -> HxWx3 float32.
    Overlapping tiles are blended with linear ramps."""
    h, w, _ = img.shape
    th, tw = min(tile, h), min(tile, w)
    if th == h and tw == w:
        out = run(img)
        if progress:
            progress(1, 1)
        return out
    stride = max(tile - overlap, 1)
    ys, xs = _starts(h, th, stride), _starts(w, tw, stride)
    acc = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w, 1), np.float32)
    total, done = len(ys) * len(xs), 0
    for y in ys:
        for x in xs:
            out = run(img[y:y + th, x:x + tw])
            wy = _ramp(th, y > 0, y + th < h, overlap)
            wx = _ramp(tw, x > 0, x + tw < w, overlap)
            wgt = (wy[:, None] * wx[None, :])[..., None]
            acc[y:y + th, x:x + tw] += out * wgt
            wsum[y:y + th, x:x + tw] += wgt
            done += 1
            if progress:
                progress(done, total)
    return acc / np.maximum(wsum, 1e-8)


def psnr_u8(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
