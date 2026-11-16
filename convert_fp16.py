"""Shrink nafnet_ft.pth (fp32, ~272 MB) to nafnet_ft_fp16.pth (~136 MB) and check the loss.

Run from this folder:
    python convert_fp16.py                 # test on a built-in synthetic image
    python convert_fp16.py my_photo.jpg    # or test on your own photo (better)

The check restores a JPEG-damaged image with both the fp32 and fp16 weights and reports the PSNR
between the two restored images. >45 dB means they are visually identical.
(Do NOT test with random noise: the network has never seen such input and the outputs blow up.)
"""
import io
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

from model import build_model

SRC, DST = "nafnet_ft.pth", "nafnet_ft_fp16.pth"

ck = torch.load(SRC, map_location="cpu")
sd = ck.get("params", ck)
torch.save({"params": {k: v.half() for k, v in sd.items()}}, DST)
print(f"saved {DST}: {os.path.getsize(DST) / 1e6:.0f} MB (from {os.path.getsize(SRC) / 1e6:.0f} MB)")

m32 = build_model(); m32.load_state_dict(sd, strict=True); m32.eval()
sd16 = {k: v.float() for k, v in torch.load(DST, map_location="cpu")["params"].items()}
m16 = build_model(); m16.load_state_dict(sd16, strict=True); m16.eval()

# --- realistic test image: a real photo if given, else smooth gradients + shapes + mild texture
if len(sys.argv) > 1:
    img = Image.open(sys.argv[1]).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((max(64, int(w * s)), max(64, int(h * s))), Image.LANCZOS)
else:
    yy, xx = np.mgrid[0:192, 0:192].astype(np.float32) / 192
    a = np.stack([0.2 + 0.6 * xx, 0.3 + 0.4 * yy, 0.5 + 0.3 * np.sin(6 * xx) * np.cos(5 * yy)], -1)
    a[(xx - 0.5) ** 2 + (yy - 0.5) ** 2 < 0.06] = [0.9, 0.2, 0.2]
    a[40:90, 100:170] = [0.1, 0.6, 0.3]
    a += np.random.default_rng(0).normal(0, 0.02, a.shape)
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
buf = io.BytesIO(); img.save(buf, "JPEG", quality=15); buf.seek(0)
x = torch.from_numpy(np.array(Image.open(buf).convert("RGB"))).permute(2, 0, 1)[None].float() / 255

with torch.inference_mode():
    a = m32(x).clamp(0, 1)
    b = m16(x).clamp(0, 1)
mse = ((a - b) ** 2).mean().item()
psnr = float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)
print(f"fp16 vs fp32 restored images: {psnr:.1f} dB PSNR (>45 dB = visually identical)")
