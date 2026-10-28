"""Pillow reference for imaging::blend::pil_gaussian_blur (Task 7 unit test).

    python tools/dump_blur_fixture.py   # writes tests/fixtures/blur.safetensors

Each case is a binary 0/255 mask and `mask.convert('RGB').filter(GaussianBlur(3))`
(one channel; the three are identical), as upstream `_post_process` does.
Shapes are [h, w] f32 holding exact byte values.
"""
import os

import numpy as np
from PIL import Image, ImageFilter
from safetensors.numpy import save_file

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "blur.safetensors")


def blur(mask):
    img = Image.fromarray(mask).convert("RGB").filter(ImageFilter.GaussianBlur(radius=3))
    return np.asarray(img)[:, :, 0].copy()


cases = {}

# 64x48: a rectangle, an isolated pixel, and mass touching the image borders.
m = np.zeros((48, 64), np.uint8)
m[10:25, 20:40] = 255
m[40, 10] = 255          # isolated pixel
m[0:6, 0:9] = 255        # top-left corner
m[44:48, 55:64] = 255    # bottom-right corner
m[20:23, 63] = 255       # right border
cases["a"] = m

# 37x23 (odd, smaller than a few box widths) random speckle.
rng = np.random.default_rng(7)
cases["b"] = ((rng.random((23, 37)) > 0.6) * 255).astype(np.uint8)

# Tiny image: box radius exceeds the extent.
cases["c"] = np.array([[0, 255, 0, 0, 255], [255, 0, 0, 0, 0], [0, 0, 255, 255, 0]], np.uint8)

t = {}
for k, mask in cases.items():
    t[f"{k}.mask"] = mask.astype(np.float32)
    t[f"{k}.blurred"] = blur(mask).astype(np.float32)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_file(t, OUT, metadata={"format": "moebius-cpu-fixture/1"})
print(f"wrote {os.path.normpath(OUT)} ({len(t)} tensors)")
