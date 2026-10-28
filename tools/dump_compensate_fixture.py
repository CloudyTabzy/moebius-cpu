"""Upstream reference for imaging::blend::paste(.., Paste::FeatherCompensate) (M8 Task 2 unit test).

    python tools/dump_compensate_fixture.py   # writes tests/fixtures/compensate.safetensors

Calls upstream `compensation_utils.paste_compensate(mask, image, result, fac=1.1)`
(Moebius/removal/v1_2, as `_post_process(paste=True, compensate=True)` does) on
synthetic cases ("a" is 96x80, "b" and "c" 48x40 to keep the file small).
Shapes are [h, w, 3] (image, result, out) and [h, w] (mask, 0/255); values are
exact bytes stored as f16 (exact up to 2048).

Cases: "a" mask in the interior (ring fully inside the image), "b" mask in the
top-left corner (ring clipped by the image edge), "c" two blobs plus a hole
(the ring excludes masked pixels inside the grown bbox).
"""
import os
import sys

import numpy as np
from PIL import Image
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.environ.get("MOEBIUS_UPSTREAM", os.path.join(HERE, "..", "..", "Moebius", "removal", "v1_2"))
sys.path.insert(0, os.path.normpath(UPSTREAM))
from compensation_utils import paste_compensate  # noqa: E402

OUT = os.path.join(HERE, "..", "tests", "fixtures", "compensate.safetensors")


def synth(seed, shift, W, H):
    """Smooth gradient + noise image, and a result with a per-channel colour offset."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    base = np.stack([xx * 2.0, yy * 3.0, (xx + yy) * 1.2], axis=-1) + 20
    img = np.clip(base + rng.normal(0, 6, (H, W, 3)), 0, 255).astype(np.uint8)
    res = np.clip(base[::-1] + np.array(shift) + rng.normal(0, 6, (H, W, 3)), 0, 255).astype(np.uint8)
    return img, res


masks = {}
m = np.zeros((80, 96), np.uint8)
m[25:50, 30:62] = 255
masks["a"] = m
m = np.zeros((40, 48), np.uint8)
m[0:8, 0:10] = 255
masks["b"] = m
m = np.zeros((40, 48), np.uint8)
m[8:16, 6:14] = 255
m[22:34, 26:42] = 255
m[26:29, 30:34] = 0
masks["c"] = m

shifts = {"a": (18, -12, 6), "b": (-25, 10, 30), "c": (40, 40, -35)}
t = {}
for i, (k, mask) in enumerate(masks.items()):
    H, W = mask.shape
    img, res = synth(10 + i, shifts[k], W, H)
    out = paste_compensate(Image.fromarray(mask), Image.fromarray(img), Image.fromarray(res), fac=1.1)
    t[f"{k}.image"] = img.astype(np.float16)
    t[f"{k}.result"] = res.astype(np.float16)
    t[f"{k}.mask"] = mask.astype(np.float16)
    t[f"{k}.out"] = np.asarray(out).astype(np.float16)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_file(t, OUT, metadata={"format": "moebius-cpu-fixture/1"})
print(f"wrote {os.path.normpath(OUT)} ({len(t)} tensors)")
