"""PyTorch reference outputs for the engine kernels (M2 parity tests).

    python tools/dump_kernel_fixtures.py   # writes tests/fixtures/kernels.safetensors

Activations are stored NHWC and weights in the moebius-cpu layouts, so this
file is also the executable spec of the converter's re-layout:
  1x1 / linear  [o, i, 1, 1] -> [o, i]
  dense 3x3     [o, i, 3, 3] -> [o, 3, 3, i]
  depthwise 3x3 [c, 1, 3, 3] -> [3, 3, c]
Channel counts are deliberately not multiples of 8 to exercise SIMD tails.
"""
import os

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

torch.manual_seed(1234)
OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "kernels.safetensors")
t = {}


def nhwc(x):  # NCHW -> NHWC
    return x.permute(0, 2, 3, 1).contiguous()


def w3x3(w):  # [o, i, 3, 3] -> [o, 3, 3, i]
    return w.permute(0, 2, 3, 1).contiguous()


def wdw(w):  # [c, 1, 3, 3] -> [3, 3, c]
    return w[:, 0].permute(1, 2, 0).contiguous()


# 1x1 conv with bias and residual: y = conv(x) + b + r
x = torch.randn(2, 20, 5, 7); w = torch.randn(36, 20, 1, 1) * 0.2; b = torch.randn(36); r = torch.randn(2, 36, 5, 7)
t |= {"c1.x": nhwc(x), "c1.w": w[:, :, 0, 0].contiguous(), "c1.b": b, "c1.r": nhwc(r),
      "c1.y": nhwc(F.conv2d(x, w, b)), "c1.yr": nhwc(F.conv2d(x, w, b) + r)}

# dense 3x3 convs
x = torch.randn(2, 13, 9, 11); w = torch.randn(22, 13, 3, 3) * 0.1; b = torch.randn(22)
t |= {"c3.x": nhwc(x), "c3.w": w3x3(w), "c3.b": b,
      "c3.same": nhwc(F.conv2d(x, w, b, padding=1)),                        # resnet / Upsample conv
      "c3.s2p1": nhwc(F.conv2d(x, w, b, stride=2, padding=1)),              # UNet Downsample2D
      "c3.s2vae": nhwc(F.conv2d(F.pad(x, (0, 1, 0, 1)), w, b, stride=2)),    # VAE Downsample2D
      "c3.up": nhwc(F.conv2d(F.interpolate(x, scale_factor=2.0, mode="nearest"), w, b, padding=1))}

# depthwise 3x3, the two fused epilogues
x = torch.randn(2, 26, 6, 9); w = torch.randn(26, 1, 3, 3) * 0.3; b = torch.randn(26)
y = F.conv2d(x, w, b, padding=1, groups=26)
a, g = y.chunk(2, dim=1)
t |= {"dw.x": nhwc(x), "dw.w": wdw(w), "dw.b": b,
      "dw.relu": nhwc(F.relu(y)), "dw.glu": nhwc(a * F.silu(g))}

# GroupNorm(32) with pre-norm per-channel add (resnet temb) and SiLU
x = torch.randn(2, 96, 5, 6) * 3 + 1; add = torch.randn(2, 96); gm = torch.randn(96); bt = torch.randn(96)
gn = F.group_norm(x + add[:, :, None, None], 32, gm, bt, eps=1e-5)
t |= {"gn.x": nhwc(x), "gn.add": add, "gn.gamma": gm, "gn.beta": bt,
      "gn.y": nhwc(F.group_norm(x, 32, gm, bt, eps=1e-6)),
      "gn.y_add_silu": nhwc(F.silu(gn))}

# LayerNorm over channels, tokens = pixels
x = torch.randn(3, 17, 44) * 2 - 0.5; gm = torch.randn(44); bt = torch.randn(44)
t |= {"ln.x": x, "ln.gamma": gm, "ln.beta": bt, "ln.y": F.layer_norm(x, (44,), gm, bt, eps=1e-5)}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_file({k: v.float().contiguous() for k, v in t.items()}, OUT, metadata={"format": "moebius-cpu-fixture/1"})
print(f"wrote {os.path.normpath(OUT)} ({len(t)} tensors)")
