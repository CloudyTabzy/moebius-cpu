"""PyTorch CPU baseline for the speed gate (PLAN §5): wall time of one UNet
step (batch 2, CFG) and the VAE passes at 512², random weights, fp32.

Needs ~3 GB of free RAM (fp32 weights + activations). Close other apps first:
it crashed with an access violation at 1.4 GB free.

    python tools/bench_torch_baseline.py [threads]
"""
import os, sys, time

import torch

threads = int(sys.argv[1]) if len(sys.argv) > 1 else 10
torch.set_num_threads(threads)
sys.path.insert(0, os.path.dirname(__file__))
from count_flops import UNet, m, AutoencoderKL  # noqa: E402  (same model construction)

import builtins
_print = builtins.print
builtins.print = lambda *a, **k: None
unet = UNet(**m).eval()
builtins.print = _print
vae = AutoencoderKL(block_out_channels=(128, 256, 512, 512), layers_per_block=2,
                    down_block_types=("DownEncoderBlock2D",) * 4,
                    up_block_types=("UpDecoderBlock2D",) * 4, latent_channels=4).eval()
emb = torch.nn.Embedding(20, 3072)


def timed(fn, n):
    fn()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t0) / n


with torch.no_grad():
    x = torch.randn(2, 9, 64, 64)
    ehs = emb(torch.arange(20).view(2, 10))
    t = torch.tensor([500])
    step = timed(lambda: unet(x, timestep=t, encoder_hidden_states=ehs), 5)
    img = torch.randn(2, 3, 512, 512)
    enc = timed(lambda: vae.encode(img).latent_dist.sample(), 2)
    z = torch.randn(1, 4, 64, 64)
    dec = timed(lambda: vae.decode(z), 2)

print(f"threads={threads}")
print(f"UNet step (B=2): {step * 1e3:7.0f} ms   x19 = {step * 19:5.1f} s")
print(f"VAE encode (B=2): {enc:5.2f} s   decode: {dec:5.2f} s")
print(f"Exact pipeline estimate: {step * 19 + enc + dec:5.1f} s")
