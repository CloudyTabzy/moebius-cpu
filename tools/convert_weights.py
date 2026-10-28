"""Convert upstream checkpoints to moebius-cpu safetensors (M1).

    python tools/convert_weights.py \
        --unet  weights/upstream/ft_ffhq/diffusion_pytorch_model.bin \
        --vae   weights/upstream/vae \
        --out   weights/ft_ffhq/

Writes `unet.safetensors` and `vae.safetensors` with metadata
`format = moebius-cpu/1` (src/weights.rs::FORMAT_VERSION), plus the source
checkpoint's SHA-256 and the VAE's scaling_factor.

Planned transformation (M1) — the Rust side relies on every item:
  1. torch.load(map_location="cpu", weights_only=True); check every name and
     shape against docs/weights-manifest.txt (strict, like upstream).
  2. Fold eval BatchNorms into the preceding conv (w' = w·γ/σ, b' = β − μ·γ/σ):
       DWConv2d bn1 → conv_dw, bn2 → conv_pw; λ norm_q → to_q, norm_v → to_v
       (cross-λ norm_v is BatchNorm1d over the 10-token sequence).
     Drop running stats and num_batches_tracked.
  3. Reshape 1x1 conv weights [o, i, 1, 1] → [o, i]; depthwise [c, 1, 3, 3] →
     [3, 3, c] (NHWC tap-major); dense 3x3 [o, i, 3, 3] → [o, 3, 3, i] so a
     row-band patch row is contiguous in i; self-λ pos_conv [k, 1, 1, 15, 15]
     → [225, k] for the A = Q·W fold.
  4. Store f16 except where |w| > 65504 or relative f16 error > 1e-3 for the
     tensor (kept f32); norms, biases and embeddings stay f32.
  5. Do NOT precompute cross-λ constants here: they are cheap at load and
     keeping raw tables keeps the format checkpoint-agnostic.

Golden tensors for parity tests come from tools/dump_golden.py (M1), which
runs the PyTorch model on data/images + data/masks with fixed noise.
"""
raise SystemExit("convert_weights.py: implemented in milestone M1 (see docstring)")
