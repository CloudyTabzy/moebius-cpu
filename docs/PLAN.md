# moebius-cpu: Design and Results

*Status: M0 and M2 complete (engine kernels parity-tested and benchmarked) · Date: 2026-10-28*

## 1. Goal

A native, **CPU-only** Rust inference engine for Moebius (0.22B latent-diffusion
inpainting, hustvl/Moebius, ECCV'26), shipped as a DLL with a C ABI for the GIMP
plugin, with a **measured and gated speed promise** (§5).

Non-goals: GPU backends, training, the PixelHacker teacher, batch inference,
non-x86 targets (x86-64 with AVX2 + FMA required: Haswell / Zen and newer).

## 2. Design choices, and why

| Choice | Why |
|---|---|
| Purpose-built engine on `gemm` + `rayon`, no ML framework | CPU only, one fixed network. Speed comes from layout, fusion and a static memory plan, which a general framework does not give on CPU (§6). |
| No einsum | All five λ contractions are batched matmuls, and two of them disappear after the folds in §6.4. |
| Reassociated 225-tap filter for the λ position conv | `Conv3d(1,k,(1,15,15))` with u=1 is a 2-D conv per value channel; folding `Q` in first cuts its FLOPs 2.5–10× (§6.4). |
| No `gather_nd` for `rel_pos_emb[n,m]` | Self-λ uses the local branch (r=15). Cross-λ indexes with an identity meshgrid. |
| Python → f16 safetensors, memory-mapped | Conversion also folds BatchNorms and re-lays weights. f16 halves resident memory. |
| Golden tests inject noise from files | Bit-matching torch's RNG isn't needed for parity, and isn't wanted for users. |
| candle not used | CPU path is NCHW with a layout copy per conv, GroupNorm in 5+ allocating passes, and grouped conv as a per-group loop. Its DDIM is off by one on `steps_offset` and the final alpha. |

## 3. Verified model facts

From `Moebius/` source, `moebius.yaml`, and the meta-device model (`tools/count_flops.py`):

- **UNet**: 226.0 M params. Three levels: 320 / 640 / 1280 channels at 64² / 32² / 16²
  latents. No mid block. 2 (down) / 3 (up) × (DWResnet, MixTransformer) per level.
  `attention_head_dim: 8` is the **head count** (diffusers naming quirk), so per-head width is C/8.
- **Square 512² only.** Cross-λ `rel_pos_emb` is a learned `(n², 10, k, 1)` table
  (n = 64/32/16), and the λ wrappers and MixFFN recover `H = W = sqrt(tokens)`.
  Upstream's resize-to-multiple-of-64 path would crash on non-square input.
- **Conditioning is constant.** `Embedding(20, 3072)` ids `10..20` (uncond) and `0..10`
  (cond) → `text_proj` 3072→768. Batch order is `[uncond, cond]`.
- **VAE**: `AutoencoderKL` f8d4 with SDXL-VAE weights (`hustvl/PixelHacker/vae`), 83.7 M params
  shipped in fp16, **scaling_factor 0.13025** (the SDXL value, not SD-1.x's 0.18215). Computed in f32.
- **Sampling**: DDIM, scaled_linear β 0.00085→0.012, `steps_offset=0`, final ᾱ = 1.
  20 steps at strength 0.99 run **19** steps from t=900, starting from
  `add_noise(z_image, noise, 900)` (ᾱ₉₀₀ = 0.0142, so 11.9% of the image signal), not pure noise.
  Noise offset 0.0357 per channel. CFG 2.5. Verified against diffusers in `src/scheduler.rs` tests.
- **Eps**: resnet GN 1e-5, transformer GN 1e-6, LayerNorm 1e-5, VAE GN 1e-6. All BatchNorms are eval-mode.
- `DWConv2d` = dw3×3 → BN+ReLU → pw1×1 → BN, **plus skip when c_in = c_out**.
- MixFFN = 1×1 (C→5C) + SiLU → dw3×3 + bias → GLU `x·SiLU(gate)` → 1×1 (2.5C→C, no bias).

## 4. Measured compute budget

Benchmark CPU: **i7-13620H** (6P + 4E, 16 threads, AVX2, no AVX-512).

### FLOPs (meta device, `tools/count_flops.py`)

| Stage | GFLOP | Breakdown |
|---|---|---|
| UNet step, batch 2 | 632 | 1×1 GEMM 62.5% · 3×3 dense 19.1% · λ pos-conv 14.0% · stride-2 1.2% · depthwise 0.5% |
| ×19 steps (Exact) | 12,008 | |
| VAE encode, batch 2 | 2,233 | 3×3 dense 90% · stride-2 5% |
| VAE decode | 2,515 | 3×3 dense 97% |
| **Exact total** | **≈16,760** | |

### Throughput (i7-13620H)

| Kernel | PyTorch 2.11 (MKL/oneDNN) | `gemm` 0.19 (`examples/gemm_probe.rs`) |
|---|---|---|
| GEMM 8192×320×1600 | 456–517 GFLOPS | 489–579 GFLOPS |
| GEMM 512×1280×6400 | 359–495 | 432–561 |
| 3×3 conv 128ch @256² (im2col tile) | 511–540 | 395–488 |
| 3×3 conv 512ch @64² | 412–555 | 339–485 |
| depthwise 3×3, 2×1600ch @64² | **19–24** (memory-bound) | own kernel (M2) |

`gemm` matches MKL. Throughput peaks at **10 threads = physical cores**; SMT adds nothing.

### Engine kernels as built (M2, `examples/kernel_bench.rs`, 10 threads)

| Kernel, Moebius shape | Result |
|---|---|
| 1×1 conv, f16 weights widened per call (MixFFN in/out, 64² and 16²) | 536–543 GFLOPS |
| 3×3 conv, Winograd F(2×2,3×3), VAE 128ch @256² / @512² | 767 / 751 GFLOPS effective |
| 3×3 conv, Winograd, VAE 512ch @64² | 1167 GFLOPS effective |
| 3×3 conv, Winograd + fused 2× upsample, UNet 1280ch 16²→32² B=2 | 1019 GFLOPS effective |
| 3×3 conv, im2col (stride 2 only) | ~350–400 GFLOPS (1% of UNet, 5% of encoder FLOPs) |
| depthwise 3×3 + GLU·SiLU, 2×64²×1600 | 3.3 ms (PyTorch 10 ms); DRAM-bound, saturates at 4–6 threads |
| GroupNorm + add + SiLU, 2×64²×320 / VAE 512²×128 | 0.83 ms / 10.2 ms |
| LayerNorm, 8192 tokens × 320 | 0.41 ms |

"Effective" = direct-convolution FLOPs ÷ time. Winograd does 2.25× fewer multiplies.
The budget below uses **520 GFLOPS** for 1×1, **~850 effective** for stride-1 3×3, **150 GFLOPS**
for the λ filter (M3, not yet built), and **0.10 s/step** for memory-bound ops.

`tools/bench_torch_baseline.py` measures the full PyTorch pipeline; it needs ~3 GB free RAM
and is still to run (the machine had 1.4 GB commit headroom during M0).

## 5. The speed promise

All targets are for one 512² inpaint on the i7-13620H, wall clock
from the `moebius_inpaint` call to return, with weights already loaded.

| Mode | Target | Projection (after M2) | What it is |
|---|---|---|---|
| **Exact** | **≤ 36 s** (stretch ≤ 30 s) | 21.7 s UNet (19 × 1.14 s) + 6.0 s VAE + 0.3 s ≈ **28 s** | Same pipeline as upstream. Parity-gated against PyTorch. |
| **Fast** | **≤ 12 s** (stretch ≤ 9 s) | **7.8–9.0 s** (by mask size) | Approximations below; quality-gated (§9). |
| Peak RSS | ≤ 1.5 GB | ≈ 1.4 GB | f16 weights (~620 MB) + static activation plan + ≤ 130 MB kernel scratch |
| Engine load | ≤ 1.5 s | mmap + constant precompute | |

The committed targets stay where M0 put them. M0 projected 35.9 s for Exact from GEMM speed
alone. Winograd brings the projection to ~28 s, but the λ layers (M3) and the assembled
UNet/VAE are not measured yet, so the margin stays in reserve until M7 measures end to end.
UNet step = 1×1 0.76 s + Winograd 3×3 0.12 s + stride-2 0.02 s + λ 0.14 s + memory-bound 0.10 s.
VAE = encode ~2.8 s (incl. stride-2 im2col) + decode ~3.0 s + norms ~0.2 s.

**Fast preset** (`src/preset.rs`; the values are starting points, M8 tunes them inside the 12 s budget):

| Lever | Saving | Effect |
|---|---|---|
| 7 steps (6 run, from t=710) | 19 → 6 UNet steps | Fewer denoising steps |
| CFG on the first 3 steps only | last 3 steps at batch 1 | Guidance matters most early |
| `single_encode`: one VAE encode (masked image) | −1.1 TFLOP (−2.3 s) | Initial noise built from masked latents (removes the 11.9% object leak, changes the trajectory) |
| `roi_decode`: decode mask bbox + 8 latents (64 px) | −25…−70% of decode | Decoder mid-attention sees the window. The window border is ≥64 px outside the mask, beyond the σ=3 feather, so it is never pasted. |

Fast projection: 3 × 1.14 + 3 × 0.58 = 5.2 s UNet, + 1.4 s encode, + 0.9–2.1 s decode, + 0.3 s.

**Gating rule:** a milestone isn't done until its time gate passes on the i7-13620H.
If Fast cannot meet both its time and quality gates, we report back instead of shipping it.

## 6. Engine design

### 6.1 Layout and memory
- **NHWC f32** activations. A 1×1 conv over P pixels is exactly `[P×Cin]·[Cin×Cout]`,
  and channel-wise kernels stream contiguous vectors.
- **Static memory plan.** Every shape is fixed (512² in, 64² latent), so all activation
  and scratch buffers are allocated once at `Engine::load`. Zero allocations per step.
- **Weights f16** (`src/weights.rs`, `engine/param.rs`). They are read from the memory-mapped file
  into f16 `Param`s at load. Each GEMM weight is widened to f32 (F16C) into a reusable panel
  (≤33 MB, the 1280→6400 layer) right before its GEMM; the 1×1 numbers in §4 include this.
  Winograd convs transform f16 weights directly. If f16 fails the parity gate, the fallback is
  f32 weights (1.24 GB, peak ≈ 2 GB).
- **Kernel scratch** (`engine::Scratch`, grown once): weight panel ≤33 MB, strided-conv patch
  band ≤32 MB, Winograd tiles / products / kernel ≤16 / 32 / 32 MB.

### 6.2 Kernels (`src/engine/`)
| Kernel | Approach |
|---|---|
All kernels in this table except the λ filter are built and parity-tested against PyTorch
(`tests/kernels.rs`, fixtures from `tools/dump_kernel_fixtures.py`, scaled error ≤ 1e-5).

| Kernel | Approach |
|---|---|
| GEMM | `gemm` crate (runtime AVX2/FMA/AVX-512 dispatch), `engine/gemm.rs` |
| 1×1 conv / linear | GEMM with bias / residual fused by pre-filling the output and accumulating (`engine/conv.rs`) |
| 3×3 conv, stride 1 | **Winograd F(2×2,3×3)** (`engine/winograd.rs`): 2.25× fewer multiplies. Banded over whole tile rows; 16 GEMMs run concurrently, one thread each (`examples/wino_probe.rs`). Nearest-2× upsample is fused into the tile gather. |
| 3×3 conv, stride 2 | Row-band implicit im2col (≤32 MB band) → GEMM. Never a full im2col. |
| depthwise 3×3 | NHWC, 32-channel register blocks × 2 pixels; row strips per task; epilogue fused: ReLU (DWConv2d) or GLU·SiLU (MixFFN) |
| GroupNorm | Per-channel vectorised stats → f64 per group; one apply pass with fused pre-norm add (resnet temb) and SiLU |
| LayerNorm, SiLU, add | Row-chunked, vectorised |
| exp / SiLU | Branch-free Cephes `exp` (≤3e-7 rel) so SiLU loops vectorise. libm `exp` would cost ~0.85 s/step |
| ISA dispatch | `multiversion!`: every hot loop is compiled for AVX2+FMA and baseline, selected at runtime |
| λ filter (M3) | Per-pixel 225-tap AXPY over the contiguous v-vector (§6.4) |

### 6.3 Threading
A dedicated rayon pool (not the host's global pool), one thread per physical core by default.
GEMM parallelism goes through the same pool. Hybrid P/E cores: M7 measures 6 vs 10 threads.

### 6.4 λ-layer algebra (details in `src/unet/lambda.rs`)
- **Self-λ:** `A[h,δ,n] = Σ_k Q W` then `Yp = Σ_δ A·V[n+δ] + Σ_k Q·b`.
  This replaces the Conv3d plus einsum: 88.5 → ~21 GFLOP/step, with no `[k·v·N]` intermediate.
- **Cross-λ:** `softmax(K)`, `V` and `λc` depend only on the fixed conditioning, so they are
  load-time constants per layer and CFG branch. Per step only `Q`, `Yc = Q·λc`, and the
  two-stage `B = Q·E`, `Yp = B·V` remain. Precomputing `λp = E·V` would cost ~0.8 GB, so it isn't done.

### 6.5 Exact-preserving savings
- CFG halves are identical until the first cross-λ: `conv_in`, the first resnet and the first
  self-λ run at batch 1.
- The time embedding and all 15 resnet `time_emb_proj` outputs are computed once per step.

## 7. Pipeline and plugin integration

Pipeline steps are in `src/pipeline.rs`. The **crop policy** (`src/imaging/crop.rs`, tested):
- Square window = mask long side + 50% margin each side, at least 512 px when the image allows (no upscaling of small objects).
- The window is **shifted, not clipped**, at image edges. The old plugin's clipped crops were non-square.
- Edge-replicated padding is used only when the image is smaller than the window.

**C ABI** (`include/moebius_cpu.h`, `src/ffi.rs`, implemented and tested):
- create / destroy / inpaint, in place on RGB(A) u8, returning the changed rect.
- Progress callback with cancel.
- Thread-local error string. Panics are caught at the boundary.
- The plugin loads the DLL at runtime (`LoadLibraryW`, Zig `std.DynLib`). An MSVC-built DLL
  exposing only C types needs no MinGW import library, and a missing DLL or unsupported CPU
  becomes a dialog instead of a failed plugin load.

## 8. Weights

`tools/convert_weights.py` (M1, spec in its docstring):
- strict name/shape check against `docs/weights-manifest.txt` (1,455 tensors)
- BatchNorm folding
- NHWC-friendly re-layout
- f16 with an f32 fallback per tensor
- metadata `format = moebius-cpu/1`

Sources: HF `hustvl/Moebius` (`ft_places2` for general scenes; `ft_celebahq` / `ft_ffhq` for
portraits) and `hustvl/PixelHacker/vae`. All Apache-2.0.

Local layout (`weights/` is git-ignored), mirroring the HF repos:

```
weights/
  upstream/<variant>/diffusion_pytorch_model.bin   UNet checkpoints as downloaded
  upstream/vae/{config.json, diffusion_pytorch_model.bin}
  <variant>/unet.safetensors, vae.safetensors      converter output, one dir per variant
```

UNet checkpoint SHA-256 (HF LFS oids, 2026-10-28). The converter records the hash in metadata:

| Variant | SHA-256 | Local |
|---|---|---|
| `pretrained` | `6c9ee08ce467af48767b74416646d82f8a0c1dcfb558c478e635cef0ca8b2ad2` | |
| `ft_places2` | `6525afb888e55f9b5c74fa0a5d19ca0762d720d6c716fb0f8422fbeb6868a09a` | ✅ validated: 1,203 tensors match the manifest (default for the plugin) |
| `ft_celebahq` | `9d26630d91cf9748ed43541669197cf5d63fbb28793d0c1c5dc7c07028ad9045` | |
| `ft_ffhq` | `ddd5b81b3034dea2bb380c574517ebbf24957903d99319fa8bbb8e69ca9d5606` | ✅ validated: 1,203 tensors match the manifest |

VAE `diffusion_pytorch_model.bin` (fp16): `a59d7ea697f2942d22002dc3469e8c53db807a6b78f7f5ec03bd4c1f70f98efe`.
✅ Validated: 248 tensors match the manifest. f16 storage is lossless.

f16 check on `ft_ffhq` and `ft_places2`: only one tensor exceeds the f16 range,
`down_blocks.0…attn2.norm_v.running_var` (9.9e4), a BatchNorm statistic that is folded away.

## 9. Verification

| Gate | Criterion |
|---|---|
| Kernels (M2) ✅ | PyTorch fixtures + naive f64 references. 1×1 ≥ 450 GFLOPS, 3×3 ≥ 420 GFLOPS. Memory-bound kernels: total ≤ 0.10 s per UNet step* |
| Blocks (M3–M6) | Golden tensors from `tools/dump_golden.py` with fixed noise. Rel L2 ≤ 1e-4 with f32 weights, ≤ 2e-3 with f16 |
| Exact e2e (M7) | Same noise as PyTorch: PSNR ≥ 45 dB (f32 weights) / ≥ 30 dB and LPIPS ≤ 0.02 (f16) inside the mask. ≤ 36 s. RSS ≤ 1.5 GB |
| ROI decode (M8) | Same latents, ROI vs full decode: PSNR ≥ 40 dB inside the mask |
| Fast preset (M8) | LPIPS-to-ground-truth ≤ 1.05 × Exact's on ≥ 100 image/mask pairs, plus visual review. ≤ 12 s |
| Speed regressions | `examples/kernel_bench.rs` now; per-stage timings recorded per milestone |

\* M0 set the depthwise gate as "≥ 3× PyTorch". Measurement showed the kernel is DRAM-bound
(it saturates at 4–6 threads), so its speed reflects memory bandwidth, not code quality. It
measures 2.7–3.2× PyTorch run to run. The gate was replaced by the per-step budget the promise
actually depends on: all depthwise convs of a UNet step ≈ 45 ms.

## 10. Milestones

| # | Deliverable | Gate |
|---|---|---|
| **M0** ✅ | Measurements, crate scaffold, scheduler, weights loader, crop, C ABI | 14 tests pass; `gemm` ≈ MKL |
| M1 | Converter + golden dumps (download weights ~1.2 GB) | Manifest match; golden set written |
| **M2** ✅ | conv1x1, Winograd 3×3 (+upsample), im2col strided 3×3, depthwise, GroupNorm, LayerNorm, SiLU | §9 kernel gates pass (§4 table) |
| M3 | Self-λ and cross-λ (folded) | Golden parity; ≤ 0.15 s/step for all λ |
| M4 | MixTransformer + MixFFN | Golden parity |
| M5 | UNet + embeddings | Parity; ≤ 1.35 s/step at batch 2 |
| M6 | VAE encoder / decoder | Parity; encode ≤ 4.8 s (B=2), decode ≤ 5.4 s |
| M7 | Pipeline, resize, blend, CLI | Exact e2e gate (§9), ≤ 36 s |
| M8 | Fast preset tuning (+ eval tooling) | Fast gates (§9), ≤ 12 s |
| M9 | Plugin hookup, packaging, cancel/progress UX | GIMP smoke test on benchmark CPU |

## 11. Risks

| Risk | Mitigation |
|---|---|
| f16 weights miss parity | Per-tensor f32 fallback in the converter; global f32 mode (+0.6 GB) |
| CPU thermal throttling over 35 s | Gate on sustained runs (3 back-to-back), not best-of |
| E-core stragglers in GEMM | Measure 6 vs 10 threads (M7); pin to P-core count if better |
| RAM pressure inside GIMP | f16 + mmap + static plan; RSS gate |
| Fast quality gate fails | Re-tune steps / CFG cutoff / ROI margin within 12 s, or report back |
| Exact needs to be faster | INT8 W8A8 1×1 GEMMs via AVX-VNNI (≈1.5–2× on 62% of UNet FLOPs), behind its own quality gate, after M9 |

## 12. Dependencies

| Crate | Use |
|---|---|
| `gemm` 0.19 | All dense math |
| `rayon` 1.11 | Engine thread pool |
| `safetensors` 0.8, `memmap2` 0.9, `half` 2.7, `bytemuck` 1 | f16 weights, mmap, zero-copy views |
| `fast_image_resize` 6.1 | Lanczos3 window ↔ 512² |
| `rand` 0.10, `rand_distr` 0.6 | Initial noise |
| `num_cpus` 1.17 | Physical-core pool size |
| `thiserror` 2 | Errors |
| `image` 0.25, `clap` 4 (feature `cli`) | Developer CLI only |

Not used: `imageproc` (the σ=3 feather is a 20-line separable blur),
`serde`/`serde_json` (no runtime config), `anyhow`.
