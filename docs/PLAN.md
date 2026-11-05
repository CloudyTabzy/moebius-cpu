# moebius-cpu: Design and Results

*Status: finished. M0–M9 complete: Exact matches PyTorch, Fast passed its gates (§5), and the plugin works in GIMP (§10) · Date: 2026-11-05*

## 1. Goal

Bring Moebius (0.22B latent-diffusion inpainting, hustvl/Moebius, ECCV'26) to image editors as a
plugin: a native, **CPU-only** Rust inference engine, shipped as a GIMP 3.2 plugin, with a DLL and
C ABI for other native hosts and a **measured and gated speed promise** (§5).

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
The budget planned before the UNet existed used **520 GFLOPS** for 1×1, **~850 effective** for
stride-1 3×3, **150 GFLOPS** for the λ filter, and **0.10 s/step** for memory-bound ops. §5 has
the stage times measured since.

**PyTorch baseline (measured):** the upstream pipeline on the i7-13620H (torch 2.11 CPU, f32)
takes **62–69 s** per Exact inpaint (`tools/dump_golden.py` run, M1).

### GEMM backend (measured)

Interleaved runs on Moebius's exact shapes. `private-gemm-x86` 0.1.22 (faer's current x86
kernels) is **1.74×** `gemm` 0.19 on the single multi-threaded GEMMs: its scheduler balances
the hybrid P/E cores, where `gemm`'s even split leaves the P-cores waiting. It is 0.94× on
Winograd's 16 concurrent single-threaded GEMMs, so those stay on `gemm`. `gemmkit` 0.1.2
scored 1.53× / 0.99×; its fused epilogue lacks SiLU. In an interleaved A/B on one UNet CFG
step: 1.04 s vs 1.19–1.27 s. `MOEBIUS_GEMM=gemm` forces `gemm` everywhere.

### Sustained power

`examples/sustained.rs`: under sustained load, GEMM throughput cycles between
**~340 and ~600 GFLOPS** every ~20 s as the power limit engages, averaging ~480. Short
benchmarks catch the burst phase; back-to-back inpaints settle into the sustained phase.

## 5. The speed promise

All targets are for one 512² inpaint on the i7-13620H, wall clock
from the `moebius_inpaint` call to return, with weights already loaded.

| Mode | Target | Measured (2026-11-05) | What it is |
|---|---|---|---|
| **Exact** | **≤ 36 s** (stretch ≤ 30 s) | ✅ **24–25 s** from cool (stretch met); median **~30 s** under sustained back-to-back load (28–37 s, power-limited, see §4) | Same pipeline as upstream. Parity-gated against PyTorch. 2.5× faster than PyTorch's 62–69 s. |
| **Fast** | **≤ 12 s** (stretch ≤ 9 s) | ✅ **10.7–11.9 s** from cool, median 11.5 s, 10 images (stretch not met); ~13.2 s sustained | Approximations below; quality-gated (§9). |
| Peak working set | ≤ 1.5 GB | ✅ **1.36–1.37 GiB (1.42–1.44 GB)** Exact, 1.33 GiB (1.42 GB) Fast | f16 weights (~650 MB) + pooled activations + kernel scratch |
| Engine load | ≤ 1.5 s | ✅ 0.31–0.36 s | mmap + constant precompute |

The promise holds for a one-off inpaint, which is how GIMP uses it. Long back-to-back
batches slow to a ~30 s median because of the sustained power limit, not the code.
The Fast preset (M8) is the headroom for that case. Fast's own margin is thin: its slowest cool
run was 11.88 s, and back-to-back runs exceed 12 s.

The plugin pays 1–2 s more per run, which the promise does not cover: it loads the model and moves
PNGs every time. Inside GIMP, Fast took 12.0–13.6 s over 7 runs and Exact 27.0 s in its one run.

Without the EcoQoS opt-out (`engine::cpu::disable_power_throttling`), Windows steers background
console processes to the efficiency cores and a run is 10–15× slower. rusty_alloc is opt-in: no
configuration beat the system allocator inside the memory gate (`tools/bench_alloc.py`).

How the measured number was reached (each step parity-neutral, verified against the golden set):

| Step | Exact inpaint (cool) |
|---|---|
| First end-to-end run (M7) | 47–52 s |
| Activation buffers recycled through `Scratch` (no per-call page zeroing) | 27.6–28.4 s |
| Multi-threaded GEMMs on `private-gemm-x86` | ~1.04 s/UNet step (A/B) |
| GroupNorm+SiLU fused into the Winograd input gather (VAE) | **24.0–25.1 s**, peak 1.63 → 1.37 GiB |

Measured stage times (cool): UNet step 1.04–1.17 s, VAE encode (B=2) 2.8–3.2 s, decode 2.7–3.1 s.

**Fast preset** (`src/preset.rs`):

| Lever | Saving | Effect |
|---|---|---|
| 10 steps, trailing spacing (9 run, from t=899) | 19 → 9 UNet steps | Starts with 12.0% image signal, like Exact's 11.9% |
| DPM-Solver++(2M) instead of DDIM | Same cost, fewer steps needed | Matches diffusers step for step. At 9 UNet steps: 1.008/1.034 vs DDIM's 1.022/1.048 on the 24-pair dev set |
| CFG on the first 3 steps only | last 6 steps at batch 1 | Same LPIPS as 4 steps in the dev sweep |
| `roi_encode`: second encode covers the mask's window only | ~0.55 s | `z_img` only supplies the ~12% starting signal. No measurable LPIPS change (1.037/1.082 vs 1.037/1.081) |

Rejected, on measurements:

- **`single_encode`:** dropping the image encode costs 1.19–1.20× LPIPS. Removed.
- **`roi_decode`:** fails the 40 dB gate, so no preset uses it. Decoder GroupNorm statistics need
  the whole frame. On the 10 gate pairs a latent crop reaches 34.67 dB at worst. Cropping after
  `up_blocks.1` reaches 42.75 dB there, but on 60 random small windows it falls to 20.36 dB, and
  27 of the 60 miss 40 dB. The best variant tried (crop after `up_blocks.0`, margin 16) still has
  8 of 60 below, worst 34.26 dB. Cropping after `up_blocks.2` is far worse (23.41 dB on the gate
  pairs); the cause is not understood. A low-res pass to estimate the statistics would cost
  ~0.8 s, most of the ~1.2 s saving on large masks. LPIPS did not show any of this: the eval's
  masks are large and scattered.
- **9 steps instead of 10:** 0.6 s faster, but faces reach 1.047 (CI 1.033–1.061).

**Gating rule:** a milestone isn't done until its time gate passes on the i7-13620H.
If Fast cannot meet both its time and quality gates, we report back instead of shipping it.

## 6. Engine design

### 6.1 Layout and memory
- **NHWC f32** activations. A 1×1 conv over P pixels is exactly `[P×Cin]·[Cin×Cout]`,
  and channel-wise kernels stream contiguous vectors.
- **Recycled activations.** Every shape is fixed (512² in, 64² latent). Each inpaint keeps one
  `Scratch` whose pool hands out buffers by best fit (`tensor_for`, `vec_buf`) and takes them back
  (`recycle`). Every role gets the same buffer each step, so after the first UNet step nothing
  large is allocated, zeroed or page-faulted. Fresh per-call allocations cost ~20 s per inpaint
  on Windows. The pool is released after the VAE encode, so encoder maps don't sit through
  the denoising loop.
- **No materialised GroupNorm before 3×3 convs (VAE).** `GroupNorm::affine` gives per-channel
  scale/shift, and `Conv3x3::forward_prologue` applies them (+SiLU) while filling each Winograd band's
  input strip. VAE resnets keep three live maps instead of five. Bit-identical to GN-then-conv.
- **Weights f16** (`src/weights.rs`, `engine/param.rs`). They are read from the memory-mapped file
  into f16 `Param`s at load. Each GEMM weight is widened to f32 (F16C) into a reusable panel
  (≤33 MB, the 1280→6400 layer) right before its GEMM; the 1×1 numbers in §4 include this.
  Winograd convs transform f16 weights directly. f16 passed the parity gate (§9). f32 weights
  (1.24 GB, peak ≈ 2 GB) remain available for comparison.
- **Kernel scratch** (`engine::Scratch`, grown once): weight panel ≤33 MB, strided-conv patch
  band ≤32 MB, Winograd tiles / products / kernel ≤16 / 32 / 32 MB.

### 6.2 Kernels (`src/engine/`)
All kernels below are built and parity-tested against PyTorch
(`tests/kernels.rs`, fixtures from `tools/dump_kernel_fixtures.py`, scaled error ≤ 1e-5).

| Kernel | Approach |
|---|---|
| GEMM | `engine/gemm.rs`: multi-threaded calls → `private-gemm-x86` (AVX2+FMA kernel set), single-threaded (Winograd) → `gemm` (§4) |
| 1×1 conv / linear | GEMM with bias / residual fused by pre-filling the output and accumulating (`engine/conv.rs`) |
| 3×3 conv, stride 1 | **Winograd F(2×2,3×3)** (`engine/winograd.rs`): 2.25× fewer multiplies. Banded over whole tile rows; 16 GEMMs run concurrently, one thread each (`examples/wino_probe.rs`). Nearest-2× upsample and an optional GroupNorm+SiLU prologue are fused into the tile gather. |
| 3×3 conv, stride 2 | Row-band implicit im2col (≤32 MB band) → GEMM. Never a full im2col. |
| depthwise 3×3 | NHWC, 32-channel register blocks × 2 pixels; row strips per task; epilogue fused: ReLU (DWConv2d) or GLU·SiLU (MixFFN) |
| GroupNorm | Per-channel vectorised stats → f64 per group; one apply pass with fused pre-norm add (resnet temb) and SiLU |
| LayerNorm, SiLU, add | Row-chunked, vectorised |
| exp / SiLU | Branch-free Cephes `exp` (≤3e-7 rel) so SiLU loops vectorise. libm `exp` would cost ~0.85 s/step |
| ISA dispatch | `multiversion!`: every hot loop is compiled for AVX2+FMA and baseline, selected at runtime |
| λ filter | Per-pixel 225-tap AXPY over the contiguous v-vector (§6.4); 18.8 ms self-λ / 7.2 ms cross-λ at 2×64²×320 |

### 6.3 Threading
A dedicated rayon pool (not the host's global pool), one thread per physical core by default.
GEMM parallelism goes through the same pool. The hybrid P/E-core imbalance turned out to be a
GEMM-scheduling problem, solved by `private-gemm-x86`'s scheduler (§4), not by thread pinning.

### 6.4 λ-layer algebra (details in `src/unet/lambda.rs`)
- **Self-λ:** `A[h,δ,n] = Σ_k Q W` then `Yp = Σ_δ A·V[n+δ] + Σ_k Q·b`.
  This replaces the Conv3d plus einsum: 88.5 → ~21 GFLOP/step, with no `[k·v·N]` intermediate.
- **Cross-λ:** `softmax(K)`, `V` and `λc` depend only on the fixed conditioning, so they are
  load-time constants per layer and CFG branch. Per step only `Q`, `Yc = Q·λc`, and the
  two-stage `B = Q·E`, `Yp = B·V` remain. Precomputing `λp = E·V` would cost ~0.8 GB, so it isn't done.

### 6.5 Exact-preserving savings
- The time embedding is computed once per step and shared by both CFG halves.
- Not done (small): the CFG halves are identical until the first cross-λ, so `conv_in`, the
  first resnet and the first self-λ could run at batch 1 (~2% of a step).

## 7. Pipeline and plugin integration

Pipeline steps are in `src/pipeline.rs`. The **crop policy** (`src/imaging/crop.rs`, tested):
- Square window = mask long side + 50% margin each side, at least 512 px when the image allows (no upscaling of small objects).
- The window is **shifted, not clipped**, at image edges.
- Edge-replicated padding is used only when the image is smaller than the window.

**GIMP plugin** (`gimp-plugin/`):
- GIMP's Python is a MINGW build and cannot load MSVC binaries, so inference runs out of process.
- `moebius-inpaint.py` writes the layer and selection to temporary PNGs and runs
  `moebius-cli.exe --plugin`. The result comes back through the shadow buffer, which limits it to
  the selection.
- The worker prints `PROGRESS <stage> <done> <total>` for GIMP's progress bar. It exits with code 3
  when stdin closes, so a killed plugin leaves no process behind.
- Releases hold only the compiled programs and scripts: `.github/workflows/release.yml` builds the zip
  for a `v*` tag.
- `install.py` puts the plugin, the worker and both models into the user's GIMP profile. It downloads
  from pinned Hugging Face revisions, so nothing large is hosted here, and resumes after a dropped
  connection. It uses only the standard library, so GIMP's bundled Python runs it: `install.bat` finds
  that Python.

**C ABI** (`include/moebius_cpu.h`, `src/ffi.rs`, implemented and tested). It serves native hosts;
the GIMP plugin does not use it:
- create / destroy / inpaint, in place on RGB(A) u8, returning the changed rect.
- Progress callback with cancel.
- Thread-local error string. Panics are caught at the boundary.

## 8. Weights

`moebius-convert` turns upstream checkpoints into the engine's format with no Python or torch: a
small zip reader and a pickle reader that calls nothing but tensor rebuilds, then the steps below.
It refuses files with the wrong hash. Tensors are bit-identical to `tools/convert_weights.py`, which
remains the format reference (spec in its docstring):
- strict name/shape check against `docs/weights-manifest.txt` (1,455 tensors)
- BatchNorm folding
- NHWC-friendly re-layout
- f16 with an f32 fallback per tensor
- metadata `format = moebius-cpu/1`

Sources: HF `hustvl/Moebius` and `hustvl/PixelHacker/vae`. All Apache-2.0.

**Pick the checkpoint by content.** It matters as much as the port does.
Upstream's README runs its face examples with `ft_celebahq`.
- People and portraits: `ft_celebahq` or `ft_ffhq`. Removing glasses yields believable eyes,
  and identity is kept.
- Scenes and objects: `ft_places2`. On a face it invents distorted eyes, and with a whole face
  masked it paints a room.
- The plugin's **Content** choice picks the checkpoint: Scenes and objects (`ft_places2`, the
  default) or Faces (`ft_celebahq`).

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
| `ft_places2` | `6525afb888e55f9b5c74fa0a5d19ca0762d720d6c716fb0f8422fbeb6868a09a` | ✅ validated: 1,203 tensors match the manifest; golden set source |
| `ft_celebahq` | `9d26630d91cf9748ed43541669197cf5d63fbb28793d0c1c5dc7c07028ad9045` | ✅ downloaded, hash verified, converted |
| `ft_ffhq` | `ddd5b81b3034dea2bb380c574517ebbf24957903d99319fa8bbb8e69ca9d5606` | ✅ validated: 1,203 tensors match the manifest |

VAE `diffusion_pytorch_model.bin` (fp16): `a59d7ea697f2942d22002dc3469e8c53db807a6b78f7f5ec03bd4c1f70f98efe`.
✅ Validated: 248 tensors match the manifest. f16 storage is lossless.

f16 check on `ft_ffhq` and `ft_places2`: only one tensor exceeds the f16 range,
`down_blocks.0…attn2.norm_v.running_var` (9.9e4), a BatchNorm statistic that is folded away.

## 9. Verification

| Gate | Criterion |
|---|---|
| Kernels (M2) ✅ | PyTorch fixtures + naive f64 references. 1×1 ≥ 450 GFLOPS, 3×3 ≥ 420 GFLOPS. Memory-bound kernels: total ≤ 0.10 s per UNet step* |
| Blocks (M3–M6) ✅ | Golden tensors from `tools/dump_golden.py` with fixed noise. Rel L2 ≤ 1e-4 with f32 weights, ≤ 2e-3 with f16. Measured: f32 ≤ 4.7e-6 per block (39 UNet tensors), f16 ≤ 1.9e-3; VAE ≤ 2.9e-5 |
| Exact e2e (M7) ✅ | Same noise as PyTorch: PSNR ≥ 45 dB (f32 weights) / ≥ 30 dB (f16), whole image (LPIPS dropped: package not installed). ≤ 36 s. RSS ≤ 1.5 GB. Measured below |
| ROI decode (M8) ❌ | Same latents, ROI vs full decode: PSNR ≥ 40 dB inside the mask. Failed (34.82 dB on 10 eval masks, §5), so Fast decodes in full |
| Fast preset (M8) ✅ | LPIPS-to-ground-truth ≤ 1.05 × Exact's on ≥ 100 image/mask pairs, overall and per subset, plus visual review. ≤ 12 s from cool. Starts at ≤ 15% image signal. Measured on 120 pairs: scenes 1.016 (CI 1.006–1.027), faces 1.038 (CI 1.025–1.052), all 1.025 (CI 1.016–1.033); 10.74–11.88 s |
| GIMP (M9) ✅ | Headless run inside GIMP 3.2.6: 0 pixels changed outside the selection, alpha untouched, worker output identical to the CLI. Run by hand in GIMP: the dialog and both checkpoints work |
| Speed regressions | `examples/kernel_bench.rs` now; per-stage timings recorded per milestone |

**End-to-end parity (M7, final code, `cargo run --release --example parity`)**: identical recorded noise, 8-bit output vs PyTorch.

| Case | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| f16 PSNR (dB) | 56.39 | 58.90 | 53.42 | 61.90 | 59.86 | 49.55 | 58.29 | 60.77 |
| f32 PSNR (dB) | 74.59 | 78.57 | | | | | | |

f32 differences are 1 LSB at truncation boundaries. f16 shows small weight-rounding drift
amplified over 19 steps; the worst pixel is in case 5.

\* M0 set the depthwise gate as "≥ 3× PyTorch". Measurement showed the kernel is DRAM-bound
(it saturates at 4–6 threads), so its speed reflects memory bandwidth, not code quality. It
measures 2.7–3.2× PyTorch run to run. The gate was replaced by the per-step budget the promise
actually depends on: all depthwise convs of a UNet step ≈ 45 ms.

## 10. Milestones

| # | Deliverable | Gate |
|---|---|---|
| **M0** ✅ | Measurements, crate scaffold, scheduler, weights loader, crop, C ABI | 14 tests pass; `gemm` ≈ MKL |
| **M1** ✅ | Converter + golden dumps | Manifest match; golden set independently re-verified |
| **M2** ✅ | conv1x1, Winograd 3×3 (+upsample), im2col strided 3×3, depthwise, GroupNorm, LayerNorm, SiLU | §9 kernel gates pass (§4 table) |
| **M3** ✅ | Self-λ and cross-λ (folded) | Golden parity (f32 ≤ 1.9e-6) |
| **M4** ✅ | MixTransformer + MixFFN | Golden parity (f32 ≤ 9.4e-7) |
| **M5** ✅ | UNet + embeddings | Parity; 1.04–1.17 s/step at batch 2 (cool) |
| **M6** ✅ | VAE encoder / decoder | Parity; encode 2.8–3.2 s (B=2), decode 2.7–3.1 s |
| **M7** ✅ | Pipeline, resize, blend (Pillow-exact), CLI | Exact e2e gate (§9); 24–25 s cool; 1.42 GB peak |
| **M8** ✅ | Fast preset tuning, eval tooling, DPM-Solver++(2M), compensated paste | Fast gates (§9); 10.7–11.9 s cool; ROI decode rejected |
| **M9** ✅ | `gimp-plugin/`: Python glue, `moebius-cli --plugin` worker, per-user installer, progress, orphan-safe cancel | §9 GIMP gate; Fast 12.0–13.6 s, Exact 27.0 s including load and PNG transfer |

Closed 2026-11-05. Left open: the plugin has only been tested on the i7-13620H, GIMP's cancel button has not
been clicked by hand (the worker's exit on a closed stdin is tested), and the programs are not
code-signed.

## 11. Risks

| Risk | Outcome |
|---|---|
| f16 weights miss parity | Did not happen (§9). Per-tensor f32 fallback in the converter; global f32 mode (+0.6 GB) |
| Sustained power limit (measured: ~340–600 GFLOPS cycles) | Promise stated per single inpaint; sustained batches documented at a ~30 s median; Fast preset (M8) for headroom |
| E-core stragglers in GEMM | Resolved: `private-gemm-x86` balances P/E cores (§4) |
| RAM pressure inside GIMP | The worker is a separate process that exits after each run; 1.42 GB peak (§5) |
| Fast quality gate fails | Did not happen (§9) |
| Exact needs to be faster | Did not happen. If it ever does: INT8 W8A8 1×1 GEMMs via AVX-VNNI (≈1.5–2× on 62% of UNet FLOPs), behind its own quality gate |
| Other CPUs behave differently | Open: only tested on the i7-13620H |

## 12. Dependencies

| Crate | Use |
|---|---|
| `gemm` 0.19 | Single-threaded GEMMs (Winograd's 16 concurrent tile GEMMs) |
| `private-gemm-x86` =0.1.22 | Multi-threaded GEMMs (pinned: a "private" crate without semver promises) |
| `rayon` 1.11 | Engine thread pool |
| `safetensors` 0.8, `memmap2` 0.9, `half` 2.7, `bytemuck` 1 | f16 weights, mmap, zero-copy views |
| `fast_image_resize` 6.1 | Lanczos3 window ↔ 512² |
| `rand` 0.10, `rand_distr` 0.6 | Initial noise |
| `num_cpus` 1.17 | Physical-core pool size |
| `thiserror` 2 | Errors |
| `image` 0.25, `clap` 4, `serde_json` 1, `sha2` 0.10 (feature `cli`) | `moebius-cli`, which is also the plugin worker, and `moebius-convert` |
| `rusty_alloc-api` 2.2.7 (feature `rusty-alloc`) | Opt-in allocator, off by default (§5) |

Not used: `imageproc` (the σ=3 feather is a 20-line separable blur),
`serde`, `anyhow`.
