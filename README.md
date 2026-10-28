# moebius-cpu

CPU-only native Rust inference engine for **Moebius**, the 0.22B diffusion
inpainting model ([hustvl/Moebius](https://github.com/hustvl/Moebius), ECCV'26).
It builds as a DLL with a C ABI (`include/moebius_cpu.h`) for the GIMP plugin.

The design, measurements and speed promise are in **[docs/PLAN.md](docs/PLAN.md)**.

## Status

Milestones M0–M7 are done: **Moebius runs end to end in pure Rust and matches PyTorch.**
Given the same noise, the 8-bit output reaches 74.6–78.6 dB PSNR vs upstream with f32 weights
and 49.6–61.9 dB with f16 (8 test pairs). Every block is checked against recorded PyTorch
tensors (`tests/parity.rs`).

Still to do:
- M8: tune and quality-gate the Fast preset.
- M9: hook up the GIMP plugin through the C ABI.

## Targets (i7-13620H, 512² window)

| Mode | Committed | Measured | Notes |
|---|---|---|---|
| Exact | ≤ 36 s | **24–25 s** (34–38 s under sustained back-to-back load) | Upstream pipeline, parity-gated. PyTorch CPU: 62–69 s |
| Fast | ≤ 12 s | M8 | Fewer steps, partial CFG, single encode, ROI decode; quality-gated |

Peak working set 1.42 GB (≤ 1.5 GB). Engine load ~0.35 s. Requires x86-64 with AVX2 + FMA.
Pure Rust: dense math on `private-gemm-x86` (multi-threaded) and `gemm` (Winograd tiles).

## Run it

```sh
# once: convert a checkpoint (see docs/PLAN.md §8 for downloads and hashes)
python tools/convert_weights.py --unet weights/upstream/ft_celebahq/diffusion_pytorch_model.bin \
    --vae weights/upstream/vae --out weights/ft_celebahq
cargo run --release --features cli -- --weights weights/ft_celebahq \
    --image photo.png --mask mask.png --output out.png      # mask: white = repaint
```

Pick the checkpoint by content: `ft_celebahq` / `ft_ffhq` for people (identity is kept),
`ft_places2` for scenes. On faces the scene model invents distorted features.

## Build

```sh
cargo test
cargo build --release                      # target/release/moebius_cpu.dll (+ .lib)
cargo build --release --features cli       # developer CLI: moebius-cli
cargo run --release --example gemm_probe   # GEMM throughput
cargo run --release --example kernel_bench # M2 kernel speed gates
cargo run --release --features cli --example profile   # per-stage timing of one inpaint
cargo run --release --example parity       # end-to-end parity vs PyTorch (needs weights/golden)
cargo test --release --test parity -- --ignored        # per-block parity vs PyTorch
```

## Tools (Python, offline)

| Script | Purpose |
|---|---|
| `tools/count_flops.py` | FLOP breakdown on the meta device; writes `docs/weights-manifest.txt` |
| `tools/bench_torch_baseline.py` | PyTorch CPU baseline timing (needs ~3 GB free RAM) |
| `tools/dump_ddim_reference.py` | diffusers values behind the scheduler tests |
| `tools/dump_kernel_fixtures.py` | PyTorch reference outputs for `tests/kernels.rs` (also the weight re-layout spec) |
| `tools/convert_weights.py` | `.bin` → safetensors (BatchNorm folding, NHWC layouts, f16/f32) |
| `tools/dump_golden.py` | Runs upstream PyTorch and records noise and intermediates for parity |
| `tools/dump_blur_fixture.py` | Pillow `GaussianBlur` reference for the byte-exact port |

## Layout

```
src/
  engine/      NHWC tensor, f16 params, buffer pool (Scratch), hybrid GEMM, conv
               (1×1, im2col), Winograd 3×3 (+GN prologue), depthwise, norms,
               vectorised exp/SiLU, runtime AVX2 dispatch
  unet/        blocks, MixTransformer, λ layers (with the fold derivations), embeddings
  vae/         AutoencoderKL encoder / decoder (SDXL-VAE weights)
  imaging/     crop policy, Lanczos resize, feathered paste
  scheduler.rs DDIM
  pipeline.rs  end-to-end Engine
  ffi.rs       C ABI
reference/     old ONNX C++ worker and Python plugin (historical)
```
