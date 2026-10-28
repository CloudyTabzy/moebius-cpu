# moebius-cpu

CPU-only native Rust inference engine for **Moebius**, the 0.22B diffusion
inpainting model ([hustvl/Moebius](https://github.com/hustvl/Moebius), ECCV'26).
It builds as a DLL with a C ABI (`include/moebius_cpu.h`) for the GIMP plugin.

The design, measurements and speed promise are in **[docs/PLAN.md](docs/PLAN.md)**.

## Status

M0 (measure + scaffold) and M2 (engine kernels) are done. Implemented and tested:
- DDIM scheduler (matches diffusers)
- f16 safetensors loader
- square crop planning
- C ABI shell
- engine kernels, parity-tested against PyTorch: 1×1 conv, Winograd and im2col 3×3 conv,
  depthwise, GroupNorm, LayerNorm, SiLU

Model assembly (λ layers, UNet, VAE, pipeline) is still stubbed, marked with its milestone
(`todo!("M…")`).

## Targets (i7-13620H, 512² window)

| Mode | Committed | Projected after M2 | Notes |
|---|---|---|---|
| Exact | ≤ 36 s | ~28 s | Upstream pipeline, parity-gated |
| Fast | ≤ 12 s | ~8–9 s | Fewer steps, partial CFG, single encode, ROI decode; quality-gated |

Peak memory ≤ 1.5 GB. Requires x86-64 with AVX2 + FMA.

## Build

```sh
cargo test
cargo build --release                      # target/release/moebius_cpu.dll (+ .lib)
cargo build --release --features cli       # developer CLI: moebius-cli
cargo run --release --example gemm_probe   # GEMM throughput
cargo run --release --example kernel_bench # M2 kernel speed gates
```

## Tools (Python, offline)

| Script | Purpose |
|---|---|
| `tools/count_flops.py` | FLOP breakdown on the meta device; writes `docs/weights-manifest.txt` |
| `tools/bench_torch_baseline.py` | PyTorch CPU baseline timing (needs ~3 GB free RAM) |
| `tools/dump_ddim_reference.py` | diffusers values behind the scheduler tests |
| `tools/dump_kernel_fixtures.py` | PyTorch reference outputs for `tests/kernels.rs` (also the weight re-layout spec) |
| `tools/convert_weights.py` | `.bin` → f16 safetensors (M1) |

## Layout

```
src/
  engine/      NHWC tensor, f16 params, gemm, conv (1×1, im2col), Winograd 3×3,
               depthwise, norms, vectorised exp/SiLU, runtime AVX2 dispatch
  unet/        blocks, MixTransformer, λ layers (with the fold derivations), embeddings
  vae/         SD AutoencoderKL encoder / decoder
  imaging/     crop policy, Lanczos resize, feathered paste
  scheduler.rs DDIM
  pipeline.rs  end-to-end Engine
  ffi.rs       C ABI
reference/     old ONNX C++ worker and Python plugin (historical)
```
