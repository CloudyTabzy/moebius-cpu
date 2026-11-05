# moebius-cpu

**Moebius inpainting as a plugin for image editors, running on the CPU.**

[Moebius](https://github.com/hustvl/Moebius) (ECCV'26) is a 0.22B diffusion inpainting model. This
project runs it in pure Rust, with no GPU, so it can work from inside an image editor. GIMP 3.2 is the
first host. Other native hosts can use the DLL and its C ABI (`include/moebius_cpu.h`).

The design, measurements and speed promise are in **[docs/PLAN.md](docs/PLAN.md)**.

## Status

**Finished.** Moebius runs in GIMP on pure Rust and matches PyTorch. Given the same noise, the 8-bit
output reaches 74.6–78.6 dB PSNR vs upstream with f32 weights and 49.6–61.9 dB with f16 (8 test
pairs). Every block is checked against recorded PyTorch tensors (`tests/parity.rs`).

Known limits:
- Tested only with Windows and GIMP 3.2.6.
- The worker's exit on cancel is tested, but GIMP's cancel button has not been clicked by hand.
- The installer needs converted weights, so there is no one-click package for other people.

## Speed

| Mode | One 512² inpaint | Quality |
|---|---|---|
| Exact | 24–25 s | Matches PyTorch, which takes 62–69 s on the same CPU |
| Fast | 10.7–11.9 s | LPIPS within 1.04× of Exact on 120 pairs |

Times vary by CPU. Back-to-back runs are slower, and GIMP
adds 1–2 s per run. Peak memory is about 1.44 GB. Requires x86-64 with AVX2 + FMA.

Hardware, method and full results: [docs/PLAN.md](docs/PLAN.md#4-measured-compute-budget) (§4 and §5).

## Use it in GIMP

Needs GIMP 3.2 on Windows and the converted `ft_places2` and `ft_celebahq` checkpoints in `weights/`.

```bat
gimp-plugin\install.bat
```

Restart GIMP, select what to remove, then use **Filters > Enhance > Moebius Inpaint...**

- **Cover the whole object with a margin**, shadows included. Anything left outside the selection
  can grow back. Grow a tight selection by 10 px or so.
- Pick **Faces** for portraits and **Scenes and objects** for everything else.
- Change **Seed** for a different result.
- Only the selection changes, and alpha is kept. The result inside the selection is 8-bit, even in
  a higher-precision image.

GIMP's Python cannot load MSVC binaries, so the plugin runs `moebius-cli.exe` as a separate process.
`include/moebius_cpu.h` remains for native hosts. `tools/gimp_smoke.py` tests an installed plugin
headlessly.

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
cargo build --release --features cli       # moebius-cli: command line and GIMP plugin worker
cargo run --release --example gemm_probe   # GEMM throughput
cargo run --release --example kernel_bench # kernel speed gates
cargo run --release --features cli --example profile   # per-stage timing of one inpaint
cargo run --release --features cli --example eval      # eval runs for Fast vs Exact (tools/eval)
cargo run --release --example sustained   # GEMM throughput under sustained load
cargo run --release --example parity       # end-to-end parity vs PyTorch (needs weights/golden)
cargo test --release --test parity -- --ignored        # per-block parity vs PyTorch
```

## Tools (Python, offline)

| Script | Purpose |
|---|---|
| `tools/count_flops.py` | FLOP breakdown on the meta device; writes `docs/weights-manifest.txt` |
| `tools/bench_torch_baseline.py` | PyTorch CPU baseline timing (needs ~3 GB free RAM) |
| `tools/dump_ddim_reference.py`, `tools/dump_dpm_reference.py` | diffusers values behind the scheduler tests |
| `tools/dump_kernel_fixtures.py` | PyTorch reference outputs for `tests/kernels.rs` (also the weight re-layout spec) |
| `tools/convert_weights.py` | `.bin` → safetensors (BatchNorm folding, NHWC layouts, f16/f32) |
| `tools/dump_golden.py` | Runs upstream PyTorch and records noise and intermediates for parity |
| `tools/dump_blur_fixture.py` | Pillow `GaussianBlur` reference for the byte-exact port |
| `tools/dump_compensate_fixture.py` | Reference for the compensated paste test |
| `tools/bench_alloc.py` | Allocator and rayon A/B |
| `tools/eval/prepare_eval_set.py`, `score.py` | Build the 120-pair eval set and score runs with LPIPS |
| `tools/eval/roi_gate.py`, `review_sheet.py` | ROI decode gate and the visual review sheet |
| `tools/gimp_smoke.py` | Headless test of the installed plugin inside real GIMP |

## Layout

```
src/
  engine/      NHWC tensor, f16 params, buffer pool (Scratch), hybrid GEMM, conv
               (1×1, im2col), Winograd 3×3 (+GN prologue), depthwise, norms,
               vectorised exp/SiLU, runtime AVX2 dispatch
  unet/        blocks, MixTransformer, λ layers (with the fold derivations), embeddings
  vae/         AutoencoderKL encoder / decoder (SDXL-VAE weights)
  imaging/     crop policy, Lanczos resize, feathered paste
  scheduler.rs DDIM and DPM-Solver++(2M)
  preset.rs    Exact and Fast
  pipeline.rs  end-to-end Engine
  ffi.rs       C ABI
  bin/         moebius-cli (`--plugin` makes it the GIMP worker)
gimp-plugin/   GIMP 3.2 plugin and per-user installer
include/       C header
```

## License

Apache-2.0, see [LICENSE](LICENSE). Moebius and its weights are by the
[hustvl/Moebius](https://github.com/hustvl/Moebius) authors, also under Apache-2.0. Weights are not in
this repo: `docs/PLAN.md` §8 has the sources and hashes.
