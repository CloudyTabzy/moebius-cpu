//! `moebius-cpu` — CPU-only native inference engine for Moebius, the 0.22B
//! latent-diffusion inpainting model (hustvl/Moebius, ECCV'26).
//!
//! The engine is purpose-built for this one network: NHWC f32 activations,
//! f16 weights memory-mapped from safetensors, dense math through
//! `private-gemm-x86` and `gemm`, and fused hand-written kernels for everything else.
//! See `docs/PLAN.md` for the design, the measured compute budget and the
//! speed promise.
//!
//! Status: finished. It matches upstream PyTorch (`tests/parity.rs`,
//! `examples/parity.rs`), the Fast preset passed its gates, and GIMP runs it
//! through the `moebius-cli` worker (`gimp-plugin/`).

/// Opt-in (`--features rusty-alloc`): the pure-Rust mimalloc remake as the
/// allocator of every Rust allocation in this crate's artifacts. Off by default:
/// measured no faster than the system allocator within the memory gate (see
/// Cargo.toml). Its thread-exit hook lives in a never-freed FLS slot, so with the
/// feature on the DLL pins itself on first use (`ffi::pin_module`).
#[cfg(feature = "rusty-alloc")]
#[global_allocator]
static ALLOC: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;

pub mod config;
pub mod engine;
pub mod error;
pub mod ffi;
pub mod imaging;
pub mod pipeline;
pub mod preset;
pub mod scheduler;
pub mod unet;
pub mod vae;
pub mod weights;

pub use error::{Error, Result};
pub use pipeline::Engine;
pub use preset::{InpaintParams, Preset};
