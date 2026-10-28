//! `moebius-cpu` — CPU-only native inference engine for Moebius, the 0.22B
//! latent-diffusion inpainting model (hustvl/Moebius, ECCV'26).
//!
//! The engine is purpose-built for this one network: NHWC f32 activations,
//! f16 weights memory-mapped from safetensors, all dense math through the
//! `gemm` crate, and fused hand-written kernels for everything else.
//! See `docs/PLAN.md` for the design, the measured compute budget and the
//! speed promise.
//!
//! Status: the compute kernels, `unet`, `vae`, `scheduler`, `imaging`,
//! the end-to-end `pipeline` (Exact preset, verified against PyTorch with
//! `examples/parity.rs`) and the C ABI are implemented. Fast-preset ROI
//! decode and paste compensation are still M8.

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
