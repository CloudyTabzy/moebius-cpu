//! `moebius-cpu` — CPU-only native inference engine for Moebius, the 0.22B
//! latent-diffusion inpainting model (hustvl/Moebius, ECCV'26).
//!
//! The engine is purpose-built for this one network: NHWC f32 activations,
//! f16 weights memory-mapped from safetensors, all dense math through the
//! `gemm` crate, and fused hand-written kernels for everything else.
//! See `docs/PLAN.md` for the design, the measured compute budget and the
//! speed promise.
//!
//! Status: runs end to end and matches upstream PyTorch (`tests/parity.rs`,
//! `examples/parity.rs`). ROI decode and compensated paste are in; still
//! open: the Fast preset's measured parameters (M8) and the GIMP plugin hookup (M9).

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
