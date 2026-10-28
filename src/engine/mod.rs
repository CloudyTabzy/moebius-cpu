//! Compute engine: NHWC f32 tensors and the handful of kernels Moebius needs.
//!
//! Layout is NHWC throughout so that every 1x1 conv / linear (62% of UNet
//! FLOPs) is a plain GEMM over pixels with no transposes, and channel-wise
//! kernels (depthwise, GroupNorm, gating) stream contiguous SIMD lanes.
//! All kernels parallelise on the current rayon pool (the engine's own).

pub mod conv;
pub mod cpu;
pub mod depthwise;
pub mod gemm;
pub mod math;
pub mod norm;
pub mod ops;
pub mod param;
pub mod simd;
pub mod tensor;
pub mod winograd;

pub use param::Param;
pub use tensor::Tensor;

/// Reusable per-engine scratch, grown on first use and kept for the run.
#[derive(Default)]
pub struct Scratch {
    /// f16 → f32 widened GEMM weights (≤ 33 MB: the 1280→6400 MixFFN layer).
    pub panel: Vec<f32>,
    /// Row-band patch matrix of the strided 3x3 convolutions (≤ 32 MB).
    pub cols: Vec<f32>,
    /// Winograd input tiles, products and transformed kernel (≤ 16/32/32 MB).
    pub wino_v: Vec<f32>,
    pub wino_m: Vec<f32>,
    pub wino_u: Vec<f32>,
}

/// Worker count of the pool the caller runs in.
pub(crate) fn threads() -> usize {
    rayon::current_num_threads()
}
