//! Moebius student UNet (M3–M5).
//!
//! ```text
//! conv_in   DWConv2d 9 → 320
//! down 0    64², 320  : 2 × (DWResnet, MixTransformer) + Downsample2D
//! down 1    32², 640  : 2 × (DWResnet, MixTransformer) + Downsample2D
//! down 2    16², 1280 : 2 × (DWResnet, MixTransformer)
//! (no mid block)
//! up 0      16², 1280 : 3 × (concat skip, DWResnet, MixTransformer) + Upsample2D
//! up 1      32², 640  : 3 × (…) + Upsample2D
//! up 2      64², 320  : 3 × (…)
//! out       GroupNorm(32)+SiLU → DWConv2d 320 → 4
//! ```
//!
//! Per step at batch 2 (CFG): 632 GFLOP — 64% 1x1 GEMM, 20% 3x3 dense
//! (Upsample2D convs), 14% λ position conv (→ ~3% after the fold in
//! `lambda.rs`), 1% stride-2, 0.5% depthwise (`tools/count_flops.py`).
//!
//! Batch layout: item 0 = unconditional, item 1 = conditional, as upstream.
//! The two items are identical until the first cross-λ, so `conv_in`, the
//! first resnet and the first self-λ run once at batch 1 and are copied.
//! When the Fast preset turns CFG off, the whole forward runs at batch 1.

pub mod blocks;
pub mod embed;
pub mod lambda;
pub mod load;
pub mod transformer;

use crate::engine::Tensor;
use crate::weights::Weights;
use crate::Result;

/// Which conditioning rows to run: `[uncond, cond]` for CFG, `[cond]` without.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Guidance {
    Cfg,
    CondOnly,
}

pub struct UNet {
    _private: (),
}

impl UNet {
    /// Loads weights and precomputes everything input-independent:
    /// conditioning tables, cross-λ constants (`lambda::CrossLambda`).
    pub fn load(_weights: &Weights) -> Result<Self> {
        todo!("M5: assemble blocks from `unet.safetensors`")
    }

    /// `x`: `[1, 64, 64, 9]` (noisy latent ‖ mask ‖ masked latent).
    /// Returns ε `[b, 64, 64, 4]` with b = 2 for `Cfg`, 1 for `CondOnly`.
    pub fn forward(&self, _x: &Tensor, _timestep: usize, _guidance: Guidance) -> Tensor {
        todo!("M5")
    }
}
