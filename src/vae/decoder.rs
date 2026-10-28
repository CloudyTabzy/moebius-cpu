//! VAE decoder (M6): post_quant_conv → conv_in → mid (resnet, attention,
//! resnet) → 4 up levels of 3 resnets (+ nearest 2× / conv) → GN+SiLU →
//! conv_out. With `roi` set (Fast preset) only that latent window is
//! decoded; its mid-block attention then sees the window instead of the full
//! 64² latent, which is why ROI decoding is gated on quality (PLAN §5, M8).

use super::LatentRect;
use crate::engine::Tensor;

pub struct Decoder;

impl Decoder {
    /// `latents`: scaled latents `[1, 64, 64, 4]`. Returns RGB in [0, 1]
    /// for the full 512² frame or for `roi` × 8.
    pub fn decode(&self, _latents: &Tensor, _roi: Option<LatentRect>) -> Tensor {
        todo!("M6")
    }
}
