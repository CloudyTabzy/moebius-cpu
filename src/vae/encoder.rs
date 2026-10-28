//! VAE encoder (M6): conv_in → 4 down levels (last without downsample) →
//! mid (resnet, attention, resnet) → GN+SiLU → conv_out (8 ch = mean ‖
//! logvar) → quant_conv. Runs image and masked image as one batch of 2
//! (Exact) or the masked image alone (Fast `single_encode`).

use crate::engine::Tensor;

pub struct Encoder;

impl Encoder {
    /// `images`: `[b, 512, 512, 3]` in [-1, 1]. Returns scaled latents
    /// `[b, 64, 64, 4]`, sampling the posterior with `noise`.
    pub fn encode(&self, _images: &Tensor, _noise: &[f32]) -> Tensor {
        todo!("M6")
    }
}
