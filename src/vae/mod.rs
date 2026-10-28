//! `AutoencoderKL` with SDXL-VAE weights (`sdvae_f8d4`, the PixelHacker VAE) (M6).
//! The SDXL VAE overflows fp16 *activations* (diffusers' `force_upcast`);
//! this engine computes in f32 throughout, so only the weights are f16.
//!
//! Hand-written on the engine kernels: block_out (128, 256, 512, 512),
//! 2 resnets per level, one single-head attention in each mid block,
//! GroupNorm eps 1e-6. 94–99% of its FLOPs are dense 3x3 convs at high
//! resolution, so it runs entirely on `engine::conv::Conv3x3`.
//!
//! | pass                   | GFLOP | time @ ~480 GFLOPS |
//! |------------------------|-------|--------------------|
//! | encode, image + masked | 2233  | ~4.6 s             |
//! | decode, full 64² latent| 2515  | ~5.2 s             |
//!
//! Encoding takes the posterior *sample* (`latent_dist.sample()`, as
//! upstream) scaled by `scaling_factor`; decoding divides by it first.
//! Peak activation memory is the decoder's last level (512² × 128 f32 =
//! 128 MB per tensor, three live).

pub mod decoder;
pub mod encoder;

/// Latent window for ROI decoding (Fast preset), in latent pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatentRect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}
