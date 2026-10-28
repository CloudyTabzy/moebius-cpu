//! VAE encoder: conv_in → 4 down levels (last without downsample) →
//! mid (resnet, attention, resnet) → GN+SiLU → conv_out (8 ch = mean ‖
//! logvar) → quant_conv. Runs image and masked image as one batch of 2
//! (Exact) or the masked image alone (Fast `single_encode`).
//!
//! `Downsample2D(padding=0)` is `F.pad(x, (0, 1, 0, 1))` then a stride-2 3x3
//! conv, i.e. `Conv3x3 { stride: 2, pad: [0, 0, 1, 1] }`.

use rayon::prelude::*;

use super::{MidBlock, Resnet, Work};
use crate::Result;
use crate::config::{LATENT_CHANNELS, VAE_BLOCK_OUT_CHANNELS, VAE_EPS, VAE_SCALING_FACTOR};
use crate::engine::conv::{Conv1x1, Conv3x3};
use crate::engine::math::exp;
use crate::engine::norm::GroupNorm;
use crate::engine::simd::multiversion;
use crate::engine::{Scratch, Tensor};
use crate::unet::load::Loader;

/// Resnets per down level (`layers_per_block`); up levels have one more.
pub(super) const LAYERS: usize = 2;
const IMAGE_CHANNELS: usize = 3;
/// diffusers' `DiagonalGaussianDistribution` clamp on the log-variance.
const LOGVAR_RANGE: (f32, f32) = (-30.0, 20.0);
/// Pixels per sampling task.
const SAMPLE_PIXELS: usize = 1024;

struct DownBlock {
    resnets: Vec<Resnet>,
    downsample: Option<Conv3x3>,
}

pub struct Encoder {
    conv_in: Conv3x3,
    down: Vec<DownBlock>,
    mid: MidBlock,
    norm_out: GroupNorm,
    conv_out: Conv3x3,
    quant_conv: Conv1x1,
}

impl Encoder {
    /// `root` is positioned at the file root.
    pub(super) fn load(root: &Loader) -> Result<Self> {
        let l = root.sub("encoder");
        let ch = VAE_BLOCK_OUT_CHANNELS;
        let mut down = Vec::new();
        let mut c_prev = ch[0];
        for (i, &c) in ch.iter().enumerate() {
            let b = l.sub(&format!("down_blocks.{i}"));
            let resnets = (0..LAYERS)
                .map(|j| Resnet::load(&b.sub(&format!("resnets.{j}")), if j == 0 { c_prev } else { c }, c))
                .collect::<Result<Vec<_>>>()?;
            let downsample = if i + 1 < ch.len() {
                Some(b.sub("downsamplers.0").conv3x3_with("conv", c, c, 2, [0, 0, 1, 1], false)?)
            } else {
                None
            };
            down.push(DownBlock { resnets, downsample });
            c_prev = c;
        }
        let c_last = ch[ch.len() - 1];
        Ok(Self {
            conv_in: l.conv3x3("conv_in", IMAGE_CHANNELS, ch[0])?,
            down,
            mid: MidBlock::load(&l.sub("mid_block"), c_last)?,
            norm_out: l.group_norm("conv_norm_out", c_last, VAE_EPS)?,
            conv_out: l.conv3x3("conv_out", c_last, 2 * LATENT_CHANNELS)?,
            quant_conv: root.conv1x1("quant_conv", 2 * LATENT_CHANNELS, 2 * LATENT_CHANNELS, true)?,
        })
    }

    /// `images`: `[b, 512, 512, 3]` in [-1, 1]. Returns scaled latents
    /// `[b, 64, 64, 4]`, sampling the posterior with `noise`.
    pub fn encode(&self, images: &Tensor, noise: &Tensor, s: &mut Scratch) -> Tensor {
        assert_eq!(images.c, IMAGE_CHANNELS, "VAE encoder expects RGB");
        assert_eq!(noise.shape(), [images.n, images.h / 8, images.w / 8, LATENT_CHANNELS], "VAE noise shape");

        let mut w = Work::default();
        self.conv_in.forward(images, &mut w.cur, s);
        for block in &self.down {
            for r in &block.resnets {
                w.resnet(r, s);
            }
            if let Some(d) = &block.downsample {
                w.conv(d, s);
            }
        }
        w.mid(&self.mid, s);
        w.norm_silu_conv(&self.norm_out, &self.conv_out, s);
        let mut moments = Tensor::default();
        self.quant_conv.forward(&w.cur, &mut moments, None, s);

        let mut z = Tensor::default();
        z.set_shape(moments.n, moments.h, moments.w, LATENT_CHANNELS);
        z.data
            .par_chunks_mut(LATENT_CHANNELS * SAMPLE_PIXELS)
            .zip(moments.data.par_chunks(2 * LATENT_CHANNELS * SAMPLE_PIXELS))
            .zip(noise.data.par_chunks(LATENT_CHANNELS * SAMPLE_PIXELS))
            .for_each(|((z, m), n)| sample(m, n, z));
        z
    }
}

multiversion! {
    /// `z = (mean + exp(0.5·clamp(logvar)) · noise) · scaling_factor` per pixel;
    /// `moments` holds `[mean(4) ‖ logvar(4)]` per pixel.
    fn sample(moments: &[f32], noise: &[f32], z: &mut [f32]) {
        const C: usize = LATENT_CHANNELS;
        for ((z, m), n) in z.chunks_exact_mut(C).zip(moments.chunks_exact(2 * C)).zip(noise.chunks_exact(C)) {
            for i in 0..C {
                let logvar = m[C + i].clamp(LOGVAR_RANGE.0, LOGVAR_RANGE.1);
                z[i] = exp(0.5 * logvar).mul_add(n[i], m[i]) * VAE_SCALING_FACTOR;
            }
        }
    }
}
