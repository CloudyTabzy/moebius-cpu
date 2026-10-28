//! VAE decoder: post_quant_conv → conv_in → mid (resnet, attention,
//! resnet) → 4 up levels of 3 resnets (+ nearest 2× / conv on the first
//! three) → GN+SiLU → conv_out. The nearest-2× upsample is fused into the
//! 3x3 conv (`Conv3x3::upsample2x`), so the 4× tensor is never stored.
//!
//! Full-frame only. ROI decoding (Fast preset, `LatentRect`) is left to
//! the milestone that gates it on quality: the mid-block attention would then
//! see the window instead of the whole 64² latent.

use rayon::prelude::*;

use super::encoder::LAYERS;
use super::{MidBlock, Resnet, Work};
use crate::Result;
use crate::config::{LATENT_CHANNELS, VAE_BLOCK_OUT_CHANNELS, VAE_EPS, VAE_SCALING_FACTOR};
use crate::engine::conv::{Conv1x1, Conv3x3};
use crate::engine::norm::GroupNorm;
use crate::engine::{Scratch, Tensor};
use crate::unet::load::Loader;

const IMAGE_CHANNELS: usize = 3;
/// `layers_per_block + 1` resnets per up level.
const UP_LAYERS: usize = LAYERS + 1;
const CHUNK: usize = 1 << 15;

struct UpBlock {
    resnets: Vec<Resnet>,
    upsample: Option<Conv3x3>,
}

pub struct Decoder {
    post_quant_conv: Conv1x1,
    conv_in: Conv3x3,
    mid: MidBlock,
    up: Vec<UpBlock>,
    norm_out: GroupNorm,
    conv_out: Conv3x3,
}

impl Decoder {
    /// `root` is positioned at the file root.
    pub(super) fn load(root: &Loader) -> Result<Self> {
        let l = root.sub("decoder");
        let ch = VAE_BLOCK_OUT_CHANNELS;
        let c_top = ch[ch.len() - 1];
        let mut up = Vec::new();
        let mut c_prev = c_top;
        for (i, &c) in ch.iter().rev().enumerate() {
            let b = l.sub(&format!("up_blocks.{i}"));
            let resnets = (0..UP_LAYERS)
                .map(|j| Resnet::load(&b.sub(&format!("resnets.{j}")), if j == 0 { c_prev } else { c }, c))
                .collect::<Result<Vec<_>>>()?;
            let upsample = if i + 1 < ch.len() {
                Some(b.sub("upsamplers.0").conv3x3_with("conv", c, c, 1, [1; 4], true)?)
            } else {
                None
            };
            up.push(UpBlock { resnets, upsample });
            c_prev = c;
        }
        Ok(Self {
            post_quant_conv: root.conv1x1("post_quant_conv", LATENT_CHANNELS, LATENT_CHANNELS, true)?,
            conv_in: l.conv3x3("conv_in", LATENT_CHANNELS, c_top)?,
            mid: MidBlock::load(&l.sub("mid_block"), c_top)?,
            up,
            norm_out: l.group_norm("conv_norm_out", ch[0], VAE_EPS)?,
            conv_out: l.conv3x3("conv_out", ch[0], IMAGE_CHANNELS)?,
        })
    }

    /// `latents`: scaled latents `[1, 64, 64, 4]`. Returns RGB `(x + 1) / 2`,
    /// unclamped, `[1, 512, 512, 3]`.
    pub fn decode(&self, latents: &Tensor, s: &mut Scratch) -> Tensor {
        assert_eq!(latents.c, LATENT_CHANNELS, "VAE decoder expects 4 latent channels");
        let mut z = latents.clone();
        z.data.par_chunks_mut(CHUNK).for_each(|c| c.iter_mut().for_each(|v| *v /= VAE_SCALING_FACTOR));

        let mut w = Work::default();
        self.post_quant_conv.forward(&z, &mut w.next, None, s);
        self.conv_in.forward(&w.next, &mut w.cur, s);
        w.mid(&self.mid, s);
        for block in &self.up {
            for r in &block.resnets {
                w.resnet(r, s);
            }
            if let Some(u) = &block.upsample {
                w.conv(u, s);
            }
        }
        w.norm_silu_conv(&self.norm_out, &self.conv_out, s);
        let mut out = w.cur;
        out.data.par_chunks_mut(CHUNK).for_each(|c| c.iter_mut().for_each(|v| *v = (*v + 1.0) * 0.5));
        out
    }
}
