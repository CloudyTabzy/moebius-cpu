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
//! Peak activation memory is the decoder's last levels: three live feature maps in
//! `Work` (up to 512² × 256 f32 = 268 MB for the upsample conv output). GroupNorm+SiLU
//! is never materialised: `GroupNorm::affine` gives per-channel scale/shift and the
//! Winograd convs apply them while gathering their input (`Conv3x3::forward_prologue`).
//!
//! Names follow diffusers' `AutoencoderKL`; the loader is positioned at the
//! file root (prefix `""`). This module holds the pieces both halves share:
//! the VAE resnet, the single-head mid-block attention and the ping-pong
//! workspace `Work`.

pub mod decoder;
pub mod encoder;

use rayon::prelude::*;

use self::decoder::Decoder;
use self::encoder::Encoder;
use crate::config::{LATENT_SIZE, VAE_EPS, VAE_SCALING_FACTOR};
use crate::engine::conv::{Conv1x1, Conv3x3};
use crate::engine::gemm::{MatRef, matmul};
use crate::engine::math::exp;
use crate::engine::norm::GroupNorm;
use crate::engine::ops::add_inplace;
use crate::engine::simd::multiversion;
use crate::engine::{Scratch, Tensor, threads};
use crate::unet::load::Loader;
use crate::weights::Weights;
use crate::{Error, Result};

/// Latent window for ROI decoding (Fast preset), in latent pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatentRect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

/// Latents of context added on every side of the mask bbox for ROI decoding. The
/// decoder's receptive field is ~4 latents (a few 3x3 convs per level, damped by the
/// upsampling), the paste feather reaches 7 px (< 1 latent), and the 8 px mask sampling
/// grid of `mask_lat` can miss up to one latent cell of mask: 8 covers all of it with
/// room for the window-local mid attention and GroupNorm statistics.
pub const ROI_MARGIN: usize = 8;

/// Bounding box of the nonzero entries of `mask_lat` (`64 x 64`, row-major, 0/1), grown by
/// `margin` latents on every side and clamped to the latent grid. `None` for an empty mask.
pub fn roi_window(mask_lat: &[f32], margin: usize) -> Option<LatentRect> {
    const N: usize = LATENT_SIZE;
    assert_eq!(mask_lat.len(), N * N, "latent mask must be {N} x {N}");
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (i, _) in mask_lat.iter().enumerate().filter(|(_, v)| **v > 0.5) {
        let (y, x) = (i / N, i % N);
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
    }
    if x0 == usize::MAX {
        return None;
    }
    let (x0, y0) = (x0.saturating_sub(margin), y0.saturating_sub(margin));
    let (x1, y1) = ((x1 + margin).min(N - 1), (y1 + margin).min(N - 1));
    Some(LatentRect { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 })
}

/// `AutoencoderKL`: encoder + decoder with the SDXL-VAE weights.
pub struct Vae {
    encoder: Encoder,
    decoder: Decoder,
}

impl Vae {
    /// Fails when the file was not converted from the 0.13025-scaled VAE.
    pub fn load(w: &Weights) -> Result<Vae> {
        let found = w.metadata("scaling_factor");
        if found.and_then(|v| v.parse::<f32>().ok()) != Some(VAE_SCALING_FACTOR) {
            return Err(Error::Weights(format!(
                "vae scaling_factor metadata {found:?}, expected \"{VAE_SCALING_FACTOR}\""
            )));
        }
        let root = Loader::new(w, "");
        Ok(Vae { encoder: Encoder::load(&root)?, decoder: Decoder::load(&root)? })
    }

    /// `images`: `[b, 512, 512, 3]` in [-1, 1]; `noise`: `[b, 64, 64, 4]` standard normal.
    /// Returns `(mean + exp(0.5·clamp(logvar, −30, 20))·noise) · 0.13025`, `[b, 64, 64, 4]`.
    pub fn encode(&self, images: &Tensor, noise: &Tensor, s: &mut Scratch) -> Tensor {
        self.encoder.encode(images, noise, s)
    }

    /// `latents`: scaled `[1, 64, 64, 4]`. Returns `(dec(latents / 0.13025) + 1) / 2`,
    /// unclamped, `[1, 512, 512, 3]`.
    pub fn decode(&self, latents: &Tensor, s: &mut Scratch) -> Tensor {
        self.decoder.decode(latents, s)
    }

    /// Decodes only `latents[roi]` (`latents`: scaled `[1, 64, 64, 4]`; `roi` inside the
    /// grid): `[1, 8·roi.h, 8·roi.w, 3]`, `(x + 1) / 2`, unclamped. The decoder is fully
    /// convolutional, so this is `decode` of the cropped latent: pixels away from the
    /// window border match the full decode closely, but the mid-block attention and the
    /// GroupNorm statistics see the window only, and the border sees zero padding. A
    /// window covering the whole latent is exactly `decode`.
    pub fn decode_roi(&self, latents: &Tensor, roi: LatentRect, s: &mut Scratch) -> Tensor {
        self.decoder.decode_roi(latents, roi, s)
    }
}

/// Three activation buffers reused by every layer: the hidden state `cur`, its
/// successor `next`, and one scratch tensor `h` for the resnet's inner feature map.
/// Layers write into `next` and swap, so no feature map is copied. GroupNorm+SiLU is
/// never materialised: it is fused into the input gather of the 3x3 convs.
pub(crate) struct Work {
    pub cur: Tensor,
    pub next: Tensor,
    h: Tensor,
}

impl Work {
    /// Takes three buffers of `len` elements (the largest feature map to come) from
    /// the scratch pool; give them back with [`Work::release`].
    pub fn new(s: &mut Scratch, len: usize) -> Self {
        Self { cur: s.tensor_for(len), next: s.tensor_for(len), h: s.tensor_for(len) }
    }

    /// Returns the buffers to the pool, except `cur`, which is handed to the caller.
    pub fn release(self, s: &mut Scratch) -> Tensor {
        s.recycle(self.next);
        s.recycle(self.h);
        self.cur
    }

    pub fn resnet(&mut self, r: &Resnet, s: &mut Scratch) {
        r.forward(&self.cur, &mut self.next, &mut self.h, s);
        std::mem::swap(&mut self.cur, &mut self.next);
    }

    pub fn conv(&mut self, c: &Conv3x3, s: &mut Scratch) {
        c.forward(&self.cur, &mut self.next, s);
        std::mem::swap(&mut self.cur, &mut self.next);
    }

    /// `cur = conv(SiLU(GN(cur)))`.
    pub fn norm_silu_conv(&mut self, norm: &GroupNorm, conv: &Conv3x3, s: &mut Scratch) {
        let aff = norm.affine(&self.cur, None, true);
        conv.forward_prologue(&self.cur, Some(&aff), &mut self.next, s);
        std::mem::swap(&mut self.cur, &mut self.next);
    }

    pub fn mid(&mut self, m: &MidBlock, s: &mut Scratch) {
        self.resnet(&m.resnet0, s);
        m.attn.forward(&self.cur, &mut self.next, s);
        std::mem::swap(&mut self.cur, &mut self.next);
        self.resnet(&m.resnet1, s);
    }
}

/// diffusers `ResnetBlock2D` without time embedding:
/// `h = conv1(SiLU(GN1(x))); h = conv2(SiLU(GN2(h))); h + (conv_shortcut(x) | x)`.
pub(crate) struct Resnet {
    norm1: GroupNorm,
    conv1: Conv3x3,
    norm2: GroupNorm,
    conv2: Conv3x3,
    shortcut: Option<Conv1x1>,
}

impl Resnet {
    /// `l` is the `resnets.j` module.
    pub fn load(l: &Loader, c_in: usize, c_out: usize) -> Result<Self> {
        Ok(Self {
            norm1: l.group_norm("norm1", c_in, VAE_EPS)?,
            conv1: l.conv3x3("conv1", c_in, c_out)?,
            norm2: l.group_norm("norm2", c_out, VAE_EPS)?,
            conv2: l.conv3x3("conv2", c_out, c_out)?,
            shortcut: if c_in != c_out { Some(l.conv1x1("conv_shortcut", c_in, c_out, true)?) } else { None },
        })
    }

    /// `out` is reshaped to `[n, h, w, c_out]`; `x`, `out`, `h` are distinct tensors
    /// (`h` is scratch). The GroupNorms are fused into the convs' input gathers, so
    /// only these three feature maps are live.
    pub fn forward(&self, x: &Tensor, out: &mut Tensor, h: &mut Tensor, s: &mut Scratch) {
        let aff = self.norm1.affine(x, None, true);
        self.conv1.forward_prologue(x, Some(&aff), h, s);
        let aff = self.norm2.affine(h, None, true);
        self.conv2.forward_prologue(h, Some(&aff), out, s);
        match &self.shortcut {
            Some(sc) => {
                // h = shortcut(x) + out, then h takes out's place.
                sc.forward(x, h, Some(out), s);
                std::mem::swap(out, h);
            }
            None => add_inplace(&mut out.data, &x.data),
        }
    }
}

/// Single-head self-attention over all pixels (`Attention`, heads 1, dim 512):
/// `to_out(softmax(q·kᵀ/√512)·v) + x` with `q, k, v = linear(GN(x))`.
pub(crate) struct MidAttention {
    norm: GroupNorm,
    to_q: Conv1x1,
    to_k: Conv1x1,
    to_v: Conv1x1,
    to_out: Conv1x1,
}

/// Query rows per softmax band: the score band is `BAND × tokens` floats
/// (8 MB at 4096 tokens).
const BAND: usize = 512;

impl MidAttention {
    /// `l` is the `attentions.0` module.
    pub fn load(l: &Loader, c: usize) -> Result<Self> {
        Ok(Self {
            norm: l.group_norm("group_norm", c, VAE_EPS)?,
            to_q: l.conv1x1("to_q", c, c, true)?,
            to_k: l.conv1x1("to_k", c, c, true)?,
            to_v: l.conv1x1("to_v", c, c, true)?,
            to_out: l.sub("to_out").conv1x1("0", c, c, true)?,
        })
    }

    /// `out` is reshaped like `x` and must not alias it.
    pub fn forward(&self, x: &Tensor, out: &mut Tensor, s: &mut Scratch) {
        let len = x.data.len();
        let (mut t, mut q, mut k, mut v) = (s.tensor_for(len), s.tensor_for(len), s.tensor_for(len), s.tensor_for(len));
        self.norm.forward(x, None, false, &mut t);
        self.to_q.forward(&t, &mut q, None, s);
        self.to_k.forward(&t, &mut k, None, s);
        self.to_v.forward(&t, &mut v, None, s);

        let (hw, c) = (x.h * x.w, x.c);
        let scale = 1.0 / (c as f32).sqrt();
        let mut scores = s.vec_buf(BAND.min(hw) * hw);
        let o = &mut t; // the GN output is dead: reuse its buffer for the attention output
        for b in 0..x.n {
            let item = b * hw * c..(b + 1) * hw * c;
            let (qi, ki, vi) = (&q.data[item.clone()], &k.data[item.clone()], &v.data[item.clone()]);
            for start in (0..hw).step_by(BAND) {
                let rows = BAND.min(hw - start);
                let band = &mut scores[..rows * hw];
                let qb = MatRef::row_major(&qi[start * c..(start + rows) * c], rows, c);
                matmul(band, qb, MatRef::transposed(ki, c, hw), false, threads());
                band.par_chunks_mut(8 * hw).for_each(|chunk| softmax_rows(chunk, hw, scale));
                let dst = &mut o.data[b * hw * c + start * c..b * hw * c + (start + rows) * c];
                matmul(dst, MatRef::row_major(band, rows, hw), MatRef::row_major(vi, hw, c), false, threads());
            }
        }
        self.to_out.forward(o, out, Some(x), s);
        s.recycle_vec(scores);
        for t in [t, q, k, v] {
            s.recycle(t);
        }
    }
}

/// Probabilities below e⁻⁴⁰ (4e-18, ≤ 1e-14 of the row sum) are flushed to zero:
/// peaky rows otherwise fill the band with subnormals, which make the AV GEMM ~20× slower.
const FLUSH: f32 = -40.0;

multiversion! {
    /// In-place `softmax(scale·row)` over each `n`-long row of `rows` (`scale > 0`).
    fn softmax_rows(rows: &mut [f32], n: usize, scale: f32) {
        for row in rows.chunks_exact_mut(n) {
            let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                let a = (*v - max) * scale;
                *v = if a < FLUSH { 0.0 } else { exp(a) };
                sum += *v;
            }
            let inv = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }
}

/// `mid_block`: resnet → attention → resnet, all `c` channels.
pub(crate) struct MidBlock {
    resnet0: Resnet,
    attn: MidAttention,
    resnet1: Resnet,
}

impl MidBlock {
    /// `l` is the `mid_block` module.
    pub fn load(l: &Loader, c: usize) -> Result<Self> {
        Ok(Self {
            resnet0: Resnet::load(&l.sub("resnets.0"), c, c)?,
            attn: MidAttention::load(&l.sub("attentions.0"), c)?,
            resnet1: Resnet::load(&l.sub("resnets.1"), c, c)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lat(cells: &[(usize, usize)]) -> Vec<f32> {
        let mut m = vec![0.0; 64 * 64];
        for &(x, y) in cells {
            m[y * 64 + x] = 1.0;
        }
        m
    }

    #[test]
    fn roi_window_of_an_empty_mask_is_none() {
        assert_eq!(roi_window(&lat(&[]), ROI_MARGIN), None);
    }

    #[test]
    fn roi_window_of_everything_is_the_full_latent() {
        let full = LatentRect { x: 0, y: 0, w: 64, h: 64 };
        assert_eq!(roi_window(&vec![1.0; 64 * 64], ROI_MARGIN), Some(full));
        assert_eq!(roi_window(&lat(&[(0, 0), (63, 63)]), ROI_MARGIN), Some(full));
    }

    #[test]
    fn roi_window_grows_by_the_margin_and_clamps_at_the_border() {
        // Interior: bbox x 20..=24, y 30..=31, grown by 8.
        assert_eq!(
            roi_window(&lat(&[(20, 30), (24, 31)]), 8),
            Some(LatentRect { x: 12, y: 22, w: 5 + 16, h: 2 + 16 })
        );
        // Top-left corner: the window starts at 0 and only grows right/down.
        assert_eq!(roi_window(&lat(&[(0, 0), (2, 1)]), 8), Some(LatentRect { x: 0, y: 0, w: 3 + 8, h: 2 + 8 }));
        // Bottom-right corner: ends at 63.
        assert_eq!(roi_window(&lat(&[(62, 63)]), 8), Some(LatentRect { x: 54, y: 55, w: 10, h: 9 }));
        // Zero margin is the bare bbox.
        assert_eq!(roi_window(&lat(&[(5, 6)]), 0), Some(LatentRect { x: 5, y: 6, w: 1, h: 1 }));
    }
}
