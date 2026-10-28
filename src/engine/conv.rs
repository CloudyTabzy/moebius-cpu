//! Dense convolutions.
//!
//! * `Conv1x1` — `[pixels × c_in]·[c_in × c_out]` through `gemm`, with bias
//!   and residual fused by pre-filling the output and accumulating into it.
//!   Also serves every linear layer (tokens = pixels).
//! * `Conv3x3` — stride 1 / padding 1 (all resnet, VAE and `Upsample2D`
//!   convs) goes through Winograd F(2×2,3×3) (`winograd.rs`). Everything
//!   else (stride-2 `Downsample2D`, the VAE's asymmetric padding) uses tiled
//!   implicit im2col: output pixels in bands whose `[band × 9·c_in]` patch
//!   matrix fits `COLS_BUDGET`, one multi-threaded GEMM per band. A fused
//!   nearest-2× upsample of the input (`Upsample2D`, so the 4× larger tensor
//!   is never stored) is just a different gather in both paths.
//!
//! Weight layouts (written by the converter): 1x1 `[c_out, c_in]`,
//! 3x3 `[c_out, 3, 3, c_in]`, both read as transposed GEMM operands.

use rayon::prelude::*;

use super::gemm::{MatRef, matmul};
use super::{Param, Scratch, Tensor, threads};

/// Patch-matrix budget in floats (32 MB). Bands are ≤ 4096 pixels, and fewer
/// when 9·c_in is large (728 px for the 1280-channel UNet upsample conv).
pub const COLS_BUDGET: usize = 8 << 20;
const MAX_BAND: usize = 4096;

pub struct Conv1x1 {
    pub c_in: usize,
    pub c_out: usize,
    /// `[c_out, c_in]`
    pub weight: Param,
    pub bias: Option<Vec<f32>>,
}

impl Conv1x1 {
    /// `out = x·Wᵀ + bias (+ residual)`; `out` is reshaped to `[n, h, w, c_out]`.
    pub fn forward(&self, x: &Tensor, out: &mut Tensor, residual: Option<&Tensor>, s: &mut Scratch) {
        assert_eq!(x.c, self.c_in, "Conv1x1 input channels");
        assert_eq!(self.weight.len(), self.c_in * self.c_out, "Conv1x1 weight size");
        out.set_shape(x.n, x.h, x.w, self.c_out);
        if let Some(r) = residual {
            assert_eq!(r.shape(), out.shape(), "Conv1x1 residual shape");
        }
        let w = self.weight.view(&mut s.panel);
        let accumulate = init_output(&mut out.data, self.c_out, self.bias.as_deref(), residual.map(|r| &r.data[..]));
        let rows = x.pixels();
        matmul(
            &mut out.data,
            MatRef::row_major(&x.data, rows, self.c_in),
            MatRef::transposed(w, self.c_in, self.c_out),
            accumulate,
            threads(),
        );
    }
}

pub struct Conv3x3 {
    pub c_in: usize,
    pub c_out: usize,
    /// `[c_out, 3, 3, c_in]`
    pub weight: Param,
    pub bias: Option<Vec<f32>>,
    pub stride: usize,
    /// Zero padding as `[top, left, bottom, right]`.
    pub pad: [usize; 4],
    /// Read the input as if nearest-upsampled 2× first.
    pub upsample2x: bool,
}

impl Conv3x3 {
    /// Stride 1, padding 1: resnet / VAE convs.
    pub fn same(c_in: usize, c_out: usize, weight: Param, bias: Option<Vec<f32>>) -> Self {
        Self { c_in, c_out, weight, bias, stride: 1, pad: [1; 4], upsample2x: false }
    }

    pub fn output_hw(&self, h: usize, w: usize) -> (usize, usize) {
        let up = if self.upsample2x { 2 } else { 1 };
        let [pt, pl, pb, pr] = self.pad;
        ((h * up + pt + pb - 3) / self.stride + 1, (w * up + pl + pr - 3) / self.stride + 1)
    }

    pub fn forward(&self, x: &Tensor, out: &mut Tensor, s: &mut Scratch) {
        assert_eq!(x.c, self.c_in, "Conv3x3 input channels");
        assert_eq!(self.weight.len(), 9 * self.c_in * self.c_out, "Conv3x3 weight size");
        if self.stride == 1 && self.pad == [1; 4] {
            super::winograd::conv3x3_same(x, self.upsample2x, &self.weight, self.bias.as_deref(), self.c_out, out, s);
        } else {
            self.forward_im2col(x, out, s);
        }
    }

    /// Direct (im2col + GEMM) path for any stride/padding; reference for
    /// the Winograd path in tests.
    pub fn forward_im2col(&self, x: &Tensor, out: &mut Tensor, s: &mut Scratch) {
        assert_eq!(x.c, self.c_in, "Conv3x3 input channels");
        let k = 9 * self.c_in;
        assert_eq!(self.weight.len(), k * self.c_out, "Conv3x3 weight size");
        let (ho, wo) = self.output_hw(x.h, x.w);
        out.set_shape(x.n, ho, wo, self.c_out);

        let Scratch { panel, cols, .. } = s;
        let w = self.weight.view(panel);
        let total = x.n * ho * wo;
        let band = (COLS_BUDGET / k).clamp(64, MAX_BAND).min(total);
        if cols.len() < band * k {
            cols.resize(band * k, 0.0);
        }
        let geo = Geometry { x, ho, wo, stride: self.stride, pad_top: self.pad[0], pad_left: self.pad[1], up: self.upsample2x };

        for start in (0..total).step_by(band) {
            let m = band.min(total - start);
            let cols = &mut cols[..m * k];
            cols.par_chunks_mut(k).with_min_len(16).enumerate().for_each(|(i, row)| geo.gather(start + i, row));
            let dst = &mut out.data[start * self.c_out..(start + m) * self.c_out];
            let accumulate = init_output(dst, self.c_out, self.bias.as_deref(), None);
            matmul(dst, MatRef::row_major(cols, m, k), MatRef::transposed(w, k, self.c_out), accumulate, threads());
        }
    }
}

struct Geometry<'a> {
    x: &'a Tensor,
    ho: usize,
    wo: usize,
    stride: usize,
    pad_top: usize,
    pad_left: usize,
    up: bool,
}

impl Geometry<'_> {
    /// Fills one patch row `[3, 3, c_in]` for output pixel `p`.
    fn gather(&self, p: usize, row: &mut [f32]) {
        let x = self.x;
        let (b, rem) = (p / (self.ho * self.wo), p % (self.ho * self.wo));
        let (oy, ox) = (rem / self.wo, rem % self.wo);
        let shift = usize::from(self.up);
        let (hi, wi) = (x.h << shift, x.w << shift);
        let c = x.c;
        for ky in 0..3 {
            let iy = (oy * self.stride + ky).wrapping_sub(self.pad_top);
            for kx in 0..3 {
                let ix = (ox * self.stride + kx).wrapping_sub(self.pad_left);
                let dst = &mut row[(ky * 3 + kx) * c..][..c];
                if iy < hi && ix < wi {
                    let src = ((b * x.h + (iy >> shift)) * x.w + (ix >> shift)) * c;
                    dst.copy_from_slice(&x.data[src..src + c]);
                } else {
                    dst.fill(0.0);
                }
            }
        }
    }
}

/// Pre-fills `out` rows with `residual + bias`; returns whether the GEMM
/// must accumulate into it.
pub(crate) fn init_output(out: &mut [f32], cols: usize, bias: Option<&[f32]>, residual: Option<&[f32]>) -> bool {
    const MIN_ROWS: usize = 64;
    match (bias, residual) {
        (None, None) => false,
        (Some(b), None) => {
            out.par_chunks_mut(cols).with_min_len(MIN_ROWS).for_each(|row| row.copy_from_slice(b));
            true
        }
        (None, Some(r)) => {
            out.par_chunks_mut(cols).with_min_len(MIN_ROWS).zip(r.par_chunks(cols)).for_each(|(o, r)| o.copy_from_slice(r));
            true
        }
        (Some(b), Some(r)) => {
            out.par_chunks_mut(cols).with_min_len(MIN_ROWS).zip(r.par_chunks(cols)).for_each(|(o, r)| {
                for ((o, &r), &b) in o.iter_mut().zip(r).zip(b) {
                    *o = r + b;
                }
            });
            true
        }
    }
}
