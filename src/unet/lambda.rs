//! LλMI blocks: multi-query lambda self- and cross-layers (M3).
//!
//! Notation per layer: C channels, h = 8 heads, k = C/8 (query/key width),
//! v = C/8 (value width), u = 1, N = H·W tokens. BatchNorms (`norm_q`,
//! `norm_v`) are eval-mode and folded into `to_q` / `to_v` by the converter.
//!
//! # Self-λ (`MultiQuerySelfLambda`, local, r = 15)
//! ```text
//! Q[h,k,n] = to_q(x)          K[k,n] = softmax_n(to_k(x))     V[v,n] = to_v(x)
//! λc[k,v]  = Σ_n K[k,n] V[v,n]                       content lambda (global)
//! Yc[h,v,n] = Σ_k Q[h,k,n] λc[k,v]
//! λp[k,v,n] = Σ_δ W[k,δ] V[v,n+δ] + b[k]              δ ∈ 15×15 window
//! Yp[h,v,n] = Σ_k Q[h,k,n] λp[k,v,n]
//! ```
//! Upstream materialises `λp` with a `Conv3d(1, k, (1,15,15))` — 14% of the
//! UNet's FLOPs and a `[k·v·N]` intermediate. Reassociating the sums:
//! ```text
//! A[h,δ,n]  = Σ_k Q[h,k,n] W[k,δ]                      small GEMM, 225 cols
//! Yp[h,v,n] = Σ_δ A[h,δ,n] V[v,n+δ]  +  Σ_k Q[h,k,n] b[k]
//! ```
//! Cost drops from k·v·225·N to (k+v)·h·225·N: 2.5× / 5× / 10× fewer FLOPs
//! at the 64² / 32² / 16² levels (88.5 → ~21 GFLOP per CFG step). The second
//! line is a per-pixel 225-tap filter over V rows — an AXPY over the
//! contiguous NHWC channel vector, vectorised over v.
//! Output `Y = Yc + Yp`, heads×v re-flattened to C (no output projection).
//!
//! # Cross-λ (`MultiQueryCrossLambda`, global, m = 10 tokens)
//! `to_k` / `to_v` read the conditioning hidden states, which are fixed
//! (embedding ids 0..20 → text_proj). So per layer and per CFG branch,
//! `softmax(K)`, `V` and `λc = K Vᵀ` are **load-time constants**. The
//! position table `E = rel_pos_emb[n, m, k]` is indexed by an identity
//! meshgrid, so no gather is needed either. Per step only:
//! ```text
//! Q = to_q(x)                                   1x1 GEMM
//! Yc[h,v,n] = Σ_k Q[h,k,n] λc[k,v]               GEMM against a constant
//! B[h,m,n]  = Σ_k Q[h,k,n] E[n,m,k]              per-pixel h×k · k×m
//! Yp[h,v,n] = Σ_m B[h,m,n] V[m,v]                per-pixel h×m · m×v
//! ```
//! Precomputing `λp = E·V` instead would cost ~0.8 GB (15 layers × 2 CFG
//! branches × N·k·v f32) for no meaningful time win; the two-stage form keeps memory at the original
//! 14.3 M-parameter table. `E` is tied to N = 64²/32²/16², hence the hard
//! 512² input size.


use rayon::prelude::*;

use super::load::Loader;
use crate::config::NUM_HEADS;
use crate::engine::conv::Conv1x1;
use crate::engine::gemm::{MatRef, matmul};
use crate::engine::math::exp;
use crate::engine::simd::multiversion;
use crate::engine::{Scratch, Tensor, threads};
use crate::{Error, Result};

const HEADS: usize = NUM_HEADS;
/// Self-λ window: 15 × 15 taps, zero padding 7.
const WIN: usize = 15;
const TAPS: usize = WIN * WIN;
const PAD: usize = WIN / 2;
/// Folded filter width: 225 taps plus one column carrying `Q·b`.
const TAPS_B: usize = TAPS + 1;
/// Conditioning tokens per branch.
const TOKENS: usize = 10;
/// Text hidden-state width.
const EHS_DIM: usize = 768;
/// Image rows per `A = Q·W` band (`≤ 16 · 64 · 8 · 226` floats = 7 MB).
const BAND_ROWS: usize = 16;
/// Pixels per rayon task in the per-pixel kernels.
const PIXEL_CHUNK: usize = 32;
/// Rows per task of the softmax statistics passes.
const SOFTMAX_CHUNK: usize = 256;

/// Multi-query self-λ: `to_q` / `to_k` / `to_v` 1×1 convs and the 15×15
/// position filter folded as `[k, 226]` (225 taps, then the bias column).
pub struct SelfLambda {
    c: usize,
    k: usize,
    to_q: Conv1x1,
    to_k: Conv1x1,
    to_v: Conv1x1,
    w_aug: Vec<f32>,
}

impl SelfLambda {
    /// `l` is the `attn1` module; `c` its channel count (heads = 8, k = v = c/8).
    pub fn load(l: &Loader, c: usize) -> Result<Self> {
        let k = head_width(c)?;
        let pos = l.sub("pos_conv");
        let w = pos.f32s("weight", &[k, TAPS])?;
        let b = pos.vec("bias", k)?;
        let mut w_aug = vec![0.0f32; k * TAPS_B];
        for kk in 0..k {
            w_aug[kk * TAPS_B..kk * TAPS_B + TAPS].copy_from_slice(&w[kk * TAPS..(kk + 1) * TAPS]);
            w_aug[kk * TAPS_B + TAPS] = b[kk];
        }
        Ok(Self {
            c,
            k,
            to_q: l.conv1x1("to_q", c, c, true)?,
            to_k: l.conv1x1("to_k", c, k, false)?,
            to_v: l.conv1x1("to_v", c, k, true)?,
            w_aug,
        })
    }

    /// `x`: LayerNormed `[b, h, w, c]`; `out` is reshaped to the same shape.
    pub fn forward(&self, x: &Tensor, out: &mut Tensor, s: &mut Scratch) {
        assert_eq!(x.c, self.c, "SelfLambda channels");
        let (hh, ww, c, k) = (x.h, x.w, self.c, self.k);
        let hw = hh * ww;
        let (mut q, mut kt, mut vt) = (s.tensor_for(x.data.len()), s.tensor_for(x.pixels() * k), s.tensor_for(x.pixels() * k));
        self.to_q.forward(x, &mut q, None, s);
        self.to_k.forward(x, &mut kt, None, s);
        self.to_v.forward(x, &mut vt, None, s);
        out.set_shape(x.n, hh, ww, c);

        let mut a = s.vec_buf(BAND_ROWS.min(hh) * ww * HEADS * TAPS_B);
        for b in 0..x.n {
            let qi = &q.data[b * hw * c..(b + 1) * hw * c];
            let vi = &vt.data[b * hw * k..(b + 1) * hw * k];
            let oi = &mut out.data[b * hw * c..(b + 1) * hw * c];

            // Content path: Yc = Q · λc with Q viewed as [(HW·8) × k].
            let lambda_c = softmax_lambda(&mut kt.data[b * hw * k..(b + 1) * hw * k], vi, hw, k, k);
            matmul(oi, MatRef::row_major(qi, hw * HEADS, k), MatRef::row_major(&lambda_c, k, k), false, threads());

            // Position path, in bands of image rows: A = Q·W, then the tap filter.
            for y0 in (0..hh).step_by(BAND_ROWS) {
                let rows = BAND_ROWS.min(hh - y0);
                let px = rows * ww;
                let a_band = &mut a[..px * HEADS * TAPS_B];
                matmul(
                    a_band,
                    MatRef::row_major(&qi[y0 * ww * c..(y0 + rows) * ww * c], px * HEADS, k),
                    MatRef::row_major(&self.w_aug, k, TAPS_B),
                    false,
                    threads(),
                );
                let a_band = &*a_band;
                oi[y0 * ww * c..(y0 + rows) * ww * c].par_chunks_mut(PIXEL_CHUNK * c).enumerate().for_each(|(i, o)| {
                    let first = i * PIXEL_CHUNK;
                    let a = &a_band[first * HEADS * TAPS_B..];
                    tap_filter(a, vi, o, y0 * ww + first, hh, ww, k);
                });
            }
        }
        s.recycle_vec(a);
        s.recycle(q);
        s.recycle(kt);
        s.recycle(vt);
    }
}

multiversion! {
    /// Adds `Yp[n,h,:] = Σ_δ A[n,h,δ]·V[n+δ,:] + A[n,h,225]` to the pixels in `o`
    /// (zero padding outside the `hh × ww` map). `a`: `[px × 8 × 226]` for the
    /// same pixels; `first`: item-level index of the first pixel.
    fn tap_filter(a: &[f32], v: &[f32], o: &mut [f32], first: usize, hh: usize, ww: usize, vd: usize) {
        let c = HEADS * vd;
        for (i, o) in o.chunks_exact_mut(c).enumerate() {
            let (y, x) = ((first + i) / ww, (first + i) % ww);
            let a = &a[i * HEADS * TAPS_B..(i + 1) * HEADS * TAPS_B];
            for h in 0..HEADS {
                let bias = a[h * TAPS_B + TAPS];
                o[h * vd..(h + 1) * vd].iter_mut().for_each(|t| *t += bias);
            }
            let kx0 = PAD.saturating_sub(x);
            let kx1 = WIN.min(ww + PAD - x);
            for ky in 0..WIN {
                let Some(iy) = (y + ky).checked_sub(PAD).filter(|&iy| iy < hh) else { continue };
                for kx in kx0..kx1 {
                    let src = (iy * ww + x + kx - PAD) * vd;
                    let vr = &v[src..src + vd];
                    let d = ky * WIN + kx;
                    for h in 0..HEADS {
                        let coef = a[h * TAPS_B + d];
                        for (t, &vv) in o[h * vd..(h + 1) * vd].iter_mut().zip(vr) {
                            *t = coef.mul_add(vv, *t);
                        }
                    }
                }
            }
        }
    }
}

/// Multi-query cross-λ against fixed conditioning: per CFG branch, the
/// softmaxed-key content lambda `λc` and the values `V` are load constants.
pub struct CrossLambda {
    c: usize,
    k: usize,
    /// Pixels the position table was trained for (64², 32², 16²).
    n: usize,
    to_q: Conv1x1,
    /// `rel_pos_emb` `[N, 10, k]`.
    e: Vec<f32>,
    /// Per branch `[k × v]`.
    lambda_c: [Vec<f32>; 2],
    /// Per branch `[10 × v]`.
    v: [Vec<f32>; 2],
}

impl CrossLambda {
    /// `l` is the `attn2` module; `ehs[0]` uncond and `ehs[1]` cond, each `[10 × 768]` row-major.
    pub fn load(l: &Loader, c: usize, ehs: &[Vec<f32>; 2]) -> Result<Self> {
        let k = head_width(c)?;
        // The table is indexed by pixel: 64² at c = 320, 32² at 640, 16² at 1280.
        let side = 64 * 320 / c;
        if side * c != 64 * 320 {
            return Err(Error::Weights(format!("cross-lambda: no position table size for c = {c}")));
        }
        let n = side * side;
        for e in ehs {
            if e.len() != TOKENS * EHS_DIM {
                return Err(Error::InvalidInput(format!(
                    "hidden states have {} values, expected {}",
                    e.len(),
                    TOKENS * EHS_DIM
                )));
            }
        }
        let wk = l.f32s("to_k.weight", &[k, EHS_DIM])?;
        let wv = l.f32s("to_v.weight", &[k, EHS_DIM])?;
        let bv = l.vec("to_v.bias", k)?;
        let project = |ehs: &[f32], w: &[f32]| {
            let mut y = vec![0.0f32; TOKENS * k];
            matmul(&mut y, MatRef::row_major(ehs, TOKENS, EHS_DIM), MatRef::transposed(w, EHS_DIM, k), false, 1);
            y
        };
        let branch = |ehs: &[f32]| {
            let mut keys = project(ehs, &wk);
            let mut v = project(ehs, &wv);
            for row in v.chunks_exact_mut(k) {
                row.iter_mut().zip(&bv).for_each(|(t, &b)| *t += b);
            }
            let lambda_c = softmax_lambda(&mut keys, &v, TOKENS, k, k);
            (lambda_c, v)
        };
        let ((lc0, v0), (lc1, v1)) = (branch(&ehs[0]), branch(&ehs[1]));
        Ok(Self {
            c,
            k,
            n,
            to_q: l.conv1x1("to_q", c, c, true)?,
            e: l.f32s("rel_pos_emb", &[n, TOKENS, k])?,
            lambda_c: [lc0, lc1],
            v: [v0, v1],
        })
    }

    /// `x`: LayerNormed `[b, h, w, c]`; item `i` uses conditioning branch `branches[i]`.
    pub fn forward(&self, x: &Tensor, branches: &[usize], out: &mut Tensor, s: &mut Scratch) {
        assert_eq!(x.c, self.c, "CrossLambda channels");
        assert_eq!(branches.len(), x.n, "one branch per batch item");
        assert_eq!(x.h * x.w, self.n, "CrossLambda input size does not match the position table");
        let (c, k, hw) = (self.c, self.k, x.h * x.w);
        let mut q = s.tensor_for(x.data.len());
        self.to_q.forward(x, &mut q, None, s);
        out.set_shape(x.n, x.h, x.w, c);
        for (b, &br) in branches.iter().enumerate() {
            assert!(br < 2, "branch must be 0 (uncond) or 1 (cond)");
            let qi = &q.data[b * hw * c..(b + 1) * hw * c];
            let oi = &mut out.data[b * hw * c..(b + 1) * hw * c];
            // Yc = Q · λc[branch]
            matmul(oi, MatRef::row_major(qi, hw * HEADS, k), MatRef::row_major(&self.lambda_c[br], k, k), false, threads());
            // Yp = (Q·Eᵀ) · V[branch], per pixel.
            oi.par_chunks_mut(PIXEL_CHUNK * c).zip(qi.par_chunks(PIXEL_CHUNK * c)).enumerate().for_each(|(i, (o, q))| {
                let e = &self.e[i * PIXEL_CHUNK * TOKENS * k..];
                position_term(q, e, &self.v[br], o, k);
            });
        }
        s.recycle(q);
    }
}

multiversion! {
    /// Per pixel: `B[h,m] = Σ_kk Q[h,kk]·E[m,kk]`, then `o[h,:] += Σ_m B[h,m]·V[m,:]`.
    /// `q`, `o`: `[px × 8·k]`; `e`: `[px × 10 × k]`; `v`: `[10 × k]` (v = k).
    fn position_term(q: &[f32], e: &[f32], v: &[f32], o: &mut [f32], k: usize) {
        let c = HEADS * k;
        for (i, (q, o)) in q.chunks_exact(c).zip(o.chunks_exact_mut(c)).enumerate() {
            let e = &e[i * TOKENS * k..(i + 1) * TOKENS * k];
            for h in 0..HEADS {
                let qh = &q[h * k..(h + 1) * k];
                let oh = &mut o[h * k..(h + 1) * k];
                for m in 0..TOKENS {
                    let b = dot(qh, &e[m * k..(m + 1) * k]);
                    for (t, &vv) in oh.iter_mut().zip(&v[m * k..(m + 1) * k]) {
                        *t = b.mul_add(vv, *t);
                    }
                }
            }
        }
    }
}

/// Dot product with 8 independent lanes so it vectorises without reassociation.
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (ac, bc) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ac.remainder().iter().zip(bc.remainder()).map(|(x, y)| x * y).sum();
    for (a, b) in ac.zip(bc) {
        for l in 0..8 {
            acc[l] = a[l].mul_add(b[l], acc[l]);
        }
    }
    acc.iter().sum::<f32>() + tail
}

fn head_width(c: usize) -> Result<usize> {
    if c % HEADS != 0 || (c / HEADS) % 8 != 0 {
        return Err(Error::Weights(format!("lambda layer: {c} channels do not split into {HEADS} heads of 8-aligned width")));
    }
    Ok(c / HEADS)
}

/// `λc[kk][j] = Σ_r softmax_r(keys)[r][kk] · v[r][j]` for `keys`: `[rows × k]`
/// (softmax over rows, per channel) and `v`: `[rows × vd]`. `keys` is
/// overwritten with `exp(key - colmax)`; the normalisation is applied to the
/// small `[k × vd]` result.
fn softmax_lambda(keys: &mut [f32], v: &[f32], rows: usize, k: usize, vd: usize) -> Vec<f32> {
    let keys = &mut keys[..rows * k];
    let max = keys
        .par_chunks(SOFTMAX_CHUNK * k)
        .map(|chunk| {
            let mut m = vec![f32::NEG_INFINITY; k];
            column_max(chunk, &mut m);
            m
        })
        .reduce(
            || vec![f32::NEG_INFINITY; k],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(a, b)| *a = a.max(b));
                a
            },
        );
    let sum = keys
        .par_chunks_mut(SOFTMAX_CHUNK * k)
        .map(|chunk| {
            let mut s = vec![0.0f32; k];
            exp_and_sum(chunk, &max, &mut s);
            s.into_iter().map(f64::from).collect::<Vec<f64>>()
        })
        .reduce(
            || vec![0.0f64; k],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
                a
            },
        );
    let mut lambda = vec![0.0f32; k * vd];
    matmul(&mut lambda, MatRef::transposed(keys, k, rows), MatRef::row_major(v, rows, vd), false, threads());
    for (row, &s) in lambda.chunks_exact_mut(vd).zip(&sum) {
        let inv = (1.0 / s) as f32;
        row.iter_mut().for_each(|t| *t *= inv);
    }
    lambda
}

multiversion! {
    fn column_max(x: &[f32], m: &mut [f32]) {
        for row in x.chunks_exact(m.len()) {
            for (m, &v) in m.iter_mut().zip(row) {
                *m = m.max(v);
            }
        }
    }
}

multiversion! {
    fn exp_and_sum(x: &mut [f32], max: &[f32], sum: &mut [f32]) {
        for row in x.chunks_exact_mut(max.len()) {
            for ((v, &m), s) in row.iter_mut().zip(max).zip(sum.iter_mut()) {
                *v = exp(*v - m);
                *s += *v;
            }
        }
    }
}
