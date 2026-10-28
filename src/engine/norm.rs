//! Normalisations, each a stats pass plus one fused apply pass.
//!
//! * `GroupNorm` — statistics are per-channel f32 partial sums over pixel
//!   chunks (vectorised across channels), folded into f64 per group. The apply
//!   pass is `x·scale[c] + shift[c]` with optional SiLU, and an optional
//!   per-item channel vector added *before* normalising (the resnet time
//!   embedding, `h + temb → norm2`) is folded into `shift` — never written.
//! * `layer_norm` — per token over channels (transformer `norm1..3`).
//!
//! BatchNorms do not appear here: all are eval-mode and folded into the
//! adjacent conv by the converter.

use rayon::prelude::*;

use super::Tensor;
use super::math::silu;
use super::simd::multiversion;

/// Pixels per stats task.
const STATS_CHUNK: usize = 256;

pub struct GroupNorm {
    pub groups: usize,
    pub eps: f32,
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
}

/// Per-item, per-channel `y = [SiLU](x·scale + shift)` that a GroupNorm applies
/// (item-major, `n·c` entries each). Splitting it from the apply pass lets a
/// consumer (the Winograd conv's input gather) apply it on the fly instead of
/// reading a materialised normalised tensor.
pub struct Affine {
    pub scale: Vec<f32>,
    pub shift: Vec<f32>,
    pub silu: bool,
}

impl GroupNorm {
    /// `out = [SiLU](GN(x + add[item]))`. `add`: `[n, c]` or `None`.
    pub fn forward(&self, x: &Tensor, add: Option<&[f32]>, silu_out: bool, out: &mut Tensor) {
        let aff = self.affine(x, add, silu_out);
        let c = x.c;
        out.set_shape(x.n, x.h, x.w, c);
        let item = x.h * x.w * c;
        for b in 0..x.n {
            let (scale, shift) = (&aff.scale[b * c..(b + 1) * c], &aff.shift[b * c..(b + 1) * c]);
            out.data[b * item..(b + 1) * item]
                .par_chunks_mut(STATS_CHUNK * c)
                .zip(x.data[b * item..(b + 1) * item].par_chunks(STATS_CHUNK * c))
                .for_each(|(dst, src)| apply_affine(src, scale, shift, silu_out, dst));
        }
    }

    /// The statistics pass: per-item channel `scale`/`shift` such that
    /// `GN(x + add) = x·scale + shift`. `add`: `[n, c]` or `None`.
    pub fn affine(&self, x: &Tensor, add: Option<&[f32]>, silu_out: bool) -> Affine {
        let (c, g) = (x.c, self.groups);
        assert!(c % g == 0, "channels not divisible by groups");
        assert_eq!(self.gamma.len(), c, "GroupNorm gamma size");
        if let Some(a) = add {
            assert_eq!(a.len(), x.n * c, "GroupNorm add size");
        }
        let item = x.h * x.w * c;
        let count = (x.h * x.w * (c / g)) as f64;
        let mut scale = vec![0.0f32; x.n * c];
        let mut shift = vec![0.0f32; x.n * c];
        for b in 0..x.n {
            let src = &x.data[b * item..(b + 1) * item];
            let add_b = add.map(|a| &a[b * c..(b + 1) * c]);
            // Per-channel f32 partials over ≤ STATS_CHUNK pixels (vectorised
            // over channels), folded into f64 per group.
            let (sum, sumsq) = src
                .par_chunks(STATS_CHUNK * c)
                .map(|chunk| {
                    let (mut s, mut q) = (vec![0.0f32; c], vec![0.0f32; c]);
                    gn_stats(chunk, add_b, &mut s, &mut q);
                    let fold = |v: Vec<f32>| -> Vec<f64> {
                        v.chunks_exact(c / g).map(|grp| grp.iter().map(|&x| x as f64).sum()).collect()
                    };
                    (fold(s), fold(q))
                })
                .reduce(
                    || (vec![0.0; g], vec![0.0; g]),
                    |(mut s, mut q), (s2, q2)| {
                        s.iter_mut().zip(s2).for_each(|(a, b)| *a += b);
                        q.iter_mut().zip(q2).for_each(|(a, b)| *a += b);
                        (s, q)
                    },
                );
            let cg = c / g;
            for gi in 0..g {
                let mean = sum[gi] / count;
                let var = (sumsq[gi] / count - mean * mean).max(0.0);
                let rstd = 1.0 / (var + self.eps as f64).sqrt();
                for ch in gi * cg..(gi + 1) * cg {
                    let sc = rstd * self.gamma[ch] as f64;
                    let pre = add_b.map_or(0.0, |a| a[ch] as f64);
                    scale[b * c + ch] = sc as f32;
                    shift[b * c + ch] = (self.beta[ch] as f64 + (pre - mean) * sc) as f32;
                }
            }
        }
        Affine { scale, shift, silu: silu_out }
    }
}

multiversion! {
    /// Per-channel Σy and Σy² over the pixels of `chunk`, y = x (+ add).
    fn gn_stats(chunk: &[f32], add: Option<&[f32]>, sum: &mut [f32], sumsq: &mut [f32]) {
        let c = sum.len();
        for px in chunk.chunks_exact(c) {
            match add {
                Some(a) => {
                    for (((s, q), &x), &a) in sum.iter_mut().zip(sumsq.iter_mut()).zip(px).zip(a) {
                        let y = x + a;
                        *s += y;
                        *q = y.mul_add(y, *q);
                    }
                }
                None => {
                    for ((s, q), &x) in sum.iter_mut().zip(sumsq.iter_mut()).zip(px) {
                        *s += x;
                        *q = x.mul_add(x, *q);
                    }
                }
            }
        }
    }
}

multiversion! {
    /// `dst = [SiLU](src·scale + shift)` per channel.
    pub(crate) fn apply_affine(src: &[f32], scale: &[f32], shift: &[f32], silu_out: bool, dst: &mut [f32]) {
        let c = scale.len();
        for (d, s) in dst.chunks_exact_mut(c).zip(src.chunks_exact(c)) {
            if silu_out {
                for (((d, &x), &a), &b) in d.iter_mut().zip(s).zip(scale).zip(shift) {
                    *d = silu(x.mul_add(a, b));
                }
            } else {
                for (((d, &x), &a), &b) in d.iter_mut().zip(s).zip(scale).zip(shift) {
                    *d = x.mul_add(a, b);
                }
            }
        }
    }
}

/// LayerNorm over the channel axis of every pixel/token.
pub fn layer_norm(x: &Tensor, gamma: &[f32], beta: &[f32], eps: f32, out: &mut Tensor) {
    let c = x.c;
    assert_eq!(gamma.len(), c, "LayerNorm gamma size");
    out.set_shape(x.n, x.h, x.w, c);
    out.data
        .par_chunks_mut(64 * c)
        .zip(x.data.par_chunks(64 * c))
        .for_each(|(dst, src)| ln_rows(src, gamma, beta, eps, dst));
}

multiversion! {
    fn ln_rows(src: &[f32], gamma: &[f32], beta: &[f32], eps: f32, dst: &mut [f32]) {
        let c = gamma.len();
        for (d, s) in dst.chunks_exact_mut(c).zip(src.chunks_exact(c)) {
            let mean = s.iter().sum::<f32>() / c as f32;
            let var = s.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
            let rstd = 1.0 / (var + eps).sqrt();
            for (((d, &v), &g), &b) in d.iter_mut().zip(s).zip(gamma).zip(beta) {
                *d = ((v - mean) * rstd).mul_add(g, b);
            }
        }
    }
}
