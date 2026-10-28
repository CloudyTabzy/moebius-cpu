//! Depthwise 3x3 convolution, stride 1, zero padding 1.
//!
//! Used by every `DWConv2d` (`conv_in`, `conv_out`, both resnet convs) and by
//! the MixFFN depth conv on 1600 / 3200 / 6400 channels. Only 0.5% of UNet
//! FLOPs but memory-bound: it saturates DRAM at 4–6
//! threads (2×64²×1600 GLU: ~3.4 ms vs PyTorch 10 ms; all depthwise convs
//! of one UNet step ≈ 45 ms). NHWC makes the inner
//! loop a contiguous channel vector FMA'd across the nine taps; each task
//! owns a strip of output rows so shared input rows are read from DRAM
//! once. Epilogues are fused:
//!
//! * `Relu` — `DWConv2d`'s folded `bn1` + ReLU, ahead of the pointwise GEMM.
//! * `GluSilu` — MixFFN: `y[..c/2] · SiLU(y[c/2..])`, so the 2× wide
//!   intermediate is never written out.

use rayon::prelude::*;

use super::Tensor;
use super::math::silu;
use super::simd::multiversion;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Epilogue {
    None,
    Relu,
    /// Output has c/2 channels.
    GluSilu,
}

/// `weight`: `[3, 3, c]`, `bias`: `[c]`.
pub fn depthwise3x3(x: &Tensor, weight: &[f32], bias: &[f32], epilogue: Epilogue, out: &mut Tensor) {
    let c = x.c;
    assert_eq!(weight.len(), 9 * c, "depthwise weight size");
    assert_eq!(bias.len(), c, "depthwise bias size");
    let c_out = if epilogue == Epilogue::GluSilu {
        assert!(c % 2 == 0, "GLU needs an even channel count");
        c / 2
    } else {
        c
    };
    out.set_shape(x.n, x.h, x.w, c_out);
    let (h, w) = (x.h, x.w);
    let item = h * w * c;
    let rows = x.n * h;
    // Contiguous strips of output rows per task: neighbouring rows share two
    // of their three input rows, which then come from this core's cache
    // instead of DRAM (one row per task re-read the input up to 3×).
    let strip = (rows / (3 * super::threads())).max(1);
    out.data.par_chunks_mut(strip * w * c_out).enumerate().for_each_init(
        || vec![0.0f32; c],
        |acc, (i, dst)| {
            for (r, dst) in dst.chunks_exact_mut(w * c_out).enumerate() {
                let row = i * strip + r;
                let (b, oy) = (row / h, row % h);
                dw_row(&x.data[b * item..(b + 1) * item], h, w, c, oy, weight, bias, epilogue, acc, dst);
            }
        },
    );
}

/// Register block: 32 channels = 4 AVX2 registers per pixel accumulator.
const CB: usize = 32;

multiversion! {
    /// One output row. Channels are processed in register blocks of `CB` for
    /// two neighbouring pixels at once, so every accumulator stays in a
    /// register across the nine taps and every tap load feeds two FMAs.
    /// Channel counts that are not multiples of `CB` finish with a slice path.
    #[allow(clippy::too_many_arguments)]
    fn dw_row(
        x: &[f32], h: usize, w: usize, c: usize, oy: usize,
        weight: &[f32], bias: &[f32], epilogue: Epilogue, acc: &mut [f32], dst: &mut [f32],
    ) {
        let glu = epilogue == Epilogue::GluSilu;
        let c_out = if glu { c / 2 } else { c };
        let full = c_out / CB * CB;
        for ox in (0..w).step_by(2) {
            let pair = ox + 1 < w;
            for j0 in (0..full).step_by(CB) {
                let (v0, v1) = acc_block(x, h, w, c, oy, ox, pair, weight, bias, j0);
                let gates = glu.then(|| acc_block(x, h, w, c, oy, ox, pair, weight, bias, c_out + j0));
                for (p, v) in [(ox, &v0), (ox + 1, &v1)].into_iter().take(1 + usize::from(pair)) {
                    let d = &mut dst[p * c_out + j0..][..CB];
                    match (epilogue, &gates) {
                        (Epilogue::GluSilu, Some((g0, g1))) => {
                            let g = if p == ox { g0 } else { g1 };
                            for ((d, &v), &g) in d.iter_mut().zip(v).zip(g) {
                                *d = v * silu(g);
                            }
                        }
                        (Epilogue::Relu, _) => {
                            for (d, &v) in d.iter_mut().zip(v) {
                                *d = v.max(0.0);
                            }
                        }
                        _ => d.copy_from_slice(v),
                    }
                }
            }
            if full < c_out {
                for p in ox..(ox + 2).min(w) {
                    tail_pixel(x, h, w, c, oy, p, weight, bias, epilogue, full, acc, dst);
                }
            }
        }
    }
}

/// Bias + nine taps for channels `ch..ch+CB` of pixels `ox` and `ox+1`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn acc_block(
    x: &[f32], h: usize, w: usize, c: usize, oy: usize, ox: usize, pair: bool,
    weight: &[f32], bias: &[f32], ch: usize,
) -> ([f32; CB], [f32; CB]) {
    let b: [f32; CB] = bias[ch..ch + CB].try_into().unwrap();
    let (mut a0, mut a1) = (b, b);
    for ky in 0..3 {
        let iy = (oy + ky).wrapping_sub(1);
        if iy >= h {
            continue;
        }
        let row = &x[iy * w * c..(iy + 1) * w * c];
        for kx in 0..3 {
            let t: &[f32; CB] = weight[(ky * 3 + kx) * c + ch..][..CB].try_into().unwrap();
            let ix0 = (ox + kx).wrapping_sub(1);
            if ix0 < w {
                let s: &[f32; CB] = row[ix0 * c + ch..][..CB].try_into().unwrap();
                for i in 0..CB {
                    a0[i] = t[i].mul_add(s[i], a0[i]);
                }
            }
            let ix1 = ox + kx;
            if pair && ix1 < w {
                let s: &[f32; CB] = row[ix1 * c + ch..][..CB].try_into().unwrap();
                for i in 0..CB {
                    a1[i] = t[i].mul_add(s[i], a1[i]);
                }
            }
        }
    }
    (a0, a1)
}

/// Output channels `from..c_out` of one pixel (channel-count remainder).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tail_pixel(
    x: &[f32], h: usize, w: usize, c: usize, oy: usize, ox: usize,
    weight: &[f32], bias: &[f32], epilogue: Epilogue, from: usize, acc: &mut [f32], dst: &mut [f32],
) {
    acc.copy_from_slice(bias);
    for ky in 0..3 {
        let iy = (oy + ky).wrapping_sub(1);
        if iy >= h {
            continue;
        }
        for kx in 0..3 {
            let ix = (ox + kx).wrapping_sub(1);
            if ix >= w {
                continue;
            }
            let src = &x[(iy * w + ix) * c..][..c];
            let tap = &weight[(ky * 3 + kx) * c..][..c];
            for ((a, &s), &k) in acc.iter_mut().zip(src).zip(tap) {
                *a = k.mul_add(s, *a);
            }
        }
    }
    match epilogue {
        Epilogue::None => dst[ox * c + from..][..c - from].copy_from_slice(&acc[from..]),
        Epilogue::Relu => {
            for (d, &a) in dst[ox * c + from..][..c - from].iter_mut().zip(&acc[from..]) {
                *d = a.max(0.0);
            }
        }
        Epilogue::GluSilu => {
            let half = c / 2;
            let (val, gate) = acc.split_at(half);
            for ((d, &v), &g) in dst[ox * half + from..][..half - from].iter_mut().zip(&val[from..]).zip(&gate[from..]) {
                *d = v * silu(g);
            }
        }
    }
}
