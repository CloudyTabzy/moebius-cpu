//! Winograd F(2×2, 3×3) for every stride-1, padding-1 3x3 conv.
//!
//! Those convs are ~40% of an Exact inpaint's FLOPs (the whole VAE plus the
//! UNet `Upsample2D` convs). Each 4×4 input tile is transformed (`Bᵀ d B`),
//! multiplied channel-wise against the transformed kernel (`G g Gᵀ`) as 16
//! independent GEMMs, and transformed back (`Aᵀ m A`) into a 2×2 output
//! tile: 16 multiplies per 4 outputs instead of 36, a 2.25× cut. F(2,3) has
//! small transform constants (0, ±1, ½), so f32 error stays at the 1e-6
//! level, well inside parity tolerances.
//!
//! NHWC makes all three transforms elementwise over contiguous channel
//! vectors. Work is banded over whole tile rows so buffers stay bounded:
//! * input tiles `V`: `[band][16][c_in]` (tile-major, filled in parallel),
//! * products `M`: `[16][band][c_out_block]` (one plane per GEMM),
//! * kernel `U`: `[c_out_block][16][c_in]`, transformed once per call when it
//!   fits `U_BUDGET`, otherwise per output-channel block (UNet 1280 conv).
//!
//! The 16 GEMMs run concurrently, single-threaded each: measured 506–680
//! GFLOPS on Moebius shapes vs 340–505 running them one after another with
//! internal parallelism (`examples/wino_probe.rs`).

use std::ops::Range;

use half::slice::HalfFloatSliceExt;
use rayon::prelude::*;

use super::gemm::{MatRef, matmul};
use super::simd::multiversion;
use super::{Param, Scratch, Tensor};

/// Floats for the `V` band (16 MB).
const V_BUDGET: usize = 4 << 20;
/// Floats for the transformed kernel block (32 MB).
const U_BUDGET: usize = 8 << 20;

/// `out = conv3x3(x [nearest-2× upsampled], pad 1) + bias`.
/// `weight`: `[c_out, 3, 3, c_in]`.
pub(crate) fn conv3x3_same(
    x: &Tensor,
    upsample2x: bool,
    weight: &Param,
    bias: Option<&[f32]>,
    c_out: usize,
    out: &mut Tensor,
    s: &mut Scratch,
) {
    let ci = x.c;
    let shift = usize::from(upsample2x);
    let (hi, wi) = (x.h << shift, x.w << shift);
    out.set_shape(x.n, hi, wi, c_out);
    let (th, tw) = (hi.div_ceil(2), wi.div_ceil(2));
    let tiles = x.n * th * tw;
    // Bands are whole tile rows so each output row pair belongs to one band.
    let band = ((V_BUDGET / (16 * ci)).clamp(256, 4096) / tw).max(1) * tw;
    let band = band.min(tiles);
    let oblk = if 16 * ci * c_out <= U_BUDGET { c_out } else { (U_BUDGET / (16 * ci) / 16 * 16).max(16) };

    grow(&mut s.wino_v, 16 * band * ci);
    grow(&mut s.wino_m, 16 * band * oblk);
    grow(&mut s.wino_u, 16 * oblk * ci);
    let zeros = vec![0.0f32; ci];
    let bias_zero;
    let bias = match bias {
        Some(b) => b,
        None => {
            bias_zero = vec![0.0f32; c_out];
            &bias_zero
        }
    };
    if oblk == c_out {
        transform_kernel(weight, ci, 0..c_out, &mut s.wino_u);
    }
    let geo = Geo { x, shift, hi, wi, th, tw };

    for t0 in (0..tiles).step_by(band) {
        let mb = band.min(tiles - t0);
        let v = &mut s.wino_v[..16 * mb * ci];
        v.par_chunks_mut(16 * ci).with_min_len(8).enumerate().for_each(|(i, vt)| {
            let rows = geo.input_tile(t0 + i, &zeros);
            input_transform(&rows, vt);
        });
        let v = &s.wino_v[..16 * mb * ci];

        for o0 in (0..c_out).step_by(oblk) {
            let ob = oblk.min(c_out - o0);
            if oblk != c_out {
                transform_kernel(weight, ci, o0..o0 + ob, &mut s.wino_u);
            }
            let u = &s.wino_u[..16 * ob * ci];
            s.wino_m[..16 * mb * ob].par_chunks_mut(mb * ob).enumerate().for_each(|(xi, plane)| {
                let a = MatRef::strided(&v[xi * ci..], mb, ci, 16 * ci, 1);
                let b = MatRef::strided(&u[xi * ci..], ci, ob, 1, 16 * ci);
                matmul(plane, a, b, false, 1);
            });
            let m = &s.wino_m[..16 * mb * ob];
            geo.output_band(t0, mb, m, &bias[o0..o0 + ob], o0, c_out, &mut out.data);
        }
    }
}

fn grow(v: &mut Vec<f32>, len: usize) {
    if v.len() < len {
        v.resize(len, 0.0);
    }
}

struct Geo<'a> {
    x: &'a Tensor,
    shift: usize,
    hi: usize,
    wi: usize,
    th: usize,
    tw: usize,
}

impl Geo<'_> {
    /// The 16 input pixel vectors of tile `t` (zero vector where padded).
    fn input_tile<'z>(&'z self, t: usize, zeros: &'z [f32]) -> [&'z [f32]; 16] {
        let x = self.x;
        let (b, rem) = (t / (self.th * self.tw), t % (self.th * self.tw));
        let (ty, tx) = (rem / self.tw, rem % self.tw);
        let c = x.c;
        std::array::from_fn(|k| {
            let iy = (2 * ty + k / 4).wrapping_sub(1);
            let ix = (2 * tx + k % 4).wrapping_sub(1);
            if iy < self.hi && ix < self.wi {
                let p = ((b * x.h + (iy >> self.shift)) * x.w + (ix >> self.shift)) * c;
                &x.data[p..p + c]
            } else {
                zeros
            }
        })
    }

    /// Writes output channels `o0..o0+bias.len()` of the tiles `t0..t0+mb`.
    #[allow(clippy::too_many_arguments)]
    fn output_band(&self, t0: usize, mb: usize, m: &[f32], bias: &[f32], o0: usize, c_out: usize, out: &mut [f32]) {
        let ob = bias.len();
        let (hi, wi, th, tw) = (self.hi, self.wi, self.th, self.tw);
        let item = hi * wi * c_out;
        let first_pair = t0 / tw;
        let pairs = mb / tw; // bands are whole tile rows
        let n = self.x.n;
        for b in 0..n {
            let lo = first_pair.max(b * th);
            let hi_pair = (first_pair + pairs).min((b + 1) * th);
            if lo >= hi_pair {
                continue;
            }
            out[b * item..(b + 1) * item]
                .par_chunks_mut(2 * wi * c_out)
                .enumerate()
                .skip(lo - b * th)
                .take(hi_pair - lo)
                .for_each_init(
                    || vec![0.0f32; 4 * ob],
                    |tmp, (ty, rows)| {
                        let two_rows = rows.len() == 2 * wi * c_out;
                        for tx in 0..tw {
                            let lt = (b * th + ty) * tw + tx - t0;
                            let planes: [&[f32]; 16] = std::array::from_fn(|xi| &m[(xi * mb + lt) * ob..][..ob]);
                            output_transform(&planes, bias, tmp);
                            for (k, y) in tmp.chunks_exact(ob).enumerate() {
                                let (dy, dx) = (k / 2, k % 2);
                                if (dy == 1 && !two_rows) || 2 * tx + dx >= wi {
                                    continue;
                                }
                                let p = (dy * wi + 2 * tx + dx) * c_out + o0;
                                rows[p..p + ob].copy_from_slice(y);
                            }
                        }
                    },
                );
        }
    }
}

/// `U = G g Gᵀ` for output channels `oc`, layout `[o][16][c_in]`.
fn transform_kernel(weight: &Param, ci: usize, oc: Range<usize>, u: &mut [f32]) {
    let ob = oc.len();
    u[..16 * ob * ci].par_chunks_mut(16 * ci).enumerate().for_each_init(
        || vec![0.0f32; 9 * ci],
        |g, (i, uo)| {
            let o = oc.start + i;
            match weight {
                Param::F32(w) => g.copy_from_slice(&w[o * 9 * ci..(o + 1) * 9 * ci]),
                Param::F16(w) => w[o * 9 * ci..(o + 1) * 9 * ci].convert_to_f32_slice(g),
            }
            kernel_transform(g, uo);
        },
    );
}

multiversion! {
    /// `g`: `[3, 3, c]` → `u`: `[16, c]`, with
    /// G = [[1,0,0], [½,½,½], [½,−½,½], [0,0,1]].
    fn kernel_transform(g: &[f32], u: &mut [f32]) {
        let c = g.len() / 9;
        let gr: [&[f32]; 9] = std::array::from_fn(|k| &g[k * c..(k + 1) * c]);
        let ur: [&mut [f32]; 16] = {
            let mut it = u.chunks_exact_mut(c);
            std::array::from_fn(|_| it.next().unwrap())
        };
        for i in 0..c {
            // T = G·g, rows a = 0..4, columns kx = 0..3
            let mut t = [[0.0f32; 3]; 4];
            for kx in 0..3 {
                let (g0, g1, g2) = (gr[kx][i], gr[3 + kx][i], gr[6 + kx][i]);
                t[0][kx] = g0;
                t[1][kx] = 0.5 * (g0 + g1 + g2);
                t[2][kx] = 0.5 * (g0 - g1 + g2);
                t[3][kx] = g2;
            }
            for (a, ta) in t.iter().enumerate() {
                ur[a * 4][i] = ta[0];
                ur[a * 4 + 1][i] = 0.5 * (ta[0] + ta[1] + ta[2]);
                ur[a * 4 + 2][i] = 0.5 * (ta[0] - ta[1] + ta[2]);
                ur[a * 4 + 3][i] = ta[2];
            }
        }
    }
}

multiversion! {
    /// `V = Bᵀ d B` per channel; `d[r*4+c]` is input row r, column c.
    /// Bᵀ = [[1,0,−1,0], [0,1,1,0], [0,−1,1,0], [0,1,0,−1]].
    fn input_transform(d: &[&[f32]; 16], v: &mut [f32]) {
        let c = v.len() / 16;
        let d: [&[f32]; 16] = std::array::from_fn(|k| &d[k][..c]);
        let vr: [&mut [f32]; 16] = {
            let mut it = v.chunks_exact_mut(c);
            std::array::from_fn(|_| it.next().unwrap())
        };
        for i in 0..c {
            let mut t = [0.0f32; 16];
            for col in 0..4 {
                let (d0, d1, d2, d3) = (d[col][i], d[4 + col][i], d[8 + col][i], d[12 + col][i]);
                t[col] = d0 - d2;
                t[4 + col] = d1 + d2;
                t[8 + col] = d2 - d1;
                t[12 + col] = d1 - d3;
            }
            for r in 0..4 {
                let q = &t[r * 4..r * 4 + 4];
                vr[r * 4][i] = q[0] - q[2];
                vr[r * 4 + 1][i] = q[1] + q[2];
                vr[r * 4 + 2][i] = q[2] - q[1];
                vr[r * 4 + 3][i] = q[1] - q[3];
            }
        }
    }
}

multiversion! {
    /// `Y = Aᵀ m A + bias` per channel into `y` = `[4][c]` (row-major 2×2).
    /// Aᵀ = [[1,1,1,0], [0,1,−1,−1]].
    fn output_transform(m: &[&[f32]; 16], bias: &[f32], y: &mut [f32]) {
        let c = bias.len();
        let m: [&[f32]; 16] = std::array::from_fn(|k| &m[k][..c]);
        let (y0, rest) = y.split_at_mut(c);
        let (y1, rest) = rest.split_at_mut(c);
        let (y2, y3) = rest.split_at_mut(c);
        for i in 0..c {
            let mut u0 = [0.0f32; 4];
            let mut u1 = [0.0f32; 4];
            for r in 0..4 {
                let (a, b, cc, d) = (m[r * 4][i], m[r * 4 + 1][i], m[r * 4 + 2][i], m[r * 4 + 3][i]);
                u0[r] = a + b + cc;
                u1[r] = b - cc - d;
            }
            y0[i] = u0[0] + u0[1] + u0[2] + bias[i];
            y1[i] = u1[0] + u1[1] + u1[2] + bias[i];
            y2[i] = u0[1] - u0[2] - u0[3] + bias[i];
            y3[i] = u1[1] - u1[2] - u1[3] + bias[i];
        }
    }
}
