//! Thin, safe wrapper over two GEMM backends.
//!
//! Every dense op in the engine ends up here: 1x1 convs and linears directly,
//! 3x3 convs through Winograd / tiled implicit im2col (`conv.rs`), the λ
//! contractions as batched calls. Activations are NHWC, so a 1x1 conv over
//! `pixels` positions is exactly `[pixels × c_in] · [c_in × c_out]`.
//!
//! Backend choice is measured, not assumed (`gemm-bench/RESULTS.md`, Moebius
//! shapes, interleaved runs across power states):
//! * multi-threaded calls → `private-gemm-x86` (faer's current x86 kernels):
//!   1.74× `gemm` 0.19 geomean on the 1x1 / linear / λ / attention shapes. Its
//!   scheduler balances the P- and E-cores of hybrid CPUs, where `gemm`'s even
//!   split leaves the P-cores waiting.
//! * single-threaded calls (Winograd's 16 concurrent GEMMs) → `gemm`, which
//!   wins there (`private-gemm-x86` is up to 42% slower at K ≥ 512).

use std::ptr::null;

use gemm::Parallelism;
use private_gemm_x86::{Accum, DType, DstKind, IType, InstrSet};

/// Row-major `[rows × cols]` view with an arbitrary column stride, so a
/// PyTorch `[out, in]` weight can be used as `[in × out]` without a copy.
#[derive(Clone, Copy, Debug)]
pub struct MatRef<'a> {
    pub data: &'a [f32],
    pub rows: usize,
    pub cols: usize,
    pub row_stride: isize,
    pub col_stride: isize,
}

impl<'a> MatRef<'a> {
    /// Dense row-major matrix.
    pub fn row_major(data: &'a [f32], rows: usize, cols: usize) -> Self {
        assert!(data.len() >= rows * cols, "matrix buffer too small");
        Self { data, rows, cols, row_stride: cols as isize, col_stride: 1 }
    }

    /// Transposed view of a dense row-major `[cols × rows]` buffer
    /// (e.g. a `[c_out, c_in]` weight read as `[c_in × c_out]`).
    pub fn transposed(data: &'a [f32], rows: usize, cols: usize) -> Self {
        assert!(data.len() >= rows * cols, "matrix buffer too small");
        Self { data, rows, cols, row_stride: 1, col_stride: rows as isize }
    }

    /// Arbitrary non-negative strides (interleaved planes, e.g. Winograd).
    pub fn strided(data: &'a [f32], rows: usize, cols: usize, row_stride: usize, col_stride: usize) -> Self {
        assert!(rows > 0 && cols > 0, "empty matrix");
        assert!(data.len() > (rows - 1) * row_stride + (cols - 1) * col_stride, "matrix buffer too small");
        Self { data, rows, cols, row_stride: row_stride as isize, col_stride: col_stride as isize }
    }
}

/// `MOEBIUS_GEMM=gemm` routes every call to the `gemm` crate (A/B timing and
/// a fallback if `private-gemm-x86` ever misbehaves on some CPU).
fn force_gemm_crate() -> bool {
    static FORCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCE.get_or_init(|| std::env::var("MOEBIUS_GEMM").is_ok_and(|v| v.eq_ignore_ascii_case("gemm")))
}

/// `c[m × n] = a · b` (or `c += a · b` when `accumulate`), `c` dense row-major.
pub fn matmul(c: &mut [f32], a: MatRef<'_>, b: MatRef<'_>, accumulate: bool, threads: usize) {
    let (m, k, n) = (a.rows, a.cols, b.cols);
    assert_eq!(b.rows, k, "inner dimensions differ");
    assert!(c.len() >= m * n, "output buffer too small");
    if m == 0 || n == 0 {
        return;
    }
    if threads > 1 && k > 0 && super::simd::avx2_fma() && !force_gemm_crate() {
        let alpha = 1.0f32;
        // SAFETY: as below; additionally AVX2+FMA (the `Avx256` kernel set) was
        // detected at runtime. Strides are in elements. The AVX-512 kernels are
        // not used: they were not benchmarked or parity-tested here.
        unsafe {
            private_gemm_x86::gemm(
                DType::F32,
                IType::U64,
                InstrSet::Avx256,
                m,
                n,
                k,
                c.as_mut_ptr().cast(),
                n as isize,
                1,
                null(),
                null(),
                DstKind::Full,
                if accumulate { Accum::Add } else { Accum::Replace },
                a.data.as_ptr().cast(),
                a.row_stride,
                a.col_stride,
                false,
                null(),
                0,
                b.data.as_ptr().cast(),
                b.row_stride,
                b.col_stride,
                false,
                (&raw const alpha).cast(),
                threads,
            );
        }
        return;
    }
    let parallelism = if threads > 1 { Parallelism::Rayon(threads) } else { Parallelism::None };
    // SAFETY: bounds of all three operands were asserted above; strides
    // describe views fully inside those buffers; `c` is uniquely borrowed.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,
            n as isize,
            accumulate,
            a.data.as_ptr(),
            a.col_stride,
            a.row_stride,
            b.data.as_ptr(),
            b.col_stride,
            b.row_stride,
            1.0,
            1.0,
            false,
            false,
            false,
            parallelism,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_naive_both_layouts() {
        let (m, k, n) = (7, 5, 3);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.37).sin()).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.11).cos()).collect(); // [n, k]
        let mut want = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                want[i * n + j] = (0..k).map(|p| a[i * k + p] * w[j * k + p]).sum();
            }
        }
        let mut got = vec![0.0f32; m * n];
        matmul(&mut got, MatRef::row_major(&a, m, k), MatRef::transposed(&w, k, n), false, 1);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5);
        }
        // accumulate doubles it
        matmul(&mut got, MatRef::row_major(&a, m, k), MatRef::transposed(&w, k, n), true, 2);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - 2.0 * w).abs() < 1e-5);
        }
    }

    /// Both backends (threads = 1 → `gemm`, > 1 → `private-gemm-x86`) on sizes
    /// past their kernel tiles, with strided A (Winograd-style interleaving) and
    /// a transposed B, plain and accumulating.
    #[test]
    fn backends_agree_on_strided_operands() {
        let (m, k, n, planes) = (67, 129, 45, 3);
        let a_buf: Vec<f32> = (0..m * planes * k).map(|i| ((i * 37) % 101) as f32 / 101.0 - 0.5).collect();
        let w: Vec<f32> = (0..n * k).map(|i| ((i * 53) % 97) as f32 / 97.0 - 0.5).collect(); // [n, k]
        let a = MatRef::strided(&a_buf[k..], m, k, planes * k, 1); // plane 1 of 3
        let mut want = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                want[i * n + j] = (0..k).map(|p| a_buf[i * planes * k + k + p] as f64 * w[j * k + p] as f64).sum();
            }
        }
        for threads in [1, 4] {
            let mut got = vec![0.5f32; m * n];
            matmul(&mut got, a, MatRef::transposed(&w, k, n), true, threads);
            for (g, w) in got.iter().zip(&want) {
                assert!((*g as f64 - (w + 0.5)).abs() < 1e-4, "threads {threads}: {g} vs {}", w + 0.5);
            }
            matmul(&mut got, a, MatRef::transposed(&w, k, n), false, threads);
            for (g, w) in got.iter().zip(&want) {
                assert!((*g as f64 - w).abs() < 1e-4, "threads {threads}");
            }
        }
    }
}
