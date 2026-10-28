//! Thin, safe wrapper over the `gemm` crate.
//!
//! Every dense op in the engine ends up here: 1x1 convs and linears directly,
//! 3x3 convs through tiled implicit im2col (`conv.rs`), the λ contractions
//! as batched calls. Activations are NHWC, so a 1x1 conv over `pixels`
//! positions is exactly `[pixels × c_in] · [c_in × c_out]`.

use gemm::Parallelism;

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

/// `c[m × n] = a · b` (or `c += a · b` when `accumulate`), `c` dense row-major.
pub fn matmul(c: &mut [f32], a: MatRef<'_>, b: MatRef<'_>, accumulate: bool, threads: usize) {
    let (m, k, n) = (a.rows, a.cols, b.cols);
    assert_eq!(b.rows, k, "inner dimensions differ");
    assert!(c.len() >= m * n, "output buffer too small");
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
}
