//! Compute engine: NHWC f32 tensors and the handful of kernels Moebius needs.
//!
//! Layout is NHWC throughout so that every 1x1 conv / linear (62% of UNet
//! FLOPs) is a plain GEMM over pixels with no transposes, and channel-wise
//! kernels (depthwise, GroupNorm, gating) stream contiguous SIMD lanes.
//! All kernels parallelise on the current rayon pool (the engine's own).

pub mod conv;
pub mod cpu;
pub mod depthwise;
pub mod gemm;
pub mod math;
pub mod norm;
pub mod ops;
pub mod param;
pub mod simd;
pub mod tensor;
pub mod winograd;

pub use param::Param;
pub use tensor::Tensor;

/// Reusable per-engine scratch, grown on first use and kept for the run.
#[derive(Default)]
pub struct Scratch {
    /// f16 → f32 widened GEMM weights (≤ 33 MB: the 1280→6400 MixFFN layer).
    pub panel: Vec<f32>,
    /// Row-band patch matrix of the strided 3x3 convolutions (≤ 32 MB).
    pub cols: Vec<f32>,
    /// Winograd input tiles, products and transformed kernel (≤ 16/32/32 MB).
    pub wino_v: Vec<f32>,
    pub wino_m: Vec<f32>,
    pub wino_u: Vec<f32>,
    /// Normalised input rows of one Winograd band (GroupNorm prologue).
    pub wino_strip: Vec<f32>,
    /// Recycled activation tensors (LIFO), see [`Scratch::tensor`].
    pool: Vec<Tensor>,
    /// Recycled plain buffers (LIFO), see [`Scratch::vec_buf`].
    vecs: Vec<Vec<f32>>,
}

impl Scratch {
    /// A recycled activation buffer (or a fresh empty one), most recently returned
    /// first. The shape is empty and the contents are unspecified: callers
    /// `set_shape` and overwrite it. Prefer [`Scratch::tensor_for`] when the size
    /// is known.
    pub fn tensor(&mut self) -> Tensor {
        self.pool.pop().unwrap_or_default()
    }

    /// Like [`Scratch::tensor`] for a tensor of about `len` elements: hands out the
    /// tightest recycled buffer that already holds `len` elements (so `set_shape`
    /// neither allocates nor writes), else one with room for `len`, else the
    /// largest. Without the size a LIFO pool lets every buffer drift to the
    /// largest role it ever served, which costs memory and re-zeroing on growth.
    ///
    /// Roles ask for the same sizes in the same order every step, so after the
    /// first step each role gets the same buffer back and nothing large is
    /// allocated, zeroed or page-faulted again (PLAN §6.1). Give every buffer back
    /// with [`Scratch::recycle`].
    pub fn tensor_for(&mut self, len: usize) -> Tensor {
        match best_fit(self.pool.iter().map(|t| (t.data.len(), t.data.capacity())), len) {
            Some(i) => self.pool.remove(i),
            None => Tensor::default(),
        }
    }

    /// Returns a buffer to the pool, keeping its allocation (and its last length,
    /// which [`Scratch::tensor_for`] uses to match it to the next role of that size).
    pub fn recycle(&mut self, mut t: Tensor) {
        (t.n, t.h, t.w, t.c) = (0, 0, 0, 0);
        self.pool.push(t);
    }

    /// A recycled `Vec<f32>` of length `len` with unspecified contents (only growth
    /// beyond a recycled buffer's old length is zero-filled). Best fit, as
    /// [`Scratch::tensor_for`].
    pub fn vec_buf(&mut self, len: usize) -> Vec<f32> {
        let mut v = match best_fit(self.vecs.iter().map(|v| (v.len(), v.capacity())), len) {
            Some(i) => self.vecs.remove(i),
            None => Vec::new(),
        };
        v.resize(len, 0.0);
        v
    }

    /// Returns a buffer taken with [`Scratch::vec_buf`].
    pub fn recycle_vec(&mut self, v: Vec<f32>) {
        self.vecs.push(v);
    }

    /// Frees every pooled buffer of fewer than `min_len` elements (capacity), e.g.
    /// the denoising loop's activations before the VAE decoder claims its memory.
    pub fn release_below(&mut self, min_len: usize) {
        self.pool.retain(|t| t.data.capacity() >= min_len);
        self.vecs.retain(|v| v.capacity() >= min_len);
    }
}

/// Index of the best buffer for `len` elements among `(len, capacity)` pairs:
/// the shortest one already holding `len`, else the one needing the least zero
/// fill within its capacity, else the largest (which must grow). Later entries
/// win ties.
fn best_fit(bufs: impl Iterator<Item = (usize, usize)>, len: usize) -> Option<usize> {
    bufs.enumerate()
        .min_by_key(|&(i, (l, cap))| {
            let key = if l >= len {
                (0, l)
            } else if cap >= len {
                (1, len - l)
            } else {
                (2, usize::MAX - cap)
            };
            (key, std::cmp::Reverse(i))
        })
        .map(|(i, _)| i)
}

/// Worker count of the pool the caller runs in.
pub(crate) fn threads() -> usize {
    rayon::current_num_threads()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf(s: &mut Scratch, len: usize) -> Tensor {
        let mut t = s.tensor_for(len);
        t.set_shape(1, 1, 1, len);
        t
    }

    #[test]
    fn tensor_for_hands_back_the_tightest_buffer_and_reuses_allocations() {
        let mut s = Scratch::default();
        let (a, b, c) = (buf(&mut s, 100), buf(&mut s, 1000), buf(&mut s, 10));
        let (pa, pb, pc) = (a.data.as_ptr(), b.data.as_ptr(), c.data.as_ptr());
        for t in [a, b, c] {
            s.recycle(t);
        }
        // Whatever the recycle order, each size gets its own buffer back untouched.
        assert_eq!(buf(&mut s, 1000).data.as_ptr(), pb);
        assert_eq!(buf(&mut s, 10).data.as_ptr(), pc);
        assert_eq!(buf(&mut s, 100).data.as_ptr(), pa);
        // A role larger than everything pooled grows the largest one.
        let big = buf(&mut s, 5000);
        assert_eq!(big.data.len(), 5000);
    }

    #[test]
    fn recycled_tensors_come_back_with_an_empty_shape() {
        let mut s = Scratch::default();
        let t = buf(&mut s, 8);
        s.recycle(t);
        let t = s.tensor();
        assert_eq!(t.shape(), [0, 0, 0, 0]);
        assert_eq!(t.data.len(), 8);
    }

    #[test]
    fn vec_buf_has_the_requested_length() {
        let mut s = Scratch::default();
        let v = s.vec_buf(64);
        assert_eq!(v.len(), 64);
        s.recycle_vec(v);
        assert_eq!(s.vec_buf(16).len(), 16);
        assert_eq!(s.vec_buf(128).len(), 128);
    }

    #[test]
    fn release_below_drops_small_buffers() {
        let mut s = Scratch::default();
        let (small, large) = (buf(&mut s, 10), buf(&mut s, 1000));
        let ptr = large.data.as_ptr();
        s.recycle(small);
        s.recycle(large);
        s.release_below(100);
        assert_eq!(s.tensor().data.as_ptr(), ptr);
        assert!(s.tensor().data.is_empty(), "the small buffer is gone");
    }
}
