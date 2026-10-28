use rayon::prelude::*;

/// Pixels per task of the channel concatenation.
const CONCAT_PIXELS: usize = 256;

/// Dense NHWC f32 activation tensor.
///
/// Buffers are meant to be allocated once and reused every step: kernels
/// call `set_shape` on their output, which only allocates when the buffer
/// has never been that large.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub n: usize,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

impl Tensor {
    pub fn zeros(n: usize, h: usize, w: usize, c: usize) -> Self {
        Self { data: vec![0.0; n * h * w * c], n, h, w, c }
    }

    pub fn from_vec(data: Vec<f32>, n: usize, h: usize, w: usize, c: usize) -> Self {
        assert_eq!(data.len(), n * h * w * c, "tensor data does not match shape");
        Self { data, n, h, w, c }
    }

    /// Reshape for use as an output; contents are unspecified afterwards.
    pub fn set_shape(&mut self, n: usize, h: usize, w: usize, c: usize) {
        self.data.resize(n * h * w * c, 0.0);
        (self.n, self.h, self.w, self.c) = (n, h, w, c);
    }

    pub fn shape(&self) -> [usize; 4] {
        [self.n, self.h, self.w, self.c]
    }

    /// Rows of the `[pixels × channels]` matrix a 1x1 conv multiplies.
    pub fn pixels(&self) -> usize {
        self.n * self.h * self.w
    }

    /// Item `i` of the batch (CFG halves are items 0 = uncond, 1 = cond).
    pub fn item(&self, i: usize) -> &[f32] {
        let len = self.h * self.w * self.c;
        &self.data[i * len..(i + 1) * len]
    }

    /// Per-pixel channel concatenation `[a ‖ b]`: `[n, h, w, a.c + b.c]`.
    pub fn concat_channels(a: &Tensor, b: &Tensor) -> Tensor {
        let mut out = Tensor::default();
        Self::concat_channels_into(a, b, &mut out);
        out
    }

    /// [`Tensor::concat_channels`] into a reusable buffer; `out` must alias neither input.
    pub fn concat_channels_into(a: &Tensor, b: &Tensor, out: &mut Tensor) {
        assert_eq!([a.n, a.h, a.w], [b.n, b.h, b.w], "concat_channels: spatial shapes differ");
        let (ca, cb) = (a.c, b.c);
        out.set_shape(a.n, a.h, a.w, ca + cb);
        out.data
            .par_chunks_mut(CONCAT_PIXELS * (ca + cb))
            .zip(a.data.par_chunks(CONCAT_PIXELS * ca))
            .zip(b.data.par_chunks(CONCAT_PIXELS * cb))
            .for_each(|((o, x), y)| {
                for ((o, x), y) in o.chunks_exact_mut(ca + cb).zip(x.chunks_exact(ca)).zip(y.chunks_exact(cb)) {
                    o[..ca].copy_from_slice(x);
                    o[ca..].copy_from_slice(y);
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn concat_channels_interleaves_per_pixel() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 1, 1, 2, 2);
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 1, 1, 2, 3);
        let c = Tensor::concat_channels(&a, &b);
        assert_eq!(c.shape(), [1, 1, 2, 5]);
        assert_eq!(c.data, vec![1.0, 2.0, 10.0, 20.0, 30.0, 3.0, 4.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn concat_channels_into_overwrites_a_dirty_larger_buffer() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 1, 1, 2, 2);
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 1, 1, 2, 3);
        let mut out = Tensor::from_vec(vec![-1.0; 40], 1, 4, 2, 5);
        Tensor::concat_channels_into(&a, &b, &mut out);
        assert_eq!(out.shape(), [1, 1, 2, 5]);
        assert_eq!(out.data, Tensor::concat_channels(&a, &b).data);
    }
}
