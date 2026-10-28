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
        assert_eq!([a.n, a.h, a.w], [b.n, b.h, b.w], "concat_channels: spatial shapes differ");
        let (ca, cb) = (a.c, b.c);
        let mut out = Tensor::zeros(a.n, a.h, a.w, ca + cb);
        for ((o, x), y) in out.data.chunks_exact_mut(ca + cb).zip(a.data.chunks_exact(ca)).zip(b.data.chunks_exact(cb)) {
            o[..ca].copy_from_slice(x);
            o[ca..].copy_from_slice(y);
        }
        out
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
}
