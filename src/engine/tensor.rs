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
}
