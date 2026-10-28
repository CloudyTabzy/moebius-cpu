//! Layer parameters in their stored precision.
//!
//! GEMM weights are kept f16 (half the resident memory) and widened into the
//! engine's reusable panel right before each GEMM; `half` uses F16C for this
//! when available. Small tensors (biases, norms, depthwise kernels) are f32.

use half::f16;
use half::slice::HalfFloatSliceExt;
use rayon::prelude::*;

use crate::Result;
use crate::weights::{TensorData, Weights};

pub enum Param {
    F32(Vec<f32>),
    F16(Vec<f16>),
}

impl Param {
    /// Copies the tensor out of the mapped file, keeping its precision.
    pub fn load(w: &Weights, name: &str, shape: &[usize]) -> Result<Self> {
        Ok(match w.get(name, shape)? {
            TensorData::F32(s) => Param::F32(s.to_vec()),
            TensorData::F16(s) => Param::F16(s.to_vec()),
        })
    }

    /// Test/bench helper: round an f32 tensor to f16 storage.
    pub fn f16_from(values: &[f32]) -> Self {
        Param::F16(values.iter().map(|&v| f16::from_f32(v)).collect())
    }

    pub fn len(&self) -> usize {
        match self {
            Param::F32(v) => v.len(),
            Param::F16(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// f32 view: the data itself, or `panel` filled with the widened values.
    pub fn view<'a>(&'a self, panel: &'a mut Vec<f32>) -> &'a [f32] {
        match self {
            Param::F32(v) => v,
            Param::F16(v) => {
                if panel.len() < v.len() {
                    panel.resize(v.len(), 0.0);
                }
                const CHUNK: usize = 1 << 16;
                v.par_chunks(CHUNK)
                    .zip(panel[..v.len()].par_chunks_mut(CHUNK))
                    .for_each(|(src, dst)| src.convert_to_f32_slice(dst));
                &panel[..v.len()]
            }
        }
    }
}

impl From<Vec<f32>> for Param {
    fn from(v: Vec<f32>) -> Self {
        Param::F32(v)
    }
}
