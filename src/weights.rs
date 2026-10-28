//! Memory-mapped safetensors produced by `tools/convert_weights.py`.
//!
//! Weights are stored f16 (≈620 MB for UNet + VAE instead of 1.24 GB f32)
//! and widened to f32 per layer when packed for GEMM; tensors whose values
//! do not survive f16 are stored f32 by the converter. Nothing is copied at
//! open: the OS pages the file in on first use.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use half::f16;
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::{Error, Result};

/// Bumped whenever the converter's key layout or folding changes.
pub const FORMAT_VERSION: &str = "moebius-cpu/1";

pub struct Weights {
    mmap: Mmap,
    data_start: usize,
    tensors: HashMap<String, Entry>,
    metadata: HashMap<String, String>,
}

struct Entry {
    dtype: Dtype,
    shape: Vec<usize>,
    range: (usize, usize),
}

/// Borrowed tensor data in its stored precision.
pub enum TensorData<'a> {
    F32(&'a [f32]),
    F16(&'a [f16]),
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_format(path, FORMAT_VERSION)
    }

    /// Opens a file whose `format` metadata must equal `expected` (test
    /// fixtures use their own format tag).
    pub fn open_with_format(path: &Path, expected: &str) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the file is opened read-only and must not be modified while
        // mapped; the weights directory is owned by the plugin install.
        let mmap = unsafe { Mmap::map(&file)? };
        let (header_len, meta) =
            SafeTensors::read_metadata(&mmap).map_err(|e| Error::Weights(format!("{}: {e}", path.display())))?;
        let metadata = meta.metadata().clone().unwrap_or_default();
        let found = metadata.get("format").map(String::as_str);
        if found != Some(expected) {
            return Err(Error::Weights(format!(
                "{}: format {found:?}, expected {expected:?} (re-run tools/convert_weights.py)",
                path.display()
            )));
        }
        let tensors = meta
            .tensors()
            .into_iter()
            .map(|(name, info)| {
                (name, Entry { dtype: info.dtype, shape: info.shape.clone(), range: info.data_offsets })
            })
            .collect();
        Ok(Self { mmap, data_start: 8 + header_len, tensors, metadata })
    }

    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Stored shape of a tensor, if present.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|e| e.shape.as_slice())
    }

    /// Tensor data, checked against the shape the model code expects.
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<TensorData<'_>> {
        let e = self.tensors.get(name).ok_or_else(|| Error::Weights(format!("missing tensor {name}")))?;
        if e.shape != shape {
            return Err(Error::Weights(format!("{name}: shape {:?}, expected {shape:?}", e.shape)));
        }
        let bytes = &self.mmap[self.data_start + e.range.0..self.data_start + e.range.1];
        let cast_err = |_| Error::Weights(format!("{name}: misaligned data"));
        match e.dtype {
            Dtype::F32 => Ok(TensorData::F32(bytemuck::try_cast_slice(bytes).map_err(cast_err)?)),
            Dtype::F16 => Ok(TensorData::F16(bytemuck::try_cast_slice(bytes).map_err(cast_err)?)),
            other => Err(Error::Weights(format!("{name}: unsupported dtype {other:?}"))),
        }
    }

    /// Widen to an owned f32 buffer (load-time constants, small tensors).
    pub fn get_f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        Ok(match self.get(name, shape)? {
            TensorData::F32(s) => s.to_vec(),
            TensorData::F16(s) => s.iter().map(|v| v.to_f32()).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;

    #[test]
    fn roundtrip_f16_and_f32() {
        let w16: Vec<f16> = [1.5f32, -2.0, 0.25, 8.0].iter().map(|&v| f16::from_f32(v)).collect();
        let w32 = [0.1f32, 0.2];
        let t16 = TensorView::new(Dtype::F16, vec![2, 2], bytemuck::cast_slice(&w16)).unwrap();
        let t32 = TensorView::new(Dtype::F32, vec![2], bytemuck::cast_slice(&w32)).unwrap();
        let meta = HashMap::from([("format".to_string(), FORMAT_VERSION.to_string())]);
        let bytes = safetensors::serialize([("a", t16), ("b", t32)], Some(meta)).unwrap();
        let path = std::env::temp_dir().join(format!("moebius-weights-test-{}.safetensors", std::process::id()));
        std::fs::write(&path, bytes).unwrap();

        let w = Weights::open(&path).unwrap();
        assert_eq!(w.get_f32("a", &[2, 2]).unwrap(), vec![1.5, -2.0, 0.25, 8.0]);
        assert_eq!(w.get_f32("b", &[2]).unwrap(), vec![0.1, 0.2]);
        assert!(w.get("a", &[4]).is_err());
        assert!(w.get("missing", &[1]).is_err());
        drop(w);
        let _ = std::fs::remove_file(path);
    }
}
