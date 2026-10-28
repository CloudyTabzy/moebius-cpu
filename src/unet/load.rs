//! Prefix-scoped access to the converted UNet weights.
//!
//! The converter's keys are dotted PyTorch module paths; a `Loader` carries
//! the path of the module being built so each layer asks for its own leaves
//! (`weight`, `bias`, ...) and gets shape-checked, precision-preserving
//! parameters back.

use crate::Result;
use crate::engine::Param;
use crate::engine::conv::{Conv1x1, Conv3x3};
use crate::engine::norm::GroupNorm;
use crate::weights::Weights;

#[derive(Clone)]
pub struct Loader<'w> {
    w: &'w Weights,
    prefix: String,
}

impl<'w> Loader<'w> {
    pub fn new(w: &'w Weights, prefix: &str) -> Self {
        Self { w, prefix: prefix.trim_end_matches('.').to_string() }
    }

    /// Loader for the child module `name` (`<prefix>.<name>`).
    pub fn sub(&self, name: &str) -> Loader<'w> {
        Loader { w: self.w, prefix: self.key(name) }
    }

    /// Full tensor key for `leaf` under this prefix.
    pub fn key(&self, leaf: &str) -> String {
        if self.prefix.is_empty() { leaf.to_string() } else { format!("{}.{leaf}", self.prefix) }
    }

    /// 1-D f32 tensor `<prefix>.<name>` of length `len` (biases, norm affine).
    pub fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        self.w.get_f32(&self.key(name), &[len])
    }

    /// Any-shape tensor `<prefix>.<name>` widened to owned f32.
    pub fn f32s(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        self.w.get_f32(&self.key(name), shape)
    }

    /// Stored shape of `<prefix>.<name>`, if present.
    pub fn shape(&self, name: &str) -> Option<&'w [usize]> {
        self.w.shape(&self.key(name))
    }

    /// `<name>.weight` `[c_out, c_in]` in stored precision, plus `<name>.bias` when `bias`.
    pub fn conv1x1(&self, name: &str, c_in: usize, c_out: usize, bias: bool) -> Result<Conv1x1> {
        let l = self.sub(name);
        let weight = Param::load(self.w, &l.key("weight"), &[c_out, c_in])?;
        let bias = if bias { Some(l.vec("bias", c_out)?) } else { None };
        Ok(Conv1x1 { c_in, c_out, weight, bias })
    }

    /// Stride-1 / padding-1 dense 3x3: `<name>.weight` `[c_out, 3, 3, c_in]` and `<name>.bias`.
    pub fn conv3x3(&self, name: &str, c_in: usize, c_out: usize) -> Result<Conv3x3> {
        let l = self.sub(name);
        let weight = Param::load(self.w, &l.key("weight"), &[c_out, 3, 3, c_in])?;
        Ok(Conv3x3::same(c_in, c_out, weight, Some(l.vec("bias", c_out)?)))
    }

    /// Depthwise 3x3 `<name>.weight` `[3, 3, c]` and `<name>.bias` `[c]`, both f32.
    pub fn depthwise3x3(&self, name: &str, c: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let l = self.sub(name);
        Ok((l.f32s("weight", &[3, 3, c])?, l.vec("bias", c)?))
    }

    /// Dense 3x3 like `conv3x3` with explicit stride, padding `[top, left, bottom, right]`
    /// and optional fused nearest-2x upsample of the input.
    pub fn conv3x3_with(
        &self,
        name: &str,
        c_in: usize,
        c_out: usize,
        stride: usize,
        pad: [usize; 4],
        upsample2x: bool,
    ) -> Result<Conv3x3> {
        let mut conv = self.conv3x3(name, c_in, c_out)?;
        conv.stride = stride;
        conv.pad = pad;
        conv.upsample2x = upsample2x;
        Ok(conv)
    }

    /// 32-group GroupNorm with affine `<name>.weight` / `<name>.bias`.
    pub fn group_norm(&self, name: &str, c: usize, eps: f32) -> Result<GroupNorm> {
        let l = self.sub(name);
        Ok(GroupNorm { groups: crate::config::GROUP_NORM_GROUPS, eps, gamma: l.vec("weight", c)?, beta: l.vec("bias", c)? })
    }
}
