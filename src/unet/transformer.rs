//! `MixTransformer2DModel` with one `MixTransformerBlock` (M4).
//!
//! ```text
//! r = x
//! x = proj_in1x1( GN₃₂(x, eps 1e-6) )
//! x = x + selfλ( LN₁(x) )
//! x = x + crossλ( LN₂(x) )
//! x = x + MixFFN( LN₃(x) )
//! y = proj_out1x1(x) + r
//! ```
//! MixFFN (Sana `GLUMBConv`, hidden = 2.5·C):
//! ```text
//! h = SiLU( inverted1x1(x) + b )          C → 5C
//! h = depthwise3x3(h) + b                 5C, fused with the GLU below
//! h = h[:, :2.5C] · SiLU(h[:, 2.5C:])     → 2.5C
//! y = point1x1(h)                         2.5C → C, no bias
//! ```
//! Residual adds of the FFN and of `proj_out` are fused into the consuming
//! 1x1 GEMM's output pre-fill; the λ outputs are added with `add_inplace`.
//! `load` takes the `Loader` at the wrapper (`...attentions.j`); the block
//! lives under `transformer_blocks.0`.

use super::lambda::{CrossLambda, SelfLambda};
use super::load::Loader;
use crate::Result;
use crate::config::{LAYER_NORM_EPS, MIX_MLP_RATIO, TRANSFORMER_GROUP_NORM_EPS};
use crate::engine::conv::Conv1x1;
use crate::engine::depthwise::{Epilogue, depthwise3x3};
use crate::engine::norm::{GroupNorm, layer_norm};
use crate::engine::ops::{add_inplace, silu_inplace};
use crate::engine::{Scratch, Tensor};

struct LayerNormParams {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

impl LayerNormParams {
    fn load(l: &Loader, name: &str, c: usize) -> Result<Self> {
        let l = l.sub(name);
        Ok(Self { gamma: l.vec("weight", c)?, beta: l.vec("bias", c)? })
    }

    fn forward(&self, x: &Tensor, out: &mut Tensor) {
        layer_norm(x, &self.gamma, &self.beta, LAYER_NORM_EPS, out);
    }
}

pub struct MixTransformer {
    c: usize,
    norm: GroupNorm,
    proj_in: Conv1x1,
    ln1: LayerNormParams,
    attn1: SelfLambda,
    ln2: LayerNormParams,
    attn2: CrossLambda,
    ln3: LayerNormParams,
    inverted: Conv1x1,
    dw_weight: Vec<f32>,
    dw_bias: Vec<f32>,
    point: Conv1x1,
    proj_out: Conv1x1,
}

impl MixTransformer {
    /// `ehs[0]` uncond, `ehs[1]` cond hidden states, each `[10 × 768]`.
    pub fn load(l: &Loader, c: usize, ehs: &[Vec<f32>; 2]) -> Result<Self> {
        let hidden = (c as f32 * MIX_MLP_RATIO) as usize;
        let b = l.sub("transformer_blocks.0");
        let ff = b.sub("ff");
        let (dw_weight, dw_bias) = ff.sub("depth_conv").depthwise3x3("conv", 2 * hidden)?;
        Ok(Self {
            c,
            norm: l.group_norm("norm", c, TRANSFORMER_GROUP_NORM_EPS)?,
            proj_in: l.conv1x1("proj_in", c, c, true)?,
            ln1: LayerNormParams::load(&b, "norm1", c)?,
            attn1: SelfLambda::load(&b.sub("attn1"), c)?,
            ln2: LayerNormParams::load(&b, "norm2", c)?,
            attn2: CrossLambda::load(&b.sub("attn2"), c, ehs)?,
            ln3: LayerNormParams::load(&b, "norm3", c)?,
            inverted: ff.sub("inverted_conv").conv1x1("conv", c, 2 * hidden, true)?,
            dw_weight,
            dw_bias,
            point: ff.sub("point_conv").conv1x1("conv", hidden, c, false)?,
            proj_out: l.conv1x1("proj_out", c, c, true)?,
        })
    }

    /// `x`: `[b, h, w, c]`; item `i` uses conditioning branch `branches[i]`.
    /// `out` is reshaped to the same shape and must not alias `x`.
    pub fn forward(&self, x: &Tensor, branches: &[usize], out: &mut Tensor, s: &mut Scratch) {
        assert_eq!(x.c, self.c, "MixTransformer channels");
        let len = x.data.len();
        let (mut t, mut u, mut h) = (s.tensor_for(len), s.tensor_for(len), s.tensor_for(len));

        self.norm.forward(x, None, false, &mut t);
        self.proj_in.forward(&t, &mut h, None, s);

        self.ln1.forward(&h, &mut t);
        self.attn1.forward(&t, &mut u, s);
        add_inplace(&mut h.data, &u.data);

        self.ln2.forward(&h, &mut t);
        self.attn2.forward(&t, branches, &mut u, s);
        add_inplace(&mut h.data, &u.data);

        self.ln3.forward(&h, &mut t);
        self.ffn(&t, Some(&h), &mut u, s);

        self.proj_out.forward(&u, out, Some(x), s);
        s.recycle(t);
        s.recycle(u);
        s.recycle(h);
    }

    /// MixFFN alone (`x` already layer-normed), no residual. Public for parity tests.
    pub fn forward_ff(&self, x: &Tensor, out: &mut Tensor, s: &mut Scratch) {
        self.ffn(x, None, out, s);
    }

    fn ffn(&self, x: &Tensor, residual: Option<&Tensor>, out: &mut Tensor, s: &mut Scratch) {
        let (mut a, mut b) = (s.tensor_for(x.pixels() * self.inverted.c_out), s.tensor_for(x.pixels() * self.point.c_in));
        self.inverted.forward(x, &mut a, None, s);
        silu_inplace(&mut a.data);
        depthwise3x3(&a, &self.dw_weight, &self.dw_bias, Epilogue::GluSilu, &mut b);
        self.point.forward(&b, out, residual, s);
        s.recycle(a);
        s.recycle(b);
    }
}
