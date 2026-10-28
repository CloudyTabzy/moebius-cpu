//! Convolutional blocks (M5).
//!
//! `DWConv2d` (timm `DepthwiseSeparableConv`, BN folded by the converter):
//! ```text
//! y = pw1x1( relu( dw3x3(x) + b1 ) ) + b2      (+ x when c_in == c_out)
//! ```
//! `DWResnetBlock2D` (diffusers ResnetBlock2D with DWConv2d convs):
//! ```text
//! h = DWConv2d₁( SiLU(GN₁(x)) )
//! h = DWConv2d₂( SiLU(GN₂(h + temb_proj)) )     temb added before GN₂
//! y = h + (1x1 shortcut(x) if c_in ≠ c_out else x)
//! ```
//! `Downsample2D`: 3x3 stride-2 conv, padding 1. `Upsample2D`: nearest 2×
//! then 3x3 conv — fused so the 4× upsampled tensor is never stored.
//! Up blocks concatenate the matching skip along channels before each
//! resnet (`Tensor::concat_channels`).
//!
//! Loaders: every `load` takes the `Loader` positioned at the module itself
//! (`...resnets.0`, `...downsamplers.0`, a `DWConv2d` prefix such as `...conv_in`).

use super::load::Loader;
use crate::Result;
use crate::config::{RESNET_EPS, TIME_EMBED_DIM};
use crate::engine::conv::{Conv1x1, Conv3x3};
use crate::engine::depthwise::{Epilogue, depthwise3x3};
use crate::engine::norm::GroupNorm;
use crate::engine::ops::{add_inplace, silu_inplace};
use crate::engine::{Scratch, Tensor};

/// Depthwise-separable conv: `conv_dw` (+ folded bn1, ReLU) then `conv_pw` (+ folded bn2).
pub struct DwConv2d {
    pub c_in: usize,
    pub c_out: usize,
    dw_weight: Vec<f32>,
    dw_bias: Vec<f32>,
    pw: Conv1x1,
}

impl DwConv2d {
    /// `l` is the `DWConv2d` module (children `conv_dw`, `conv_pw`).
    pub fn load(l: &Loader, c_in: usize, c_out: usize) -> Result<Self> {
        let (dw_weight, dw_bias) = l.depthwise3x3("conv_dw", c_in)?;
        Ok(Self { c_in, c_out, dw_weight, dw_bias, pw: l.conv1x1("conv_pw", c_in, c_out, true)? })
    }

    /// `out` is reshaped to `[n, h, w, c_out]`; must not alias `x`.
    pub fn forward(&self, x: &Tensor, out: &mut Tensor, s: &mut Scratch) {
        let mut mid = s.tensor_for(x.data.len());
        depthwise3x3(x, &self.dw_weight, &self.dw_bias, Epilogue::Relu, &mut mid);
        let skip = (self.c_in == self.c_out).then_some(x);
        self.pw.forward(&mid, out, skip, s);
        s.recycle(mid);
    }
}

pub struct DwResnet {
    pub c_in: usize,
    pub c_out: usize,
    norm1: GroupNorm,
    conv1: DwConv2d,
    time_emb_proj: Conv1x1,
    norm2: GroupNorm,
    conv2: DwConv2d,
    shortcut: Option<Conv1x1>,
}

impl DwResnet {
    /// `l` is the `resnets.j` module.
    pub fn load(l: &Loader, c_in: usize, c_out: usize) -> Result<Self> {
        Ok(Self {
            c_in,
            c_out,
            norm1: l.group_norm("norm1", c_in, RESNET_EPS)?,
            conv1: DwConv2d::load(&l.sub("conv1"), c_in, c_out)?,
            time_emb_proj: l.conv1x1("time_emb_proj", TIME_EMBED_DIM, c_out, true)?,
            norm2: l.group_norm("norm2", c_out, RESNET_EPS)?,
            conv2: DwConv2d::load(&l.sub("conv2"), c_out, c_out)?,
            shortcut: if c_in != c_out { Some(l.conv1x1("conv_shortcut", c_in, c_out, true)?) } else { None },
        })
    }

    /// `temb`: `[n, 1, 1, 1280]` raw time embedding (SiLU is applied here).
    /// `out` is reshaped to `[n, h, w, c_out]`; must not alias `x`.
    pub fn forward(&self, x: &Tensor, temb: &Tensor, out: &mut Tensor, s: &mut Scratch) {
        assert_eq!(x.c, self.c_in, "DwResnet input channels");
        assert_eq!((temb.n, temb.h * temb.w, temb.c), (x.n, 1, TIME_EMBED_DIM), "DwResnet temb shape");

        let (mut a, mut b) = (s.tensor_for(x.data.len()), s.tensor_for(x.pixels() * self.c_out));
        self.norm1.forward(x, None, true, &mut a);
        self.conv1.forward(&a, &mut b, s);

        let mut t = s.tensor_for(temb.data.len());
        t.set_shape(temb.n, temb.h, temb.w, temb.c);
        t.data.copy_from_slice(&temb.data);
        silu_inplace(&mut t.data);
        let mut tp = s.tensor_for(temb.n * self.c_out);
        self.time_emb_proj.forward(&t, &mut tp, None, s);
        self.norm2.forward(&b, Some(&tp.data), true, &mut a);
        s.recycle(t);
        s.recycle(tp);
        self.conv2.forward(&a, out, s);

        match &self.shortcut {
            Some(sc) => {
                sc.forward(x, &mut b, None, s);
                add_inplace(&mut out.data, &b.data);
            }
            None => add_inplace(&mut out.data, &x.data),
        }
        s.recycle(a);
        s.recycle(b);
    }
}

/// 3x3 stride-2 conv, padding 1.
pub struct Downsample {
    conv: Conv3x3,
}

impl Downsample {
    /// `l` is the `downsamplers.0` module (child `conv`).
    pub fn load(l: &Loader, c: usize) -> Result<Self> {
        Ok(Self { conv: l.conv3x3_with("conv", c, c, 2, [1; 4], false)? })
    }

    pub fn forward(&self, x: &Tensor, out: &mut Tensor, s: &mut Scratch) {
        self.conv.forward(x, out, s);
    }
}

/// Nearest 2× upsample fused into a stride-1 3x3 conv.
pub struct Upsample {
    conv: Conv3x3,
}

impl Upsample {
    /// `l` is the `upsamplers.0` module (child `conv`).
    pub fn load(l: &Loader, c: usize) -> Result<Self> {
        Ok(Self { conv: l.conv3x3_with("conv", c, c, 1, [1; 4], true)? })
    }

    pub fn forward(&self, x: &Tensor, out: &mut Tensor, s: &mut Scratch) {
        self.conv.forward(x, out, s);
    }
}
