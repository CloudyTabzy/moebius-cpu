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
//! resnet; in NHWC the concat is written directly by the producer into
//! the right channel slice.

pub struct DwConv2d;
pub struct DwResnet;
pub struct Downsample;
pub struct Upsample;
