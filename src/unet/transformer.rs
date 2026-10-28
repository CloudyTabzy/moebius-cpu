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
//! The three LayerNorms fuse into the GEMM that consumes them (normalise
//! while packing the LHS panel); residual adds fuse into GEMM epilogues.

pub struct MixTransformer;
