//! Window ↔ model resampling (M7) via `fast_image_resize` Lanczos3, which
//! uses the same convolution formulation as PIL's `LANCZOS` used upstream.
//! The source window is read with edge replication where it extends past the
//! image (`crop::CropPlan`); the mask is resized with nearest-neighbour and
//! re-thresholded, as upstream.

use super::crop::CropPlan;
use super::{ImageMut, MaskRef};

/// Square window → 512² RGB in [-1, 1] (NHWC) and binary 512² mask.
pub fn to_model(_img: &ImageMut<'_>, _mask: MaskRef<'_>, _plan: &CropPlan) -> (Vec<f32>, Vec<u8>) {
    todo!("M7")
}

/// 512² RGB in [0, 1] → window side, 8-bit, for blending.
pub fn from_model(_rgb: &[f32], _side: usize) -> Vec<u8> {
    todo!("M7")
}
