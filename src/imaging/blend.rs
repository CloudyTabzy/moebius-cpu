//! Paste-back (M7), mirroring `RemovalSDXLPipeline_BatchMode._post_process`:
//!
//! * `Feather`: `out = result·m + src·(1 − m)` with `m` the binary mask
//!   blurred at σ = 3 (PIL `GaussianBlur(radius=3)`; a separable f32 Gaussian
//!   here — sub-1/255 differences at the feather edge are accepted).
//! * `FeatherCompensate`: additionally shift the result by 1.1 × the mean
//!   colour difference measured in a 10 px ring around the mask's bbox
//!   (`compensation_utils.paste_compensate`).
//!
//! Only pixels inside `CropPlan::source_rect` are written.

use super::crop::CropPlan;
use super::{ImageMut, MaskRef, Rect};
use crate::preset::Paste;

pub fn paste(_img: &mut ImageMut<'_>, _mask: MaskRef<'_>, _result: &[u8], _plan: &CropPlan, _mode: Paste) -> Rect {
    todo!("M7")
}
