//! Paste-back (M7), mirroring `RemovalSDXLPipeline_BatchMode._post_process`:
//!
//! * `Feather`: `out = result·m + src·(1 − m)` with `m` the binary mask
//!   blurred with PIL `GaussianBlur(radius=3)` (reproduced byte-exactly by
//!   [`pil_gaussian_blur`]), arithmetic in f64 and truncated like numpy.
//! * `FeatherCompensate`: upstream additionally shifts the result by 1.1 × the
//!   mean colour difference in a 10 px ring around the mask. Not ported yet
//!   (M8); it currently behaves like `Feather`.
//!
//! Only pixels inside `CropPlan::source_rect` are written.

use super::crop::CropPlan;
use super::{ImageMut, MaskRef, Rect};
use crate::preset::Paste;

/// Model output in [0, 1] → 8 bit, as upstream `(x.clamp(0, 1) * 255).astype(np.uint8)`:
/// truncating, not rounding.
pub fn to_u8(v: f32) -> u8 {
    // `as u8` saturates and maps NaN to 0, so the clamp only has to bound above.
    (v.clamp(0.0, 1.0) * 255.0) as u8
}

/// Pillow's `_gaussian_blur_radius`: the fractional box radius whose
/// `passes`-fold box blur approximates a Gaussian of `sigma`
/// (Kovesi, "Fast Almost-Gaussian Filtering", eqs 7, 8, 11). The C code
/// works in `float` with double `sqrt`/`floor`.
fn gaussian_box_radius(sigma: f32, passes: u32) -> f32 {
    let sigma2 = sigma * sigma / passes as f32;
    let big_l = (12.0 * sigma2 as f64 + 1.0).sqrt() as f32;
    let l = ((big_l as f64 - 1.0) / 2.0).floor() as f32;
    let mut a = (2.0 * l + 1.0) * (l * (l + 1.0) - 3.0 * sigma2);
    a /= 6.0 * (sigma2 - (l + 1.0) * (l + 1.0));
    l + a
}

/// One fractional-radius box pass over a line (Pillow `BoxBlur.c`,
/// `ImagingLineBoxBlur8`): window `[x−r, x+r]` with edge pixels extended,
/// plus the two fractional taps `fw` just outside it, in 24-bit fixed point.
/// A running sum replaces the C code's edge-split loops; the result is the
/// same clamped-index formula.
fn box_line(src: &[u8], dst: &mut [u8], radius: usize, ww: u32, fw: u32) {
    let n = src.len();
    let at = |i: isize| src[i.clamp(0, n as isize - 1) as usize] as u32;
    let r = radius as isize;
    // Sum of the window for x = 0.
    let mut acc: u32 = (-r..=r).map(at).sum();
    for x in 0..n as isize {
        let far = at(x - r - 1) + at(x + r + 1);
        // acc·ww + far·fw ≤ 255·2²⁴ < 2³²: the weights sum to ≤ 2²⁴.
        let bulk = acc * ww + far * fw;
        dst[x as usize] = ((bulk + (1 << 23)) >> 24) as u8;
        acc = acc + at(x + r + 1) - at(x - r);
    }
}

/// `ImagingHorizontalBoxBlur`: `passes` box passes over every row of `buf`.
fn box_rows(buf: &mut [u8], w: usize, fradius: f32, passes: u32) {
    let radius = fradius as usize;
    let ww = ((1u32 << 24) as f32 / (fradius * 2.0 + 1.0)) as u32;
    let fw = ((1u32 << 24) - (radius as u32 * 2 + 1) * ww) / 2;
    let mut tmp = vec![0u8; w];
    for row in buf.chunks_exact_mut(w) {
        for _ in 0..passes {
            box_line(row, &mut tmp, radius, ww, fw);
            row.copy_from_slice(&tmp);
        }
    }
}

fn transpose(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    for y in 0..h {
        for x in 0..w {
            out[x * h + y] = src[y * w + x];
        }
    }
    out
}

/// PIL `ImageFilter.GaussianBlur(radius)` of a single-channel 8-bit image:
/// three fractional-radius box blurs per axis (`ImagingGaussianBlur`,
/// passes = 3), horizontal first, vertical via transposes, with 8-bit
/// rounding after every pass. Byte-exact with Pillow 10.4 (see the
/// fixture test).
pub fn pil_gaussian_blur(mask: &[u8], w: usize, h: usize, radius: f32) -> Vec<u8> {
    assert_eq!(mask.len(), w * h, "mask size");
    const PASSES: u32 = 3;
    if radius <= 0.0 || w == 0 || h == 0 {
        return mask.to_vec();
    }
    let fr = gaussian_box_radius(radius, PASSES);
    let mut buf = mask.to_vec();
    box_rows(&mut buf, w, fr, PASSES);
    let mut t = transpose(&buf, w, h);
    box_rows(&mut t, h, fr, PASSES);
    transpose(&t, h, w)
}

/// Blur margin kept around the pasted rectangle so the feather matches a
/// whole-image blur there (3 passes of radius ≈ 2.4 per axis reach ≈ 7 px).
const BLUR_MARGIN: usize = 16;

/// `result`: the window resized back, `side × side` RGB 8-bit (`CropPlan::side`).
/// Returns the rectangle of `img` that was written.
pub fn paste(img: &mut ImageMut<'_>, mask: MaskRef<'_>, result: &[u8], plan: &CropPlan, mode: Paste) -> Rect {
    let src = plan.source_rect(img.width, img.height);
    assert_eq!(result.len(), plan.side * plan.side * 3, "result must be side × side RGB");
    if src.w == 0 || src.h == 0 {
        return Rect::default();
    }
    let (ox, oy) = ((src.x as isize - plan.x) as usize, (src.y as isize - plan.y) as usize);
    let ch = img.channels;
    let result_px = |x: usize, y: usize| &result[((oy + y) * plan.side + ox + x) * 3..][..3];

    match mode {
        Paste::None => {
            for y in 0..src.h {
                for x in 0..src.w {
                    let o = (src.y + y) * img.stride + (src.x + x) * ch;
                    img.data[o..o + 3].copy_from_slice(result_px(x, y));
                }
            }
        }
        // FeatherCompensate: M8 (mean-colour compensation); feathers for now.
        Paste::Feather | Paste::FeatherCompensate => {
            // Blur the binary mask on the rectangle plus a margin; the margin
            // keeps the image-edge clamp of PIL where the real image ends.
            let rx0 = src.x.saturating_sub(BLUR_MARGIN);
            let ry0 = src.y.saturating_sub(BLUR_MARGIN);
            let rx1 = (src.x + src.w + BLUR_MARGIN).min(img.width);
            let ry1 = (src.y + src.h + BLUR_MARGIN).min(img.height);
            let (rw, rh) = (rx1 - rx0, ry1 - ry0);
            let mut region = Vec::with_capacity(rw * rh);
            for y in ry0..ry1 {
                let row = &mask.data[y * mask.stride + rx0..y * mask.stride + rx1];
                region.extend(row.iter().map(|&v| if v >= 128 { 255u8 } else { 0 }));
            }
            let blurred = pil_gaussian_blur(&region, rw, rh, 3.0);
            for y in 0..src.h {
                for x in 0..src.w {
                    // numpy: m = blur/255; (res/255)·m + (1−m)·(src/255), then ·255 → uint8.
                    let m = blurred[(src.y + y - ry0) * rw + src.x + x - rx0] as f64 / 255.0;
                    let r = result_px(x, y);
                    let o = (src.y + y) * img.stride + (src.x + x) * ch;
                    for (dst, &res) in img.data[o..o + 3].iter_mut().zip(r) {
                        let (res, orig) = (res as f64 / 255.0, *dst as f64 / 255.0);
                        *dst = ((res * m + (1.0 - m) * orig) * 255.0) as u8;
                    }
                }
            }
        }
    }
    src
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::Weights;
    use std::path::Path;

    #[test]
    fn to_u8_truncates_and_clamps() {
        assert_eq!(to_u8(0.99999), 254);
        assert_eq!(to_u8(1.2), 255);
        assert_eq!(to_u8(-0.1), 0);
        assert_eq!(to_u8(0.5), 127);
        assert_eq!(to_u8(1.0), 255);
        assert_eq!(to_u8(0.0), 0);
        assert_eq!(to_u8(f32::NAN), 0);
    }

    fn fixture() -> Weights {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/blur.safetensors");
        Weights::open_with_format(&path, "moebius-cpu-fixture/1").expect("run tools/dump_blur_fixture.py")
    }

    /// Byte-exact against Pillow 10.4 (`tools/dump_blur_fixture.py`).
    #[test]
    fn gaussian_blur_is_byte_exact_vs_pillow() {
        let f = fixture();
        for (k, (w, h)) in [("a", (64, 48)), ("b", (37, 23)), ("c", (5, 3))] {
            let mask: Vec<u8> = f.get_f32(&format!("{k}.mask"), &[h, w]).unwrap().iter().map(|&v| v as u8).collect();
            let want: Vec<u8> = f.get_f32(&format!("{k}.blurred"), &[h, w]).unwrap().iter().map(|&v| v as u8).collect();
            let got = pil_gaussian_blur(&mask, w, h, 3.0);
            let bad = got.iter().zip(&want).filter(|(g, w)| g != w).count();
            assert_eq!(bad, 0, "case {k}: {bad} of {} pixels differ from Pillow", want.len());
        }
    }

    #[test]
    fn feather_paste_keeps_outside_and_blends_edge() {
        let (w, h) = (40, 30);
        let mut data = vec![10u8; w * h * 3];
        let mut m = vec![0u8; w * h];
        for y in 5..25 {
            for x in 10..30 {
                m[y * w + x] = 255;
            }
        }
        let plan = CropPlan { x: 0, y: 0, side: 40 };
        // Window larger than the image: only the inside is written.
        let result = vec![250u8; 40 * 40 * 3];
        let mut img = ImageMut { data: &mut data, width: w, height: h, stride: w * 3, channels: 3 };
        let r = paste(&mut img, MaskRef { data: &m, width: w, height: h, stride: w }, &result, &plan, Paste::Feather);
        assert_eq!(r, Rect { x: 0, y: 0, w: 40, h: 30 });
        let px = |x: usize, y: usize| data[(y * w + x) * 3];
        assert_eq!(px(0, 0), 10, "far from the mask the source is untouched");
        assert_eq!(px(20, 15), 250, "deep inside the mask the result wins");
        let edge = px(9, 15);
        assert!(edge > 10 && edge < 250, "feathered edge: {edge}");
    }
}
