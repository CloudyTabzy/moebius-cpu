//! Paste-back (M7), mirroring `RemovalSDXLPipeline_BatchMode._post_process`:
//!
//! * `Feather`: `out = result·m + src·(1 − m)` with `m` the binary mask
//!   blurred with PIL `GaussianBlur(radius=3)` (reproduced byte-exactly by
//!   [`pil_gaussian_blur`]), arithmetic in f64 and truncated like numpy.
//! * `FeatherCompensate`: upstream `compensation_utils.paste_compensate(.., fac=1.1)`
//!   additionally shifts the result by 1.1 × the per-channel difference of the
//!   mean colour (source − result) over a ring around the mask: the mask bbox
//!   grown by 10 px (centred, clipped to the window) minus the mask. The mask
//!   is thresholded at `> 128` there (Feather: `>= 128`). With no ring pixels
//!   (upstream: 0/0 = NaN) it falls back to `Feather`.
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

/// Upstream `paste_compensate(.., fac=1.1)` strength.
const COMPENSATE_FAC: f64 = 1.1;
/// Upstream `get_bbox_unmask(off=10)`: total growth of the mask bbox.
const RING_OFF: usize = 10;

/// Per-channel `mean(src over ring) − mean(result over ring)` of upstream
/// `get_unmasked_stat` / `get_bbox_unmask`, on the source-rect window.
/// `set(x, y)` is the binary mask in window coordinates, `src_px` / `res_px` the
/// window's pixels. `None` when the mask is empty or the ring has no pixels.
fn ring_delta<'a, 'b>(
    w: usize,
    h: usize,
    set: impl Fn(usize, usize) -> bool,
    src_px: impl Fn(usize, usize) -> &'a [u8],
    res_px: impl Fn(usize, usize) -> &'b [u8],
) -> Option<[f64; 3]> {
    let (mut r0, mut r1, mut c0, mut c1) = (usize::MAX, 0, usize::MAX, 0);
    for y in 0..h {
        for x in 0..w {
            if set(x, y) {
                (r0, r1, c0, c1) = (r0.min(y), r1.max(y), c0.min(x), c1.max(x));
            }
        }
    }
    if r0 == usize::MAX {
        return None;
    }
    // New size = old + off; centre = (min + max) // 2; half = new // 2 (numpy int arithmetic).
    let (hh, hw) = ((r1 - r0 + 1 + RING_OFF) / 2, (c1 - c0 + 1 + RING_OFF) / 2);
    let (cr, cc) = ((r0 + r1) / 2, (c0 + c1) / 2);
    let (y0, y1) = (cr.saturating_sub(hh), (cr + hh).min(h - 1));
    let (x0, x1) = (cc.saturating_sub(hw), (cc + hw).min(w - 1));
    let (mut sum_src, mut sum_res, mut n) = ([0u64; 3], [0u64; 3], 0u64);
    for y in y0..=y1 {
        for x in x0..=x1 {
            if !set(x, y) {
                n += 1;
                for c in 0..3 {
                    sum_src[c] += src_px(x, y)[c] as u64;
                    sum_res[c] += res_px(x, y)[c] as u64;
                }
            }
        }
    }
    (n > 0).then(|| std::array::from_fn(|c| sum_src[c] as f64 / n as f64 - sum_res[c] as f64 / n as f64))
}

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
        Paste::Feather | Paste::FeatherCompensate => {
            let compensate = mode == Paste::FeatherCompensate;
            // numpy: Feather binarises at `>= 128`, upstream's compensation at `> 128`.
            let lo = if compensate { 129 } else { 128 };
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
                region.extend(row.iter().map(|&v| if v >= lo { 255u8 } else { 0 }));
            }
            let blurred = pil_gaussian_blur(&region, rw, rh, 3.0);
            // Per-channel shift `fac · delta / 255`; `None` = plain Feather.
            let shift = compensate
                .then(|| {
                    let set = |x: usize, y: usize| region[(src.y + y - ry0) * rw + src.x + x - rx0] != 0;
                    let src_px = |x: usize, y: usize| &img.data[(src.y + y) * img.stride + (src.x + x) * ch..][..3];
                    ring_delta(src.w, src.h, set, src_px, result_px)
                })
                .flatten()
                .map(|d| d.map(|d| COMPENSATE_FAC * d / 255.0));
            for y in 0..src.h {
                for x in 0..src.w {
                    // numpy: m = blur/255; (res/255 [+ shift])·m + (1−m)·(src/255), clip, ·255 → uint8.
                    let m = blurred[(src.y + y - ry0) * rw + src.x + x - rx0] as f64 / 255.0;
                    let r = result_px(x, y);
                    let o = (src.y + y) * img.stride + (src.x + x) * ch;
                    for (c, (dst, &res)) in img.data[o..o + 3].iter_mut().zip(r).enumerate() {
                        let (mut res, orig) = (res as f64 / 255.0, *dst as f64 / 255.0);
                        if let Some(shift) = shift {
                            res += shift[c];
                        }
                        let out = res * m + (1.0 - m) * orig;
                        *dst = (if shift.is_some() { out.clamp(0.0, 1.0) } else { out } * 255.0) as u8;
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

    fn compensate_fixture() -> Weights {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/compensate.safetensors");
        Weights::open_with_format(&path, "moebius-cpu-fixture/1").expect("run tools/dump_compensate_fixture.py")
    }

    /// Pastes `result` (w × h RGB) over `image` through a window at the origin that is
    /// as large as the image's long side (`result` is padded to side × side).
    fn paste_at_origin(image: &[u8], result: &[u8], mask: &[u8], w: usize, h: usize, mode: Paste) -> Vec<u8> {
        let side = w.max(h);
        let mut padded = vec![0u8; side * side * 3];
        for y in 0..h {
            padded[y * side * 3..(y * side + w) * 3].copy_from_slice(&result[y * w * 3..(y + 1) * w * 3]);
        }
        let mut data = image.to_vec();
        let plan = CropPlan { x: 0, y: 0, side };
        let mut img = ImageMut { data: &mut data, width: w, height: h, stride: w * 3, channels: 3 };
        paste(&mut img, MaskRef { data: mask, width: w, height: h, stride: w }, &padded, &plan, mode);
        data
    }

    /// Upstream `paste_compensate(.., fac=1.1)` (`tools/dump_compensate_fixture.py`), byte-exact.
    #[test]
    fn compensated_paste_is_byte_exact_vs_upstream() {
        let f = compensate_fixture();
        for (k, (w, h)) in [("a", (96, 80)), ("b", (48, 40)), ("c", (48, 40))] {
            let bytes = |name: &str, shape: &[usize]| -> Vec<u8> {
                f.get_f32(&format!("{k}.{name}"), shape).unwrap().iter().map(|&v| v as u8).collect()
            };
            let (image, result) = (bytes("image", &[h, w, 3]), bytes("result", &[h, w, 3]));
            let (mask, want) = (bytes("mask", &[h, w]), bytes("out", &[h, w, 3]));
            let got = paste_at_origin(&image, &result, &mask, w, h, Paste::FeatherCompensate);
            let bad = got.iter().zip(&want).filter(|(g, w)| g != w).count();
            let max = got.iter().zip(&want).map(|(&g, &w)| g.abs_diff(w)).max().unwrap();
            assert_eq!(bad, 0, "case {k}: {bad} of {} bytes differ from upstream (max diff {max})", want.len());
            let plain = paste_at_origin(&image, &result, &mask, w, h, Paste::Feather);
            assert_ne!(plain, got, "case {k}: compensation must change the output");
        }
    }

    /// A mask covering the whole image leaves no neighbour ring: upstream divides 0/0;
    /// we fall back to plain Feather.
    #[test]
    fn compensation_with_empty_ring_falls_back_to_feather() {
        let (w, h) = (24, 20);
        let image: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
        let result: Vec<u8> = (0..w * h * 3).map(|i| (i * 13 % 241) as u8).collect();
        let mask = vec![255u8; w * h];
        let feather = paste_at_origin(&image, &result, &mask, w, h, Paste::Feather);
        let comp = paste_at_origin(&image, &result, &mask, w, h, Paste::FeatherCompensate);
        assert_eq!(comp, feather);
    }

    /// An empty mask inside the window has no bbox to grow: also plain Feather.
    #[test]
    fn compensation_with_empty_mask_falls_back_to_feather() {
        let (w, h) = (16, 16);
        let image = vec![90u8; w * h * 3];
        let result = vec![200u8; w * h * 3];
        let mask = vec![0u8; w * h];
        let feather = paste_at_origin(&image, &result, &mask, w, h, Paste::Feather);
        assert_eq!(paste_at_origin(&image, &result, &mask, w, h, Paste::FeatherCompensate), feather);
    }
}
