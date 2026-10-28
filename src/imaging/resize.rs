//! Window ↔ model resampling (M7) via `fast_image_resize` Lanczos3, which
//! uses the same convolution formulation as PIL's `LANCZOS` used upstream.
//! The source window is read with edge replication where it extends past the
//! image (`crop::CropPlan`); the mask is resized with nearest-neighbour and
//! re-thresholded, as upstream.

use fast_image_resize::images::Image;
use fast_image_resize::{PixelType, ResizeOptions, Resizer};

use super::blend::to_u8;
use super::crop::CropPlan;
use super::{ImageMut, MaskRef};
use crate::config::IMAGE_SIZE;

/// Lanczos3 resize of a tightly packed 8-bit image (`pixel` = U8 or U8x3).
/// Same-size requests are a copy, as in PIL.
fn resize_u8(src: Vec<u8>, from: usize, to: usize, pixel: PixelType) -> Vec<u8> {
    if from == to {
        return src;
    }
    let src = Image::from_vec_u8(from as u32, from as u32, src, pixel).expect("source buffer size");
    let mut dst = Image::new(to as u32, to as u32, pixel);
    Resizer::new().resize(&src, &mut dst, &ResizeOptions::new()).expect("resize of a valid 8-bit image");
    dst.into_vec()
}

/// Square window → 512² RGB in [-1, 1] (NHWC) and binary (0/255) 512² mask.
///
/// The window is read with edge replication for pixels past the image, the
/// mask with zeros. Values ≥ 128 are masked (upstream `x < 255/2 → 0`),
/// before and after the resize.
pub fn to_model(img: &ImageMut<'_>, mask: MaskRef<'_>, plan: &CropPlan) -> (Vec<f32>, Vec<u8>) {
    let side = plan.side;
    let (w, h, ch) = (img.width as isize, img.height as isize, img.channels);
    let mut rgb = Vec::with_capacity(side * side * 3);
    let mut bin = Vec::with_capacity(side * side);
    for wy in 0..side as isize {
        let y = plan.y + wy;
        let yc = y.clamp(0, h - 1) as usize;
        for wx in 0..side as isize {
            let x = plan.x + wx;
            let xc = x.clamp(0, w - 1) as usize;
            let o = yc * img.stride + xc * ch;
            rgb.extend_from_slice(&img.data[o..o + 3]);
            let inside = (0..h).contains(&y) && (0..w).contains(&x);
            bin.push(if inside && mask.data[yc * mask.stride + xc] >= 128 { 255 } else { 0 });
        }
    }
    let rgb = resize_u8(rgb, side, IMAGE_SIZE, PixelType::U8x3);
    let mut bin = resize_u8(bin, side, IMAGE_SIZE, PixelType::U8);
    for v in &mut bin {
        *v = if *v >= 128 { 255 } else { 0 };
    }
    // np.asarray(img) / 255. * 2 - 1 in f64, stored as f32.
    let rgb = rgb.iter().map(|&v| (v as f64 / 255.0 * 2.0 - 1.0) as f32).collect();
    (rgb, bin)
}

/// 512² RGB in [0, 1] → window side, 8-bit (truncating, then Lanczos3 as
/// upstream), for blending.
pub fn from_model(rgb: &[f32], side: usize) -> Vec<u8> {
    assert_eq!(rgb.len(), IMAGE_SIZE * IMAGE_SIZE * 3, "model output must be 512² RGB");
    let bytes: Vec<u8> = rgb.iter().map(|&v| to_u8(v)).collect();
    resize_u8(bytes, IMAGE_SIZE, side, PixelType::U8x3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imaging::Rect;

    fn model_value(v: u8) -> f32 {
        (v as f64 / 255.0 * 2.0 - 1.0) as f32
    }

    #[test]
    fn same_size_window_is_an_exact_copy() {
        let n = 512 * 512;
        let img: Vec<u8> = (0..n * 3).map(|i| (i * 7 % 256) as u8).collect();
        let mut mask = vec![0u8; n];
        mask[10 * 512 + 10] = 127; // below threshold
        mask[10 * 512 + 11] = 128; // masked
        mask[10 * 512 + 12] = 255;
        let mut data = img.clone();
        let im = ImageMut { data: &mut data, width: 512, height: 512, stride: 512 * 3, channels: 3 };
        let m = MaskRef { data: &mask, width: 512, height: 512, stride: 512 };
        // A mask covering most of the image would make a larger padded window
        // for other sizes; a 512² image is always taken whole.
        let plan = CropPlan::new(Rect { x: 0, y: 0, w: 500, h: 400 }, 512, 512);
        assert_eq!((plan.x, plan.y, plan.side), (0, 0, 512));
        let (rgb, mk) = to_model(&im, m, &plan);
        assert_eq!(rgb.len(), n * 3);
        for (i, (&v, &s)) in rgb.iter().zip(&img).enumerate() {
            assert_eq!(v, model_value(s), "value {i}");
        }
        // Threshold: 127 -> 0, 128 -> masked.
        assert_eq!((mk[10 * 512 + 10], mk[10 * 512 + 11], mk[10 * 512 + 12]), (0, 255, 255));
        assert_eq!(mk.iter().filter(|&&v| v != 0).count(), 2);
    }

    #[test]
    fn rgba_alpha_is_ignored_and_stride_respected() {
        let (w, h) = (512, 512);
        let stride = w * 4 + 8;
        let mut data = vec![99u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                let o = y * stride + x * 4;
                data[o..o + 4].copy_from_slice(&[(x % 256) as u8, (y % 256) as u8, 7, 200]);
            }
        }
        let mask = vec![255u8; w * h];
        let im = ImageMut { data: &mut data, width: w, height: h, stride, channels: 4 };
        let plan = CropPlan { x: 0, y: 0, side: 512 };
        let (rgb, _) = to_model(&im, MaskRef { data: &mask, width: w, height: h, stride: w }, &plan);
        let o = (5 * 512 + 9) * 3;
        assert_eq!(rgb[o..o + 3], [model_value(9), model_value(5), model_value(7)]);
    }

    #[test]
    fn non_512_window_resizes_and_replicates_edges() {
        // 100x60 image, window 100 (padded by 20 rows top and bottom): a flat
        // colour must stay flat through Lanczos, including the padded rows.
        let (w, h) = (100, 60);
        let mut data: Vec<u8> = [40u8, 120, 200].repeat(w * h);
        let mask = vec![255u8; w * h];
        let im = ImageMut { data: &mut data, width: w, height: h, stride: w * 3, channels: 3 };
        let plan = CropPlan { x: 0, y: -20, side: 100 };
        let (rgb, mk) = to_model(&im, MaskRef { data: &mask, width: w, height: h, stride: w }, &plan);
        for px in rgb.chunks_exact(3) {
            for (&v, c) in px.iter().zip([40u8, 120, 200]) {
                assert!((v - model_value(c)).abs() < 2.0 / 255.0, "{v}");
            }
        }
        // Mask: 0 in the padded rows (top 20% of the window), 255 inside.
        assert_eq!(mk[2 * 512 + 256], 0);
        assert_eq!(mk[256 * 512 + 256], 255);
        assert_eq!(mk[509 * 512 + 256], 0);

        // And back: 512² in [0,1] → side × side 8-bit.
        let back = from_model(&vec![0.5; 512 * 512 * 3], 100);
        assert_eq!(back.len(), 100 * 100 * 3);
        assert!(back.iter().all(|&v| v == 127));
        let same = from_model(&vec![0.999; 512 * 512 * 3], 512);
        assert!(same.iter().all(|&v| v == 254));
    }
}
