//! Square context window around the mask.
//!
//! The model only accepts square 512² input, so the region sent to it must
//! be square in source pixels too or the result is distorted. The old
//! Python plugin clipped its window at image edges, producing non-square
//! crops; here the window is *shifted* to stay inside the image and only
//! padded (edge-replicated) when the image itself is smaller than the
//! window on that axis.
//!
//! Size: the mask's long side plus 50% margin on each side (as the old
//! plugin), but never smaller than 512 px when the image allows it, so small
//! objects are inpainted at native resolution instead of being upscaled.

use super::Rect;
use crate::config::IMAGE_SIZE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropPlan {
    /// Top-left of the square window in image coordinates; negative when the
    /// window extends past the image (padded area).
    pub x: isize,
    pub y: isize,
    pub side: usize,
}

impl CropPlan {
    pub fn new(mask_bbox: Rect, img_w: usize, img_h: usize) -> Self {
        let long = mask_bbox.w.max(mask_bbox.h);
        let side = (long + 2 * (long / 2)).max(IMAGE_SIZE.min(img_w.max(img_h)));
        Self {
            x: place(mask_bbox.x, mask_bbox.w, side, img_w),
            y: place(mask_bbox.y, mask_bbox.h, side, img_h),
            side,
        }
    }

    /// Part of the window that lies inside the image (what gets pasted back).
    pub fn source_rect(&self, img_w: usize, img_h: usize) -> Rect {
        let x0 = self.x.max(0) as usize;
        let y0 = self.y.max(0) as usize;
        let x1 = ((self.x + self.side as isize) as usize).min(img_w);
        let y1 = ((self.y + self.side as isize) as usize).min(img_h);
        Rect { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }
    }
}

/// Window start on one axis: centred on the mask, shifted inside the image
/// when it fits, otherwise centred on the image.
fn place(start: usize, len: usize, side: usize, extent: usize) -> isize {
    if side <= extent {
        let centred = (2 * start + len) as isize / 2 - side as isize / 2;
        centred.clamp(0, (extent - side) as isize)
    } else {
        -(((side - extent) / 2) as isize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(x: usize, y: usize, w: usize, h: usize) -> Rect {
        Rect { x, y, w, h }
    }

    #[test]
    fn small_mask_uses_native_512_window() {
        assert_eq!(CropPlan::new(r(100, 100, 40, 40), 2000, 2000), CropPlan { x: 0, y: 0, side: 512 });
        let p = CropPlan::new(r(900, 900, 40, 40), 2000, 2000);
        assert_eq!((p.x, p.y, p.side), (664, 664, 512));
    }

    #[test]
    fn large_mask_gets_half_margin() {
        assert_eq!(CropPlan::new(r(500, 500, 600, 400), 2000, 1500), CropPlan { x: 200, y: 100, side: 1200 });
    }

    #[test]
    fn shifted_not_clipped_at_edges() {
        assert_eq!(CropPlan::new(r(1900, 10, 80, 80), 2000, 2000), CropPlan { x: 1488, y: 0, side: 512 });
    }

    #[test]
    fn padded_when_image_too_small() {
        let p = CropPlan::new(r(100, 50, 100, 100), 300, 200);
        assert_eq!(p, CropPlan { x: 0, y: -50, side: 300 });
        assert_eq!(p.source_rect(300, 200), r(0, 0, 300, 200));
    }

    #[test]
    fn window_always_contains_mask() {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = |m: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % m as u64) as usize
        };
        for _ in 0..10_000 {
            let (iw, ih) = (1 + next(3000), 1 + next(3000));
            let (x, y) = (next(iw), next(ih));
            let b = r(x, y, 1 + next(iw - x), 1 + next(ih - y));
            let p = CropPlan::new(b, iw, ih);
            assert!(p.x <= b.x as isize && p.y <= b.y as isize, "{b:?} {p:?}");
            assert!(p.x + p.side as isize >= (b.x + b.w) as isize, "{b:?} {iw}x{ih} {p:?}");
            assert!(p.y + p.side as isize >= (b.y + b.h) as isize, "{b:?} {iw}x{ih} {p:?}");
            let s = p.source_rect(iw, ih);
            assert!(s.x + s.w <= iw && s.y + s.h <= ih);
        }
    }
}
