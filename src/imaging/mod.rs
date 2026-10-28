//! Pixel-space pre/post-processing around the 512² model window.

pub mod blend;
pub mod crop;
pub mod resize;

/// Axis-aligned rectangle in image pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

/// 8-bit grayscale mask; pixels ≥ 128 are inpainted (upstream thresholds at
/// 255 / 2).
#[derive(Clone, Copy)]
pub struct MaskRef<'a> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
    /// Bytes per row.
    pub stride: usize,
}

impl MaskRef<'_> {
    pub fn is_set(&self, x: usize, y: usize) -> bool {
        self.data[y * self.stride + x] >= 128
    }

    /// Bounding box of the inpaint region, `None` if the mask is empty.
    pub fn bbox(&self) -> Option<Rect> {
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        for y in 0..self.height {
            let row = &self.data[y * self.stride..y * self.stride + self.width];
            if let Some(first) = row.iter().position(|&v| v >= 128) {
                let last = row.iter().rposition(|&v| v >= 128).unwrap_or(first);
                x0 = x0.min(first);
                x1 = x1.max(last);
                y0 = y0.min(y);
                y1 = y;
            }
        }
        (x0 != usize::MAX).then(|| Rect { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 })
    }
}

/// Square-kernel dilation of the binarised mask (`cv2.dilate` with a
/// `k × k` ones kernel, anchor at `k / 2`), as `0`/`255` with stride = width.
pub fn dilate(mask: MaskRef<'_>, k: usize) -> Vec<u8> {
    let (w, h) = (mask.width, mask.height);
    let mut cur = Vec::with_capacity(w * h);
    for y in 0..h {
        cur.extend(mask.data[y * mask.stride..y * mask.stride + w].iter().map(|&v| if v >= 128 { 255u8 } else { 0 }));
    }
    if k <= 1 {
        return cur;
    }
    // Separable running max over [i − k/2, i + (k − 1 − k/2)], out-of-range ignored.
    let (before, after) = (k / 2, k - 1 - k / 2);
    let mut tmp = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let lo = x.saturating_sub(before);
            let hi = (x + after).min(w - 1);
            tmp[y * w + x] = cur[y * w + lo..=y * w + hi].iter().copied().max().unwrap_or(0);
        }
    }
    for y in 0..h {
        let (lo, hi) = (y.saturating_sub(before), (y + after).min(h - 1));
        for x in 0..w {
            cur[y * w + x] = (lo..=hi).map(|yy| tmp[yy * w + x]).max().unwrap_or(0);
        }
    }
    cur
}

/// 8-bit RGB or RGBA image, edited in place (alpha is preserved).
pub struct ImageMut<'a> {
    pub data: &'a mut [u8],
    pub width: usize,
    pub height: usize,
    /// Bytes per row.
    pub stride: usize,
    /// 3 (RGB) or 4 (RGBA).
    pub channels: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_bbox() {
        let (w, h) = (6, 5);
        let mut m = vec![0u8; w * h];
        assert_eq!(MaskRef { data: &m, width: w, height: h, stride: w }.bbox(), None);
        m[w + 2] = 255;
        m[3 * w + 4] = 200;
        m[2 * w + 1] = 127; // below threshold
        let b = MaskRef { data: &m, width: w, height: h, stride: w }.bbox();
        assert_eq!(b, Some(Rect { x: 2, y: 1, w: 3, h: 3 }));
    }

    #[test]
    fn dilate_matches_cv2_anchor() {
        let (w, h) = (9, 7);
        let mut m = vec![0u8; w * h];
        m[3 * w + 4] = 255;
        m[0] = 200;
        fn mr(d: &[u8]) -> MaskRef<'_> {
            MaskRef { data: d, width: 9, height: 7, stride: 9 }
        }
        // k = 3: one pixel each side.
        let d = dilate(mr(&m), 3);
        assert_eq!(d.iter().filter(|&&v| v == 255).count(), 9 + 4);
        assert_eq!(d[2 * w + 3], 255);
        assert_eq!(d[4 * w + 5], 255);
        assert_eq!(d[2 * w + 6], 0);
        // k = 2 (even): anchor 1 -> window [x-1, x] (reaches up-left only).
        let d = dilate(mr(&m), 2);
        assert_eq!(d[3 * w + 4], 255);
        assert_eq!(d[2 * w + 3], 0, "up-left pixel is not covered");
        assert_eq!(d[4 * w + 5], 255);
        // k = 0/1: binarise only.
        assert_eq!(dilate(mr(&m), 1).iter().filter(|&&v| v == 255).count(), 2);
        let sub = [127u8; 63];
        assert!(dilate(mr(&sub), 3).iter().all(|&v| v == 0));
    }
}
