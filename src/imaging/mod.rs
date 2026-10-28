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
}
