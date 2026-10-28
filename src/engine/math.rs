//! Branch-free transcendental functions that vectorise.
//!
//! A Moebius step evaluates ~170 M SiLUs (MixFFN alone is 7.5·C per token);
//! `f32::exp` is a scalar libm call (~5 ns), which would cost ~0.85 s/step.
//! `exp` here is the Cephes range reduction + degree-6 polynomial in pure
//! arithmetic, so LLVM turns loops calling it into 8-wide AVX2 code.

const LOG2_E: f32 = std::f32::consts::LOG2_E;
// ln 2 split so that n·LN2_HI is exact for |n| ≤ 128.
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;
// Keeps n ≤ 127 so the exponent bits never overflow into inf.
const MAX_X: f32 = 88.0;
const MIN_X: f32 = -87.336_55;

/// eˣ with ≤ 2 ulp error over the f32 range (clamped to the normal range).
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    let x = x.clamp(MIN_X, MAX_X);
    let n = (x * LOG2_E).round();
    let r = n.mul_add(-LN2_LO, n.mul_add(-LN2_HI, x));
    let mut p = 1.987_569_1e-4f32;
    p = p.mul_add(r, 1.398_199_9e-3);
    p = p.mul_add(r, 8.333_452e-3);
    p = p.mul_add(r, 4.166_579_6e-2);
    p = p.mul_add(r, 1.666_666_5e-1);
    p = p.mul_add(r, 5e-1);
    let poly = (p * r).mul_add(r, r) + 1.0;
    poly * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// x · σ(x).
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + exp(-x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_std() {
        let mut worst = 0.0f64;
        let mut x = -87.0f32;
        while x < 88.0 {
            let (got, want) = (exp(x) as f64, (x as f64).exp());
            worst = worst.max(((got - want) / want).abs());
            x += 0.0137;
        }
        assert!(worst < 3e-7, "worst relative error {worst:e}");
        assert_eq!(exp(-1000.0), exp(MIN_X));
        assert!(exp(1000.0).is_finite());
    }

    #[test]
    fn silu_matches_definition() {
        for i in -2000..2000 {
            let x = i as f32 * 0.01;
            let want = x as f64 / (1.0 + (-x as f64).exp());
            assert!((silu(x) as f64 - want).abs() <= 1e-6 * (1.0 + want.abs()), "x={x}");
        }
        assert!(silu(-200.0).abs() < 1e-30);
        assert_eq!(silu(200.0), 200.0);
    }
}
