//! Runtime ISA dispatch for hand-written kernels.
//!
//! The crate targets baseline x86-64 so the DLL loads anywhere and can
//! report an unsupported CPU cleanly. Kernels are written as plain loops
//! over contiguous channel slices; `multiversion!` compiles each one twice —
//! once with `avx2,fma` enabled (where LLVM vectorises the loops and lowers
//! `mul_add` to `vfmadd`) and once for baseline — and picks at runtime.
//!
//! Dispatch happens per chunk, inside rayon closures: closures do not
//! inherit `#[target_feature]`, so the feature-enabled function must be the
//! one doing the inner loops.

use std::sync::OnceLock;

pub fn avx2_fma() -> bool {
    static DETECTED: OnceLock<bool> = OnceLock::new();
    *DETECTED.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        false
    })
}

/// Defines `fn $name(args) -> ret` that runs `$body` compiled with AVX2+FMA
/// when the CPU has them, and the baseline build otherwise.
macro_rules! multiversion {
    ($(#[$meta:meta])* $vis:vis fn $name:ident($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block) => {
        $(#[$meta])*
        $vis fn $name($($arg: $ty),*) $(-> $ret)? {
            #[inline(always)]
            #[allow(clippy::too_many_arguments)]
            fn body($($arg: $ty),*) $(-> $ret)? $body

            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2,fma")]
                #[allow(clippy::too_many_arguments)]
                unsafe fn avx2($($arg: $ty),*) $(-> $ret)? {
                    body($($arg),*)
                }
                if $crate::engine::simd::avx2_fma() {
                    // SAFETY: AVX2 and FMA support was detected at runtime.
                    return unsafe { avx2($($arg),*) };
                }
            }
            body($($arg),*)
        }
    };
}
pub(crate) use multiversion;
