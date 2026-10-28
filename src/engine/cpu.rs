//! CPU capability checks and thread-pool sizing.
//!
//! The crate is built for baseline x86-64 so the DLL always loads; hot
//! kernels are compiled with `#[target_feature(enable = "avx2,fma")]` and
//! only reached after `require_avx2_fma` succeeds at engine creation.
//! (`gemm` does its own runtime dispatch.)

use crate::{Error, Result};

pub fn require_avx2_fma() -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            return Ok(());
        }
        Err(Error::UnsupportedCpu("AVX2 and FMA are required (Intel Haswell / AMD Zen or newer)".into()))
    }
    #[cfg(not(target_arch = "x86_64"))]
    Err(Error::UnsupportedCpu("only x86-64 is supported".into()))
}

/// Default worker count: one per physical core. Measured (`docs/PLAN.md` §4): GEMM peaks at one
/// thread per physical core, and SMT siblings add nothing for FMA-bound kernels (`examples/gemm_probe.rs`).
pub fn default_threads() -> usize {
    num_cpus::get_physical().max(1)
}
