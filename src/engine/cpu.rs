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

/// Opts the calling thread, and once the whole process, out of Windows 11
/// power throttling (EcoQoS). On the Balanced plan Windows moves a process it
/// deems "background" (for example, not the focused window: a plugin process
/// while the user waits in GIMP) onto efficiency cores at low priority after a
/// moment. Measured: a UNet step went from 1.1 s to
/// 14.5 s within seconds, with the CPU only 25–40% busy. Inference is
/// foreground work by definition, so the engine declares it.
/// `MOEBIUS_POWER_THROTTLING=allow` keeps the OS default (A/B runs).
pub fn disable_power_throttling() {
    #[cfg(windows)]
    {
        use std::ffi::c_void;
        use std::sync::Once;

        if std::env::var("MOEBIUS_POWER_THROTTLING").is_ok_and(|v| v.eq_ignore_ascii_case("allow")) {
            return;
        }
        #[repr(C)]
        struct ThrottlingState {
            version: u32,
            control_mask: u32,
            state_mask: u32,
        }
        const VERSION: u32 = 1;
        const EXECUTION_SPEED: u32 = 0x1;
        const PROCESS_POWER_THROTTLING: i32 = 4; // PROCESS_INFORMATION_CLASS::ProcessPowerThrottling
        const THREAD_POWER_THROTTLING: i32 = 3; // THREAD_INFORMATION_CLASS::ThreadPowerThrottling
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut c_void;
            fn GetCurrentThread() -> *mut c_void;
            fn SetProcessInformation(process: *mut c_void, class: i32, info: *const c_void, size: u32) -> i32;
            fn SetThreadInformation(thread: *mut c_void, class: i32, info: *const c_void, size: u32) -> i32;
        }
        // Control the execution-speed policy, with the throttled bit clear: always high QoS.
        let state = ThrottlingState { version: VERSION, control_mask: EXECUTION_SPEED, state_mask: 0 };
        let info = (&raw const state).cast::<c_void>();
        let size = size_of::<ThrottlingState>() as u32;
        static PROCESS: Once = Once::new();
        // SAFETY: pseudo-handles of the current process / thread; `info` points to a
        // correctly laid out, live struct of `size` bytes. Failure (older Windows)
        // just leaves the OS default in place.
        PROCESS.call_once(|| unsafe {
            SetProcessInformation(GetCurrentProcess(), PROCESS_POWER_THROTTLING, info, size);
        });
        unsafe { SetThreadInformation(GetCurrentThread(), THREAD_POWER_THROTTLING, info, size) };
    }
}

/// Default worker count: one per physical core. Measured (`docs/PLAN.md` §4): GEMM peaks at one
/// thread per physical core, and SMT siblings add nothing for FMA-bound kernels (`examples/gemm_probe.rs`).
pub fn default_threads() -> usize {
    num_cpus::get_physical().max(1)
}
