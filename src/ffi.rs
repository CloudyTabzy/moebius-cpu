//! C ABI for the GIMP plugin. Mirrors `include/moebius_cpu.h`; keep the two
//! in sync and bump `MOEBIUS_ABI_VERSION` on any change.
//!
//! Conventions: every entry point returns a status code; on failure the
//! message is available from `moebius_last_error()` on the same thread.
//! Panics never cross the boundary — they are caught and reported as
//! `MOEBIUS_ERR_INTERNAL`. The library is MSVC-built but exposes only C
//! types, so the Zig/MinGW plugin links it as a plain DLL.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use crate::imaging::{ImageMut, MaskRef};
use crate::pipeline::{Control, Engine, Progress};
use crate::preset::{Paste, Preset};
use crate::Error;

pub const MOEBIUS_ABI_VERSION: u32 = 1;

pub const MOEBIUS_OK: i32 = 0;
pub const MOEBIUS_ERR_INVALID_ARGUMENT: i32 = 1;
pub const MOEBIUS_ERR_WEIGHTS: i32 = 2;
pub const MOEBIUS_ERR_UNSUPPORTED_CPU: i32 = 3;
pub const MOEBIUS_ERR_CANCELLED: i32 = 4;
pub const MOEBIUS_ERR_IO: i32 = 5;
pub const MOEBIUS_ERR_INTERNAL: i32 = 6;

pub const MOEBIUS_PRESET_EXACT: u32 = 0;
pub const MOEBIUS_PRESET_FAST: u32 = 1;

pub const MOEBIUS_PASTE_DEFAULT: u32 = 0;
pub const MOEBIUS_PASTE_NONE: u32 = 1;
pub const MOEBIUS_PASTE_FEATHER: u32 = 2;
/// Not ported yet (M8): currently identical to `MOEBIUS_PASTE_FEATHER`.
pub const MOEBIUS_PASTE_FEATHER_COMPENSATE: u32 = 3;

/// Opaque handle.
pub struct MoebiusEngine(Engine);

/// Zero-initialised = preset defaults.
#[repr(C)]
pub struct MoebiusParams {
    pub preset: u32,
    /// 0 = preset default.
    pub steps: u32,
    /// <= 0 = preset default.
    pub guidance: f32,
    pub seed: u64,
    pub paste: u32,
    pub mask_dilate: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct MoebiusRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// `done`/`total` count denoising steps; `stage` 0 = encode, 1 = denoise,
/// 2 = decode. Return non-zero to cancel.
pub type MoebiusProgressFn = Option<unsafe extern "C" fn(user: *mut c_void, stage: u32, done: u32, total: u32) -> i32>;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn fail(code: i32, msg: impl Into<String>) -> i32 {
    let msg = CString::new(msg.into().replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = msg);
    code
}

fn fail_with(err: Error) -> i32 {
    let code = match err {
        Error::InvalidInput(_) => MOEBIUS_ERR_INVALID_ARGUMENT,
        Error::Weights(_) => MOEBIUS_ERR_WEIGHTS,
        Error::Io(_) => MOEBIUS_ERR_IO,
        Error::UnsupportedCpu(_) => MOEBIUS_ERR_UNSUPPORTED_CPU,
        Error::Cancelled => MOEBIUS_ERR_CANCELLED,
    };
    fail(code, err.to_string())
}

/// Runs `f`, converting errors and panics into status codes.
fn guard(f: impl FnOnce() -> crate::Result<()>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => MOEBIUS_OK,
        Ok(Err(e)) => fail_with(e),
        Err(panic) => {
            let msg = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("panic");
            fail(MOEBIUS_ERR_INTERNAL, format!("internal error: {msg}"))
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn moebius_abi_version() -> u32 {
    MOEBIUS_ABI_VERSION
}

/// Message for the last failed call on this thread (valid until the next
/// call on this thread). Never NULL.
#[unsafe(no_mangle)]
pub extern "C" fn moebius_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// # Safety
/// `weights_dir` must be a NUL-terminated UTF-8 path; `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn moebius_engine_create(
    weights_dir: *const c_char,
    threads: u32,
    out: *mut *mut MoebiusEngine,
) -> i32 {
    if weights_dir.is_null() || out.is_null() {
        return fail(MOEBIUS_ERR_INVALID_ARGUMENT, "null argument");
    }
    // SAFETY: caller contract.
    let Ok(dir) = unsafe { CStr::from_ptr(weights_dir) }.to_str() else {
        return fail(MOEBIUS_ERR_INVALID_ARGUMENT, "weights_dir is not UTF-8");
    };
    pin_module();
    guard(|| {
        let engine = Engine::load(Path::new(dir), threads as usize)?;
        // SAFETY: caller contract.
        unsafe { *out = Box::into_raw(Box::new(MoebiusEngine(engine))) };
        Ok(())
    })
}

/// Keeps this DLL loaded for the life of the process. The allocator's
/// thread-exit callback sits in an FLS slot that is never freed, so unloading
/// the DLL while other threads live would make their exit call into unmapped
/// code. Pinning makes any later `FreeLibrary` by the host a no-op.
#[cfg(all(windows, feature = "rusty-alloc"))]
fn pin_module() {
    use std::sync::Once;
    static PIN: Once = Once::new();
    PIN.call_once(|| {
        const FROM_ADDRESS: u32 = 0x4;
        const PIN_FLAG: u32 = 0x1;
        unsafe extern "system" {
            fn GetModuleHandleExW(flags: u32, module_name: *const u16, module: *mut *mut c_void) -> i32;
        }
        let mut module = std::ptr::null_mut();
        // SAFETY: with FROM_ADDRESS the "name" is an address inside this module
        // (this function); `module` is a valid out-pointer. Failure only means
        // the DLL stays unpinned, which is the status quo.
        unsafe { GetModuleHandleExW(FROM_ADDRESS | PIN_FLAG, pin_module as *const u16, &mut module) };
    });
}

#[cfg(not(all(windows, feature = "rusty-alloc")))]
fn pin_module() {}

/// # Safety
/// `engine` must come from `moebius_engine_create` and not be used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn moebius_engine_destroy(engine: *mut MoebiusEngine) {
    if !engine.is_null() {
        // SAFETY: caller contract.
        drop(unsafe { Box::from_raw(engine) });
    }
}

/// Inpaints `pixels` (RGB or RGBA, 8-bit, `stride` bytes per row) in place
/// wherever `mask` (8-bit, `mask_stride` bytes per row) is ≥ 128.
///
/// # Safety
/// Buffers must cover `height` rows of their strides; `params` may be NULL
/// (Exact defaults); `changed` may be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn moebius_inpaint(
    engine: *const MoebiusEngine,
    pixels: *mut u8,
    width: u32,
    height: u32,
    stride: u32,
    channels: u32,
    mask: *const u8,
    mask_stride: u32,
    params: *const MoebiusParams,
    progress: MoebiusProgressFn,
    user: *mut c_void,
    changed: *mut MoebiusRect,
) -> i32 {
    if engine.is_null() || pixels.is_null() || mask.is_null() {
        return fail(MOEBIUS_ERR_INVALID_ARGUMENT, "null argument");
    }
    let (w, h, c) = (width as usize, height as usize, channels as usize);
    if w == 0 || h == 0 || (stride as usize) < w * c || (mask_stride as usize) < w {
        return fail(MOEBIUS_ERR_INVALID_ARGUMENT, "bad dimensions or stride");
    }
    // SAFETY: caller contract on buffer extents and pointer validity.
    let (engine, pixels, mask, params) = unsafe {
        (
            &(*engine).0,
            std::slice::from_raw_parts_mut(pixels, (h - 1) * stride as usize + w * c),
            std::slice::from_raw_parts(mask, (h - 1) * mask_stride as usize + w),
            params.as_ref(),
        )
    };
    let p = match resolve_params(params) {
        Ok(p) => p,
        Err(e) => return fail_with(e),
    };
    guard(|| {
        let mut img = ImageMut { data: pixels, width: w, height: h, stride: stride as usize, channels: c };
        let mask = MaskRef { data: mask, width: w, height: h, stride: mask_stride as usize };
        let mut report = |pr: Progress| match progress {
            // SAFETY: the callback and `user` are supplied together by the caller.
            Some(cb) if unsafe { cb(user, pr.stage as u32, pr.done as u32, pr.total as u32) } != 0 => Control::Cancel,
            _ => Control::Continue,
        };
        let rect = engine.inpaint(&mut img, mask, &p, &mut report)?;
        if !changed.is_null() {
            // SAFETY: caller contract.
            unsafe {
                *changed = MoebiusRect { x: rect.x as u32, y: rect.y as u32, width: rect.w as u32, height: rect.h as u32 }
            };
        }
        Ok(())
    })
}

fn resolve_params(p: Option<&MoebiusParams>) -> crate::Result<crate::InpaintParams> {
    let Some(p) = p else { return Ok(Preset::Exact.params()) };
    let mut out = match p.preset {
        MOEBIUS_PRESET_EXACT => Preset::Exact.params(),
        MOEBIUS_PRESET_FAST => Preset::Fast.params(),
        other => return Err(Error::InvalidInput(format!("unknown preset {other}"))),
    };
    if p.steps > 0 {
        out.steps = p.steps as usize;
    }
    if p.guidance > 0.0 {
        out.guidance = p.guidance;
    }
    out.seed = p.seed;
    out.mask_dilate = p.mask_dilate;
    out.paste = match p.paste {
        MOEBIUS_PASTE_DEFAULT => out.paste,
        MOEBIUS_PASTE_NONE => Paste::None,
        MOEBIUS_PASTE_FEATHER => Paste::Feather,
        MOEBIUS_PASTE_FEATHER_COMPENSATE => Paste::FeatherCompensate,
        other => return Err(Error::InvalidInput(format!("unknown paste mode {other}"))),
    };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_arguments_are_reported_not_panicked() {
        let mut px = [0u8; 12];
        let m = [0u8; 4];
        // SAFETY: null engine is rejected before any dereference.
        let rc = unsafe {
            moebius_inpaint(std::ptr::null(), px.as_mut_ptr(), 2, 2, 6, 3, m.as_ptr(), 2, std::ptr::null(), None, std::ptr::null_mut(), std::ptr::null_mut())
        };
        assert_eq!(rc, MOEBIUS_ERR_INVALID_ARGUMENT);
        let msg = unsafe { CStr::from_ptr(moebius_last_error()) };
        assert_eq!(msg.to_str().unwrap(), "null argument");
    }

    #[test]
    fn panics_become_internal_errors() {
        assert_eq!(guard(|| panic!("boom")), MOEBIUS_ERR_INTERNAL);
        let msg = unsafe { CStr::from_ptr(moebius_last_error()) };
        assert!(msg.to_str().unwrap().contains("boom"));
    }

    #[test]
    fn zeroed_params_are_exact_defaults() {
        let p = MoebiusParams { preset: 0, steps: 0, guidance: 0.0, seed: 0, paste: 0, mask_dilate: 0 };
        assert_eq!(resolve_params(Some(&p)).unwrap(), Preset::Exact.params());
        let fast = MoebiusParams { preset: 1, steps: 9, ..p };
        let r = resolve_params(Some(&fast)).unwrap();
        assert_eq!((r.steps, r.roi_decode), (9, true));
        assert!(resolve_params(Some(&MoebiusParams { preset: 7, ..p })).is_err());
    }
}
