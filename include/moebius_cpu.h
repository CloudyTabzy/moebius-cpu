/* moebius-cpu C API — keep in sync with src/ffi.rs (MOEBIUS_ABI_VERSION). */
#ifndef MOEBIUS_CPU_H
#define MOEBIUS_CPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MOEBIUS_ABI_VERSION 1u

#define MOEBIUS_OK                   0
#define MOEBIUS_ERR_INVALID_ARGUMENT 1
#define MOEBIUS_ERR_WEIGHTS          2
#define MOEBIUS_ERR_UNSUPPORTED_CPU  3
#define MOEBIUS_ERR_CANCELLED        4
#define MOEBIUS_ERR_IO               5
#define MOEBIUS_ERR_INTERNAL         6

#define MOEBIUS_PRESET_EXACT 0u
#define MOEBIUS_PRESET_FAST  1u

#define MOEBIUS_PASTE_DEFAULT            0u
#define MOEBIUS_PASTE_NONE               1u
#define MOEBIUS_PASTE_FEATHER            2u
#define MOEBIUS_PASTE_FEATHER_COMPENSATE 3u

typedef struct MoebiusEngine MoebiusEngine;

/* Zero-initialised = Exact preset defaults. */
typedef struct MoebiusParams {
    uint32_t preset;      /* MOEBIUS_PRESET_* */
    uint32_t steps;       /* 0 = preset default */
    float    guidance;    /* <= 0 = preset default */
    uint64_t seed;
    uint32_t paste;       /* MOEBIUS_PASTE_* */
    uint32_t mask_dilate; /* px, 0 = off */
} MoebiusParams;

typedef struct MoebiusRect {
    uint32_t x, y, width, height;
} MoebiusRect;

/* stage: 0 encode, 1 denoise, 2 decode. Return non-zero to cancel. */
typedef int32_t (*MoebiusProgressFn)(void *user, uint32_t stage, uint32_t done, uint32_t total);

uint32_t moebius_abi_version(void);

/* Message for the last failed call on the calling thread. Never NULL. */
const char *moebius_last_error(void);

/* weights_dir: UTF-8 path holding unet.safetensors + vae.safetensors.
 * threads: 0 = one per physical core. */
int32_t moebius_engine_create(const char *weights_dir, uint32_t threads, MoebiusEngine **out);
void moebius_engine_destroy(MoebiusEngine *engine);

/* In-place inpaint of 8-bit RGB/RGBA `pixels` where `mask` >= 128.
 * params, progress, changed may be NULL. Alpha is preserved.
 * On MOEBIUS_ERR_CANCELLED the image is untouched. */
int32_t moebius_inpaint(const MoebiusEngine *engine,
                        uint8_t *pixels, uint32_t width, uint32_t height,
                        uint32_t stride, uint32_t channels,
                        const uint8_t *mask, uint32_t mask_stride,
                        const MoebiusParams *params,
                        MoebiusProgressFn progress, void *user,
                        MoebiusRect *changed);

#ifdef __cplusplus
}
#endif

#endif /* MOEBIUS_CPU_H */
