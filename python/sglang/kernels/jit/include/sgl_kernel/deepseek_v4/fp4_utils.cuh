#pragma once

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#ifndef USE_ROCM
#include <cuda_fp4.h>
#endif

#include <cstdint>

// FP4 (e2m1) helpers: per-32 UE8M0 for the indexer, per-16 E4M3 for compressed KV.

namespace sglang {

namespace deepseek_v4::fp4 {

/// Largest finite e2m1 value.
constexpr float kMax = 6.0f;
/// `6 * 2^-126`, the amax floor `torch_quant.fake_quant_fp4` clamps to.
constexpr float kAmaxFloor = 6.0f * 1.1754943508222875e-38f;
/// Elements sharing one ue8m0 scale.
constexpr uint32_t kBlockSize = 32;
/// Compressed-KV elements sharing one E4M3 scale.
constexpr uint32_t kCompressedKVBlockSize = 16;
#ifdef USE_ROCM
/// \brief `cvt.rn.satfinite.e2m1x2.f32` for one value, in software: RNE onto the
/// e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}, saturated at 6, sign kept (also for -0).
SGL_DEVICE uint32_t e2m1_code(float x) {
  const float mag = fminf(fabsf(x), kMax);
  const float step = mag < 2.0f ? 0.5f : (mag < 4.0f ? 1.0f : 2.0f);
  const float q = rintf(mag / step) * step;
  // q in {0, .5, 1, 1.5} -> 2q; {2, 3} -> q + 2; {4, 6} -> q / 2 + 4.
  const uint32_t idx = q < 2.0f ? static_cast<uint32_t>(q * 2.0f)
                                : (q < 4.0f ? static_cast<uint32_t>(q) + 2u : static_cast<uint32_t>(q * 0.5f) + 4u);
  return idx | (__float_as_uint(x) >> 31 << 3);
}

/// \brief The e2m1 grid value a code stands for, exact in fp32.
SGL_DEVICE float e2m1_value(uint32_t code) {
  const uint32_t idx = code & 0x7u;
  const float mag = idx < 4u ? static_cast<float>(idx) * 0.5f
                             : (idx < 6u ? static_cast<float>(idx) - 2.0f : static_cast<float>(idx - 4u) * 2.0f);
  return (code & 0x8u) ? -mag : mag;
}

/// \brief `__nv_cvt_float2_to_fp4x2(.., __NV_E2M1, cudaRoundNearest)`: `.x` in
/// the low nibble, `.y` in the high one.
SGL_DEVICE uint32_t e2m1x2_code(fp32x2_t x) {
  return e2m1_code(x.x) | (e2m1_code(x.y) << 4);
}

/// \brief `float(__nv_fp8_e4m3(raw))` for `raw` in `[2^-9, 448]`: RNE onto the E4M3FN
/// grid, gfx950's `v_cvt_pk_fp8_f32` in hardware, software elsewhere (gfx942's fp8 is FNUZ).
SGL_DEVICE float e4m3_round_rn(float raw) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
  const auto packed = __builtin_amdgcn_cvt_pk_fp8_f32(raw, raw, 0, false);
  return __builtin_amdgcn_cvt_f32_fp8(packed, 0);
#else
  const int exponent = ilogbf(raw);
  const float step = exponent < -6 ? 0x1p-9f : ldexpf(1.0f, exponent - 3);
  return rintf(raw / step) * step;
#endif
}
#endif

/// \brief Round amax / 6 to a positive finite E4M3 scale, ties to even.
SGL_DEVICE float compressed_kv_scale(float amax) {
  const auto raw = fminf(fmaxf(amax * (1.0f / kMax), 0x1p-9f), 448.0f);
#ifdef USE_ROCM
  return e4m3_round_rn(raw);
#else
  return static_cast<float>(__nv_fp8_e4m3(raw));
#endif
}

/// \brief Quantize compressed KV with its E4M3 scale and return dequantized values.
SGL_DEVICE fp32x2_t fake_quant_compressed_kv_x2(fp32x2_t x, float scale) {
  const fp32x2_t scaled{__fdiv_rn(x.x, scale) + 0.0f, __fdiv_rn(x.y, scale) + 0.0f};
#ifdef USE_ROCM
  const fp32x2_t grid{e2m1_value(e2m1_code(scaled.x)), e2m1_value(e2m1_code(scaled.y))};
#else
  const auto code = __nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest);
  const auto grid = device::cast<fp32x2_t>(fp16x2_t{__nv_cvt_fp4x2_to_halfraw2(code, __NV_E2M1)});
#endif
  return {grid.x * scale, grid.y * scale};
}

/// \brief Per-block ue8m0 scale and its reciprocal, from the block's absmax.
SGL_DEVICE fp32x2_t block_scale(float amax) {
  const auto exponent = fp8::cast_to_ue8m0(fmaxf(amax, kAmaxFloor) * (1.0f / kMax));
  return {__uint_as_float(static_cast<uint32_t>(exponent) << 23), fp8::inv_scale_ue8m0(exponent)};
}

/// \brief Round a pair onto the e2m1 grid and back, through `scale`.
///
/// Every e2m1 value is exact in fp16, so the roundtrip is lossless. Adding `0.0f` during
/// scaling clears negative zero to match `torch.sign(0) == 0` in `torch_quant.round_fp4`.
SGL_DEVICE fp32x2_t fake_quant_x2(fp32x2_t x, float scale, float inv_scale) {
  const fp32x2_t scaled{__fmaf_rn(x.x, inv_scale, 0.0f), __fmaf_rn(x.y, inv_scale, 0.0f)};
#ifdef USE_ROCM
  const fp32x2_t grid{e2m1_value(e2m1_code(scaled.x)), e2m1_value(e2m1_code(scaled.y))};
#else
  const auto code = __nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest);
  const auto grid = device::cast<fp32x2_t>(fp16x2_t{__nv_cvt_fp4x2_to_halfraw2(code, __NV_E2M1)});
#endif
  return {grid.x * scale, grid.y * scale};
}

// The packed FP32 -> E2M1 conversion is an architecture-specific instruction.
// SM121 (GB10) is deliberately excluded: unlike SM120, it does not implement
// cvt.rn.satfinite.e2m1x2.f32.
#if !defined(USE_ROCM) && CUDA_VERSION >= 12080 && \
    ((SGL_CUDA_ARCH >= 1000 && SGL_CUDA_ARCH < 1200) || SGL_CUDA_ARCH == 1200)
#define SGL_DSV4_HAS_NATIVE_E2M1X2 1
#else
#define SGL_DSV4_HAS_NATIVE_E2M1X2 0
#endif

SGL_DEVICE uint8_t quant_e2m1(float x) {
  const float ax = fminf(fabsf(x), 6.0f);
  uint8_t idx = 0;
  idx += ax > 0.25f;
  idx += ax >= 0.75f;
  idx += ax > 1.25f;
  idx += ax >= 1.75f;
  idx += ax > 2.5f;
  idx += ax >= 3.5f;
  idx += ax > 5.0f;
  if (x < 0.0f && idx != 0) idx |= 0x8;
  return idx;
}

// Return two E2M1 values in the cache's native byte order: lo in bits [3:0]
// and hi in bits [7:4]. PTX packs its first source into the high nibble, hence
// the intentionally reversed source operands below.
SGL_DEVICE uint8_t quant_e2m1x2(float lo, float hi) {
#if SGL_DSV4_HAS_NATIVE_E2M1X2
  // Inline PTX has no 8-bit output constraint. Replicate the packed byte into
  // a 16-bit register and return its low byte, matching CUTLASS's conversion.
  uint16_t result;
  asm volatile(
      "{\n\t"
      ".reg .b8 fp4;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 fp4, %2, %1;\n\t"
      "mov.b16 %0, {fp4, fp4};\n\t"
      "}"
      : "=h"(result)
      : "f"(lo), "f"(hi));
  return static_cast<uint8_t>(result);
#else
  return quant_e2m1(lo) | (quant_e2m1(hi) << 4);
#endif
}


#undef SGL_DSV4_HAS_NATIVE_E2M1X2

}  // namespace deepseek_v4::fp4

}  // namespace sglang
