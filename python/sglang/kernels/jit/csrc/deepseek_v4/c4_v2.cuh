/**
 * \brief Here's some dimension info for the main buffer used in C4 prefill and decode.
 *
 * kv_buffer: [num_indices, 8, head_dim * 4]
 * - last dimension layout: | kv overlap | kv | score overlap | score |
 * kv_input: [batch_size, head_dim * 4]
 * kv_output: [batch_size, head_dim]
 * score_bias (ape): [8, head_dim]
 * plan_c/plan_w: [variable length]
 *
 * For prefill, batch_size = num_q_tokens
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/compress_v2.cuh>
#include <sgl_kernel/deepseek_v4/fp4_utils.cuh>
#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/object.h>

#include <bit>
#include <cfloat>
#include <cstdint>
#include <type_traits>

namespace sglang {

using PlanD = device::compress::DecodePlan;
using PlanC = device::compress::CompressPlan;
using PlanW = device::compress::WritePlan;

/// \brief Each thread will handle this many elements (split along head_dim)
constexpr int32_t kTileElements = 4;

/// \brief Need to improve register usage to reduce latency
#define C4_KERNEL __global__ __launch_bounds__(128, 4)
#define WRITE_KERNEL __global__ __launch_bounds__(128, 16)

struct Compress4DecodeParams {
  void* __restrict__ kv_buffer;
  const void* __restrict__ kv_input;
  void* __restrict__ kv_output;
  const void* __restrict__ score_bias;
  const PlanD* __restrict__ plan_d;
  uint32_t batch_size;
};

struct Compress4PrefillParams {
  void* __restrict__ kv_buffer;
  const void* __restrict__ kv_input;
  void* __restrict__ kv_output;
  const void* __restrict__ score_bias;
  const PlanC* __restrict__ plan_c;
  const PlanW* __restrict__ plan_w;
  uint32_t num_compress;
  uint32_t num_write;
};

struct Compress4Fp4StoreDecodeParams {
  void* __restrict__ kv_buffer;
  const void* __restrict__ kv_input;
  const void* __restrict__ score_bias;
  const PlanD* __restrict__ plan_d;
  const void* __restrict__ norm_weight;
  const float* __restrict__ freqs_cis;
  const int64_t* __restrict__ out_loc;
  uint8_t* __restrict__ kvcache;
  float norm_eps;
  uint32_t batch_size;
};

struct Compress4Fp4StorePrefillParams {
  void* __restrict__ kv_buffer;
  const void* __restrict__ kv_input;
  const void* __restrict__ score_bias;
  const PlanC* __restrict__ plan_c;
  const PlanW* __restrict__ plan_w;
  const void* __restrict__ norm_weight;
  const float* __restrict__ freqs_cis;
  const int64_t* __restrict__ out_loc;
  uint8_t* __restrict__ kvcache;
  float norm_eps;
  uint32_t num_compress;
  uint32_t num_write;
};

template <int64_t kHeadDim_>
struct C4Trait {
  static constexpr int64_t kTileDim = kTileElements * device::kWarpThreads;  // 128
  static constexpr int64_t kHeadDim = kHeadDim_;
  static constexpr int64_t kOverlapOffset = kHeadDim;
  static constexpr int64_t kScoreOffset = kHeadDim * 2;
  static constexpr int64_t kElementSize = kHeadDim * 4;
  static constexpr int64_t kPageElementSize = 4 * kElementSize;  // page size = 4
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static_assert(kHeadDim % kTileDim == 0);
};

SGL_DEVICE uint32_t c4_warp_broadcast_u32(uint32_t value) {
#ifndef USE_ROCM
  return __shfl_sync(device::warp::kFullMask, value, 0, device::kWarpThreads);
#else
  return __shfl(value, 0, device::kWarpThreads);
#endif
}

SGL_DEVICE PlanD c4_load_plan_d(const PlanD* plans, uint32_t plan_id) {
  PlanD plan{};
  if (threadIdx.x % device::kWarpThreads == 0) plan = plans[plan_id];
  plan.seq_len = c4_warp_broadcast_u32(plan.seq_len);
  plan.write_loc = static_cast<int32_t>(
      c4_warp_broadcast_u32(static_cast<uint32_t>(plan.write_loc)));
  plan.read_page_0 = static_cast<int32_t>(
      c4_warp_broadcast_u32(static_cast<uint32_t>(plan.read_page_0)));
  plan.read_page_1 = static_cast<int32_t>(
      c4_warp_broadcast_u32(static_cast<uint32_t>(plan.read_page_1)));
  return plan;
}

SGL_DEVICE PlanC c4_load_plan_c(const PlanC* plans, uint32_t plan_id) {
  PlanC plan{};
  if (threadIdx.x % device::kWarpThreads == 0) plan = plans[plan_id];
  plan.seq_len = c4_warp_broadcast_u32(plan.seq_len);
  uint32_t ragged_buffer = static_cast<uint32_t>(plan.ragged_id) |
                           (static_cast<uint32_t>(plan.buffer_len) << 16);
  ragged_buffer = c4_warp_broadcast_u32(ragged_buffer);
  plan.ragged_id = static_cast<uint16_t>(ragged_buffer);
  plan.buffer_len = static_cast<uint16_t>(ragged_buffer >> 16);
  plan.read_page_0 = static_cast<int32_t>(
      c4_warp_broadcast_u32(static_cast<uint32_t>(plan.read_page_0)));
  plan.read_page_1 = static_cast<int32_t>(
      c4_warp_broadcast_u32(static_cast<uint32_t>(plan.read_page_1)));
  return plan;
}

SGL_DEVICE PlanW c4_load_plan_w(const PlanW* plans, uint32_t plan_id) {
  PlanW plan{};
  if (threadIdx.x % device::kWarpThreads == 0) plan = plans[plan_id];
  plan.ragged_id = c4_warp_broadcast_u32(plan.ragged_id);
  plan.write_loc = static_cast<int32_t>(
      c4_warp_broadcast_u32(static_cast<uint32_t>(plan.write_loc)));
  return plan;
}

SGL_DEVICE int64_t c4_load_out_loc(const int64_t* out_loc, uint32_t index) {
  uint64_t bits = 0;
  if (threadIdx.x % device::kWarpThreads == 0) {
    bits = static_cast<uint64_t>(out_loc[index]);
  }
  const auto low = c4_warp_broadcast_u32(static_cast<uint32_t>(bits));
  const auto high = c4_warp_broadcast_u32(static_cast<uint32_t>(bits >> 32));
  return static_cast<int64_t>((static_cast<uint64_t>(high) << 32) | low);
}

template <typename RoundFloat, typename LoadFloat>
SGL_DEVICE auto c4_load_fp32(const LoadFloat* ptr)
    -> device::AlignedVector<float, kTileElements> {
  using namespace device;
  using StorageLoad = AlignedVector<LoadFloat, kTileElements>;
  using Float4 = AlignedVector<float, kTileElements>;
  const auto loaded = tile::Memory<StorageLoad>::warp().load(ptr);
  Float4 result;
#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    // Preserve the standalone C4 path's BufferFloat -> InputFloat rounding
    // before promoting scores to fp32.
    result[i] = cast<float>(cast<RoundFloat>(loaded[i]));
  }
  return result;
}

template <typename Trait, typename BufferFloat, typename InputFloat, typename OutFloat>
SGL_DEVICE auto c4_compute(
    const BufferFloat* kv_buf_0,  // overlap [4n - 4, 4n - 1]
    const BufferFloat* kv_buf_1,  // normal [4n + 0, 4n + 3]
    const InputFloat* kv_src,     // ragged pointer at position = 4n + 3
    const InputFloat* score_bias,
    const bool should_overlap,
    const int32_t buffer_len) -> device::AlignedVector<OutFloat, kTileElements> {
  using namespace device;

  using StorageIn = AlignedVector<InputFloat, kTileElements>;
  using Float4 = AlignedVector<float, kTileElements>;
  const auto gmem_in = tile::Memory<StorageIn>::warp();
  StorageIn kv[8];
  Float4 score[8];

  if constexpr (std::is_same_v<BufferFloat, InputFloat>) {
    if (should_overlap) {
      const auto kv_start = kv_src - 7 * Trait::kElementSize;  // point to start
#pragma unroll
      for (int32_t i = 0; i < 4; ++i) {
        const auto src = i < buffer_len ? kv_buf_0 : kv_start;
        const auto base = src + i * Trait::kElementSize;
        kv[i] = gmem_in.load(base);
        score[i] = c4_load_fp32<InputFloat>(base + Trait::kScoreOffset);
      }
    } else {
      [[unlikely]];
      constexpr float kFloatNegInf = -FLT_MAX;
#pragma unroll
      for (int32_t i = 0; i < 4; ++i) {
        kv[i].fill(cast<InputFloat>(0.0f));
        score[i].fill(cast<float>(cast<InputFloat>(kFloatNegInf)));
      }
    }

    const auto kv_start = kv_src - 3 * Trait::kElementSize;  // point to start
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
      const auto src = i + 4 < buffer_len ? kv_buf_1 : kv_start;
      const auto base = src + i * Trait::kElementSize + Trait::kOverlapOffset;
      kv[i + 4] = gmem_in.load(base);
      score[i + 4] = c4_load_fp32<InputFloat>(base + Trait::kScoreOffset);
    }
  } else {  // mixed dtype
    using StorageBuffer = AlignedVector<BufferFloat, kTileElements>;
    const auto gmem_buffer = tile::Memory<StorageBuffer>::warp();
    const auto kv_start_0 = kv_src - 7 * Trait::kElementSize;  // point to start

#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
      if (should_overlap && i < buffer_len) {
        const auto base = kv_buf_0 + i * Trait::kElementSize;
        const auto kv_tmp = gmem_buffer.load(base);
        score[i] = c4_load_fp32<InputFloat>(base + Trait::kScoreOffset);
#pragma unroll
        for (int32_t j = 0; j < kTileElements; ++j) {
          kv[i][j] = cast<InputFloat>(kv_tmp[j]);
        }
      } else if (should_overlap) {
        const auto base = kv_start_0 + i * Trait::kElementSize;
        kv[i] = gmem_in.load(base);
        score[i] = c4_load_fp32<InputFloat>(base + Trait::kScoreOffset);
      } else {
        [[unlikely]];
        constexpr float kFloatNegInf = -FLT_MAX;
        kv[i].fill(cast<InputFloat>(0.0f));
        score[i].fill(cast<float>(cast<InputFloat>(kFloatNegInf)));
      }
    }

    const auto kv_start = kv_src - 3 * Trait::kElementSize;  // point to start
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
      if (i + 4 < buffer_len) {
        const auto base = kv_buf_1 + i * Trait::kElementSize + Trait::kOverlapOffset;
        const auto kv_tmp = gmem_buffer.load(base);
        score[i + 4] = c4_load_fp32<InputFloat>(base + Trait::kScoreOffset);
#pragma unroll
        for (int32_t j = 0; j < kTileElements; ++j) {
          kv[i + 4][j] = cast<InputFloat>(kv_tmp[j]);
        }
      } else {
        const auto base = kv_start + i * Trait::kElementSize + Trait::kOverlapOffset;
        kv[i + 4] = gmem_in.load(base);
        score[i + 4] = c4_load_fp32<InputFloat>(base + Trait::kScoreOffset);
      }
    }
  }

  // Fold APE into the fp32 score registers one row at a time. The temporary
  // bias vector dies each iteration, avoiding eight live bias vectors and the
  // second 4x8 fp32 score array used by the previous implementation.
#pragma unroll
  for (int32_t j = 0; j < 8; ++j) {
    const auto bias = c4_load_fp32<InputFloat>(score_bias + j * Trait::kHeadDim);
#pragma unroll
    for (int32_t i = 0; i < kTileElements; ++i) {
      score[j][i] += bias[i];
    }
  }

  /// NOTE: part 2: safe online softmax + weighted sum
  using StorageOut = AlignedVector<OutFloat, kTileElements>;
  StorageOut result;

#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    float max_value = score[0][i];
    float sum_exp_value = 0.0f;

#pragma unroll
    for (int32_t j = 1; j < 8; ++j) {
      const auto fp32_score = score[j][i];
      max_value = fmaxf(max_value, fp32_score);
    }

    float sum_product = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      const auto fp32_score = score[j][i];
      const auto exp_score = __expf(fp32_score - max_value);
      sum_product += cast<float>(kv[j][i]) * exp_score;
      sum_exp_value += exp_score;
    }

    result[i] = cast<OutFloat>(sum_product / sum_exp_value);
  }

  return result;
}

template <typename Trait, bool kUsePDL, typename BufferFloat, typename InputFloat, typename OutFloat>
SGL_DEVICE void c4_forward(
    const BufferFloat* kv_buf_0,
    const BufferFloat* kv_buf_1,
    const InputFloat* kv_src,
    OutFloat* kv_out,
    const InputFloat* score_bias,
    const bool should_overlap,
    const int32_t buffer_len) {
  using namespace device;
  using StorageOut = AlignedVector<OutFloat, kTileElements>;
  const auto gmem_out = tile::Memory<StorageOut>::warp();
  const auto result = c4_compute<Trait, BufferFloat, InputFloat, OutFloat>(
      kv_buf_0, kv_buf_1, kv_src, score_bias, should_overlap, buffer_len);

  // overlap the store with the next iteration's load
  PDLTriggerSecondary<kUsePDL>();
  gmem_out.store(kv_out, result);
}

template <typename DType, uint32_t kPageSize, bool kUsePDL>
SGL_DEVICE void c4_norm_rope_hadamard_fp4_store(
    device::AlignedVector<DType, kTileElements> compressed,
    const DType* norm_weight,
    const float* freqs_cis,
    const int64_t out_loc,
    uint8_t* kvcache,
    const float norm_eps) {
  using namespace device;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr uint32_t kRopeLanes = kRopeDim / kTileElements;
  constexpr int64_t kPageBytes = 68ll * kPageSize;
  static_assert(kHeadDim == kWarpThreads * kTileElements);
  static_assert(std::has_single_bit(kPageSize));

  using Storage = AlignedVector<DType, kTileElements>;
  using Float4 = AlignedVector<float, kTileElements>;

  const auto lane_id = threadIdx.x % kWarpThreads;
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeLanes;
  Storage weight_vec;
  Float4 data, freq;
  weight_vec.load(norm_weight, lane_id);
  if (is_rope_lane) {
    freq.load(freqs_cis, lane_id - (kWarpThreads - kRopeLanes));
  }

  float sum_of_squares = 0.0f;
#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    const auto value = cast<float>(compressed[i]);
    sum_of_squares += value * value;
  }
  sum_of_squares = warp::reduce_sum(sum_of_squares);
  const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + norm_eps);

#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    data[i] = cast<float>(compressed[i]) * norm_factor * cast<float>(weight_vec[i]);
  }

  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    data[0] = x_real * freq[0] - x_imag * freq[1];
    data[1] = x_real * freq[1] + x_imag * freq[0];
    data[2] = y_real * freq[2] - y_imag * freq[3];
    data[3] = y_real * freq[3] + y_imag * freq[2];
  }

  {
    const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a1;
    data[1] = a0 - a1;
    data[2] = a2 + a3;
    data[3] = a2 - a3;
  }
  {
    const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a2;
    data[1] = a1 + a3;
    data[2] = a0 - a2;
    data[3] = a1 - a3;
  }
#pragma unroll
  for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
    for (int32_t i = 0; i < kTileElements; ++i) {
#ifndef USE_ROCM
      const float other = __shfl_xor_sync(warp::kFullMask, data[i], mask, kWarpThreads);
#else
      const float other = __shfl_xor(data[i], mask, kWarpThreads);
#endif
      data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
    }
  }
  const float hadamard_scale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    data[i] *= hadamard_scale;
  }

  float local_max = math::abs(data[0]);
#pragma unroll
  for (int32_t i = 1; i < kTileElements; ++i) {
    local_max = math::max(local_max, math::abs(data[i]));
  }
  local_max = warp::reduce_max<8>(local_max);
  const auto scale_raw = fmaxf(1e-4f, local_max) / 6.0f;
  const auto scale_ue8m0 =
      static_cast<uint8_t>(deepseek_v4::fp8::cast_to_ue8m0(scale_raw));
  const auto inv_scale = deepseek_v4::fp8::inv_scale_ue8m0(scale_ue8m0);

  const uint8_t packed0 =
      deepseek_v4::fp4::quant_e2m1x2(data[0] * inv_scale, data[1] * inv_scale);
  const uint8_t packed1 =
      deepseek_v4::fp4::quant_e2m1x2(data[2] * inv_scale, data[3] * inv_scale);
  const uint16_t packed =
      static_cast<uint16_t>(packed0) | (static_cast<uint16_t>(packed1) << 8);

  const int64_t page = out_loc / kPageSize;
  const int64_t offset = out_loc % kPageSize;
  const auto page_ptr = kvcache + page * kPageBytes;
  const auto value_ptr = page_ptr + offset * 64;
  const auto scale_ptr = page_ptr + 64 * kPageSize + offset * 4;

  PDLTriggerSecondary<kUsePDL>();
  reinterpret_cast<uint16_t*>(value_ptr)[lane_id] = packed;
  if ((lane_id & 7) == 0) {
    scale_ptr[lane_id >> 3] = scale_ue8m0;
  }
}

template <typename Trait, typename BufferFloat, typename InputFloat>
SGL_DEVICE void c4_write_decode(BufferFloat* kv_buf, const InputFloat* kv_src) {
  using namespace device;

  using StorageInput = AlignedVector<InputFloat, kTileElements>;
  const auto gmem_input = tile::Memory<StorageInput>::warp();

  StorageInput data[4];
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    data[i] = gmem_input.load(kv_src + Trait::kHeadDim * i);
  }

  if constexpr (std::is_same_v<BufferFloat, InputFloat>) {
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
      gmem_input.store(kv_buf + Trait::kHeadDim * i, data[i]);
    }
  } else {
    using StorageBuffer = AlignedVector<BufferFloat, kTileElements>;
    const auto gmem_buffer = tile::Memory<StorageBuffer>::warp();

    StorageBuffer data_cast[4];
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
      for (int32_t j = 0; j < kTileElements; ++j) {
        data_cast[i][j] = cast<BufferFloat>(data[i][j]);
      }
      gmem_buffer.store(kv_buf + Trait::kHeadDim * i, data_cast[i]);
    }
  }
}

template <int64_t kHeadDim, typename BufferFloat, typename InputFloat, typename OutFloat, bool kUsePDL>
C4_KERNEL void flash_c4_decode(const __grid_constant__ Compress4DecodeParams params) {
  using namespace device;
  using Trait = C4Trait<kHeadDim>;

  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_wid = global_tid / kWarpThreads;      // warp id
  const uint32_t global_bid = global_wid / Trait::kNumSplit;  // batch id
  const uint32_t global_sid = global_wid % Trait::kNumSplit;  // split id
  const int64_t split_offset = global_sid * Trait::kTileDim;
  if (global_bid >= params.batch_size) return;

  const auto plan = c4_load_plan_d(params.plan_d, global_bid);
  const auto kv_input = static_cast<const InputFloat*>(params.kv_input) + split_offset;
  const auto kv_output = static_cast<OutFloat*>(params.kv_output) + split_offset;
  const auto kv_buffer = static_cast<BufferFloat*>(params.kv_buffer) + split_offset;
  const auto score_bias = static_cast<const InputFloat*>(params.score_bias) + split_offset;

  const auto kv_src = kv_input + global_bid * Trait::kElementSize;
  const auto kv_out = kv_output + global_bid * Trait::kHeadDim;
  const auto kv_buf_0 = kv_buffer + plan.read_page_0 * Trait::kPageElementSize;
  const auto kv_buf_1 = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  const auto kv_dst = kv_buffer + plan.write_loc * Trait::kElementSize;

  PDLWaitPrimary<kUsePDL>();
  c4_write_decode<Trait, BufferFloat, InputFloat>(kv_dst, kv_src);
  if (plan.seq_len % 4 == 0) {
    const auto need_overlap = plan.seq_len > 4;
    c4_forward<Trait, kUsePDL, BufferFloat, InputFloat, OutFloat>(
        kv_buf_0, kv_buf_1, kv_src, kv_out, score_bias, need_overlap, 8);
  }
}

template <int64_t kHeadDim, typename BufferFloat, typename InputFloat, typename OutFloat, bool kUsePDL>
C4_KERNEL void flash_c4_prefill(const __grid_constant__ Compress4PrefillParams params) {
  using namespace device;
  using Trait = C4Trait<kHeadDim>;

  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_wid = global_tid / kWarpThreads;      // warp id
  const uint32_t global_pid = global_wid / Trait::kNumSplit;  // plan id
  const uint32_t global_sid = global_wid % Trait::kNumSplit;  // split id
  const int64_t split_offset = global_sid * Trait::kTileDim;
  if (global_pid >= params.num_compress) return;

  const auto plan = c4_load_plan_c(params.plan_c, global_pid);
  const auto kv_input = static_cast<const InputFloat*>(params.kv_input) + split_offset;
  const auto kv_output = static_cast<OutFloat*>(params.kv_output) + split_offset;
  const auto kv_buffer = static_cast<BufferFloat*>(params.kv_buffer) + split_offset;
  const auto score_bias = static_cast<const InputFloat*>(params.score_bias) + split_offset;
  if (plan.is_invalid()) return;

  const auto kv_src = kv_input + plan.ragged_id * Trait::kElementSize;
  // Compact output: one row per compress plan, indexed by `global_pid`.
  const auto kv_out = kv_output + global_pid * Trait::kHeadDim;
  const auto kv_buf_0 = kv_buffer + plan.read_page_0 * Trait::kPageElementSize;
  const auto kv_buf_1 = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  const bool need_overlap = plan.seq_len > 4;
  PDLWaitPrimary<kUsePDL>();
  c4_forward<Trait, kUsePDL, BufferFloat, InputFloat, OutFloat>(
      kv_buf_0, kv_buf_1, kv_src, kv_out, score_bias, need_overlap, plan.buffer_len);
}

template <int64_t kHeadDim, typename BufferFloat, typename InputFloat, typename OutFloat, bool kUsePDL>
WRITE_KERNEL void write_c4_prefill(const __grid_constant__ Compress4PrefillParams params) {
  using namespace device;
  using Trait = C4Trait<kHeadDim>;
  using StorageInput = AlignedVector<InputFloat, kTileElements>;

  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_wid = global_tid / kWarpThreads;      // warp id
  const uint32_t global_pid = global_wid / Trait::kNumSplit;  // plan id
  const uint32_t global_sid = global_wid % Trait::kNumSplit;  // split id
  // split the contiguous `kHeadDim * 4` into `kNumSplit` tiles
  // each warp handles 1 contiguous tile (in contrast, decode handle the strided head_dim)
  const int64_t split_offset = global_sid * (Trait::kTileDim * 4);
  if (global_pid >= params.num_write) return;

  const auto plan = c4_load_plan_w(params.plan_w, global_pid);
  const auto kv_input = static_cast<const InputFloat*>(params.kv_input) + split_offset;
  const auto kv_buffer = static_cast<BufferFloat*>(params.kv_buffer) + split_offset;
  if (plan.is_invalid()) return;

  // each warp will handle a contiguous region
  const auto kv_src = kv_input + plan.ragged_id * Trait::kElementSize;
  const auto kv_buf = kv_buffer + plan.write_loc * Trait::kElementSize;
  const auto gmem_input = tile::Memory<StorageInput>::warp();

  PDLWaitPrimary<kUsePDL>();
  StorageInput data[4];
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    data[i] = gmem_input.load(kv_src, i);
  }

  if constexpr (std::is_same_v<BufferFloat, InputFloat>) {
    PDLTriggerSecondary<kUsePDL>();
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
      gmem_input.store(kv_buf, data[i], i);
    }
  } else {
    using StorageBuffer = AlignedVector<BufferFloat, kTileElements>;
    const auto gmem_buffer = tile::Memory<StorageBuffer>::warp();

    StorageBuffer data_cast[4];
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
      for (int32_t j = 0; j < kTileElements; ++j) {
        data_cast[i][j] = cast<BufferFloat>(data[i][j]);
      }
    }
    PDLTriggerSecondary<kUsePDL>();
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
      gmem_buffer.store(kv_buf, data_cast[i], i);
    }
  }
}

template <
    int64_t kHeadDim,
    typename BufferFloat,
    typename InputFloat,
    uint32_t kPageSize,
    bool kUsePDL>
C4_KERNEL void flash_c4_norm_rope_fp4_store_decode(
    const __grid_constant__ Compress4Fp4StoreDecodeParams params) {
  using namespace device;
  using Trait = C4Trait<kHeadDim>;
  static_assert(kHeadDim == 128 && Trait::kNumSplit == 1);

  const uint32_t global_wid =
      (blockIdx.x * blockDim.x + threadIdx.x) / kWarpThreads;
  if (global_wid >= params.batch_size) return;

  const auto plan = c4_load_plan_d(params.plan_d, global_wid);
  const auto out_loc = c4_load_out_loc(params.out_loc, global_wid);
  const auto kv_input = static_cast<const InputFloat*>(params.kv_input);
  const auto kv_buffer = static_cast<BufferFloat*>(params.kv_buffer);
  const auto score_bias = static_cast<const InputFloat*>(params.score_bias);
  const auto kv_src = kv_input + global_wid * Trait::kElementSize;
  const auto kv_buf_0 = kv_buffer + plan.read_page_0 * Trait::kPageElementSize;
  const auto kv_buf_1 = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  const auto kv_dst = kv_buffer + plan.write_loc * Trait::kElementSize;

  PDLWaitPrimary<kUsePDL>();
  c4_write_decode<Trait, BufferFloat, InputFloat>(kv_dst, kv_src);
  if (plan.seq_len % 4 != 0) return;

  const auto compressed = c4_compute<Trait, BufferFloat, InputFloat, InputFloat>(
      kv_buf_0, kv_buf_1, kv_src, score_bias, plan.seq_len > 4, 8);
  const auto freqs_cis = params.freqs_cis + (plan.seq_len - 4) * 64;
  c4_norm_rope_hadamard_fp4_store<InputFloat, kPageSize, kUsePDL>(
      compressed,
      static_cast<const InputFloat*>(params.norm_weight),
      freqs_cis,
      out_loc,
      params.kvcache,
      params.norm_eps);
}

template <
    int64_t kHeadDim,
    typename BufferFloat,
    typename InputFloat,
    uint32_t kPageSize,
    bool kUsePDL>
C4_KERNEL void flash_c4_norm_rope_fp4_store_prefill(
    const __grid_constant__ Compress4Fp4StorePrefillParams params) {
  using namespace device;
  using Trait = C4Trait<kHeadDim>;
  static_assert(kHeadDim == 128 && Trait::kNumSplit == 1);

  const uint32_t global_wid =
      (blockIdx.x * blockDim.x + threadIdx.x) / kWarpThreads;
  if (global_wid >= params.num_compress) return;

  const auto plan = c4_load_plan_c(params.plan_c, global_wid);
  if (plan.is_invalid()) return;
  const auto out_loc = c4_load_out_loc(params.out_loc, plan.ragged_id);
  const auto kv_input = static_cast<const InputFloat*>(params.kv_input);
  const auto kv_buffer = static_cast<BufferFloat*>(params.kv_buffer);
  const auto score_bias = static_cast<const InputFloat*>(params.score_bias);
  const auto kv_src = kv_input + plan.ragged_id * Trait::kElementSize;
  const auto kv_buf_0 = kv_buffer + plan.read_page_0 * Trait::kPageElementSize;
  const auto kv_buf_1 = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;

  PDLWaitPrimary<kUsePDL>();
  const auto compressed = c4_compute<Trait, BufferFloat, InputFloat, InputFloat>(
      kv_buf_0,
      kv_buf_1,
      kv_src,
      score_bias,
      plan.seq_len > 4,
      plan.buffer_len);
  const auto freqs_cis = params.freqs_cis + (plan.seq_len - 4) * 64;
  c4_norm_rope_hadamard_fp4_store<InputFloat, kPageSize, kUsePDL>(
      compressed,
      static_cast<const InputFloat*>(params.norm_weight),
      freqs_cis,
      out_loc,
      params.kvcache,
      params.norm_eps);
}

template <int64_t kHeadDim, typename BufferFloat, typename InputFloat, typename OutFloat, bool kUsePDL>
struct FlashCompress4Kernel {
  static constexpr auto decode_kernel = flash_c4_decode<kHeadDim, BufferFloat, InputFloat, OutFloat, kUsePDL>;
  static constexpr auto prefill_c_kernel = flash_c4_prefill<kHeadDim, BufferFloat, InputFloat, OutFloat, kUsePDL>;
  static constexpr auto prefill_w_kernel = write_c4_prefill<kHeadDim, BufferFloat, InputFloat, OutFloat, kUsePDL>;
  static constexpr uint32_t kBlockSize = 128;
  static constexpr uint32_t kTileDim = kTileElements * device::kWarpThreads;
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static constexpr uint32_t kWarpsPerBlock = kBlockSize / device::kWarpThreads;
  using Trait = C4Trait<kHeadDim>;

  static void run_decode(
      const tvm::ffi::TensorView kv_buffer,
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView kv_output,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView plan_d_) {
    using namespace host;

    auto N = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({-1, 4, Trait::kElementSize})  // kv score
        .with_dtype<BufferFloat>()
        .with_device(device_)
        .verify(kv_buffer);
    TensorMatcher({N, Trait::kElementSize})  // kv score input
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(kv_input);
    TensorMatcher({N, kHeadDim})  // kv compressed output
        .with_dtype<OutFloat>()
        .with_device(device_)
        .verify(kv_output);
    TensorMatcher({8, kHeadDim})  // ape
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(ape);

    const auto plan_d = compress::verify_plan_d(plan_d_, N, device_);
    const auto batch_size = static_cast<uint32_t>(N.unwrap());
    const auto params = Compress4DecodeParams{
        .kv_buffer = kv_buffer.data_ptr(),
        .kv_input = kv_input.data_ptr(),
        .kv_output = kv_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .plan_d = plan_d,
        .batch_size = batch_size,
    };
    const uint32_t num_blocks = div_ceil(batch_size * kNumSplit, kWarpsPerBlock);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(decode_kernel, params);
  }

  static void run_prefill(
      const tvm::ffi::TensorView kv_buffer,
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView kv_output,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView plan_c_,
      const tvm::ffi::TensorView plan_w_) {
    using namespace host;

    auto N = SymbolicSize{"num_q_tokens"};
    auto C = SymbolicSize{"num_c_plans"};
    auto W = SymbolicSize{"num_w_plans"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({-1, 4, Trait::kElementSize})  // kv score
        .with_dtype<BufferFloat>()
        .with_device(device_)
        .verify(kv_buffer);
    TensorMatcher({N, Trait::kElementSize})  // kv score input (ragged)
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(kv_input);
    TensorMatcher({C, kHeadDim})  // kv compressed output (compact)
        .with_dtype<OutFloat>()
        .with_device(device_)
        .verify(kv_output);
    TensorMatcher({8, kHeadDim})  // ape
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(ape);
    const auto plan_c = compress::verify_plan_c(plan_c_, C, device_);
    const auto plan_w = compress::verify_plan_w(plan_w_, W, device_);
    const auto device = device_.unwrap();
    const auto num_q_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_c = static_cast<uint32_t>(C.unwrap());
    const auto num_w = static_cast<uint32_t>(W.unwrap());
    const auto params = Compress4PrefillParams{
        .kv_buffer = kv_buffer.data_ptr(),
        .kv_input = kv_input.data_ptr(),
        .kv_output = kv_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .plan_c = plan_c,
        .plan_w = plan_w,
        .num_compress = num_c,
        .num_write = num_w,
    };
    RuntimeCheck(num_q_tokens >= num_w, "invalid prefill plan: num_q < num_w");
    if (const auto num_c_blocks = div_ceil(num_c * kNumSplit, kWarpsPerBlock)) {
      LaunchKernel(num_c_blocks, kBlockSize, device)  //
          .enable_pdl(kUsePDL)(prefill_c_kernel, params);
    }
    if (const auto num_w_blocks = div_ceil(num_w * kNumSplit, kWarpsPerBlock)) {
      LaunchKernel(num_w_blocks, kBlockSize, device)  //
          .enable_pdl(kUsePDL)(prefill_w_kernel, params);
    }
  }
};

template <
    int64_t kHeadDim,
    typename BufferFloat,
    typename InputFloat,
    uint32_t kPageSize,
    bool kUsePDL>
struct FlashCompress4Fp4StoreKernel {
  static_assert(kHeadDim == 128, "fused C4 FP4 store is indexer-only");
  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  using Trait = C4Trait<kHeadDim>;
  static constexpr uint32_t kBlockSize = 128;
  static constexpr uint32_t kWarpsPerBlock =
      kBlockSize / device::kWarpThreads;
  static constexpr int64_t kFp4PageBytes = 68ll * kPageSize;
  static constexpr auto decode_kernel =
      flash_c4_norm_rope_fp4_store_decode<
          kHeadDim,
          BufferFloat,
          InputFloat,
          kPageSize,
          kUsePDL>;
  static constexpr auto prefill_c_kernel =
      flash_c4_norm_rope_fp4_store_prefill<
          kHeadDim,
          BufferFloat,
          InputFloat,
          kPageSize,
          kUsePDL>;
  static constexpr auto prefill_w_kernel =
      write_c4_prefill<
          kHeadDim,
          BufferFloat,
          InputFloat,
          InputFloat,
          kUsePDL>;

  static void run_decode(
      const tvm::ffi::TensorView kv_buffer,
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView plan_d_,
      const tvm::ffi::TensorView norm_weight,
      const float norm_eps,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache) {
    using namespace host;

    auto N = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({-1, 4, Trait::kElementSize})
        .with_dtype<BufferFloat>()
        .with_device(device_)
        .verify(kv_buffer);
    TensorMatcher({N, Trait::kElementSize})
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(kv_input);
    TensorMatcher({8, kHeadDim})
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(ape);
    TensorMatcher({kHeadDim})
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(norm_weight);
    TensorMatcher({-1, 64})
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    TensorMatcher({N})
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(out_loc);
    TensorMatcher({-1, -1})
        .with_strides({kFp4PageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(kvcache);

    const auto plan_d = compress::verify_plan_d(plan_d_, N, device_);
    const auto batch_size = static_cast<uint32_t>(N.unwrap());
    if (batch_size == 0) return;
    const auto params = Compress4Fp4StoreDecodeParams{
        .kv_buffer = kv_buffer.data_ptr(),
        .kv_input = kv_input.data_ptr(),
        .score_bias = ape.data_ptr(),
        .plan_d = plan_d,
        .norm_weight = norm_weight.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .out_loc = static_cast<const int64_t*>(out_loc.data_ptr()),
        .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
        .norm_eps = norm_eps,
        .batch_size = batch_size,
    };
    const uint32_t num_blocks = div_ceil(batch_size, kWarpsPerBlock);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap())
        .enable_pdl(kUsePDL)(decode_kernel, params);
  }

  static void run_prefill(
      const tvm::ffi::TensorView kv_buffer,
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView plan_c_,
      const tvm::ffi::TensorView plan_w_,
      const tvm::ffi::TensorView norm_weight,
      const float norm_eps,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache) {
    using namespace host;

    auto N = SymbolicSize{"num_q_tokens"};
    auto C = SymbolicSize{"num_c_plans"};
    auto W = SymbolicSize{"num_w_plans"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({-1, 4, Trait::kElementSize})
        .with_dtype<BufferFloat>()
        .with_device(device_)
        .verify(kv_buffer);
    TensorMatcher({N, Trait::kElementSize})
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(kv_input);
    TensorMatcher({8, kHeadDim})
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(ape);
    TensorMatcher({kHeadDim})
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(norm_weight);
    TensorMatcher({-1, 64})
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    TensorMatcher({N})
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(out_loc);
    TensorMatcher({-1, -1})
        .with_strides({kFp4PageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(kvcache);

    const auto plan_c = compress::verify_plan_c(plan_c_, C, device_);
    const auto plan_w = compress::verify_plan_w(plan_w_, W, device_);
    const auto num_q_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_c = static_cast<uint32_t>(C.unwrap());
    const auto num_w = static_cast<uint32_t>(W.unwrap());
    RuntimeCheck(num_q_tokens >= num_w, "invalid prefill plan: num_q < num_w");
    const auto device = device_.unwrap();
    const auto params = Compress4Fp4StorePrefillParams{
        .kv_buffer = kv_buffer.data_ptr(),
        .kv_input = kv_input.data_ptr(),
        .score_bias = ape.data_ptr(),
        .plan_c = plan_c,
        .plan_w = plan_w,
        .norm_weight = norm_weight.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .out_loc = static_cast<const int64_t*>(out_loc.data_ptr()),
        .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
        .norm_eps = norm_eps,
        .num_compress = num_c,
        .num_write = num_w,
    };
    if (const auto num_c_blocks = div_ceil(num_c, kWarpsPerBlock)) {
      LaunchKernel(num_c_blocks, kBlockSize, device)
          .enable_pdl(kUsePDL)(prefill_c_kernel, params);
    }
    if (const auto num_w_blocks = div_ceil(num_w, kWarpsPerBlock)) {
      const auto write_params = Compress4PrefillParams{
          .kv_buffer = kv_buffer.data_ptr(),
          .kv_input = kv_input.data_ptr(),
          .kv_output = nullptr,
          .score_bias = ape.data_ptr(),
          .plan_c = plan_c,
          .plan_w = plan_w,
          .num_compress = num_c,
          .num_write = num_w,
      };
      LaunchKernel(num_w_blocks, kBlockSize, device)
          .enable_pdl(kUsePDL)(prefill_w_kernel, write_params);
    }
  }
};

}  // namespace sglang
