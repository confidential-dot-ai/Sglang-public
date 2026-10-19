<!-- Preserved from /data/dsv41-bringup/profile-dspark-b64-20260928-143420/kernel-breakdown.md on 2026-09-28. -->

# Individual CUDA kernel breakdown

40 DSpark iterations at batch 64. Values are averaged across TP0/GPU2 and TP1/GPU3. Each row is one exact kernel symbol; short display names below only abbreviate template arguments. No distinct kernel symbols are merged.

One iteration includes target verification and draft generation. Per-iteration GPU time is the sum of kernel durations, not elapsed wall time; concurrent GPU streams may overlap. The two ranks run concurrently and are averaged, not added.

There are 152 distinct symbols. Summed kernel time averages 41.257 ms per GPU per iteration. The observed iteration interval was about 63.5 ms. The top 20 kernels account for 90.58% of summed kernel time.

| # | Kernel (short name) | Calls/iteration/GPU | Mean µs/call | ms/iteration/GPU | Kernel time % |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `FP4 expert GEMM + SwiGLU: bmm_MxE4m3…t128x16x256u2` | 40 | 278.92 | 11.157 | 27.04% |
| 2 | `FP4 expert GEMM: bmm_Bfloat16…t128x16x256` | 43 | 135.42 | 5.823 | 14.11% |
| 3 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 88 | 57.59 | 5.068 | 12.28% |
| 4 | `_hc_mix_stats_partial_kernel` | 86 | 36.11 | 3.105 | 7.53% |
| 5 | `flash_fwd_splitkv_mla_fp8_sparse_kernel` | 43 | 66.14 | 2.844 | 6.89% |
| 6 | `Sm100BlockScaledPersistentDenseGemmKernel` | 230 | 11.23 | 2.582 | 6.26% |
| 7 | `_hc_mix_reduce_sinkhorn_kernel` | 86 | 12.09 | 1.039 | 2.52% |
| 8 | `at::native::direct_copy_kernel_cuda (BF16, vectorized)` | 99 | 10.01 | 0.991 | 2.40% |
| 9 | `nvjet_sm103_tst_128x96_64x9_2x2_2cta_h_bz_TNT` | 40 | 18.06 | 0.722 | 1.75% |
| 10 | `RMSNormKernel (5120-wide specialization)` | 89 | 6.93 | 0.617 | 1.50% |
| 11 | `_mhc_post_split_h_kernel` | 86 | 6.29 | 0.541 | 1.31% |
| 12 | `moe::dev::finalize::finalizeKernelVecLoad` | 40 | 12.15 | 0.486 | 1.18% |
| 13 | `ncclDevKernel_AllGather_RING_LL` | 6 | 77.61 | 0.466 | 1.13% |
| 14 | `flash_fwd_mla_combine_kernel` | 43 | 8.67 | 0.373 | 0.90% |
| 15 | `MXFP8QuantizeSwizzledKernel (5120-wide specialization)` | 90 | 4.04 | 0.364 | 0.88% |
| 16 | `nvjet_sm103_tss_64x32_64x16_2x1_2cta_v_bz_splitK_TNT` | 40 | 6.32 | 0.253 | 0.61% |
| 17 | `FP4 expert GEMM + SwiGLU: bmm_MxE4m3…t128x16x256 (no u2)` | 3 | 82.45 | 0.247 | 0.60% |
| 18 | `_router_triton_kernel` | 43 | 5.46 | 0.235 | 0.57% |
| 19 | `_hc_combine_kernel` | 86 | 2.71 | 0.233 | 0.56% |
| 20 | `get_mla_metadata_kernel` | 4 | 56.21 | 0.225 | 0.54% |

## Complete symbol list

The CSV and JSON files contain the full exact symbols, call counts on each rank, rank-specific time per iteration, and call-duration mean/median/p95. These include all 152 symbols, including the long tail below the top 20.

### 1. FP4 expert GEMM + SwiGLU: bmm_MxE4m3…t128x16x256u2

TP0: 10.974893 ms/iteration (1600 calls); TP1: 11.338343 ms/iteration (1600 calls). Mean: 278.915 µs/call; median: 278.690; p95: 310.466. Share: 27.0421%.

```text
bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x16x256u2_s6_et128x16_m256x16x32_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_tma_ldgstsSf_rgTma_clmp_swiGlu_dynB_sm100f
```

### 2. FP4 expert GEMM: bmm_Bfloat16…t128x16x256

TP0: 5.737038 ms/iteration (1720 calls); TP1: 5.909498 ms/iteration (1720 calls). Mean: 135.425 µs/call; median: 141.441; p95: 158.209. Share: 14.1148%.

```text
bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x16x256_s6_et128x16_m256x16x32_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f
```

### 3. ncclDevKernel_AllReduce_Sum_bf16_RING_LL

TP0: 5.481644 ms/iteration (3520 calls); TP1: 4.654045 ms/iteration (3520 calls). Mean: 57.589 µs/call; median: 43.200; p95: 93.121. Share: 12.2837%.

```text
ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)
```

### 4. _hc_mix_stats_partial_kernel

TP0: 3.108079 ms/iteration (3440 calls); TP1: 3.102789 ms/iteration (3440 calls). Mean: 36.110 µs/call; median: 36.448; p95: 37.217. Share: 7.5271%.

```text
_hc_mix_stats_partial_kernel
```

### 5. flash_fwd_splitkv_mla_fp8_sparse_kernel

TP0: 2.845321 ms/iteration (1720 calls); TP1: 2.843078 ms/iteration (1720 calls). Mean: 66.144 µs/call; median: 70.560; p95: 71.553. Share: 6.8939%.

```text
void sm100::decode::head64::flash_fwd_splitkv_mla_fp8_sparse_kernel<sm100::decode::head64::KernelTemplate<(ModelType)1>, sm100::decode::head64::KernelTemplate<(ModelType)1>::TmaParams<cute::tuple<int, int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > >, cute::tuple<int, int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_STORE, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<1> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<0> > > > >, cute::tuple<cute::C<64>, cute::C<64> > > > >(SparseAttnDecodeParams, sm100::decode::head64::KernelTemplate<(ModelType)1>::TmaParams<cute::tuple<int, int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > >, cute::tuple<int, int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_STORE, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<1> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<0> > > > >, cute::tuple<cute::C<64>, cute::C<64> > > >)
```

### 6. Sm100BlockScaledPersistentDenseGemmKernel

TP0: 2.589347 ms/iteration (9200 calls); TP1: 2.575068 ms/iteration (9200 calls). Mean: 11.227 µs/call; median: 11.840; p95: 12.832. Share: 6.2589%.

```text
kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDenseGemmKernel_object_at__TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____MMAAtom_ThrID1_0
```

### 7. _hc_mix_reduce_sinkhorn_kernel

TP0: 1.042469 ms/iteration (3440 calls); TP1: 1.036199 ms/iteration (3440 calls). Mean: 12.085 µs/call; median: 12.544; p95: 13.248. Share: 2.5192%.

```text
_hc_mix_reduce_sinkhorn_kernel
```

### 8. at::native::direct_copy_kernel_cuda (BF16, vectorized)

TP0: 0.994854 ms/iteration (3960 calls); TP1: 0.988109 ms/iteration (3960 calls). Mean: 10.015 µs/call; median: 10.544; p95: 13.344. Share: 2.4032%.

```text
void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})
```

### 9. nvjet_sm103_tst_128x96_64x9_2x2_2cta_h_bz_TNT

TP0: 0.718298 ms/iteration (1600 calls); TP1: 0.726647 ms/iteration (1600 calls). Mean: 18.062 µs/call; median: 18.144; p95: 18.816. Share: 1.7512%.

```text
nvjet_sm103_tst_128x96_64x9_2x2_2cta_h_bz_TNT
```

### 10. RMSNormKernel (5120-wide specialization)

TP0: 0.614290 ms/iteration (3560 calls); TP1: 0.619461 ms/iteration (3560 calls). Mean: 6.931 µs/call; median: 6.816; p95: 9.824. Share: 1.4952%.

```text
kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64512051201_tensorptrbf16gmemalign16o51201_tensorptrbf16gmemalign128oi64512051201___T_0
```

### 11. _mhc_post_split_h_kernel

TP0: 0.541445 ms/iteration (3440 calls); TP1: 0.540475 ms/iteration (3440 calls). Mean: 6.290 µs/call; median: 6.304; p95: 6.656. Share: 1.3112%.

```text
_mhc_post_split_h_kernel
```

### 12. moe::dev::finalize::finalizeKernelVecLoad

TP0: 0.480196 ms/iteration (1600 calls); TP1: 0.491986 ms/iteration (1600 calls). Mean: 12.152 µs/call; median: 12.128; p95: 13.440. Share: 1.1782%.

```text
void moe::dev::finalize::finalizeKernelVecLoad<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cutlass::bfloat16_t, 2, true> >(moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cutlass::bfloat16_t, 2, true>)
```

### 13. ncclDevKernel_AllGather_RING_LL

TP0: 0.468325 ms/iteration (240 calls); TP1: 0.463008 ms/iteration (240 calls). Mean: 77.611 µs/call; median: 63.361; p95: 150.241. Share: 1.1287%.

```text
ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)
```

### 14. flash_fwd_mla_combine_kernel

TP0: 0.373703 ms/iteration (1720 calls); TP1: 0.372202 ms/iteration (1720 calls). Mean: 8.673 µs/call; median: 9.024; p95: 9.472. Share: 0.9040%.

```text
void smxx::decode::flash_fwd_mla_combine_kernel<cutlass::bfloat16_t, 512, 8, 160, 256>(CombineParams)
```

### 15. MXFP8QuantizeSwizzledKernel (5120-wide specialization)

TP0: 0.362231 ms/iteration (3600 calls); TP1: 0.365729 ms/iteration (3600 calls). Mean: 4.044 µs/call; median: 3.328; p95: 9.024. Share: 0.8822%.

```text
kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o512051201_tensorptri8gmemalign16o512051201_tensorptri8gmem_0
```

### 16. nvjet_sm103_tss_64x32_64x16_2x1_2cta_v_bz_splitK_TNT

TP0: 0.253363 ms/iteration (1600 calls); TP1: 0.252480 ms/iteration (1600 calls). Mean: 6.323 µs/call; median: 6.304; p95: 6.656. Share: 0.6130%.

```text
nvjet_sm103_tss_64x32_64x16_2x1_2cta_v_bz_splitK_TNT
```

### 17. FP4 expert GEMM + SwiGLU: bmm_MxE4m3…t128x16x256 (no u2)

TP0: 0.260692 ms/iteration (120 calls); TP1: 0.233991 ms/iteration (120 calls). Mean: 82.447 µs/call; median: 76.833; p95: 116.352. Share: 0.5995%.

```text
bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x16x256_s6_et128x16_m256x16x32_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_tma_ldgstsSf_rgTma_clmp_swiGlu_dynB_sm100f
```

### 18. _router_triton_kernel

TP0: 0.234573 ms/iteration (1720 calls); TP1: 0.234937 ms/iteration (1720 calls). Mean: 5.459 µs/call; median: 5.568; p95: 5.792. Share: 0.5690%.

```text
_router_triton_kernel
```

### 19. _hc_combine_kernel

TP0: 0.232750 ms/iteration (3440 calls); TP1: 0.232873 ms/iteration (3440 calls). Mean: 2.707 µs/call; median: 2.687; p95: 2.913. Share: 0.5643%.

```text
_hc_combine_kernel
```

### 20. get_mla_metadata_kernel

TP0: 0.224967 ms/iteration (160 calls); TP1: 0.224720 ms/iteration (160 calls). Mean: 56.211 µs/call; median: 57.921; p95: 59.872. Share: 0.5450%.

```text
smxx::decode::get_mla_metadata_kernel(GetDecodeSchedMetaParams)
```

### 21. void moe::dev::routing::routingCustom::routingIndicesClusterKernel<moe::dev::routing::routingCustom:

TP0: 0.212992 ms/iteration (1600 calls); TP1: 0.213815 ms/iteration (1600 calls). Mean: 5.335 µs/call; median: 5.313; p95: 5.633. Share: 0.5173%.

```text
void moe::dev::routing::routingCustom::routingIndicesClusterKernel<moe::dev::routing::routingCustom::KernelParams<__nv_bfloat16, __nv_bfloat16, 512, 8, moe::dev::routing::TopKExpertSelect<moe::dev::routing::NoOpPreprocess, moe::dev::routing::SoftmaxPostprocess> >, 1024>(moe::dev::routing::routingCustom::KernelParams<__nv_bfloat16, __nv_bfloat16, 512, 8, moe::dev::routing::TopKExpertSelect<moe::dev::routing::NoOpPreprocess, moe::dev::routing::SoftmaxPostprocess> >)
```

### 22. void sglang::deepseek_rope_kernel<true, false, long>(sglang::FusedQKRopeParams)

TP0: 0.201687 ms/iteration (1720 calls); TP1: 0.201590 ms/iteration (1720 calls). Mean: 4.689 µs/call; median: 4.640; p95: 5.792. Share: 0.4887%.

```text
void sglang::deepseek_rope_kernel<true, false, long>(sglang::FusedQKRopeParams)
```

### 23. void sglang::deepseek_rope_kernel<true, true, long>(sglang::FusedQKRopeParams)

TP0: 0.198057 ms/iteration (1720 calls); TP1: 0.197266 ms/iteration (1720 calls). Mean: 4.597 µs/call; median: 4.609; p95: 4.896. Share: 0.4791%.

```text
void sglang::deepseek_rope_kernel<true, true, long>(sglang::FusedQKRopeParams)
```

### 24. void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorB

TP0: 0.160337 ms/iteration (120 calls); TP1: 0.160303 ms/iteration (120 calls). Mean: 53.440 µs/call; median: 16.704; p95: 129.473. Share: 0.3886%.

```text
void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)
```

### 25. nvjet_sm103_tst_128x192_64x7_2x1_2cta_v_bz_TNT

TP0: 0.155904 ms/iteration (40 calls); TP1: 0.155576 ms/iteration (40 calls). Mean: 155.740 µs/call; median: 155.537; p95: 158.465. Share: 0.3775%.

```text
nvjet_sm103_tst_128x192_64x7_2x1_2cta_v_bz_TNT
```

### 26. void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, flo

TP0: 0.148096 ms/iteration (1720 calls); TP1: 0.147119 ms/iteration (1720 calls). Mean: 3.433 µs/call; median: 3.424; p95: 3.744. Share: 0.3578%.

```text
void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, float const*, float*, float*, float const*, float const*, float const*, float const*, float*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)
```

### 27. kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_

TP0: 0.140155 ms/iteration (1720 calls); TP1: 0.138958 ms/iteration (1720 calls). Mean: 3.246 µs/call; median: 3.232; p95: 3.424. Share: 0.3383%.

```text
kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign16o409640961_tensorptri8gmem_0
```

### 28. nvjet_sm103_tst_128x160_64x8_2x1_2cta_v_bz_TNT

TP0: 0.139576 ms/iteration (40 calls); TP1: 0.138817 ms/iteration (40 calls). Mean: 139.197 µs/call; median: 138.912; p95: 142.881. Share: 0.3374%.

```text
nvjet_sm103_tst_128x160_64x8_2x1_2cta_v_bz_TNT
```

### 29. kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_

TP0: 0.133261 ms/iteration (2040 calls); TP1: 0.132102 ms/iteration (2040 calls). Mean: 2.602 µs/call; median: 2.496; p95: 3.680. Share: 0.3216%.

```text
kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o128012801_tensorptri8gmemalign16o128012801_tensorptri8gmem_0
```

### 30. kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeLinearKernel_object_at

TP0: 0.127925 ms/iteration (1720 calls); TP1: 0.127767 ms/iteration (1720 calls). Mean: 2.973 µs/call; median: 2.976; p95: 3.200. Share: 0.3099%.

```text
kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o512051201_tensorptri8gmemalign16o512051201_tensorptri8gmemal_0
```

### 31. void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::a

TP0: 0.121811 ms/iteration (1720 calls); TP1: 0.122435 ms/iteration (1720 calls). Mean: 2.840 µs/call; median: 2.848; p95: 3.072. Share: 0.2960%.

```text
void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)
```

### 32. void sglang::fused_k_norm_rope_flashmla<__nv_bfloat16, 512l, 64l, long, 8, true>(sglang::FusedKNormR

TP0: 0.110543 ms/iteration (1840 calls); TP1: 0.110480 ms/iteration (1840 calls). Mean: 2.402 µs/call; median: 2.400; p95: 2.720. Share: 0.2679%.

```text
void sglang::fused_k_norm_rope_flashmla<__nv_bfloat16, 512l, 64l, long, 8, true>(sglang::FusedKNormRopeFlashMLAParams)
```

### 33. kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gmemalign16o

TP0: 0.107189 ms/iteration (1600 calls); TP1: 0.106883 ms/iteration (1600 calls). Mean: 2.676 µs/call; median: 2.656; p95: 2.944. Share: 0.2594%.

```text
kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gmemalign16oi641280i64div81_tensorptrbf16gmemalign16o12801_tensorptrbf16gmemalign16oi641280i64div81_0
```

### 34. void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_co

TP0: 0.099390 ms/iteration (200 calls); TP1: 0.099673 ms/iteration (200 calls). Mean: 19.906 µs/call; median: 19.904; p95: 20.192. Share: 0.2413%.

```text
void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1})
```

### 35. kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_

TP0: 0.098125 ms/iteration (1720 calls); TP1: 0.098373 ms/iteration (1720 calls). Mean: 2.285 µs/call; median: 2.272; p95: 2.464. Share: 0.2381%.

```text
kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o115211521_tensorptri8gmemalign16o115211521_tensorptri8gmem_0
```

### 36. void sglang::silu_mul_clamp_kernel<__nv_bfloat16, true>(sglang::SiluAndMulClampParams)

TP0: 0.096598 ms/iteration (1720 calls); TP1: 0.097032 ms/iteration (1720 calls). Mean: 2.252 µs/call; median: 2.240; p95: 2.432. Share: 0.2347%.

```text
void sglang::silu_mul_clamp_kernel<__nv_bfloat16, true>(sglang::SiluAndMulClampParams)
```

### 37. void deep_gemm::sm100_paged_mqa_logits<true, 1u, 32u, 128u, 64u, true, false, 3u, 10u, 256u, 16u, 12

TP0: 0.079118 ms/iteration (320 calls); TP1: 0.078865 ms/iteration (320 calls). Mean: 9.874 µs/call; median: 10.144; p95: 10.816. Share: 0.1915%.

```text
void deep_gemm::sm100_paged_mqa_logits<true, 1u, 32u, 128u, 64u, true, false, 3u, 10u, 256u, 16u, 128u, 256u, float, float, 2u>(unsigned int, unsigned int, unsigned int, unsigned int const*, float*, unsigned int const*, unsigned int const*, unsigned int const*, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)
```

### 38. void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<float, 4, at::na

TP0: 0.075161 ms/iteration (200 calls); TP1: 0.074997 ms/iteration (200 calls). Mean: 15.016 µs/call; median: 15.008; p95: 15.200. Share: 0.1820%.

```text
void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<float, 4, at::native::templates::cuda::uniform_and_transform<float, float, at::CUDAGeneratorImpl*, at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::CUDAGeneratorImpl*, at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1})::{lambda(curandStatePhilox4_32_10*)#2}, at::native::(anonymous namespace)::distribution_nullary_kernel<float, float, float4, at::CUDAGeneratorImpl*, at::native::templates::cuda::uniform_and_transform<float, float, at::CUDAGeneratorImpl*, at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::CUDAGeneratorImpl*, at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1})::{lambda(curandStatePhilox4_32_10*)#2}, at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::CUDAGeneratorImpl*, at::native::templates::cuda::uniform_and_transform<float, float, at::CUDAGeneratorImpl*, at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::CUDAGeneratorImpl*, at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1})::{lambda(curandStatePhilox4_32_10*)#2} const&, at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1})::{lambda(int, float)#1}>(long, at::PhiloxCudaState, at::native::templates::cuda::uniform_and_transform<float, float, at::CUDAGeneratorImpl*, at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::CUDAGeneratorImpl*, at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1})::{lambda(curandStatePhilox4_32_10*)#2}, at::native::(anonymous namespace)::distribution_nullary_kernel<float, float, float4, at::CUDAGeneratorImpl*, at::native::templates::cuda::uniform_and_transform<float, float, at::CUDAGeneratorImpl*, at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::CUDAGeneratorImpl*, at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1})::{lambda(curandStatePhilox4_32_10*)#2}, at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::CUDAGeneratorImpl*, at::native::templates::cuda::uniform_and_transform<float, float, at::CUDAGeneratorImpl*, at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::CUDAGeneratorImpl*, at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1})::{lambda(curandStatePhilox4_32_10*)#2} const&, at::native::templates::cuda::exponential_kernel<at::CUDAGeneratorImpl*>(at::TensorIteratorBase&, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1})::{lambda(int, float)#1})
```

### 39. _online_partial_kernel

TP0: 0.070755 ms/iteration (200 calls); TP1: 0.070571 ms/iteration (200 calls). Mean: 14.133 µs/call; median: 14.144; p95: 14.592. Share: 0.1713%.

```text
_online_partial_kernel
```

### 40. _pack_topk_ids_triton_kernel

TP0: 0.066534 ms/iteration (1600 calls); TP1: 0.066265 ms/iteration (1600 calls). Mean: 1.660 µs/call; median: 1.632; p95: 1.824. Share: 0.1609%.

```text
_pack_topk_ids_triton_kernel
```

### 41. void at::native::sbtopk::gatherTopK<float, unsigned int, 2, false>(at::cuda::detail::TensorInfo<floa

TP0: 0.066196 ms/iteration (40 calls); TP1: 0.066226 ms/iteration (40 calls). Mean: 66.211 µs/call; median: 66.177; p95: 66.433. Share: 0.1605%.

```text
void at::native::sbtopk::gatherTopK<float, unsigned int, 2, false>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, float*)
```

### 42. ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>)

TP0: 0.055589 ms/iteration (320 calls); TP1: 0.066632 ms/iteration (320 calls). Mean: 7.639 µs/call; median: 7.200; p95: 10.784. Share: 0.1481%.

```text
ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>)
```

### 43. void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, uns

TP0: 0.059063 ms/iteration (40 calls); TP1: 0.059027 ms/iteration (40 calls). Mean: 59.045 µs/call; median: 58.945; p95: 60.224. Share: 0.1431%.

```text
void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, 4, 4> >(at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, 4, 4>)
```

### 44. nvjet_sm103_tst_448x64_64x3_2x1_2cta_v_bz_TNT

TP0: 0.053244 ms/iteration (200 calls); TP1: 0.053060 ms/iteration (200 calls). Mean: 10.630 µs/call; median: 10.784; p95: 11.296. Share: 0.1288%.

```text
nvjet_sm103_tst_448x64_64x3_2x1_2cta_v_bz_TNT
```

### 45. void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonymous namesp

TP0: 0.050888 ms/iteration (40 calls); TP1: 0.050896 ms/iteration (40 calls). Mean: 50.892 µs/call; median: 50.864; p95: 51.520. Share: 0.1234%.

```text
void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonymous namespace)::OpaqueType<4u>, unsigned int, 2, 128, 1, 16, 4>(char*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<4u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int)
```

### 46. nvjet_sm103_tst_128x80_64x10_2x2_2cta_h_bz_TNT

TP0: 0.046088 ms/iteration (120 calls); TP1: 0.047101 ms/iteration (120 calls). Mean: 15.531 µs/call; median: 15.520; p95: 16.224. Share: 0.1129%.

```text
nvjet_sm103_tst_128x80_64x10_2x2_2cta_h_bz_TNT
```

### 47. void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_co

TP0: 0.045907 ms/iteration (760 calls); TP1: 0.045229 ms/iteration (760 calls). Mean: 2.398 µs/call; median: 2.240; p95: 3.360. Share: 0.1105%.

```text
void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int)#1})
```

### 48. nvjet_sm103_tss_64x64_64x16_2x2_2cta_h_bz_TNT

TP0: 0.043371 ms/iteration (120 calls); TP1: 0.044819 ms/iteration (120 calls). Mean: 14.698 µs/call; median: 16.128; p95: 17.120. Share: 0.1069%.

```text
nvjet_sm103_tss_64x64_64x16_2x2_2cta_h_bz_TNT
```

### 49. nvjet_sm103_tst_64x32_64x16_1x4_h_bz_splitK_TNT

TP0: 0.039301 ms/iteration (320 calls); TP1: 0.039282 ms/iteration (320 calls). Mean: 4.911 µs/call; median: 4.896; p95: 5.280. Share: 0.0952%.

```text
nvjet_sm103_tst_64x32_64x16_1x4_h_bz_splitK_TNT
```

### 50. void sglang::flash_index_q_kernel<true, 128l, 64l, long, true>(sglang::IndexQParams)

TP0: 0.039162 ms/iteration (320 calls); TP1: 0.039173 ms/iteration (320 calls). Mean: 4.896 µs/call; median: 4.896; p95: 5.152. Share: 0.0949%.

```text
void sglang::flash_index_q_kernel<true, 128l, 64l, long, true>(sglang::IndexQParams)
```

### 51. void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymou

TP0: 0.038900 ms/iteration (80 calls); TP1: 0.038929 ms/iteration (80 calls). Mean: 19.457 µs/call; median: 19.456; p95: 19.680. Share: 0.0943%.

```text
void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(bool, unsigned short, unsigned short)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(bool, unsigned short, unsigned short)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(bool, unsigned short, unsigned short)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(bool, unsigned short, unsigned short)#1} const&)::{lambda(int)#1})
```

### 52. void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_ker

TP0: 0.038703 ms/iteration (337 calls); TP1: 0.038330 ms/iteration (337 calls). Mean: 4.572 µs/call; median: 4.768; p95: 7.136. Share: 0.0934%.

```text
void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})
```

### 53. _candidate_mask_kernel

TP0: 0.035391 ms/iteration (160 calls); TP1: 0.035244 ms/iteration (160 calls). Mean: 8.829 µs/call; median: 8.832; p95: 9.056. Share: 0.0856%.

```text
_candidate_mask_kernel
```

### 54. void sglang::topk_main_kernel<true, 2, (sglang::TopKMode)2>(sglang::TopKPagedParams)

TP0: 0.034943 ms/iteration (160 calls); TP1: 0.034984 ms/iteration (160 calls). Mean: 8.741 µs/call; median: 8.416; p95: 10.272. Share: 0.0847%.

```text
void sglang::topk_main_kernel<true, 2, (sglang::TopKMode)2>(sglang::TopKPagedParams)
```

### 55. void at_cuda_detail::cub::detail::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::det

TP0: 0.034820 ms/iteration (232 calls); TP1: 0.034909 ms/iteration (232 calls). Mean: 6.011 µs/call; median: 6.048; p95: 6.495. Share: 0.0845%.

```text
void at_cuda_detail::cub::detail::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::detail::radix::policy_hub<long, at_cuda_detail::cub::NullType, unsigned long long>::Policy1000, (at_cuda_detail::cub::SortOrder)0, long, at_cuda_detail::cub::NullType, unsigned long long, int, int, at_cuda_detail::cub::detail::identity_decomposer_t>(int*, int*, unsigned long long*, unsigned long long const*, long*, long const*, at_cuda_detail::cub::NullType*, at_cuda_detail::cub::NullType const*, int, int, int, at_cuda_detail::cub::detail::identity_decomposer_t)
```

### 56. void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIt

TP0: 0.034699 ms/iteration (40 calls); TP1: 0.034588 ms/iteration (40 calls). Mean: 34.643 µs/call; median: 34.432; p95: 35.456. Share: 0.0840%.

```text
void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>)
```

### 57. void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::MeanOps<c10::

TP0: 0.033606 ms/iteration (120 calls); TP1: 0.033410 ms/iteration (120 calls). Mean: 11.169 µs/call; median: 11.168; p95: 11.488. Share: 0.0812%.

```text
void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::MeanOps<c10::BFloat16, float, float, c10::BFloat16>, unsigned int, c10::BFloat16, 4, 8> >(at::native::ReduceOp<c10::BFloat16, at::native::MeanOps<c10::BFloat16, float, float, c10::BFloat16>, unsigned int, c10::BFloat16, 4, 8>)
```

### 58. void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, f

TP0: 0.025833 ms/iteration (360 calls); TP1: 0.025511 ms/iteration (360 calls). Mean: 2.852 µs/call; median: 2.912; p95: 3.552. Share: 0.0622%.

```text
void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, float*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)
```

### 59. void moe::dev::finalize::finalizeKernelVecLoad<moe::dev::finalize::KernelParams<cutlass::bfloat16_t,

TP0: 0.024462 ms/iteration (120 calls); TP1: 0.024178 ms/iteration (120 calls). Mean: 8.107 µs/call; median: 7.824; p95: 10.304. Share: 0.0589%.

```text
void moe::dev::finalize::finalizeKernelVecLoad<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cutlass::bfloat16_t, 1, true> >(moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cutlass::bfloat16_t, 1, true>)
```

### 60. void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorB

TP0: 0.024195 ms/iteration (520 calls); TP1: 0.024146 ms/iteration (520 calls). Mean: 1.859 µs/call; median: 1.856; p95: 2.208. Share: 0.0586%.

```text
void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)
```

### 61. _engram_gate_kernel

TP0: 0.024053 ms/iteration (80 calls); TP1: 0.023911 ms/iteration (80 calls). Mean: 11.991 µs/call; median: 11.968; p95: 12.384. Share: 0.0581%.

```text
_engram_gate_kernel
```

### 62. void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorB

TP0: 0.023804 ms/iteration (430 calls); TP1: 0.023857 ms/iteration (430 calls). Mean: 2.217 µs/call; median: 2.208; p95: 3.424. Share: 0.0578%.

```text
void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)
```

### 63. _build_step_local_kernel

TP0: 0.022223 ms/iteration (200 calls); TP1: 0.022698 ms/iteration (200 calls). Mean: 4.492 µs/call; median: 4.480; p95: 4.672. Share: 0.0544%.

```text
_build_step_local_kernel
```

### 64. void sglang::topk_main_kernel<true, 1, (sglang::TopKMode)1>(sglang::TopKPagedParams)

TP0: 0.020406 ms/iteration (120 calls); TP1: 0.020355 ms/iteration (120 calls). Mean: 6.794 µs/call; median: 6.880; p95: 7.936. Share: 0.0494%.

```text
void sglang::topk_main_kernel<true, 1, (sglang::TopKMode)1>(sglang::TopKPagedParams)
```

### 65. nvjet_sm103_tst_64x8_64x16_2x4_h_bz_TNT

TP0: 0.018827 ms/iteration (160 calls); TP1: 0.019889 ms/iteration (160 calls). Mean: 4.840 µs/call; median: 5.056; p95: 5.792. Share: 0.0469%.

```text
nvjet_sm103_tst_64x8_64x16_2x4_h_bz_TNT
```

### 66. nvjet_sm103_tss_64x32_64x16_2x4_2cta_h_bz_splitK_TNT

TP0: 0.017294 ms/iteration (120 calls); TP1: 0.017335 ms/iteration (120 calls). Mean: 5.772 µs/call; median: 5.760; p95: 5.984. Share: 0.0420%.

```text
nvjet_sm103_tss_64x32_64x16_2x4_2cta_h_bz_splitK_TNT
```

### 67. void sglang::flash_c2_decode_kernel<true, true, true, 512l, 64l, 7, long, int>(sglang::C2Params)

TP0: 0.015783 ms/iteration (120 calls); TP1: 0.015772 ms/iteration (120 calls). Mean: 5.259 µs/call; median: 5.248; p95: 6.048. Share: 0.0382%.

```text
void sglang::flash_c2_decode_kernel<true, true, true, 512l, 64l, 7, long, int>(sglang::C2Params)
```

### 68. void moe::dev::routing::routingCustom::routingIndicesClusterKernel<moe::dev::routing::routingCustom:

TP0: 0.013807 ms/iteration (120 calls); TP1: 0.013793 ms/iteration (120 calls). Mean: 4.600 µs/call; median: 4.608; p95: 4.896. Share: 0.0334%.

```text
void moe::dev::routing::routingCustom::routingIndicesClusterKernel<moe::dev::routing::routingCustom::KernelParams<__nv_bfloat16, __nv_bfloat16, 128, 4, moe::dev::routing::TopKExpertSelect<moe::dev::routing::NoOpPreprocess, moe::dev::routing::SoftmaxPostprocess> >, 1024>(moe::dev::routing::routingCustom::KernelParams<__nv_bfloat16, __nv_bfloat16, 128, 4, moe::dev::routing::TopKExpertSelect<moe::dev::routing::NoOpPreprocess, moe::dev::routing::SoftmaxPostprocess> >)
```

### 69. _publish_candidate_mask_kernel

TP0: 0.013505 ms/iteration (40 calls); TP1: 0.013442 ms/iteration (40 calls). Mean: 13.473 µs/call; median: 13.440; p95: 13.632. Share: 0.0327%.

```text
_publish_candidate_mask_kernel
```

### 70. _engram_gather_kernel

TP0: 0.012603 ms/iteration (80 calls); TP1: 0.012553 ms/iteration (80 calls). Mean: 6.289 µs/call; median: 6.272; p95: 6.432. Share: 0.0305%.

```text
_engram_gather_kernel
```

### 71. memcpy32_post

TP0: 0.012503 ms/iteration (360 calls); TP1: 0.012500 ms/iteration (360 calls). Mean: 1.389 µs/call; median: 1.312; p95: 1.760. Share: 0.0303%.

```text
memcpy32_post
```

### 72. void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long

TP0: 0.011742 ms/iteration (200 calls); TP1: 0.012339 ms/iteration (200 calls). Mean: 2.408 µs/call; median: 2.176; p95: 3.777. Share: 0.0292%.

```text
void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)
```

### 73. nvjet_sm103_tst_64x64_64x16_2x1_2cta_v_bz_splitK_TNT

TP0: 0.009754 ms/iteration (40 calls); TP1: 0.009822 ms/iteration (40 calls). Mean: 9.788 µs/call; median: 10.064; p95: 11.744. Share: 0.0237%.

```text
nvjet_sm103_tst_64x64_64x16_2x1_2cta_v_bz_splitK_TNT
```

### 74. _online_combine_kernel

TP0: 0.009749 ms/iteration (200 calls); TP1: 0.009822 ms/iteration (200 calls). Mean: 1.957 µs/call; median: 1.984; p95: 2.112. Share: 0.0237%.

```text
_online_combine_kernel
```

### 75. _expand_prefill_causally_kernel

TP0: 0.009644 ms/iteration (40 calls); TP1: 0.009634 ms/iteration (40 calls). Mean: 9.639 µs/call; median: 9.600; p95: 9.856. Share: 0.0234%.

```text
_expand_prefill_causally_kernel
```

### 76. void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_sc

TP0: 0.009526 ms/iteration (240 calls); TP1: 0.009478 ms/iteration (240 calls). Mean: 1.584 µs/call; median: 1.472; p95: 2.112. Share: 0.0230%.

```text
void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul>)
```

### 77. _filter_topk_pages

TP0: 0.009210 ms/iteration (160 calls); TP1: 0.009299 ms/iteration (160 calls). Mean: 2.314 µs/call; median: 2.304; p95: 2.496. Share: 0.0224%.

```text
_filter_topk_pages
```

### 78. void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_put

TP0: 0.008906 ms/iteration (106 calls); TP1: 0.008487 ms/iteration (106 calls). Mean: 3.282 µs/call; median: 2.880; p95: 7.360. Share: 0.0211%.

```text
void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})
```

### 79. void sglang::flash_index_k_kernel<true, 128l, 64l, 64u, 2u, long>(sglang::IndexKParams)

TP0: 0.008825 ms/iteration (120 calls); TP1: 0.008501 ms/iteration (120 calls). Mean: 2.888 µs/call; median: 2.720; p95: 3.744. Share: 0.0210%.

```text
void sglang::flash_index_k_kernel<true, 128l, 64l, 64u, 2u, long>(sglang::IndexKParams)
```

### 80. void sglang::topk_main_kernel<true, 2, (sglang::TopKMode)1>(sglang::TopKPagedParams)

TP0: 0.008454 ms/iteration (40 calls); TP1: 0.008409 ms/iteration (40 calls). Mean: 8.432 µs/call; median: 8.384; p95: 8.896. Share: 0.0204%.

```text
void sglang::topk_main_kernel<true, 2, (sglang::TopKMode)1>(sglang::TopKPagedParams)
```

### 81. void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array

TP0: 0.008044 ms/iteration (40 calls); TP1: 0.008053 ms/iteration (40 calls). Mean: 8.048 µs/call; median: 8.000; p95: 8.224. Share: 0.0195%.

```text
void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)
```

### 82. void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gather_interna

TP0: 0.007969 ms/iteration (40 calls); TP1: 0.007870 ms/iteration (40 calls). Mean: 7.919 µs/call; median: 7.904; p95: 8.416. Share: 0.0192%.

```text
void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1})
```

### 83. kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gmemalign128

TP0: 0.007807 ms/iteration (120 calls); TP1: 0.007770 ms/iteration (120 calls). Mean: 2.596 µs/call; median: 2.592; p95: 2.753. Share: 0.0189%.

```text
kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64128012801_tensorptrbf16gmemalign16o12801_tensorptrbf16gmemalign128oi64128012801___T_0
```

### 84. void at::native::vectorized_elementwise_kernel<2, at::native::BUnaryFunctor<long, long, long, at::na

TP0: 0.007101 ms/iteration (109 calls); TP1: 0.007059 ms/iteration (109 calls). Mean: 2.598 µs/call; median: 2.528; p95: 3.071. Share: 0.0172%.

```text
void at::native::vectorized_elementwise_kernel<2, at::native::BUnaryFunctor<long, long, long, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long, long)#1}>, std::array<char*, 2ul> >(int, at::native::BUnaryFunctor<long, long, long, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long, long)#1}>, std::array<char*, 2ul>)
```

### 85. kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_

TP0: 0.006570 ms/iteration (80 calls); TP1: 0.006567 ms/iteration (80 calls). Mean: 3.284 µs/call; median: 3.296; p95: 3.424. Share: 0.0159%.

```text
kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o614461441_tensorptri8gmemalign16o614461441_tensorptri8gmem_0
```

### 86. void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunct

TP0: 0.006519 ms/iteration (120 calls); TP1: 0.006516 ms/iteration (120 calls). Mean: 2.173 µs/call; median: 2.143; p95: 2.368. Share: 0.0158%.

```text
void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<long> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<long> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<long> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<long> const&)::{lambda(int)#1})
```

### 87. _engram_hash_kernel

TP0: 0.006444 ms/iteration (40 calls); TP1: 0.006438 ms/iteration (40 calls). Mean: 6.441 µs/call; median: 6.432; p95: 6.720. Share: 0.0156%.

```text
_engram_hash_kernel
```

### 88. void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1u

TP0: 0.005922 ms/iteration (200 calls); TP1: 0.005905 ms/iteration (200 calls). Mean: 1.183 µs/call; median: 1.120; p95: 1.664. Share: 0.0143%.

```text
void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)
```

### 89. _vocab_parallel_embedding_kernel

TP0: 0.006246 ms/iteration (80 calls); TP1: 0.005551 ms/iteration (80 calls). Mean: 2.949 µs/call; median: 2.848; p95: 3.648. Share: 0.0143%.

```text
_vocab_parallel_embedding_kernel
```

### 90. void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonymous namesp

TP0: 0.005713 ms/iteration (40 calls); TP1: 0.005675 ms/iteration (40 calls). Mean: 5.694 µs/call; median: 5.664; p95: 6.080. Share: 0.0138%.

```text
void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1, 16, 8>(char*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int)
```

### 91. kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_

TP0: 0.005589 ms/iteration (40 calls); TP1: 0.005664 ms/iteration (40 calls). Mean: 5.626 µs/call; median: 5.600; p95: 5.888. Share: 0.0136%.

```text
kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o15360153601_tensorptri8gmemalign16o15360153601_tensorptri8_0
```

### 92. _candidate_scores_kernel

TP0: 0.005482 ms/iteration (40 calls); TP1: 0.005590 ms/iteration (40 calls). Mean: 5.536 µs/call; median: 5.521; p95: 5.792. Share: 0.0134%.

```text
_candidate_scores_kernel
```

### 93. void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scal

TP0: 0.005467 ms/iteration (201 calls); TP1: 0.005406 ms/iteration (201 calls). Mean: 1.082 µs/call; median: 1.120; p95: 1.216. Share: 0.0132%.

```text
void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, function_traits<at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>::result_type*)
```

### 94. void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<long>, std::array<char*, 1

TP0: 0.005255 ms/iteration (160 calls); TP1: 0.005281 ms/iteration (160 calls). Mean: 1.317 µs/call; median: 1.248; p95: 1.696. Share: 0.0128%.

```text
void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<long>, std::array<char*, 1ul> >(int, at::native::FillFunctor<long>, std::array<char*, 1ul>)
```

### 95. void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::nativ

TP0: 0.005094 ms/iteration (90 calls); TP1: 0.005063 ms/iteration (90 calls). Mean: 2.257 µs/call; median: 2.304; p95: 2.688. Share: 0.0123%.

```text
void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul> >(int, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul>)
```

### 96. _page_table_positions_kernel

TP0: 0.004888 ms/iteration (80 calls); TP1: 0.004837 ms/iteration (80 calls). Mean: 2.431 µs/call; median: 2.496; p95: 2.720. Share: 0.0118%.

```text
_page_table_positions_kernel
```

### 97. void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymou

TP0: 0.004618 ms/iteration (40 calls); TP1: 0.004602 ms/iteration (40 calls). Mean: 4.610 µs/call; median: 4.576; p95: 4.864. Share: 0.0112%.

```text
void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<long> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<long> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<long> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<long> const&)::{lambda(int)#1})
```

### 98. void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<long>, std::arra

TP0: 0.004534 ms/iteration (120 calls); TP1: 0.004490 ms/iteration (120 calls). Mean: 1.504 µs/call; median: 1.440; p95: 1.856. Share: 0.0109%.

```text
void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>)
```

### 99. _rmsnorm_fp32_kernel

TP0: 0.004350 ms/iteration (80 calls); TP1: 0.004509 ms/iteration (80 calls). Mean: 2.215 µs/call; median: 2.385; p95: 2.816. Share: 0.0107%.

```text
_rmsnorm_fp32_kernel
```

### 100. void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::nativ

TP0: 0.004326 ms/iteration (80 calls); TP1: 0.004413 ms/iteration (80 calls). Mean: 2.185 µs/call; median: 2.176; p95: 2.368. Share: 0.0106%.

```text
void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::native::remainder_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul> >(int, at::native::BUnaryFunctor<int, int, int, at::native::remainder_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul>)
```

### 101. void at::native::index_elementwise_kernel<128, 4, at::native::index_fill_kernel_impl<at::native::Opa

TP0: 0.004308 ms/iteration (42 calls); TP1: 0.004285 ms/iteration (42 calls). Mean: 4.092 µs/call; median: 4.096; p95: 4.256. Share: 0.0104%.

```text
void at::native::index_elementwise_kernel<128, 4, at::native::index_fill_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, long, long, long, at::native::OpaqueType<8>)::{lambda(int)#1}>(long, at::native::index_fill_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, long, long, long, at::native::OpaqueType<8>)::{lambda(int)#1})
```

### 102. sglang::paged_mqa_metadata_small_kernel(sglang::MetadataParams)

TP0: 0.004230 ms/iteration (80 calls); TP1: 0.004222 ms/iteration (80 calls). Mean: 2.113 µs/call; median: 2.144; p95: 2.208. Share: 0.0102%.

```text
sglang::paged_mqa_metadata_small_kernel(sglang::MetadataParams)
```

### 103. void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace):

TP0: 0.004167 ms/iteration (80 calls); TP1: 0.004158 ms/iteration (80 calls). Mean: 2.081 µs/call; median: 2.080; p95: 2.304. Share: 0.0101%.

```text
void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::TensorListMetadata<2>, at::native::(anonymous namespace)::UnaryOpFunctor<long, 2, 1, 1>, at::native::Copy<long, long> >(at::native::(anonymous namespace)::TensorListMetadata<2>, at::native::(anonymous namespace)::UnaryOpFunctor<long, 2, 1, 1>, at::native::Copy<long, long>)
```

### 104. sglang::topk_plan_cluster(unsigned int const*, sglang::PlanItem*, unsigned int, int)

TP0: 0.004098 ms/iteration (80 calls); TP1: 0.004113 ms/iteration (80 calls). Mean: 2.053 µs/call; median: 2.144; p95: 2.176. Share: 0.0100%.

```text
sglang::topk_plan_cluster(unsigned int const*, sglang::PlanItem*, unsigned int, int)
```

### 105. void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::(anonymou

TP0: 0.003918 ms/iteration (80 calls); TP1: 0.003924 ms/iteration (80 calls). Mean: 1.960 µs/call; median: 1.921; p95: 2.208. Share: 0.0095%.

```text
void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(bool, unsigned long, unsigned long)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(bool, unsigned long, unsigned long)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(bool, unsigned long, unsigned long)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(bool, unsigned long, unsigned long)#1} const&)::{lambda(int)#1})
```

### 106. void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array

TP0: 0.003559 ms/iteration (80 calls); TP1: 0.003534 ms/iteration (80 calls). Mean: 1.773 µs/call; median: 1.760; p95: 2.144. Share: 0.0086%.

```text
void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul>)
```

### 107. void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scal

TP0: 0.003580 ms/iteration (122 calls); TP1: 0.003467 ms/iteration (122 calls). Mean: 1.155 µs/call; median: 1.168; p95: 1.600. Share: 0.0085%.

```text
void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>(int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}, function_traits<at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>::result_type*)
```

### 108. _swa_page_indices_kernel

TP0: 0.003247 ms/iteration (40 calls); TP1: 0.003333 ms/iteration (40 calls). Mean: 3.290 µs/call; median: 3.296; p95: 3.424. Share: 0.0080%.

```text
_swa_page_indices_kernel
```

### 109. void at::native::vectorized_elementwise_kernel<8, at::native::AUnaryFunctor<long, long, bool, at::na

TP0: 0.003273 ms/iteration (80 calls); TP1: 0.003261 ms/iteration (80 calls). Mean: 1.633 µs/call; median: 1.600; p95: 1.952. Share: 0.0079%.

```text
void at::native::vectorized_elementwise_kernel<8, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul>)
```

### 110. void at_cuda_detail::cub::detail::select::DeviceSelectSweepKernel<at_cuda_detail::cub::detail::selec

TP0: 0.003333 ms/iteration (29 calls); TP1: 0.003190 ms/iteration (29 calls). Mean: 4.498 µs/call; median: 4.401; p95: 5.281. Share: 0.0079%.

```text
void at_cuda_detail::cub::detail::select::DeviceSelectSweepKernel<at_cuda_detail::cub::detail::select::policy_hub<long, at_cuda_detail::cub::NullType, int, false, (at_cuda_detail::cub::SelectImpl)0>::Policy1000, long const*, at_cuda_detail::cub::NullType*, long*, long*, at_cuda_detail::cub::ScanTileState<int, true>, at_cuda_detail::cub::NullType, cuda::std::__4::equal_to<void>, int, at_cuda_detail::cub::detail::select::streaming_context_t<long, true>, (at_cuda_detail::cub::SelectImpl)0>(long const*, at_cuda_detail::cub::NullType*, long*, long*, at_cuda_detail::cub::ScanTileState<int, true>, at_cuda_detail::cub::NullType, cuda::std::__4::equal_to<void>, int, int, at_cuda_detail::cub::detail::select::streaming_context_t<long, true>, at_cuda_detail::cub::detail::vsmem_t)
```

### 111. void at_cuda_detail::cub::detail::select::DeviceSelectSweepKernel<at_cuda_detail::cub::detail::selec

TP0: 0.003338 ms/iteration (29 calls); TP1: 0.003172 ms/iteration (29 calls). Mean: 4.490 µs/call; median: 4.449; p95: 5.248. Share: 0.0079%.

```text
void at_cuda_detail::cub::detail::select::DeviceSelectSweepKernel<at_cuda_detail::cub::detail::select::policy_hub<long, bool, int, false, (at_cuda_detail::cub::SelectImpl)0>::Policy1000, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::counting_iterator<long, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::use_default, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::use_default, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::use_default>, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::transform_iterator<at::native::(anonymous namespace)::NonZeroOp<bool>, bool const*, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::use_default, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::use_default>, long*, int*, at_cuda_detail::cub::ScanTileState<int, true>, at_cuda_detail::cub::NullType, at_cuda_detail::cub::NullType, int, at_cuda_detail::cub::detail::select::streaming_context_t<long, true>, (at_cuda_detail::cub::SelectImpl)0>(thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::counting_iterator<long, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::use_default, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::use_default, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::use_default>, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::transform_iterator<at::native::(anonymous namespace)::NonZeroOp<bool>, bool const*, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::use_default, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::use_default>, long*, int*, at_cuda_detail::cub::ScanTileState<int, true>, at_cuda_detail::cub::NullType, at_cuda_detail::cub::NullType, int, int, at_cuda_detail::cub::detail::select::streaming_context_t<long, true>, at_cuda_detail::cub::detail::vsmem_t)
```

### 112. void at::native::vectorized_elementwise_kernel<8, at::native::AUnaryFunctor<int, int, bool, at::nati

TP0: 0.003136 ms/iteration (80 calls); TP1: 0.003137 ms/iteration (80 calls). Mean: 1.568 µs/call; median: 1.568; p95: 1.632. Share: 0.0076%.

```text
void at::native::vectorized_elementwise_kernel<8, at::native::AUnaryFunctor<int, int, bool, at::native::(anonymous namespace)::CompareEqFunctor<int> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<int, int, bool, at::native::(anonymous namespace)::CompareEqFunctor<int> >, std::array<char*, 2ul>)
```

### 113. void at::native::vectorized_elementwise_kernel<2, at::native::AUnaryFunctor<long, long, long, at::na

TP0: 0.003105 ms/iteration (80 calls); TP1: 0.003153 ms/iteration (80 calls). Mean: 1.564 µs/call; median: 1.536; p95: 1.696. Share: 0.0076%.

```text
void at::native::vectorized_elementwise_kernel<2, at::native::AUnaryFunctor<long, long, long, at::native::binary_internal::MulFunctor<long> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<long, long, long, at::native::binary_internal::MulFunctor<long> >, std::array<char*, 2ul>)
```

### 114. alloc_extend_kernel

TP0: 0.003153 ms/iteration (52 calls); TP1: 0.003054 ms/iteration (52 calls). Mean: 2.387 µs/call; median: 2.400; p95: 2.656. Share: 0.0075%.

```text
alloc_extend_kernel
```

### 115. _causal_swa_page_indices_kernel

TP0: 0.002929 ms/iteration (40 calls); TP1: 0.002900 ms/iteration (40 calls). Mean: 2.914 µs/call; median: 2.880; p95: 3.136. Share: 0.0071%.

```text
_causal_swa_page_indices_kernel
```

### 116. _rope_fake_quant_pack_indexer_kernel

TP0: 0.002666 ms/iteration (40 calls); TP1: 0.002724 ms/iteration (40 calls). Mean: 2.695 µs/call; median: 2.656; p95: 2.848. Share: 0.0065%.

```text
_rope_fake_quant_pack_indexer_kernel
```

### 117. void at_cuda_detail::cub::detail::radix_sort::DeviceRadixSortHistogramKernel<at_cuda_detail::cub::de

TP0: 0.002657 ms/iteration (29 calls); TP1: 0.002657 ms/iteration (29 calls). Mean: 3.665 µs/call; median: 3.617; p95: 3.808. Share: 0.0064%.

```text
void at_cuda_detail::cub::detail::radix_sort::DeviceRadixSortHistogramKernel<at_cuda_detail::cub::detail::radix::policy_hub<long, at_cuda_detail::cub::NullType, unsigned long long>::Policy1000, (at_cuda_detail::cub::SortOrder)0, long, unsigned long long, at_cuda_detail::cub::detail::identity_decomposer_t>(unsigned long long*, long const*, unsigned long long, int, int, at_cuda_detail::cub::detail::identity_decomposer_t)
```

### 118. void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<

TP0: 0.002637 ms/iteration (40 calls); TP1: 0.002589 ms/iteration (40 calls). Mean: 2.613 µs/call; median: 2.624; p95: 2.816. Share: 0.0063%.

```text
void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<long> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<long> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<long> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<long> const&)::{lambda(int)#1})
```

### 119. assign_extend_cache_locs

TP0: 0.002420 ms/iteration (40 calls); TP1: 0.002441 ms/iteration (40 calls). Mean: 2.430 µs/call; median: 2.432; p95: 2.560. Share: 0.0059%.

```text
assign_extend_cache_locs
```

### 120. void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<bool>, std::array<char*, 1

TP0: 0.002340 ms/iteration (40 calls); TP1: 0.002339 ms/iteration (40 calls). Mean: 2.340 µs/call; median: 2.336; p95: 2.464. Share: 0.0057%.

```text
void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<bool>, std::array<char*, 1ul> >(int, at::native::FillFunctor<bool>, std::array<char*, 1ul>)
```

### 121. void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::Opaqu

TP0: 0.002292 ms/iteration (40 calls); TP1: 0.002323 ms/iteration (40 calls). Mean: 2.308 µs/call; median: 2.368; p95: 2.560. Share: 0.0056%.

```text
void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<8u>, unsigned int, 2, 64, 64>(at::native::(anonymous namespace)::OpaqueType<8u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<8u>, unsigned int, 64, 64>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int)
```

### 122. void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunct

TP0: 0.002221 ms/iteration (42 calls); TP1: 0.002320 ms/iteration (42 calls). Mean: 2.162 µs/call; median: 2.160; p95: 2.304. Share: 0.0055%.

```text
void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int)#1})
```

### 123. _engram_commit_history_kernel

TP0: 0.002261 ms/iteration (40 calls); TP1: 0.002134 ms/iteration (40 calls). Mean: 2.197 µs/call; median: 2.176; p95: 2.432. Share: 0.0053%.

```text
_engram_commit_history_kernel
```

### 124. _rope_tail_fake_quant_fp4_kernel

TP0: 0.002177 ms/iteration (40 calls); TP1: 0.002117 ms/iteration (40 calls). Mean: 2.147 µs/call; median: 2.048; p95: 2.976. Share: 0.0052%.

```text
_rope_tail_fake_quant_fp4_kernel
```

### 125. void sglang::fused_store_flashmla_cache<__nv_bfloat16, long, 8u, true>(sglang::FusedStoreCacheParam)

TP0: 0.001992 ms/iteration (40 calls); TP1: 0.001926 ms/iteration (40 calls). Mean: 1.959 µs/call; median: 1.856; p95: 2.496. Share: 0.0047%.

```text
void sglang::fused_store_flashmla_cache<__nv_bfloat16, long, 8u, true>(sglang::FusedStoreCacheParam)
```

### 126. void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_sc

TP0: 0.002018 ms/iteration (40 calls); TP1: 0.001866 ms/iteration (40 calls). Mean: 1.942 µs/call; median: 1.984; p95: 2.208. Share: 0.0047%.

```text
void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>)
```

### 127. void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<int>(at::TensorI

TP0: 0.002057 ms/iteration (40 calls); TP1: 0.001771 ms/iteration (40 calls). Mean: 1.914 µs/call; median: 1.872; p95: 2.272. Share: 0.0046%.

```text
void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<int>(at::TensorIteratorBase&, at::native::(anonymous namespace)::OpType, int)::{lambda(int)#1}, std::array<char*, 2ul> >(int, at::native::compare_scalar_kernel<int>(at::TensorIteratorBase&, at::native::(anonymous namespace)::OpType, int)::{lambda(int)#1}, std::array<char*, 2ul>)
```

### 128. void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::where_kernel_im

TP0: 0.001915 ms/iteration (40 calls); TP1: 0.001899 ms/iteration (40 calls). Mean: 1.907 µs/call; median: 1.888; p95: 2.080. Share: 0.0046%.

```text
void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(bool, unsigned int, unsigned int)#1}, std::array<char*, 4ul> >(int, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(bool, unsigned int, unsigned int)#1}, std::array<char*, 4ul>)
```

### 129. void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<int, int, int, at::nativ

TP0: 0.001854 ms/iteration (42 calls); TP1: 0.001859 ms/iteration (42 calls). Mean: 1.768 µs/call; median: 1.920; p95: 1.952. Share: 0.0045%.

```text
void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<int, int, int, at::native::binary_internal::MulFunctor<int> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<int, int, int, at::native::binary_internal::MulFunctor<int> >, std::array<char*, 2ul>)
```

### 130. void at::native::(anonymous namespace)::CatArrayBatchedCopy_alignedK_contig<at::native::(anonymous n

TP0: 0.001812 ms/iteration (40 calls); TP1: 0.001826 ms/iteration (40 calls). Mean: 1.819 µs/call; median: 1.760; p95: 2.016. Share: 0.0044%.

```text
void at::native::(anonymous namespace)::CatArrayBatchedCopy_alignedK_contig<at::native::(anonymous namespace)::OpaqueType<8u>, unsigned int, 2, 128, 1, 16>(at::native::(anonymous namespace)::OpaqueType<8u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<8u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int)
```

### 131. _build_out_tokens_kernel

TP0: 0.001820 ms/iteration (40 calls); TP1: 0.001807 ms/iteration (40 calls). Mean: 1.814 µs/call; median: 1.792; p95: 2.048. Share: 0.0044%.

```text
_build_out_tokens_kernel
```

### 132. void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_co

TP0: 0.001745 ms/iteration (40 calls); TP1: 0.001733 ms/iteration (40 calls). Mean: 1.739 µs/call; median: 1.729; p95: 1.824. Share: 0.0042%.

```text
void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int)#1})
```

### 133. _window_gather_kernel

TP0: 0.001728 ms/iteration (40 calls); TP1: 0.001738 ms/iteration (40 calls). Mean: 1.733 µs/call; median: 1.696; p95: 1.920. Share: 0.0042%.

```text
_window_gather_kernel
```

### 134. assign_req_to_token_pool

TP0: 0.001730 ms/iteration (26 calls); TP1: 0.001734 ms/iteration (26 calls). Mean: 2.665 µs/call; median: 2.688; p95: 2.848. Share: 0.0042%.

```text
assign_req_to_token_pool
```

### 135. void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<int>, std::array<char*

TP0: 0.001665 ms/iteration (40 calls); TP1: 0.001701 ms/iteration (40 calls). Mean: 1.683 µs/call; median: 1.760; p95: 2.016. Share: 0.0041%.

```text
void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>)
```

### 136. void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFun

TP0: 0.001713 ms/iteration (40 calls); TP1: 0.001645 ms/iteration (40 calls). Mean: 1.679 µs/call; median: 1.633; p95: 1.824. Share: 0.0041%.

```text
void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<int, int, int, at::native::binary_internal::MulFunctor<int> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<int, int, int, at::native::binary_internal::MulFunctor<int> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<int, int, int, at::native::binary_internal::MulFunctor<int> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<int, int, int, at::native::binary_internal::MulFunctor<int> > const&)::{lambda(int)#1})
```

### 137. get_last_loc_kernel

TP0: 0.001574 ms/iteration (26 calls); TP1: 0.001608 ms/iteration (26 calls). Mean: 2.447 µs/call; median: 2.496; p95: 2.784. Share: 0.0039%.

```text
get_last_loc_kernel
```

### 138. _fused_correct_drafts_and_bonus_kernel

TP0: 0.001550 ms/iteration (40 calls); TP1: 0.001539 ms/iteration (40 calls). Mean: 1.545 µs/call; median: 1.504; p95: 1.760. Share: 0.0037%.

```text
_fused_correct_drafts_and_bonus_kernel
```

### 139. _finalize_accept_lens_kernel

TP0: 0.001496 ms/iteration (40 calls); TP1: 0.001513 ms/iteration (40 calls). Mean: 1.505 µs/call; median: 1.472; p95: 1.696. Share: 0.0036%.

```text
_finalize_accept_lens_kernel
```

### 140. void at_cuda_detail::cub::detail::reduce::DeviceReduceKernel<at_cuda_detail::cub::detail::reduce::po

TP0: 0.001503 ms/iteration (29 calls); TP1: 0.001496 ms/iteration (29 calls). Mean: 2.068 µs/call; median: 2.112; p95: 2.176. Share: 0.0036%.

```text
void at_cuda_detail::cub::detail::reduce::DeviceReduceKernel<at_cuda_detail::cub::detail::reduce::policy_hub<int, unsigned long long, cuda::std::__4::plus<void> >::Policy1000, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::transform_iterator<at::native::(anonymous namespace)::NonZeroOp<bool>, bool const*, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::use_default, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::use_default>, unsigned long long, cuda::std::__4::plus<void>, int, cuda::std::__4::__identity>(thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::transform_iterator<at::native::(anonymous namespace)::NonZeroOp<bool>, bool const*, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::use_default, thrust::THRUST_300001_SM_750_800_860_900_1000_1200_NS::use_default>, int*, unsigned long long, at_cuda_detail::cub::GridEvenShare<unsigned long long>, cuda::std::__4::plus<void>, cuda::std::__4::__identity)
```

### 141. void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<int, int, int, at::nativ

TP0: 0.001435 ms/iteration (40 calls); TP1: 0.001466 ms/iteration (40 calls). Mean: 1.450 µs/call; median: 1.425; p95: 1.664. Share: 0.0035%.

```text
void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<int, int, int, at::native::minimum_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<int, int, int, at::native::minimum_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 3ul>)
```

### 142. void at_cuda_detail::cub::detail::reduce::DeviceReduceSingleTileKernel<at_cuda_detail::cub::detail::

TP0: 0.001502 ms/iteration (29 calls); TP1: 0.001322 ms/iteration (29 calls). Mean: 1.948 µs/call; median: 1.936; p95: 2.304. Share: 0.0034%.

```text
void at_cuda_detail::cub::detail::reduce::DeviceReduceSingleTileKernel<at_cuda_detail::cub::detail::reduce::policy_hub<int, unsigned long long, cuda::std::__4::plus<void> >::Policy1000, int*, int*, int, cuda::std::__4::plus<void>, int, int, cuda::std::__4::__identity>(int*, int*, int, cuda::std::__4::plus<void>, int, cuda::std::__4::__identity)
```

### 143. void at_cuda_detail::cub::detail::radix_sort::DeviceRadixSortExclusiveSumKernel<at_cuda_detail::cub:

TP0: 0.001303 ms/iteration (29 calls); TP1: 0.001304 ms/iteration (29 calls). Mean: 1.798 µs/call; median: 1.888; p95: 1.952. Share: 0.0032%.

```text
void at_cuda_detail::cub::detail::radix_sort::DeviceRadixSortExclusiveSumKernel<at_cuda_detail::cub::detail::radix::policy_hub<long, at_cuda_detail::cub::NullType, unsigned long long>::Policy1000, unsigned long long>(unsigned long long*)
```

### 144. void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<long>(at::Tensor

TP0: 0.001303 ms/iteration (29 calls); TP1: 0.001293 ms/iteration (29 calls). Mean: 1.790 µs/call; median: 1.792; p95: 1.889. Share: 0.0031%.

```text
void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<long>(at::TensorIteratorBase&, at::native::(anonymous namespace)::OpType, long)::{lambda(long)#1}, std::array<char*, 2ul> >(int, at::native::compare_scalar_kernel<long>(at::TensorIteratorBase&, at::native::(anonymous namespace)::OpType, long)::{lambda(long)#1}, std::array<char*, 2ul>)
```

### 145. _block_seq_lens_casual_kernel

TP0: 0.001310 ms/iteration (40 calls); TP1: 0.001282 ms/iteration (40 calls). Mean: 1.296 µs/call; median: 1.248; p95: 1.409. Share: 0.0031%.

```text
_block_seq_lens_casual_kernel
```

### 146. void at::native::unrolled_elementwise_kernel<at::native::BUnaryFunctor<int, int, int, at::native::bi

TP0: 0.001251 ms/iteration (22 calls); TP1: 0.001226 ms/iteration (22 calls). Mean: 2.252 µs/call; median: 2.272; p95: 2.495. Share: 0.0030%.

```text
void at::native::unrolled_elementwise_kernel<at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast)
```

### 147. void at_cuda_detail::cub::detail::scan::DeviceCompactInitKernel<at_cuda_detail::cub::ScanTileState<i

TP0: 0.001051 ms/iteration (29 calls); TP1: 0.001039 ms/iteration (29 calls). Mean: 1.442 µs/call; median: 1.296; p95: 1.665. Share: 0.0025%.

```text
void at_cuda_detail::cub::detail::scan::DeviceCompactInitKernel<at_cuda_detail::cub::ScanTileState<int, true>, long*>(at_cuda_detail::cub::ScanTileState<int, true>, int, long*)
```

### 148. void at_cuda_detail::cub::detail::scan::DeviceCompactInitKernel<at_cuda_detail::cub::ScanTileState<i

TP0: 0.001023 ms/iteration (29 calls); TP1: 0.001026 ms/iteration (29 calls). Mean: 1.413 µs/call; median: 1.280; p95: 1.664. Share: 0.0025%.

```text
void at_cuda_detail::cub::detail::scan::DeviceCompactInitKernel<at_cuda_detail::cub::ScanTileState<int, true>, int*>(at_cuda_detail::cub::ScanTileState<int, true>, int, int*)
```

### 149. void at::native::vectorized_elementwise_kernel<2, at::native::BUnaryFunctor<int, int, int, at::nativ

TP0: 0.000663 ms/iteration (10 calls); TP1: 0.000655 ms/iteration (10 calls). Mean: 2.637 µs/call; median: 2.672; p95: 2.816. Share: 0.0016%.

```text
void at::native::vectorized_elementwise_kernel<2, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul> >(int, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul>)
```

### 150. void at::native::radixSortKVInPlace<-2, -1, 32, 32, long, long, unsigned int>(at::cuda::detail::Tens

TP0: 0.000614 ms/iteration (1 calls); TP1: 0.000595 ms/iteration (1 calls). Mean: 24.176 µs/call; median: 24.176; p95: 24.544. Share: 0.0015%.

```text
void at::native::radixSortKVInPlace<-2, -1, 32, 32, long, long, unsigned int>(at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, bool)
```

### 151. void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonymous namesp

TP0: 0.000441 ms/iteration (9 calls); TP1: 0.000439 ms/iteration (9 calls). Mean: 1.956 µs/call; median: 1.952; p95: 2.080. Share: 0.0011%.

```text
void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonymous namespace)::OpaqueType<8u>, unsigned int, 1, 128, 1, 16, 2>(char*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<8u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int)
```

### 152. void at::native::(anonymous namespace)::CatArrayBatchedCopy_alignedK_contig<at::native::(anonymous n

TP0: 0.000055 ms/iteration (1 calls); TP1: 0.000055 ms/iteration (1 calls). Mean: 2.208 µs/call; median: 2.208; p95: 2.208. Share: 0.0001%.

```text
void at::native::(anonymous namespace)::CatArrayBatchedCopy_alignedK_contig<at::native::(anonymous namespace)::OpaqueType<8u>, unsigned int, 1, 128, 1, 16>(at::native::(anonymous namespace)::OpaqueType<8u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<8u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int)
```
