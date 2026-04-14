# HIP 优化策略参考

## 默认策略（按算法类型）

### MatMul
P0: LDS Tiling (TILE_SIZE=16/32) → P1: Bank Conflict Padding (+1) → P2: Vectorized Load (float4)

### Reduction
Warp Shuffle (`__shfl_down`, 64-thread wavefront) + 多轮规约

### Element-wise
Grid-Stride Loop + Vectorized Load (float4/dwordx4)

## LDS Tiling 模板
```cpp
__shared__ float As[TILE][TILE + 1];  // +1 消除 bank conflict
__shared__ float Bs[TILE][TILE + 1];
```

## Workgroup 配置
| 场景 | 大小 | 说明 |
|------|------|------|
| 2D 矩阵 | (16,16) 或 (32,32) | TILE 对齐 |
| Reduction | (256,1) | 4 waves，利于 shuffle |
| 通用 | (64,1) | 1 wave，保守 |

Workgroup 大小应为 64 的倍数（CDNA wavefront = 64 threads）。

## CUDA → HIP 映射
| CUDA | HIP |
|------|-----|
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `__shfl_down_sync(mask,v,d)` | `__shfl_down(v,d)` |
| `nvcc -arch=sm_XX` | `hipcc --offload-arch=gfxXXX` |

## 瓶颈 → 优化映射
| 瓶颈 | 优先优化 |
|------|---------|
| VRAM_BOUND | LDS Tiling → Vectorized Load → Prefetching |
| LDS_BOUND | Padding → Swizzle → 改用寄存器 |
| LATENCY_BOUND | Double Buffering → ILP → Unrolling |
| COMPUTE_BOUND | MFMA → Packed Math → 强度削减 |
| OCCUPANCY_BOUND | 调整 Block Size → `__launch_bounds__` → 减少 LDS |
