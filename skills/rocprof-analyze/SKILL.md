---
name: rocprof-analyze
description: Profiles a HIP kernel with rocprof and analyzes performance counters to diagnose bottlenecks. Use when asking for performance analysis, rocprof profiling, or bottleneck diagnosis on AMD GPUs.
---

# rocprof Profiling 与性能分析

## AMD GPU 术语映射

| NVIDIA | AMD |
|--------|-----|
| SM | CU (Compute Unit) |
| Warp (32) | Wavefront (64, CDNA) |
| Shared Memory | LDS (Local Data Share) |
| Tensor Core | Matrix Core / MFMA |

## Profiling 命令

```bash
rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR \
    SQ_INSTS_LDS SQ_BUSY_CYCLES SQ_WAIT_ANY \
    -- python3 skills/kernel-benchmark/scripts/benchmark.py <cpp_file> --repeat=5
```

## 瓶颈诊断逻辑

```
wait_ratio = SQ_WAIT_ANY / SQ_BUSY_CYCLES
valu_busy  = SQ_ACTIVE_INST_VALU / SQ_BUSY_CYCLES

IF wait_ratio > 60%:
    IF vmem > lds:  → VRAM_MEMORY_BOUND
    ELIF lds > vmem: → LDS_PRESSURE_BOUND
    ELSE:            → LATENCY_BOUND
ELIF valu_busy > 70%: → COMPUTE_BOUND
ELIF occupancy < 30%: → OCCUPANCY_BOUND
```

## 优化策略速查

| 瓶颈 | P0 | P1 | P2 |
|------|----|----|-----|
| VRAM_BOUND | LDS Tiling | Vectorized Load | Prefetching |
| LDS_BOUND | LDS Padding | Swizzle | 改用寄存器 |
| LATENCY_BOUND | Double Buffering | ILP | Loop Unrolling |
| COMPUTE_BOUND | MFMA | Packed Math | 强度削减 |
| OCCUPANCY_BOUND | 调整 Workgroup Size | 减少 VGPR | 减少 LDS |
