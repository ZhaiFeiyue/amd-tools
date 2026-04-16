---
name: amd-kernel-profiling
description: Profiles HIP kernels on MI355X (gfx950) using rocprofv3. Collects kernel timing, computes TFLOPS/BW, and analyzes compute/MFMA utilization, L1/L2 cache hit rate, data volume, and latency. Use when profiling GPU kernels, collecting performance counters, or diagnosing bottlenecks on MI355X.
---

# MI355X (gfx950) Kernel Profiling

Platform: AMD Instinct MI355X, CDNA4, gfx950. Wave size=64, CU_NUM=256, SIMD_NUM=512.

## Profiling Command

One command collects kernel timing + all counters (compute, MFMA, L1, L2). Each `--pmc` flag is a separate profiling pass:

```bash
rocprofv3 \
  --kernel-trace \
  --pmc SQ_WAVES,SQ_BUSY_CYCLES,SQ_WAVE_CYCLES,SQ_INSTS_VALU,SQ_INSTS_SALU,SQ_ACTIVE_INST_VALU,SQ_THREAD_CYCLES_VALU,SQ_INST_CYCLES_SALU \
  --pmc SQ_WAIT_ANY,SQ_WAIT_INST_ANY,SQ_ACTIVE_INST_ANY,SQ_INSTS_VMEM_RD,SQ_INSTS_VMEM_WR,SQ_INSTS_LDS,SQ_INSTS_SMEM,SQ_INSTS_FLAT \
  --pmc GRBM_GUI_ACTIVE,GRBM_COUNT \
  --pmc TCC_HIT,TCC_MISS,TCC_EA_RDREQ,TCC_EA_RDREQ_32B,TCC_EA_WRREQ,TCC_EA_WRREQ_64B,TCC_EA_RDREQ_LEVEL,TCC_EA_WRREQ_LEVEL \
  --pmc TCP_TOTAL_READ,TCP_TOTAL_WRITE,TCP_TOTAL_CACHE_ACCESSES,TCP_TCC_READ_REQ,TCP_TCC_WRITE_REQ,TCP_TCP_LATENCY,TCP_TA_TCP_STATE_READ,TCP_TCC_READ_REQ_LATENCY,TCP_TCC_WRITE_REQ_LATENCY \
  --pmc SQ_INSTS_VALU_MFMA_MOPS_F16,SQ_INSTS_VALU_MFMA_MOPS_BF16,SQ_INSTS_VALU_MFMA_MOPS_F32,SQ_INSTS_VALU_MFMA_MOPS_F64,SQ_VALU_MFMA_BUSY_CYCLES \
  --kernel-include-regex "your_kernel" \
  --output-format csv \
  -d ./prof_out \
  -- ./your_app
```

`--kernel-trace` produces `DurationNs` per dispatch. Each `--pmc` group reruns the app once (6 passes total).

---

## TFLOPS and Bandwidth Calculation

From `DurationNs` in the kernel-trace CSV + known problem dimensions:

```
TFLOPS = total_flops / duration_ns / 1e3
BW_GBs = total_bytes / duration_ns          (bytes/ns = GB/s)
```

| Operation | total_flops | total_bytes |
|-----------|-------------|-------------|
| GEMM C=A×B | `2 × M × N × K` | `M×K×bs_A + K×N×bs_B + M×N×bs_C` |
| Batched GEMM | `batch × 2 × M × N × K` | `batch × (M×K×bs_A + K×N×bs_B + M×N×bs_C)` |
| Elementwise | `numel × ops_per_elem` | `numel × (bs_in + bs_out)` |
| Reduction | `numel` | `numel × bs_in + out_size × bs_out` |

`bs` = bytes per element: fp4=0.5, fp8/int8=1, fp16/bf16=2, fp32=4, fp64=8.

---

## Counter Meaning and Derived Metrics

### Compute

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| GPU Busy % | `100 × GRBM_GUI_ACTIVE / GRBM_COUNT` | Overall GPU utilization |
| VALU Busy % | `100 × SQ_ACTIVE_INST_VALU × 4 / 512 / GRBM_GUI_ACTIVE` | Vector ALU pipe utilization |
| SALU Busy % | `100 × SQ_INST_CYCLES_SALU × 4 / 512 / GRBM_GUI_ACTIVE` | Scalar ALU pipe utilization |
| VALU Thread Util % | `100 × SQ_THREAD_CYCLES_VALU / (SQ_ACTIVE_INST_VALU × 64)` | Thread divergence: 100%=all 64 lanes active |
| MFMA Util % | `100 × SQ_VALU_MFMA_BUSY_CYCLES / (GRBM_GUI_ACTIVE × 256 × 4)` | Matrix core utilization |
| MFMA TFLOPS | `(MFMA_MOPS_F16+BF16+F32+F64) × 512 / duration_ns / 1e3` | Actual matrix TFLOPS achieved |
| Wave Exec % | `100 × SQ_ACTIVE_INST_ANY / SQ_WAVE_CYCLES` | % time wave is executing |
| Wave Wait % | `100 × SQ_WAIT_ANY / SQ_WAVE_CYCLES` | % time wave stalled on data dependency |
| Wave Issue Wait % | `100 × SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES` | % time stalled on instruction issue |
| Compute Intensity | `SQ_INSTS_VALU / (SQ_INSTS_VMEM_RD + SQ_INSTS_VMEM_WR)` | ALU vs memory instruction ratio |

Exec% + Wait% + IssueWait% ≈ 100%: shows where wave time goes.

### L2 Cache (TCC)

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| L2 Hit Rate % | `100 × TCC_HIT / (TCC_HIT + TCC_MISS)` | Cache effectiveness |
| L2 Read Volume | `TCC_EA_RDREQ_32B×32 + (TCC_EA_RDREQ−TCC_EA_RDREQ_32B)×64` bytes | Data read from HBM→L2 |
| L2 Write Volume | `(TCC_EA_WRREQ−TCC_EA_WRREQ_64B)×32 + TCC_EA_WRREQ_64B×64` bytes | Data written L2→HBM |
| L2 Read BW | `L2_Read_Volume / duration_ns` | Actual HBM read bandwidth (GB/s) |
| L2 Write BW | `L2_Write_Volume / duration_ns` | Actual HBM write bandwidth (GB/s) |
| L2 Read Latency | `TCC_EA_RDREQ_LEVEL / TCC_EA_RDREQ` | Avg cycles per HBM read |
| L2 Write Latency | `TCC_EA_WRREQ_LEVEL / TCC_EA_WRREQ` | Avg cycles per HBM write |

### L1 Cache (TCP / vL1D)

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| L1 Read Hit Rate % | `100 × (1 − TCP_TCC_READ_REQ / TCP_TOTAL_READ)` | L1 data reuse |
| L1 Miss Rate % | `100 × (TCP_TCC_READ_REQ + TCP_TCC_WRITE_REQ) / TCP_TOTAL_CACHE_ACCESSES` | Overall miss rate |
| L1 Read Accesses | `TCP_TOTAL_READ` | Total L1 read requests |
| L1 Write Accesses | `TCP_TOTAL_WRITE` | Total L1 write requests |
| L1 Read Misses→L2 | `TCP_TCC_READ_REQ` | Reads forwarded to L2 |
| L1 Write Misses→L2 | `TCP_TCC_WRITE_REQ` | Writes forwarded to L2 |
| L1 Avg Wave Latency | `TCP_TCP_LATENCY / TCP_TA_TCP_STATE_READ` | Avg cycles per wave in L1 |
| L1→L2 Read Latency | `TCP_TCC_READ_REQ_LATENCY / TCP_TCC_READ_REQ` | Avg cycles for L1 miss served by L2 |
| L1→L2 Write Latency | `TCP_TCC_WRITE_REQ_LATENCY / TCP_TCC_WRITE_REQ` | Avg cycles for write to L2 |

### Instruction Mix (per wave)

```
VALU/wave    = SQ_INSTS_VALU / SQ_WAVES
SALU/wave    = SQ_INSTS_SALU / SQ_WAVES
VMEM_RD/wave = SQ_INSTS_VMEM_RD / SQ_WAVES
VMEM_WR/wave = SQ_INSTS_VMEM_WR / SQ_WAVES
LDS/wave     = SQ_INSTS_LDS / SQ_WAVES
SMEM/wave    = SQ_INSTS_SMEM / SQ_WAVES
```

---

## Bottleneck Diagnosis

```
1. GPU Busy < 50%
   → Launch overhead or insufficient parallelism. Increase problem size or reduce host sync.

2. Wave Wait % > 60%  (SQ_WAIT_ANY / SQ_WAVE_CYCLES)
   → Stalled on data. Check what type:
   ├─ VMEM_RD+WR >> LDS   → VRAM MEMORY BOUND
   ├─ LDS >> VMEM          → LDS PRESSURE (bank conflicts, queue full)
   └─ Both moderate        → LATENCY BOUND (not enough ILP to hide latency)

3. VALU Busy > 70%
   → COMPUTE BOUND. Kernel is ALU-limited.

4. L2 Hit Rate < 50%
   → CACHE THRASHING. Working set too large for L2.

5. Compute Intensity < 10
   → Memory-bound character. Optimization should focus on data movement.
   Compute Intensity > 50
   → Compute-bound character. Optimization should focus on ALU efficiency.

6. MFMA Util < 30% (for GEMM-class kernels)
   → Matrix cores underutilized. Check data feeding pipeline.

7. VALU Thread Util < 80%
   → Thread divergence. Workgroup size not multiple of 64, or conditional branches.

8. None dominant
   → OCCUPANCY BOUND. Check VGPR/SGPR/LDS pressure limiting wave count.
```

## Optimization Actions

| Bottleneck | Actions |
|------------|---------|
| VRAM Memory Bound | LDS tiling, cooperative load, vectorized loads (uint4), prefetch/double buffer |
| LDS Pressure | Pad LDS for bank conflicts, swizzle access, move data to registers |
| Latency Bound | Double buffering (async copy + compute overlap), increase ILP, unroll loops |
| Compute Bound | Use MFMA instead of VALU, packed math (fp16/bf16), strength reduction |
| Occupancy Bound | Reduce VGPR usage, tune workgroup size, reduce LDS allocation |
| Cache Thrashing | Tile for L2 residency, change access order, shrink working set |
| Low MFMA Util | Improve data prefetch pipeline, overlap VMEM loads with MFMA via `__builtin_amdgcn_sched_group_barrier` |
| Thread Divergence | Ensure workgroup size is multiple of 64, minimize conditional branches |

---

## Memory Hierarchy (MI355X)

```
           ┌──────────┐
           │  Kernel   │ ← DurationNs → TFLOPS, BW
           └──┬────┬───┘
     compute  │    │  memory
  ┌───────────▼┐  ┌▼───────────┐
  │  CU (256)  │  │  L1 (TCP)  │ ← TCP_TOTAL_READ/WRITE, TCP_TCP_LATENCY
  │ VALU Busy% │  └─────┬──────┘
  │ MFMA Util% │   miss │
  │ WaveExec%  │  ┌─────▼──────┐
  │ WaveWait%  │  │  L2 (TCC)  │ ← TCC_HIT/MISS → hit rate
  └────────────┘  │ RDREQ_LEVEL│ → latency
                  └─────┬──────┘
                   miss │
                  ┌─────▼──────┐
                  │    HBM3e   │ ← TCC_EA_RDREQ/WRREQ → volume, BW
                  └────────────┘
```
