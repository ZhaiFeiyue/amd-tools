---
name: rocprof-analyze
description: Profiles HIP kernels on MI355X (gfx950) using rocprofv3. Roofline analysis (timing + HBM bandwidth + arithmetic intensity) first, then detailed counter analysis. Use when profiling GPU kernels, collecting performance counters, or diagnosing bottlenecks on MI355X.
---

# MI355X (gfx950) Kernel Profiling

Platform: AMD Instinct MI355X, CDNA4, gfx950. Wave size=64, CU_NUM=256, SIMDs_per_CU=4, SIMD_NUM=1024.

## MI355X Peak Specs

| Precision | Peak | Unit |
|-----------|------|------|
| FP4 Matrix (MXFP4) | 10,100 | TFLOPS |
| FP16 Matrix | 2,500 | TFLOPS |
| FP32 Matrix | 157.3 | TFLOPS |
| FP32 Vector | 157.3 | TFLOPS |
| HBM3e Bandwidth | 8,000 | GB/s |

---

## Roofline Analysis

First step. Collects kernel timing (100 iterations) + HBM load/store. Computes actual TFLOPS, theoretical Arithmetic Intensity, and diagnoses efficiency gap.

### L2 Cache Flush

MI355X has 256 MB total L2 (8 XCDs × 32 MB). Without flushing, repeated kernel runs keep weights hot in L2, making HBM Load much lower than theoretical — hiding real memory traffic.

**Flush before every kernel dispatch** in both benchmark and profiling scripts:

```python
def flush_l2():
    """Fill 256 MB to evict all L2 contents. Call before each kernel run."""
    buf = torch.zeros(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    buf.fill_(1.0)
    torch.cuda.synchronize()
    del buf

# Benchmark / profiling loop:
for i in range(100):
    flush_l2()
    run_kernel()
    torch.cuda.synchronize()
```

HIP C++ equivalent:

```cpp
void flush_l2() {
    void* buf;
    size_t sz = 256 * 1024 * 1024;
    hipMalloc(&buf, sz);
    hipMemset(buf, 1, sz);
    hipDeviceSynchronize();
    hipFree(buf);
}
```

### Commands (2 passes)

App must run the kernel at least 100 times with L2 flush before each dispatch.

```bash
# Pass 1: Kernel timing
rocprofv3 --kernel-trace --kernel-include-regex "your_kernel" \
  --output-format csv -d ./prof/roofline_time -- ./your_app

# Pass 2: L2 total read (includes L2 hits — reflects all data the kernel accessed)
rocprofv3 --pmc TCC_READ_SECTORS \
  --kernel-include-regex "your_kernel" \
  --output-format csv -d ./prof/roofline_l2read -- ./your_app

# Pass 3: HBM actual read (L2 misses only — actual DRAM traffic)
rocprofv3 --pmc TCC_EA0_RDREQ_DRAM_32B \
  --kernel-include-regex "your_kernel" \
  --output-format csv -d ./prof/roofline_hbm_rd -- ./your_app

# Pass 4: HBM write
rocprofv3 --pmc TCC_EA0_WRREQ,TCC_EA0_WRREQ_64B \
  --kernel-include-regex "your_kernel" \
  --output-format csv -d ./prof/roofline_hbm_wr -- ./your_app
```

Note: `--kernel-include-regex` filters out the flush kernel dispatches from results.

### Collected Metrics

**1. Actual TFLOPS** (from timing):

```
TFLOPS = total_flops / avg_duration_ns / 1e3
```

| Operation | total_flops |
|-----------|-------------|
| GEMM C=A×B | `2 × M × N × K` |
| Batched GEMM | `batch × 2 × M × N × K` |
| Elementwise | `numel × ops_per_elem` |
| Reduction | `numel` |

**2. Memory Traffic** — two levels:

```
L2 Total Read  = TCC_READ_SECTORS × 32                                                    [bytes]
HBM Actual Read = TCC_EA0_RDREQ_DRAM_32B × 32                                             [bytes]
HBM Write      = (TCC_EA0_WRREQ − TCC_EA0_WRREQ_64B) × 32 + TCC_EA0_WRREQ_64B × 64      [bytes]
L2 Read Hit Rate = (1 − HBM_Actual_Read / L2_Total_Read) × 100%
```

- **`TCC_READ_SECTORS`**: all 32B sectors read at L2 (hit + miss). Should be >= theoretical IO. Use to validate kernel reads the expected amount.
- **`TCC_EA0_RDREQ_DRAM_32B`**: only L2 misses that go to HBM. Can be < theoretical IO if workgroups share L2 cache within the same dispatch. Use for actual HBM bandwidth calculation.

IMPORTANT: `TCC_EA0_RDREQ_DRAM_32B` counts 32B-equivalent requests (128B request = 4, 64B = 2). This is the most reliable counter for HBM bytes on gfx950.

Do NOT use `TCC_EA0_RDREQ_128B × 128 + 64B × 64 + 32B × 32` for HBM bytes — this formula gives L2→EA requests, not DRAM requests. Some EA requests may be served by other caches or coherency probes without going to DRAM.

**3. Theoretical Arithmetic Intensity** (from algorithm, not measured):

```
Theoretical IO Read  = input tensor bytes   (e.g., GEMM: M×K×bs_A + K×N×bs_B)
Theoretical IO Write = output tensor bytes  (e.g., GEMM: M×N×bs_C)
Theoretical IO Total = IO_Read + IO_Write

Theoretical AI = total_flops / Theoretical_IO_Total   [FLOPs/Byte]
```

`bs` = bytes per element: fp4=0.5, fp8/int8=1, fp16/bf16=2, fp32=4, fp64=8.

### Roofline Ridge Points (Hardware AI)

| Precision | Ridge Point = Peak TFLOPS / Peak BW |
|-----------|-------------------------------------|
| FP4 MFMA | 10100 PFLOPS / 8 TB/s = **1262.5** FLOPs/Byte |
| FP16 MFMA | 2500 / 8 = **312.5** FLOPs/Byte |
| FP32 Vector | 157.3 / 8 = **19.7** FLOPs/Byte |

### Regime 判定 + 诊断

```
IF Theoretical_AI < Ridge_Point:
    → MEMORY-BOUND: kernel limited by HBM bandwidth
    → 诊断:
      1. Validate: L2_Total_Read ≈ Theoretical_IO_Read (should be >= 1.0x)
         IF L2_Total_Read << Theoretical → counter issue or kernel doesn't read all data
         IF L2_Total_Read >> Theoretical → redundant reads (poor coalescing)
      2. L2 efficiency: L2_Read_Hit_Rate = (1 - HBM_Read / L2_Total_Read)
         High hit rate → good L2 reuse (e.g., CK cooperative loading)
         Low hit rate  → every L2 access goes to HBM

ELSE:
    → COMPUTE-BOUND: kernel limited by peak TFLOPS
    → 诊断: Compute Efficiency = actual_TFLOPS / Peak_TFLOPS × 100%
```

### Roofline Report Template

```markdown
## Roofline Analysis

| Metric | Value |
|--------|-------|
| Kernel | {name} |
| Problem | {op} M={M} N={N} K={K} dtype={dtype} |
| Dispatches | 100 (warmup=10, measured=90) |
| Duration avg | {X} us |
| Duration p50/p95 | {X} / {X} us |
| **Actual TFLOPS** | {X} |
| **L2 Total Read** | {X} MB (should ≈ theoretical) |
| **HBM Actual Read** | {X} MB (L2 misses only) |
| **HBM Write** | {X} MB |
| Theoretical IO Read | {X} MB |
| Theoretical IO Write | {X} MB |
| **L2 Read Hit Rate** | {X}% |
| **Theoretical AI** | {X} FLOPs/Byte |
| Ridge Point ({dtype}) | {X} FLOPs/Byte |
| **Regime** | **MEMORY-BOUND / COMPUTE-BOUND** |
| HBM Read BW | {X} GB/s ({X}% of 8000 peak) |

Notes:
- L2 Total Read should be >= Theoretical IO Read (validates data was actually read)
- HBM Actual Read can be < L2 Total Read due to L2 cache reuse within dispatch
- L2 Read Hit Rate shows how much data was served from L2 vs HBM
```

---

## Detailed Counter Analysis

Run after Roofline to investigate specific bottlenecks. Each counter group is a **separate rocprofv3 invocation** (multi-pass in one command only saves the last pass).

gfx950 uses `TCC_EA0_*` (not `TCC_EA_*`). Hardware counter slots limited per block — keep ≤4 same-block counters per pass.

```bash
# SQ compute
rocprofv3 --pmc SQ_WAVES,SQ_BUSY_CYCLES,SQ_WAVE_CYCLES,SQ_INSTS_VALU,SQ_INSTS_SALU,SQ_ACTIVE_INST_VALU,SQ_THREAD_CYCLES_VALU,SQ_INST_CYCLES_SALU \
  --kernel-include-regex "your_kernel" --output-format csv -d ./prof/sq1 -- ./your_app

# SQ stall + instruction mix
rocprofv3 --pmc SQ_WAIT_ANY,SQ_WAIT_INST_ANY,SQ_ACTIVE_INST_ANY,SQ_INSTS_VMEM_RD,SQ_INSTS_VMEM_WR,SQ_INSTS_LDS,SQ_INSTS_SMEM,SQ_INSTS_FLAT \
  --kernel-include-regex "your_kernel" --output-format csv -d ./prof/sq2 -- ./your_app

# GRBM
rocprofv3 --pmc GRBM_GUI_ACTIVE,GRBM_COUNT \
  --kernel-include-regex "your_kernel" --output-format csv -d ./prof/grbm -- ./your_app

# L2 cache hit/miss
rocprofv3 --pmc TCC_HIT,TCC_MISS \
  --kernel-include-regex "your_kernel" --output-format csv -d ./prof/l2hit -- ./your_app

# L2 latency
rocprofv3 --pmc TCC_EA0_RDREQ_LEVEL,TCC_EA0_WRREQ_LEVEL \
  --kernel-include-regex "your_kernel" --output-format csv -d ./prof/l2lat -- ./your_app

# MFMA
rocprofv3 --pmc SQ_INSTS_VALU_MFMA_MOPS_F16,SQ_INSTS_VALU_MFMA_MOPS_BF16,SQ_INSTS_VALU_MFMA_MOPS_F32,SQ_INSTS_VALU_MFMA_MOPS_F64,SQ_VALU_MFMA_BUSY_CYCLES \
  --kernel-include-regex "your_kernel" --output-format csv -d ./prof/mfma -- ./your_app
```

### Compute Metrics

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| GPU Busy % | `100 × GRBM_GUI_ACTIVE / GRBM_COUNT` | Overall GPU utilization |
| VALU Busy % | `100 × SQ_ACTIVE_INST_VALU × 4 / 1024 / GRBM_GUI_ACTIVE` | Vector ALU pipe utilization |
| SALU Busy % | `100 × SQ_INST_CYCLES_SALU × 4 / 1024 / GRBM_GUI_ACTIVE` | Scalar ALU pipe utilization |
| VALU Thread Util % | `100 × SQ_THREAD_CYCLES_VALU / (SQ_ACTIVE_INST_VALU × 64)` | Thread divergence: 100%=all 64 lanes active |
| MFMA Util % | `100 × SQ_VALU_MFMA_BUSY_CYCLES / (GRBM_GUI_ACTIVE × 256 × 4)` | Matrix core utilization |
| Wave Exec % | `100 × SQ_ACTIVE_INST_ANY / SQ_WAVE_CYCLES` | % time wave is executing |
| Wave Wait % | `100 × SQ_WAIT_ANY / SQ_WAVE_CYCLES` | % time wave stalled on data dependency |
| Wave Issue Wait % | `100 × SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES` | % time stalled on instruction issue |

Exec% + Wait% + IssueWait% ≈ 100%: shows where wave time goes.

FP4 MFMA (`mfma_f32_16x16x128_f8f6f4`) is NOT counted by `MFMA_MOPS_F16/BF16/F32/F64`. Use `SQ_VALU_MFMA_BUSY_CYCLES`.

### L2 Cache (TCC)

MI355X: 8 XCDs, each with its own L2 (TCC) channels. CSV = summed across all 8 XCDs. JSON with `DIMENSION_XCC[0:7]` for per-XCD breakdown.

| Metric | Formula |
|--------|---------|
| L2 Hit Rate % | `100 × TCC_HIT / (TCC_HIT + TCC_MISS)` |
| L2 Read Latency | `TCC_EA0_RDREQ_LEVEL / TCC_EA0_RDREQ` cycles |
| L2 Write Latency | `TCC_EA0_WRREQ_LEVEL / TCC_EA0_WRREQ` cycles |

### L1 Cache (TCP)

MI355X: 8 XCDs × 32 CUs = 256 L1 caches. CSV = summed across all 256 instances. JSON with `DIMENSION_XCC[0:7]` for per-XCD.

| Metric | Formula |
|--------|---------|
| L1 Read Hit Rate % | `100 × (1 − TCP_TCC_READ_REQ / TCP_TOTAL_READ)` |
| L1 Avg Wave Latency | `TCP_TCP_LATENCY / TCP_TA_TCP_STATE_READ` cycles |
| L1→L2 Read Latency | `TCP_TCC_READ_REQ_LATENCY / TCP_TCC_READ_REQ` cycles |

### Instruction Mix (per wave)

```
VALU/wave = SQ_INSTS_VALU / SQ_WAVES    SALU/wave = SQ_INSTS_SALU / SQ_WAVES
VMEM_RD   = SQ_INSTS_VMEM_RD / SQ_WAVES VMEM_WR   = SQ_INSTS_VMEM_WR / SQ_WAVES
LDS       = SQ_INSTS_LDS / SQ_WAVES     SMEM      = SQ_INSTS_SMEM / SQ_WAVES
```

---

## Bottleneck Diagnosis

```
1. GPU Busy < 50%
   → Launch overhead or insufficient parallelism.

2. Wave Wait % > 60%
   ├─ VMEM_RD+WR >> LDS   → VRAM MEMORY BOUND
   ├─ LDS >> VMEM          → LDS PRESSURE
   └─ Both moderate        → LATENCY BOUND

3. VALU Busy > 70%  → COMPUTE BOUND

4. L2 Hit Rate < 50% → CACHE THRASHING

5. MFMA Util < 30% (GEMM kernels) → Matrix cores underutilized

6. VALU Thread Util < 80% → Thread divergence

7. None dominant → OCCUPANCY BOUND
   Confirm with: SPI_RA_VGPR_SIMD_FULL_CSN, SPI_RA_LDS_CU_FULL_CSN, SPI_RA_WVLIM_STALL_CSN
```

## Optimization Actions

| Bottleneck | Actions |
|------------|---------|
| VRAM Memory Bound | LDS tiling, cooperative load, vectorized loads (uint4), prefetch/double buffer |
| LDS Pressure | Pad LDS for bank conflicts, swizzle access, move data to registers |
| Latency Bound | Double buffering, increase ILP, unroll loops |
| Compute Bound | Use MFMA, packed math (fp16/bf16), strength reduction |
| Occupancy Bound | Reduce VGPR, tune workgroup size, reduce LDS |
| Cache Thrashing | Tile for L2 residency, change access order, shrink working set |
| Low MFMA Util | Prefetch pipeline, overlap VMEM+MFMA via `__builtin_amdgcn_sched_group_barrier` |

---

## Common Pitfalls

1. **DurationNs ≠ wall-clock time** — For splitK kernels (multiple dispatches per op), sum dispatch durations or use benchmark timing.

2. **gfx950 TCC counter names** — Use `TCC_EA0_RDREQ` not `TCC_EA_RDREQ`. Verify with `rocprofv3-avail list --pmc | grep TCC`.

3. **Counter slot limits** — Keep ≤4 same-block counters per pass. Extras are silently dropped.

4. **Multi-pass must be separate invocations** — Multiple `--pmc` in one command only saves last pass.

5. **FP4 MFMA not in MFMA_MOPS** — Use `SQ_VALU_MFMA_BUSY_CYCLES` for FP4 utilization.

6. **Wave occupancy is not "higher is better"** — More waves = fewer VGPRs per wave = possible scratch spilling. Target minimum needed to hide latency.

---

## Memory Hierarchy (MI355X)

```
           ┌──────────┐
           │  Kernel   │ ← timing (100x) → TFLOPS
           └──┬────┬───┘
     compute  │    │  memory
  ┌───────────▼┐  ┌▼───────────┐
  │  CU (256)  │  │  L1 (TCP)  │
  │ VALU Busy% │  └─────┬──────┘
  │ MFMA Util% │   miss │
  │ WaveExec%  │  ┌─────▼──────┐
  │ WaveWait%  │  │  L2 (TCC)  │ ← TCC_HIT/MISS
  └────────────┘  └─────┬──────┘
                   miss │
                  ┌─────▼──────┐
                  │   HBM3e    │ ← TCC_EA0_RDREQ/WRREQ → Load/Store → BW
                  │   8 TB/s   │ → Arithmetic Intensity = FLOPS / HBM_bytes
                  └────────────┘
```
