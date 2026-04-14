---
name: fp4-gemm-m1-optimize
description: Optimizes FP4 (MXFP4 e2m1) GEMM kernels for small batch sizes (M=1..16) on AMD MI355X (gfx950). Covers MFMA intrinsics, preshuffled data layouts, scale shuffle addressing, CK compiler flags, and the full optimization path from scalar baseline to 2.51 TFLOPS. Use when optimizing FP4 GEMM decode kernels, writing MFMA-based HIP kernels, or tuning aiter's gemm_a4w4 for small M on CDNA4.
---

# FP4 GEMM M=1 Optimization Skill

## Problem

FP4 (MXFP4 e2m1) GEMM with M=1 (decode-phase single-token) is memory-bound and underutilized on MI355X. Default aiter paths (ASM/Unified) achieve only 1.4 TFLOPS. CK with optimal config reaches 4.2 TFLOPS.

## Best Result

**v13 preshuffled: 2.51 TFLOPS** — 62.8x over scalar baseline, 73% faster than ASM default, 60% of CK best. Precision: max_diff=0.000000 (exact match to golden reference).

---

## Optimization Path (each step verified for precision)

### Step 1: Scalar Baseline → 0.04 TFLOPS

Standard scalar FP4 dequant kernel. Each thread unpacks fp4x2 bytes into float, multiplies, accumulates. Serves as correctness golden reference.

### Step 2: Vectorized Loads → 0.29 TFLOPS (7.3x)

- Replace byte-by-byte load with `uint32_t` dword loads (4x fewer load ops)
- Use all 256 threads for K-parallel reduction (not just 128)
- Warp shuffle `__shfl_down` for cross-thread reduction

### Step 3: MFMA Instruction → 1.42 TFLOPS (4.9x)

Replace scalar FP4 dequant+multiply with single MFMA instruction:

```cpp
c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
    a_data, b_data, c,
    4,        // cbsz = FP4 for A
    4,        // blgp = FP4 for B
    0,        // opsel_a
    scale_a,  // E8M0 scale byte for A
    0,        // opsel_b
    scale_b   // E8M0 scale byte for B
);
```

**Critical precision findings:**

1. **MFMA output mapping**: For 16x16x128 FP4, lane `l` (0..15) holds `C[l, ...]` in `c[0]`. Only `c[0]` contains valid output for M=1. Output: `D[n_start + lane] = bf16(c[0])` for `lane < 16`.

2. **Scale must be per-lane single byte, not packed 4-byte**: Each lane provides ONE E8M0 scale byte corresponding to its K-group. NOT all 4 K-group scales packed.

```cpp
// WRONG: all lanes get same 4-byte packed scale
int32_t scale = ScaleA[0] | (ScaleA[1]<<8) | (ScaleA[2]<<16) | (ScaleA[3]<<24);

// CORRECT: each lane gets its K-group's scale byte
int32_t scale = (int32_t)ScaleA[ki * 4 + k_row_group];
```

### Step 4: 4-Wave K-Split + LDS Reduce → 2.08 TFLOPS (1.46x)

- 256 threads = 4 waves, each processes K/4 elements independently
- LDS reduction across waves (64 bytes, no atomicAdd needed)
- Single kernel launch, no temp buffer allocation

### Step 5: K-Loop Unroll + Preshuffled B → 2.51 TFLOPS (1.17x)

**Preshuffled B data** (`shuffle_weight(w, layout=(16,16))`) gives coalesced memory access:

```
B flat offset = n_block * stride_B * 16
              + k_block * 512
              + k_half * 256
              + m_index * 16       // 16 lanes read consecutive 16-byte chunks!
```

Where `k_block = ki*2 + k_row_group/2`, `k_half = k_row_group % 2`.

**Shuffled scale** (`e8m0_shuffle`) addressing formula:

```cpp
__device__ int shuffled_scale_offset(int r, int c, int s_n) {
    int i0 = r / 32, i1 = (r%32) / 16, i2 = r % 16;
    int j0 = c / 8, j1 = (c%8) / 4, j2 = c % 4;
    return i0 * 32 * s_n + 256*j0 + 64*j2 + 4*i2 + 2*j1 + i1;
}
```

**CK compiler flags** that improve instruction scheduling:

```bash
hipcc -O3 \
  -mllvm -amdgpu-early-inline-all=true \
  -mllvm -amdgpu-function-calls=false \
  -mllvm -enable-post-misched=0 \
  -fno-gpu-rdc
```

- `-enable-post-misched=0`: Preserves source-order instruction scheduling
- `-amdgpu-early-inline-all=true`: Eliminates function call overhead
- `-amdgpu-function-calls=false`: Forces inlining of all device functions

---

## MFMA 16x16x128 FP4 Lane Mapping (gfx950)

```
Lane l (0..63): M_index = l % 16, K_row_group = l / 16 (0..3)

Each lane holds 4 VGPRs (v0-v3), each = 4 bytes = 8 FP4 (fp4x2 packed)
Total per lane: 32 FP4 for K range [K_row_group*32 .. +31]

For A[16×128]: all lanes broadcast row 0 (M=1)
For B[128×16]: lane l loads B[n_start + m_index, K_row_group*32..+31]

Output C[16×16]: lane l holds c[0] = C[l % 16, ...]
  → For M=1: only c[0] of lanes 0-15 has valid output
```

---

## Preshuffled Data Layouts

### B Data: `shuffle_weight(w, layout=(16,16))` on uint8

**Python transform:**

```python
# w shape: [N, K_bytes] where K_bytes = K/2 (fp4x2 packed)
BN, BK_bytes, K_elem = 16, 32, 16  # layout=(16,16) → BK = 16*2 = 32 bytes
x = w.view(-1, N//BN, BN, K_bytes//BK_bytes, BK_bytes//K_elem, K_elem)
#  shape: [batch, N//16, 16(=n_local), K_bytes//32(=k_block), 2(=k_half), 16(=k_local)]
x = x.permute(0, 1, 3, 4, 2, 5).contiguous()
#  shape: [batch, N//16, K_bytes//32, 2, 16(=n_local), 16(=k_local)]
x = x.view(-1, N, K_bytes)
```

**Address mapping** — original `B[n, k_byte]` maps to shuffled flat offset:

```
n_block  = n / 16           (which 16-row N block)
n_local  = n % 16           (row within block, 0..15)
k_block  = k_byte / 32      (which 32-byte K block)
k_half   = (k_byte % 32) / 16   (0 or 1, which half of 32-byte block)
k_local  = k_byte % 16      (byte within 16-byte chunk, 0..15)

flat_offset = n_block * K_bytes * 16    // start of this N-block
            + k_block * 512             // 32-byte K block = 2 halves × 16 n_local × 16 k_local
            + k_half * 256              // which half
            + n_local * 16              // which N-row within block
            + k_local                   // which byte
```

**For MFMA lane loading** (m_index, k_row_group, k_iter):

```
k_byte = k_iter * 64 + k_row_group * 16    (128 FP4 = 64 bytes per K-iter)
k_block = k_iter * 2 + k_row_group / 2
k_half = k_row_group % 2

b_offset = n_block * stride_B * 16         // stride_B = K_bytes per row
         + k_block * 512
         + k_half * 256
         + m_index * 16                    // m_index = lane % 16
```

**Why coalesced**: 16 lanes (m_index=0..15) read addresses `base + 0*16, base + 1*16, ..., base + 15*16` = 256 consecutive bytes = perfectly aligned to cache lines.

**Incremental stepping** in K-loop (no div/mod needed):

```cpp
int b_base = n_block * stride_B * 16
           + (k_row_group/2) * 512
           + (k_row_group%2) * 256
           + m_index * 16;
int b_off = b_base + k_begin * 1024;   // each k_iter advances by 2 k_blocks * 512 = 1024
// In loop: b_off += 1024;
```

### Scale: `e8m0_shuffle` (6-dim permute)

**Python transform:**

```python
# scale shape: [s_m, s_n] where s_m = N (or M_pad), s_n = K/32 (padded to multiple of 8)
scale = scale.view(s_m//32, 2, 16, s_n//8, 2, 4)
#  dims: [i0, i1, i2, j0, j1, j2]
#  i0 = r/32, i1 = (r%32)/16, i2 = r%16
#  j0 = c/8, j1 = (c%8)/4, j2 = c%4
scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
#  dims: [i0, j0, j2, i2, j1, i1]
scale = scale.view(s_m, s_n)  # flatten back to original shape
```

**Address mapping** — original `scale[r, c]` maps to shuffled flat offset:

```
i0 = r / 32           i1 = (r % 32) / 16     i2 = r % 16
j0 = c / 8            j1 = (c % 8) / 4       j2 = c % 4

inner_offset = 256 * j0 + 64 * j2 + 4 * i2 + 2 * j1 + i1
flat_offset = i0 * 32 * s_n + inner_offset
```

**HIP device function:**

```cpp
__device__ __forceinline__ int shuffled_scale_offset(int r, int c, int s_n) {
    int i0 = r / 32, i1 = (r % 32) / 16, i2 = r % 16;
    int j0 = c / 8, j1 = (c % 8) / 4, j2 = c % 4;
    return i0 * 32 * s_n + 256 * j0 + 64 * j2 + 4 * i2 + 2 * j1 + i1;
}
```

**Simplified for ScaleA with r=0 (M=1)**: i0=i1=i2=0 →

```cpp
__device__ __forceinline__ int scale_off_r0(int c) {
    return 256 * (c >> 3) + 64 * (c & 3) + 2 * ((c >> 2) & 1);
}
```

**For ScaleB, precompute per-lane row constants once:**

```cpp
int sb_i0 = b_n >> 5;            // b_n / 32
int sb_i1 = (b_n >> 4) & 1;     // (b_n % 32) / 16
int sb_i2 = b_n & 15;           // b_n % 16
// In loop: ScaleB[scale_off(sb_i0, sb_i1, sb_i2, k_scale_idx, stride_SB)]
```

**stride_SB** is the padded column count of the shuffled scale tensor (NOT K/32). Obtain from Python: `w_sc_s.view(torch.uint8).stride(0)`.

### A Data and ScaleA (M=1 special case)

- **A data**: `quant_func(x, shuffle=True)` with M=1 produces identical bytes as `shuffle=False`. No special handling needed.
- **A scale**: IS shuffled (padded to 256 rows). Use `scale_off_r0(c)` with `stride_SA = x_sc_s.view(torch.uint8).stride(0)`.

### Tensor Shapes (M=1, N=4096, K=7168)

| Tensor | Unshuffled Shape | Shuffled Shape | stride(0) |
|--------|-----------------|----------------|-----------|
| A data | [1, 3584] uint8 | [1, 3584] uint8 | 3584 |
| B data | [4096, 3584] uint8 | [4096, 3584] uint8 | 3584 |
| ScaleA | [1, 224] uint8 | [256, 224] uint8 | 224 |
| ScaleB | [4096, 224] uint8 | [4096, 224] uint8 | 224 |
| Output | [1, 4096] bf16 | same | 4096 |

---

## What CK Does Beyond v13 (remaining 1.7x gap)

1. **Cooperative B loading via LDS**: 256 threads jointly load B tile to LDS, then each thread reads from LDS. Reduces global memory transactions.
2. **Compile-time unrolled pipeline** (`BlockwiseGemmPipeline_v3`): Uses `__builtin_amdgcn_sched_group_barrier` to interleave `buffer_load` and `v_mfma` instructions at precise cycle granularity.
3. **Internal K-split** via CK's `KBatch`: No atomicAdd or extra kernel needed — reduction happens in the epilogue within the same kernel.
4. **`amd_buffer_load`** intrinsic for async VMEM loads with buffer resource descriptors.

---

## Quick Start: Tuning Existing CK/ASM for M=1

Before writing custom kernels, try CK config tuning first:

```python
# Best CK config for M=1, N=4096, K=7168
aiter.gemm_a4w4_blockscale_tune(x, w, x_sc, w_sc, out, kernelId=14, splitK=2)
# → 4.2 TFLOPS (MPerBLOCK=32, NPerBLOCK=128)
```

For Triton path, sweep `NUM_KSPLIT=1` (not split-K) with small block sizes:

```json
{"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
 "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4,
 "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16}
```

---

## Files

- Kernel source: `kernel/f4gemm_m1/f4gemm_m1_v13.cpp`
- Benchmark: `kernel/f4gemm_m1/bench_v13.py`
- Shuffle debug tools: `kernel/f4gemm_m1/debug_shuffle*.py`, `verify_scale_formula.py`
- ISA reference: `skills/isa-reference/04_matrix_mfma.md`
