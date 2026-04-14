# CDNA4 LDS / Data Share Instructions (gfx950)

Source: AMD CDNA4 ISA — Chapter 11 (Data Share Operations) and §12.12 (LDS Instructions).

## Overview

**Local data share (LDS)** is an on-chip, low-latency scratchpad with much higher effective bandwidth than uncached global memory. Work-items in a work-group share it; it supports full gather/read and scatter/write, unlike read-only caches.

**Capacity and banking (per compute unit):** 160 kB segmented into **64 banks** of 640 dwords; each bank is **32 bits** wide. Dwords are placed in banks serially, but all banks can load or store in parallel. A work-group may use up to 160 kB. Reads across a wavefront are issued over **four cycles** in a waterfall pattern.

**Throughput and conflicts:** Up to 32 concurrent read or write operations (each nominally 32 bits; **read2/write2** can be 64 bits per op). If multiple accesses hit the **same bank** in the same cycle, **bank conflicts** occur; for indexed and atomics, hardware **serializes** those accesses, reducing effective bandwidth.

**Dataflow:** Typical path is global → VGPR → LDS, or LDS → VGPR → global; some paths can fill LDS while bypassing VGPRs. **LDS atomics** execute in LDS hardware.

**Indexed access:** `LDS_Addr = LDS_BASE + VGPR[ADDR] + {OFFSET1, OFFSET0}` (byte offsets). Double-address ops use two offsets; **ST64** variants scale each offset as an index × 64 elements (256 B per 32-bit element slot, 512 B per 64-bit slot in ISA pseudocode).

**Atomics (§11.3):** `ADDR` is a **dword** address; 64-bit data uses **paired VGPRs**. Compare-swap: **`DATA` = compare, `DATA2` = new value** — order is **opposite** to `BUFFER_ATOMIC_CMPSWAP`. FP atomics: **NaN/Inf/denorm** per `MODE.FP_DENORM`; rounding **round-to-nearest-even** where applicable.

**M0:** Initialize before use; `M0[16:0]` = LDS segment byte size for clamping; **`0xFFFFFFFF`** disables clamping.

**MFMA transpose (`DS_READ_*_TR_*`):** **`EXEC` = all 1s**; LDS address **aligned to element size**; 64b+ ops use **even-aligned** VGPR pairs except **`DS_READ_B96_TR_B6`**.

**`RTN` mnemonics:** §12.12 — **`RTN`** explicitly denotes returning the **pre-operation** memory value to `VDST`. Non-`RTN` twins often show identical `RETURN_DATA` pseudocode; follow your assembler/encoding reference for `VDST` writes.

---

## Operand summary (DS encoding)

| Field | Role |
|-------|------|
| `VDST` | Destination VGPR |
| `ADDR` | Address VGPR (indexed byte path; dword address for atomics) |
| `DATA` / `DATA2` | Source VGPRs (`DATA0` / `DATA1` in encoding) |
| `OFFSET0`, `OFFSET1` | 8-bit immediates |
| `M0` | Implied segment limit |

**Syntax legend:** `v_dst`, `v_addr`, `v_d0`, `v_d1`; offsets `off0`, `off1`; `{off}` = combined 16-bit byte offset unless noted.

---
## DS Read Instructions

| Instruction | Syntax | Description | Size | Notes |
|---|---|---|---|---|
| `DS_READ_B32` | `v_dst, v_addr` + `{off}` | Load dword from LDS | 32 b |  |
| `DS_READ_B64` | `v_dst` (64b), `v_addr` + `{off}` | Load qword (two dwords) | 64 b |  |
| `DS_READ_B96` | `v_dst` (96b), `v_addr` + `{off}` | Load three dwords | 96 b |  |
| `DS_READ_B128` | `v_dst` (128b), `v_addr` + `{off}` | Load four dwords | 128 b |  |
| `DS_READ2_B32` | `v_dst` (64b), `v_addr`; dword slots `off0*4`, `off1*4` | Two dword loads | 2×32 b |  |
| `DS_READ2_B64` | `v_dst` (128b), `v_addr`; qword slots `off0*8`, `off1*8` | Two qword loads | 2×64 b |  |
| `DS_READ2ST64_B32` | `v_dst` (64b), `v_addr`; + `off*256` B indexing | Two dword loads, ×64-element stride | 2×32 b | ST64 |
| `DS_READ2ST64_B64` | `v_dst` (128b), `v_addr`; + `off*512` B indexing | Two qword loads, ST64 | 2×64 b |  |
| `DS_READ_I8` | `v_dst, v_addr` | Signed byte → sign-extended dword | 8→32 b |  |
| `DS_READ_U8` | `v_dst, v_addr` | Byte → zero-extended dword | 8→32 b |  |
| `DS_READ_I16` | `v_dst, v_addr` | Signed word → sign-extended dword | 16→32 b |  |
| `DS_READ_U16` | `v_dst, v_addr` | Word → zero-extended dword | 16→32 b |  |
| `DS_READ_U8_D16` | `v_dst, v_addr` | Byte → zero-extended to **low 16b** | 8→16 b | `[31:16]` preserved |
| `DS_READ_U8_D16_HI` | `v_dst, v_addr` | Byte → zero-extended to **high 16b** | 8→16 b | Low half preserved |
| `DS_READ_I8_D16` | `v_dst, v_addr` | Byte → sign-extended to low 16b | 8→16 b |  |
| `DS_READ_I8_D16_HI` | `v_dst, v_addr` | Byte → sign-extended to high 16b | 8→16 b |  |
| `DS_READ_U16_D16` | `v_dst, v_addr` | Word to low 16b | 16 b | Upper preserved |
| `DS_READ_U16_D16_HI` | `v_dst, v_addr` | Word to high 16b | 16 b |  |
| `DS_READ_ADDTID_B32` | `v_dst`; addr = `{off}+M0[15:0]+laneId*4` | Dword load; immediate base + lane offset | 32 b | Not `CalcDsAddr`+`v_addr` |
| `DS_READ_B64_TR_B4` | `v_dst`, LDS addr | Transpose matrix load, **4b** elements | 64 b/lane | MFMA; 2-instr typical |
| `DS_READ_B64_TR_B8` | `v_dst`, LDS addr | Transpose, **8b** elements | 64 b/lane |  |
| `DS_READ_B64_TR_B16` | `v_dst`, LDS addr | Transpose, **16b** elements | 64 b/lane |  |
| `DS_READ_B96_TR_B6` | `v_dst` (3 VGPRs), LDS addr | Transpose, **6b** elements | 96 b/lane | No even-VGPR rule |

## DS Write Instructions

| Instruction | Syntax | Description | Size | Notes |
|---|---|---|---|---|
| `DS_WRITE_B32` | `v_addr`, `v_d0` + `{off}` | Store dword | 32 b |  |
| `DS_WRITE_B64` | `v_addr`, `v_d0` (64b) + `{off}` | Store qword | 64 b |  |
| `DS_WRITE_B96` | `v_addr`, `v_d0` (96b) + `{off}` | Store 96 bits | 96 b |  |
| `DS_WRITE_B128` | `v_addr`, `v_d0` (128b) + `{off}` | Store 128 bits | 128 b |  |
| `DS_WRITE_B8` | `v_addr`, `v_d0` | Store low 8 bits | 8 b |  |
| `DS_WRITE_B16` | `v_addr`, `v_d0` | Store low 16 bits | 16 b |  |
| `DS_WRITE_B8_D16_HI` | `v_addr`, `v_d0` | Store from `v_d0[23:16]` | 8 b |  |
| `DS_WRITE_B16_D16_HI` | `v_addr`, `v_d0` | Store from `v_d0[31:16]` | 16 b |  |
| `DS_WRITE2_B32` | `v_addr`, `v_d0`, `v_d1`; `off0*4`, `off1*4` | Two dword stores | 2×32 b |  |
| `DS_WRITE2_B64` | `v_addr`, two 64b; `off0*8`, `off1*8` | Two qword stores | 2×64 b |  |
| `DS_WRITE2ST64_B32` | `v_addr`, `v_d0`, `v_d1`; ST64 dword stride | Two dword stores | 2×32 b | ×64 elements |
| `DS_WRITE2ST64_B64` | `v_addr`, two 64b; ST64 stride | Two qword stores | 2×64 b |  |
| `DS_WRITE_ADDTID_B32` | `v_d0`; addr = `{off}+M0[15:0]+laneId*4` | Store dword; imm base + lane | 32 b |  |

## DS Read-Write (Atomic) Instructions

Operands use `CalcDsAddr(v_addr, off0, off1)` except **ADD TID** / **CONDXCHG** / **CONSUME** / **APPEND** special cases. **`v_dst`**: returned pre-op value where the ISA defines `RETURN_DATA`.

| Instruction | Syntax | Description | Size | Notes |
|---|---|---|---|---|
| `DS_ADD_U32` | `v_dst, v_addr, v_d0` + offs | MEM += `v_d0` (u32) | 32 b | RMW |
| `DS_SUB_U32` | `v_dst, v_addr, v_d0` + offs | MEM -= `v_d0` (u32) | 32 b |  |
| `DS_RSUB_U32` | `v_dst, v_addr, v_d0` + offs | MEM = `v_d0` - MEM (u32) | 32 b |  |
| `DS_INC_U32` | `v_dst, v_addr, v_d0` + offs | Inc; wrap to 0 if ≥ `v_d0` | 32 b |  |
| `DS_DEC_U32` | `v_dst, v_addr, v_d0` + offs | Dec; wrap per ISA | 32 b |  |
| `DS_MIN_I32` | `v_dst, v_addr, v_d0` + offs | Signed min into MEM | 32 b |  |
| `DS_MAX_I32` | `v_dst, v_addr, v_d0` + offs | Signed max into MEM | 32 b |  |
| `DS_MIN_U32` | `v_dst, v_addr, v_d0` + offs | Unsigned min | 32 b |  |
| `DS_MAX_U32` | `v_dst, v_addr, v_d0` + offs | Unsigned max | 32 b |  |
| `DS_AND_B32` | `v_dst, v_addr, v_d0` + offs | MEM &= `v_d0` | 32 b |  |
| `DS_OR_B32` | `v_dst, v_addr, v_d0` + offs | MEM |= `v_d0` | 32 b |  |
| `DS_XOR_B32` | `v_dst, v_addr, v_d0` + offs | MEM ^= `v_d0` | 32 b |  |
| `DS_MSKOR_B32` | `v_dst, v_addr, v_d0` mask, `v_d1` or + offs | (MEM & ~mask) | or_bits | 32 b |  |
| `DS_ADD_RTN_U32` | `v_dst, v_addr, v_d0` + offs | ADD u32; explicit pre-op return | 32 b | RTN |
| `DS_SUB_RTN_U32` | `v_dst, v_addr, v_d0` + offs | SUB; pre-op return | 32 b | RTN |
| `DS_RSUB_RTN_U32` | `v_dst, v_addr, v_d0` + offs | RSUB; pre-op return | 32 b | RTN |
| `DS_INC_RTN_U32` | `v_dst, v_addr, v_d0` + offs | INC; pre-op return | 32 b | RTN |
| `DS_DEC_RTN_U32` | `v_dst, v_addr, v_d0` + offs | DEC; pre-op return | 32 b | RTN |
| `DS_MIN_RTN_I32` | `v_dst, v_addr, v_d0` + offs | MIN signed; pre-op | 32 b | RTN |
| `DS_MAX_RTN_I32` | `v_dst, v_addr, v_d0` + offs | MAX signed; pre-op | 32 b | RTN |
| `DS_MIN_RTN_U32` | `v_dst, v_addr, v_d0` + offs | MIN unsigned; pre-op | 32 b | RTN |
| `DS_MAX_RTN_U32` | `v_dst, v_addr, v_d0` + offs | MAX unsigned; pre-op | 32 b | RTN |
| `DS_AND_RTN_B32` | `v_dst, v_addr, v_d0` + offs | AND; pre-op | 32 b | RTN |
| `DS_OR_RTN_B32` | `v_dst, v_addr, v_d0` + offs | OR; pre-op | 32 b | RTN |
| `DS_XOR_RTN_B32` | `v_dst, v_addr, v_d0` + offs | XOR; pre-op | 32 b | RTN |
| `DS_MSKOR_RTN_B32` | `v_dst, v_addr, v_d0`, `v_d1` + offs | MSKOR; pre-op | 32 b | RTN |
| `DS_CMPST_B32` | `v_dst, v_addr, v_cmp, v_new` + offs | If MEM==cmp then MEM=new | 32 b | vs buffer atomic order |
| `DS_CMPST_RTN_B32` | `v_dst, v_addr, v_cmp, v_new` + offs | Compare-swap; pre-op | 32 b |  |
| `DS_CMPST_F32` | `v_dst, v_addr, v_cmp, v_new` + offs | FP32 compare-swap | 32 b |  |
| `DS_CMPST_RTN_F32` | `v_dst, v_addr, v_cmp, v_new` + offs | FP32 cmpswap; pre-op | 32 b |  |
| `DS_MIN_F32` | `v_dst, v_addr, v_d0` + offs | FP32 min into MEM | 32 b | NaN/Inf/denorm |
| `DS_MAX_F32` | `v_dst, v_addr, v_d0` + offs | FP32 max | 32 b | NaN/Inf/denorm |
| `DS_MIN_RTN_F32` | `v_dst, v_addr, v_d0` + offs | FP32 min; pre-op | 32 b | RTN |
| `DS_MAX_RTN_F32` | `v_dst, v_addr, v_d0` + offs | FP32 max; pre-op | 32 b | RTN |
| `DS_ADD_F32` | `v_dst, v_addr, v_d0` + offs | MEM += FP32 | 32 b | NaN/Inf/denorm |
| `DS_ADD_RTN_F32` | `v_dst, v_addr, v_d0` + offs | FP32 add; pre-op | 32 b | RTN |
| `DS_PK_ADD_F16` | `v_dst, v_addr, v_d0` | Packed **two F16** add to MEM | 2×16 in 32 |  |
| `DS_PK_ADD_RTN_F16` | `v_dst, v_addr, v_d0` | Packed F16 add; pre-op dword | 32 b | RTN |
| `DS_PK_ADD_BF16` | `v_dst, v_addr, v_d0` | Packed **two BF16** add | 2×16 in 32 |  |
| `DS_PK_ADD_RTN_BF16` | `v_dst, v_addr, v_d0` | Packed BF16 add; pre-op | 32 b | RTN |
| `DS_WRXCHG_RTN_B32` | `v_dst, v_addr, v_d0` + offs | Swap MEM ↔ `v_d0` | 32 b |  |
| `DS_WRXCHG2_RTN_B32` | `v_dst` (64b), `v_addr`, `v_d0`, `v_d1` | Two dword swaps | 2×32 b | Offsets ×4 |
| `DS_WRXCHG2ST64_RTN_B32` | `v_dst` (64b), `v_addr`, `v_d0`, `v_d1` + ST64 | Two dword swaps, ST64 | 2×32 b |  |
| `DS_WRAP_RTN_B32` | `v_dst, v_addr, v_sub, v_add` + offs | If MEM≥sub MEM-=sub else MEM+=add | 32 b | Ring buffer |
| `DS_ADD_U64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | MEM += u64 | 64 b |  |
| `DS_SUB_U64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | MEM -= u64 | 64 b |  |
| `DS_RSUB_U64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | MEM = src - MEM | 64 b |  |
| `DS_INC_U64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | 64b inc/wrap | 64 b |  |
| `DS_DEC_U64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | 64b dec/wrap | 64 b |  |
| `DS_MIN_I64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | Signed min | 64 b |  |
| `DS_MAX_I64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | Signed max | 64 b |  |
| `DS_MIN_U64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | Unsigned min | 64 b |  |
| `DS_MAX_U64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | Unsigned max | 64 b |  |
| `DS_AND_B64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | AND | 64 b |  |
| `DS_OR_B64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | OR | 64 b |  |
| `DS_XOR_B64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | XOR | 64 b |  |
| `DS_MSKOR_B64` | `v_dst` (64), `v_addr`, mask, or + offs | Masked OR | 64 b |  |
| `DS_ADD_RTN_U64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | ADD; pre-op | 64 b | RTN |
| `DS_SUB_RTN_U64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | SUB; pre-op | 64 b | RTN |
| `DS_RSUB_RTN_U64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | RSUB; pre-op | 64 b | RTN |
| `DS_INC_RTN_U64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | INC; pre-op | 64 b | RTN |
| `DS_DEC_RTN_U64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | DEC; pre-op | 64 b | RTN |
| `DS_MIN_RTN_I64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | MIN signed; pre-op | 64 b | RTN |
| `DS_MAX_RTN_I64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | MAX signed; pre-op | 64 b | RTN |
| `DS_MIN_RTN_U64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | MIN u64; pre-op | 64 b | RTN |
| `DS_MAX_RTN_U64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | MAX u64; pre-op | 64 b | RTN |
| `DS_AND_RTN_B64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | AND; pre-op | 64 b | RTN |
| `DS_OR_RTN_B64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | OR; pre-op | 64 b | RTN |
| `DS_XOR_RTN_B64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | XOR; pre-op | 64 b | RTN |
| `DS_MSKOR_RTN_B64` | `v_dst` (64), `v_addr`, `v_d0`, `v_d1` + offs | MSKOR; pre-op | 64 b | RTN |
| `DS_CMPST_B64` | `v_dst` (64), `v_addr`, cmp, new + offs | 64b compare-swap | 64 b | vs buffer atomic |
| `DS_CMPST_RTN_B64` | `v_dst` (64), `v_addr`, cmp, new + offs | 64b cmpswap; pre-op | 64 b |  |
| `DS_CMPST_F64` | `v_dst` (64), `v_addr`, cmp, new + offs | FP64 cmpswap | 64 b |  |
| `DS_CMPST_RTN_F64` | `v_dst` (64), `v_addr`, cmp, new + offs | FP64 cmpswap; pre-op | 64 b |  |
| `DS_MIN_F64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | FP64 min | 64 b |  |
| `DS_MAX_F64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | FP64 max | 64 b |  |
| `DS_MIN_RTN_F64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | FP64 min; pre-op | 64 b | RTN |
| `DS_MAX_RTN_F64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | FP64 max; pre-op | 64 b | RTN |
| `DS_ADD_F64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | MEM += FP64 | 64 b |  |
| `DS_ADD_RTN_F64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | FP64 add; pre-op | 64 b | RTN |
| `DS_WRXCHG_RTN_B64` | `v_dst` (64), `v_addr`, `v_d0` (64) + offs | 64b swap | 64 b |  |
| `DS_WRXCHG2_RTN_B64` | `v_dst` (128), `v_addr`, two 64b | Two 64b swaps | 2×64 b | Offsets ×8 |
| `DS_WRXCHG2ST64_RTN_B64` | `v_dst` (128), `v_addr`, two 64b + ST64 | Two 64b swaps ST64 | 2×64 b |  |
| `DS_CONSUME` | `v_dst`, offset/M0 | MEM -= popcount(EXEC) | 32 b | GDS/LDS addr rules; pre-op |
| `DS_APPEND` | `v_dst`, offset/M0 | MEM += popcount(EXEC) | 32 b | GDS/LDS addr rules; pre-op |
| `DS_CONDXCHG32_RTN_B64` | `v_dst` (2×32), `v_addr`, `v_d0` (64), offs | Two cond. writes if MSB each half | 2×32 b | Addr &0xfff8; clears MSB on write |

## DS Permute Instructions

| Instruction | Syntax | Description | Size | Notes |
|---|---|---|---|---|
| `DS_SWIZZLE_B32` | `v_dst, v_d0`, imm pattern | Cross-lane dword swizzle | 32 b | No LDS bank access; FFT / rotate / 4-lane / 32-lane; bad lane → 0 |
| `DS_PERMUTE_B32` | `v_dst, v_addr, v_d0` + off | Forward permute (scatter to lanes) | 32 b | No LDS alloc; `v_addr` ≡ lane×4; highest src wins |
| `DS_BPERMUTE_B32` | `v_dst, v_addr, v_d0` + off | Backward permute (gather) | 32 b | EXEC on src; disabled → 0 |

## DS Special Instructions

| Instruction | Syntax | Description | Size | Notes |
|---|---|---|---|---|
| `DS_NOP` | (none) | No operation | — |  |


---

**Total:** 126 distinct `DS_*` mnemonics in §12.12 (CDNA4 ISA text).
