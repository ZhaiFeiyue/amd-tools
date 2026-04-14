# CDNA4 Scalar Memory (SMEM) — Reference

Source: AMD CDNA4 ISA (Ch. 8 *Scalar Memory Operations*, §12.6 *SMEM Instructions*).

## Overview (Chapter 8)

- **SMEM** moves data between **memory and SGPRs** through the **scalar data cache** (no format conversion for plain load/store).
- **Loads**: 1–16 dwords per instruction; **stores**: 1–4 dwords.
- **Addressing**: `ADDR = SGPR[base] + inst_offset + { M0 or SGPR[offset] or zero }` (bytes; **two LSBs ignored** for dword alignment). **S_SCRATCH_*** uses **64-byte-scaled** scratch addressing (`CalcScalarScratchAddr`).
- **Buffer (S_BUFFER_*)**: **SBASE** is a **4-SGPR-aligned** resource constant; used fields include **base, stride, num_records** (stride used for bounds, not address swizzle).
- **Atomics**: same family as vector atomics; **GLC=1** returns **pre-op** value to **SDATA** (when enabled).
- **S_DCACHE_INV / S_DCACHE_WB**: whole scalar L0; **S_DCACHE_DISCARD(_X2)**: drop **1 (or 2)** dirty **64-byte** lines without writeback (address like store; **6 LSBs** cleared).
- **S_MEMTIME / S_MEMREALTIME**: **64-bit** counters into **SDATA, SDATA+1** (RTC is **100 MHz**).
- **Dependencies**: use **LGKM_CNT** / **`S_WAITCNT`**; clause rules apply (see Ch. 8).

### SMEM operand pattern (encoding summary)

| Role | Meaning |
|------|---------|
| **SDATA** | First SGPR for load **destination** or store **source** (multi-dword spans consecutive SGPRs); atomics also use for **return** when GLC requests pre-op value |
| **SBASE** | **Global/scratch**: SGPR pair base (even). **Buffer**: **4** consecutive SGPRs (**T#/V#** resource) |
| **OFFSET** | **IMM=1**: 21-bit **byte** offset (unsigned for SGPR offset case per opcode notes; **signed** when documented for immediate). **IMM=0**: SGPR or **M0** holding byte offset. **Stores/atomics**: offset must be **imm or M0**, not SGPR (per Ch. 8) |

---

## Global loads (`S_LOAD_DWORD*`)

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `S_LOAD_DWORD` | `SDATA, SBASE, offset` | Load one dword from scalar global addr | **4** (read) |
| `S_LOAD_DWORDX2` | `SDATA, SBASE, offset` | Load two consecutive dwords | **8** |
| `S_LOAD_DWORDX4` | `SDATA, SBASE, offset` | Load four consecutive dwords | **16** |
| `S_LOAD_DWORDX8` | `SDATA, SBASE, offset` | Load eight consecutive dwords | **32** |
| `S_LOAD_DWORDX16` | `SDATA, SBASE, offset` | Load sixteen consecutive dwords | **64** |

---

## Scratch loads (`S_SCRATCH_LOAD_DWORD*`)

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `S_SCRATCH_LOAD_DWORD` | `SDATA, SBASE, offset` | Load dword from **scalar scratch** aperture | **4** |
| `S_SCRATCH_LOAD_DWORDX2` | `SDATA, SBASE, offset` | Load 2 dwords from scratch | **8** |
| `S_SCRATCH_LOAD_DWORDX4` | `SDATA, SBASE, offset` | Load 4 dwords from scratch | **16** |

*Note: SGPR offset is an **unsigned 64-byte** scratch offset when used; immediate offset semantics per §12.6 notes.*

---

## Buffer loads (`S_BUFFER_LOAD_DWORD*`)

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `S_BUFFER_LOAD_DWORD` | `SDATA, SBASE, offset` | Load dword via **buffer resource** | **4** |
| `S_BUFFER_LOAD_DWORDX2` | `SDATA, SBASE, offset` | Load 2 dwords | **8** |
| `S_BUFFER_LOAD_DWORDX4` | `SDATA, SBASE, offset` | Load 4 dwords | **16** |
| `S_BUFFER_LOAD_DWORDX8` | `SDATA, SBASE, offset` | Load 8 dwords | **32** |
| `S_BUFFER_LOAD_DWORDX16` | `SDATA, SBASE, offset` | Load 16 dwords | **64** |

---

## Global stores (`S_STORE_DWORD*`)

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `S_STORE_DWORD` | `SDATA, SBASE, offset` | Store one dword to scalar global addr | **4** (write) |
| `S_STORE_DWORDX2` | `SDATA, SBASE, offset` | Store two dwords | **8** |
| `S_STORE_DWORDX4` | `SDATA, SBASE, offset` | Store four dwords | **16** |

---

## Scratch stores (`S_SCRATCH_STORE_DWORD*`)

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `S_SCRATCH_STORE_DWORD` | `SDATA, SBASE, offset` | Store dword to scratch aperture | **4** |
| `S_SCRATCH_STORE_DWORDX2` | `SDATA, SBASE, offset` | Store 2 dwords | **8** |
| `S_SCRATCH_STORE_DWORDX4` | `SDATA, SBASE, offset` | Store 4 dwords | **16** |

---

## Buffer stores (`S_BUFFER_STORE_DWORD*`)

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `S_BUFFER_STORE_DWORD` | `SDATA, SBASE, offset` | Store dword via buffer resource | **4** |
| `S_BUFFER_STORE_DWORDX2` | `SDATA, SBASE, offset` | Store 2 dwords | **8** |
| `S_BUFFER_STORE_DWORDX4` | `SDATA, SBASE, offset` | Store 4 dwords | **16** |

---

## Scalar L0 cache and time

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `S_DCACHE_INV` | *(no SDATA return)* | Invalidate entire scalar L0 | — |
| `S_DCACHE_WB` | *(no SDATA return)* | Write back dirty scalar L0 | — |
| `S_DCACHE_INV_VOL` | *(no SDATA return)* | Invalidate **volatile** lines in scalar L0 | — |
| `S_DCACHE_WB_VOL` | *(no SDATA return)* | Write back **volatile** dirty lines | — |
| `S_MEMTIME` | `SDATA` (pair) | Read **64-bit** timestamp | **8** (to SGPR pair) |
| `S_MEMREALTIME` | `SDATA` (pair) | Read **64-bit** real-time clock (**100 MHz**) | **8** |
| `S_DCACHE_DISCARD` | `SBASE, offset` (like store) | Discard **one** aligned **64-byte** L0 line (no WB) | **64** line |
| `S_DCACHE_DISCARD_X2` | `SBASE, offset` | Discard **two** consecutive **64-byte** lines | **128** (two lines) |

---

## Buffer atomics — 32-bit (`S_BUFFER_ATOMIC_*`)

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `S_BUFFER_ATOMIC_SWAP` | `SDATA, SBASE, offset` | Atomic exchange **u32**; optional pre-op in SDATA | **4** (memory) |
| `S_BUFFER_ATOMIC_CMPSWAP` | `SDATA, SBASE, offset` | Compare (**DATA[63:32]**), swap (**DATA[31:0]**) if equal | **4** |
| `S_BUFFER_ATOMIC_ADD` | `SDATA, SBASE, offset` | Atomic **u32** add | **4** |
| `S_BUFFER_ATOMIC_SUB` | `SDATA, SBASE, offset` | Atomic **u32** subtract | **4** |
| `S_BUFFER_ATOMIC_SMIN` | `SDATA, SBASE, offset` | Atomic **s32** minimum | **4** |
| `S_BUFFER_ATOMIC_UMIN` | `SDATA, SBASE, offset` | Atomic **u32** minimum | **4** |
| `S_BUFFER_ATOMIC_SMAX` | `SDATA, SBASE, offset` | Atomic **s32** maximum | **4** |
| `S_BUFFER_ATOMIC_UMAX` | `SDATA, SBASE, offset` | Atomic **u32** maximum | **4** |
| `S_BUFFER_ATOMIC_AND` | `SDATA, SBASE, offset` | Atomic **u32** AND | **4** |
| `S_BUFFER_ATOMIC_OR` | `SDATA, SBASE, offset` | Atomic **u32** OR | **4** |
| `S_BUFFER_ATOMIC_XOR` | `SDATA, SBASE, offset` | Atomic **u32** XOR | **4** |
| `S_BUFFER_ATOMIC_INC` | `SDATA, SBASE, offset` | **u32** increment with wrap at limit in DATA | **4** |
| `S_BUFFER_ATOMIC_DEC` | `SDATA, SBASE, offset` | **u32** decrement with wrap using limit in DATA | **4** |

---

## Buffer atomics — 64-bit (`S_BUFFER_ATOMIC_*_X2`)

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `S_BUFFER_ATOMIC_SWAP_X2` | `SDATA, SBASE, offset` | Atomic exchange **u64** | **8** |
| `S_BUFFER_ATOMIC_CMPSWAP_X2` | `SDATA, SBASE, offset` | **u64** compare/swap (cmp **DATA[127:64]**, src **DATA[63:0]**) | **8** |
| `S_BUFFER_ATOMIC_ADD_X2` | `SDATA, SBASE, offset` | Atomic **u64** add | **8** |
| `S_BUFFER_ATOMIC_SUB_X2` | `SDATA, SBASE, offset` | Atomic **u64** subtract | **8** |
| `S_BUFFER_ATOMIC_SMIN_X2` | `SDATA, SBASE, offset` | Atomic **s64** minimum | **8** |
| `S_BUFFER_ATOMIC_UMIN_X2` | `SDATA, SBASE, offset` | Atomic **u64** minimum | **8** |
| `S_BUFFER_ATOMIC_SMAX_X2` | `SDATA, SBASE, offset` | Atomic **s64** maximum | **8** |
| `S_BUFFER_ATOMIC_UMAX_X2` | `SDATA, SBASE, offset` | Atomic **u64** maximum | **8** |
| `S_BUFFER_ATOMIC_AND_X2` | `SDATA, SBASE, offset` | Atomic **u64** AND | **8** |
| `S_BUFFER_ATOMIC_OR_X2` | `SDATA, SBASE, offset` | Atomic **u64** OR | **8** |
| `S_BUFFER_ATOMIC_XOR_X2` | `SDATA, SBASE, offset` | Atomic **u64** XOR | **8** |
| `S_BUFFER_ATOMIC_INC_X2` | `SDATA, SBASE, offset` | **u64** increment with wrap | **8** |
| `S_BUFFER_ATOMIC_DEC_X2` | `SDATA, SBASE, offset` | **u64** decrement with wrap | **8** |

---

## Global atomics — 32-bit (`S_ATOMIC_*`)

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `S_ATOMIC_SWAP` | `SDATA, SBASE, offset` | Atomic exchange **u32** in scalar global memory | **4** |
| `S_ATOMIC_CMPSWAP` | `SDATA, SBASE, offset` | **u32** compare/swap | **4** |
| `S_ATOMIC_ADD` | `SDATA, SBASE, offset` | Atomic **u32** add | **4** |
| `S_ATOMIC_SUB` | `SDATA, SBASE, offset` | Atomic **u32** subtract | **4** |
| `S_ATOMIC_SMIN` | `SDATA, SBASE, offset` | Atomic **s32** minimum | **4** |
| `S_ATOMIC_UMIN` | `SDATA, SBASE, offset` | Atomic **u32** minimum | **4** |
| `S_ATOMIC_SMAX` | `SDATA, SBASE, offset` | Atomic **s32** maximum | **4** |
| `S_ATOMIC_UMAX` | `SDATA, SBASE, offset` | Atomic **u32** maximum | **4** |
| `S_ATOMIC_AND` | `SDATA, SBASE, offset` | Atomic **u32** AND | **4** |
| `S_ATOMIC_OR` | `SDATA, SBASE, offset` | Atomic **u32** OR | **4** |
| `S_ATOMIC_XOR` | `SDATA, SBASE, offset` | Atomic **u32** XOR | **4** |
| `S_ATOMIC_INC` | `SDATA, SBASE, offset` | **u32** increment with wrap | **4** |
| `S_ATOMIC_DEC` | `SDATA, SBASE, offset` | **u32** decrement with wrap | **4** |

---

## Global atomics — 64-bit (`S_ATOMIC_*_X2`)

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `S_ATOMIC_SWAP_X2` | `SDATA, SBASE, offset` | Atomic exchange **u64** | **8** |
| `S_ATOMIC_CMPSWAP_X2` | `SDATA, SBASE, offset` | **u64** compare/swap | **8** |
| `S_ATOMIC_ADD_X2` | `SDATA, SBASE, offset` | Atomic **u64** add | **8** |
| `S_ATOMIC_SUB_X2` | `SDATA, SBASE, offset` | Atomic **u64** subtract | **8** |
| `S_ATOMIC_SMIN_X2` | `SDATA, SBASE, offset` | Atomic **s64** minimum | **8** |
| `S_ATOMIC_UMIN_X2` | `SDATA, SBASE, offset` | Atomic **u64** minimum | **8** |
| `S_ATOMIC_SMAX_X2` | `SDATA, SBASE, offset` | Atomic **s64** maximum | **8** |
| `S_ATOMIC_UMAX_X2` | `SDATA, SBASE, offset` | Atomic **u64** maximum | **8** |
| `S_ATOMIC_AND_X2` | `SDATA, SBASE, offset` | Atomic **u64** AND | **8** |
| `S_ATOMIC_OR_X2` | `SDATA, SBASE, offset` | Atomic **u64** OR | **8** |
| `S_ATOMIC_XOR_X2` | `SDATA, SBASE, offset` | Atomic **u64** XOR | **8** |
| `S_ATOMIC_INC_X2` | `SDATA, SBASE, offset` | **u64** increment with wrap | **8** |
| `S_ATOMIC_DEC_X2` | `SDATA, SBASE, offset` | **u64** decrement with wrap | **8** |
