# CDNA4 Vector Memory — Reference (MUBUF, MTBUF, Flat, Scratch, Global)

Source: AMD CDNA4 ISA (Ch. 9 *Vector Memory Operations*, Ch. 10 *Flat Memory Instructions*, §12.13–12.15).

## Overview

### Chapter 9 — Vector memory (buffer path)

- **Per-lane** loads/stores/atomics: **VGPR** addresses and data; **T#/V#** buffer resource in **4 (or 8) aligned SGPRs** (`SRSRC`).
- **MTBUF**: **DFMT/NFMT** in the instruction (typed); **MUBUF**: untyped opcodes or **FORMAT** from resource.
- **Address**: `CalcBufferAddr(VADDR, SRSRC, SOFFSET, inst_offset)` with **IDXEN/OFFEN** selecting **index/offset VGPRs**; **element size** for transfers is **1, 2, 4, 8, or 16 bytes** (opcode or DFMT).
- **D16** variants pack **16-bit** results into **VGPR** lanes (low/high half).
- **Scope/cache**: **SC[1:0]** (wave/group/device/system), **NT** non-temporal; atomics use **SC0** for **return pre-op**, **SC1** device vs system (see Ch. 9 tables).
- **LDS**: some **MUBUF** loads can target **LDS** (**LDS=1**); **M0** provides LDS offset (see Ch. 9.1.9).

### Chapter 10 — Flat / Global / Scratch

- **Flat**: `SEG=FLAT`; aperture dispatch per lane (**global / scratch / LDS**). Uses **`CalcFlatAddr(ADDR, OFFSET)`**; increments **both VM_CNT and LGKM_CNT** — prefer **`S_WAITCNT 0`** after flat.
- **Global**: `SEG=GLOBAL`; **`CalcGlobalAddr(ADDR, SADDR, OFFSET)`** — must not target LDS (**MEM_VIOL** if so). **VM_CNT** only.
- **Scratch**: `SEG=SCRATCH`; **`CalcScratchAddr(ADDR, SADDR, OFFSET)`**, swizzled private space. **VM_CNT** only.
- **Flat/Scratch/Global** micro fields: **ADDR** (address VGPR, **64-bit** = pair), **VDST/DATA**, **SADDR** (scratch/global; `0x7F` disables), **OFFSET** immediate, **SC**, **NT**, **NV**, **SVE** (scratch VGPR enable), **LDS** for load-to-LDS encodings.

### Operand shorthand (buffer instructions)

| Operand | Role |
|---------|------|
| **VDATA** | Load **destination** / store **source** / atomic **DATA** and optional **RETURN_DATA** |
| **VADDR** | First address VGPR (**index** and/or **offset** per **IDXEN/OFFEN**) |
| **SRSRC** | SGPR specifying **128-bit resource** (4-SGPR aligned) |
| **SOFFSET** | SGPR or **M0**: unsigned **byte** offset added to address |
| **offset** | 12-bit unsigned **instruction byte offset** |

---

## MUBUF — Formatted load/store (`BUFFER_LOAD_FORMAT_*`, `BUFFER_STORE_FORMAT_*`, D16)

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `BUFFER_LOAD_FORMAT_X` | `VDATA, VADDR, SRSRC, SOFFSET, offset` | Load **1** component; convert using **resource** DFMT/NFMT | **format-dependent** |
| `BUFFER_LOAD_FORMAT_XY` | same | Load **2** components as **32-bit** each in VGPRs | **format-dependent** |
| `BUFFER_LOAD_FORMAT_XYZ` | same | Load **3** components | **format-dependent** |
| `BUFFER_LOAD_FORMAT_XYZW` | same | Load **4** components | **format-dependent** |
| `BUFFER_STORE_FORMAT_X` | same | Store **1** component from **VDATA[31:0]** | **format-dependent** |
| `BUFFER_STORE_FORMAT_XY` | same | Store **2** components | **format-dependent** |
| `BUFFER_STORE_FORMAT_XYZ` | same | Store **3** components | **format-dependent** |
| `BUFFER_STORE_FORMAT_XYZW` | same | Store **4** components | **format-dependent** |
| `BUFFER_LOAD_FORMAT_D16_X` | same | Load 1 component → **16-bit** in **VDATA[15:0]** | **format-dependent** |
| `BUFFER_LOAD_FORMAT_D16_XY` | same | 2× **16-bit** packed in one **VGPR** | **format-dependent** |
| `BUFFER_LOAD_FORMAT_D16_XYZ` | same | 3× **16-bit** across VGPRs | **format-dependent** |
| `BUFFER_LOAD_FORMAT_D16_XYZW` | same | 4× **16-bit** packed | **format-dependent** |
| `BUFFER_STORE_FORMAT_D16_X` | same | Store from **VDATA[15:0]** | **format-dependent** |
| `BUFFER_STORE_FORMAT_D16_XY` | same | 2 components from **16-bit** lanes | **format-dependent** |
| `BUFFER_STORE_FORMAT_D16_XYZ` | same | 3 components | **format-dependent** |
| `BUFFER_STORE_FORMAT_D16_XYZW` | same | 4 components | **format-dependent** |
| `BUFFER_LOAD_FORMAT_D16_HI_X` | same | Load 1 component → **VDATA[31:16]** | **format-dependent** |
| `BUFFER_STORE_FORMAT_D16_HI_X` | same | Store from **VDATA[31:16]** | **format-dependent** |

*Memory bytes per lane follow **resource data format** (see Ch. 9); doc states “Mem access size depends on format”. **BUFFER_STORE_FORMAT_** variants take **DFMT/NFMT from the instruction**, overriding the resource (§12.13).*

---

## MUBUF — Untyped load/store

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `BUFFER_LOAD_UBYTE` | `VDATA, VADDR, SRSRC, SOFFSET, offset` | Load **u8**, zero-extend to **u32** | **1** |
| `BUFFER_LOAD_SBYTE` | same | Load **s8**, sign-extend to **s32** | **1** |
| `BUFFER_LOAD_USHORT` | same | Load **u16**, zero-extend | **2** |
| `BUFFER_LOAD_SSHORT` | same | Load **s16**, sign-extend | **2** |
| `BUFFER_LOAD_DWORD` | same | Load **32-bit** | **4** |
| `BUFFER_LOAD_DWORDX2` | same | Load **64-bit** (2 dwords) | **8** |
| `BUFFER_LOAD_DWORDX3` | same | Load **96-bit** (3 dwords) | **12** |
| `BUFFER_LOAD_DWORDX4` | same | Load **128-bit** (4 dwords) | **16** |
| `BUFFER_STORE_BYTE` | same | Store low **8 bits** of VDATA | **1** |
| `BUFFER_STORE_BYTE_D16_HI` | same | Store **bits [23:16]** as byte | **1** |
| `BUFFER_STORE_SHORT` | same | Store **16-bit** from VDATA | **2** |
| `BUFFER_STORE_SHORT_D16_HI` | same | Store **high half** as **16-bit** | **2** |
| `BUFFER_STORE_DWORD` | same | Store **32-bit** | **4** |
| `BUFFER_STORE_DWORDX2` | same | Store **64-bit** | **8** |
| `BUFFER_STORE_DWORDX3` | same | Store **96-bit** | **12** |
| `BUFFER_STORE_DWORDX4` | same | Store **128-bit** | **16** |
| `BUFFER_LOAD_UBYTE_D16` | same | **u8** → **16-bit** in **VDATA[15:0]** | **1** |
| `BUFFER_LOAD_UBYTE_D16_HI` | same | **u8** → **16-bit** in **VDATA[31:16]** | **1** |
| `BUFFER_LOAD_SBYTE_D16` | same | **s8** → **16-bit** low | **1** |
| `BUFFER_LOAD_SBYTE_D16_HI` | same | **s8** → **16-bit** high | **1** |
| `BUFFER_LOAD_SHORT_D16` | same | Load **16-bit** to **VDATA[15:0]** | **2** |
| `BUFFER_LOAD_SHORT_D16_HI` | same | Load **16-bit** to **VDATA[31:16]** | **2** |

---

## MUBUF — Cache control

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `BUFFER_WBL2` | `SRSRC`, **SC** (per §9.1.10) | **L2 write-back** / NOP per **SC** | — |
| `BUFFER_INV` | `SRSRC`, **SC** | **CU/L2 invalidate** per **SC** | — |

*These opcodes use the **MUBUF** encoding; assemblers may require the full operand list (**VDATA/VADDR**/imm) even when unused—**SRSRC** and **SC** bits define behavior (§9.1.10, §12.13).*

---

## MUBUF — Integer and FP atomics (32-bit and 64-bit)

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `BUFFER_ATOMIC_SWAP` | `VDATA, VADDR, SRSRC, SOFFSET, offset` | Atomic **u32** exchange; pre-op to VDATA if **SC0** | **4** |
| `BUFFER_ATOMIC_CMPSWAP` | same | **u32** CAS (**DATA**: compare high, new low) | **4** |
| `BUFFER_ATOMIC_ADD` | same | **u32** add | **4** |
| `BUFFER_ATOMIC_SUB` | same | **u32** sub | **4** |
| `BUFFER_ATOMIC_SMIN` | same | **s32** min | **4** |
| `BUFFER_ATOMIC_UMIN` | same | **u32** min | **4** |
| `BUFFER_ATOMIC_SMAX` | same | **s32** max | **4** |
| `BUFFER_ATOMIC_UMAX` | same | **u32** max | **4** |
| `BUFFER_ATOMIC_AND` | same | **u32** AND | **4** |
| `BUFFER_ATOMIC_OR` | same | **u32** OR | **4** |
| `BUFFER_ATOMIC_XOR` | same | **u32** XOR | **4** |
| `BUFFER_ATOMIC_INC` | same | **u32** inc with wrap | **4** |
| `BUFFER_ATOMIC_DEC` | same | **u32** dec with wrap | **4** |
| `BUFFER_ATOMIC_ADD_F32` | same | **f32** add (NaN/Inf/denorm rules) | **4** |
| `BUFFER_ATOMIC_PK_ADD_F16` | same | **2×f16** packed add | **4** |
| `BUFFER_ATOMIC_ADD_F64` | same | **f64** add | **8** |
| `BUFFER_ATOMIC_MIN_F64` | same | **f64** minimum | **8** |
| `BUFFER_ATOMIC_MAX_F64` | same | **f64** maximum | **8** |
| `BUFFER_ATOMIC_PK_ADD_BF16` | same | **2×bf16** packed add | **4** |
| `BUFFER_ATOMIC_SWAP_X2` | same | **u64** exchange | **8** |
| `BUFFER_ATOMIC_CMPSWAP_X2` | same | **u64** CAS | **8** |
| `BUFFER_ATOMIC_ADD_X2` | same | **u64** add | **8** |
| `BUFFER_ATOMIC_SUB_X2` | same | **u64** sub | **8** |
| `BUFFER_ATOMIC_SMIN_X2` | same | **s64** min | **8** |
| `BUFFER_ATOMIC_UMIN_X2` | same | **u64** min | **8** |
| `BUFFER_ATOMIC_SMAX_X2` | same | **s64** max | **8** |
| `BUFFER_ATOMIC_UMAX_X2` | same | **u64** max | **8** |
| `BUFFER_ATOMIC_AND_X2` | same | **u64** AND | **8** |
| `BUFFER_ATOMIC_OR_X2` | same | **u64** OR | **8** |
| `BUFFER_ATOMIC_XOR_X2` | same | **u64** XOR | **8** |
| `BUFFER_ATOMIC_INC_X2` | same | **u64** inc with wrap | **8** |
| `BUFFER_ATOMIC_DEC_X2` | same | **u64** dec with wrap | **8** |

---

## MTBUF — Typed buffer (`TBUFFER_*`)

*Same opcode numbering and semantics as MUBUF formatted family; **DFMT/NFMT** come from the **instruction** (override resource). Operand pattern unchanged.*

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `TBUFFER_LOAD_FORMAT_X` | `VDATA, VADDR, SRSRC, SOFFSET, offset` + **DFMT,NFMT** | Typed load **1** component | **format-dependent** |
| `TBUFFER_LOAD_FORMAT_XY` | same | **2** components | **format-dependent** |
| `TBUFFER_LOAD_FORMAT_XYZ` | same | **3** components | **format-dependent** |
| `TBUFFER_LOAD_FORMAT_XYZW` | same | **4** components | **format-dependent** |
| `TBUFFER_STORE_FORMAT_X` | same | Typed store **1** component | **format-dependent** |
| `TBUFFER_STORE_FORMAT_XY` | same | **2** components | **format-dependent** |
| `TBUFFER_STORE_FORMAT_XYZ` | same | **3** components | **format-dependent** |
| `TBUFFER_STORE_FORMAT_XYZW` | same | **4** components | **format-dependent** |
| `TBUFFER_LOAD_FORMAT_D16_X` | same | D16 typed load **1** | **format-dependent** |
| `TBUFFER_LOAD_FORMAT_D16_XY` | same | D16 **2** | **format-dependent** |
| `TBUFFER_LOAD_FORMAT_D16_XYZ` | same | D16 **3** | **format-dependent** |
| `TBUFFER_LOAD_FORMAT_D16_XYZW` | same | D16 **4** | **format-dependent** |
| `TBUFFER_STORE_FORMAT_D16_X` | same | D16 store **1** | **format-dependent** |
| `TBUFFER_STORE_FORMAT_D16_XY` | same | D16 **2** | **format-dependent** |
| `TBUFFER_STORE_FORMAT_D16_XYZ` | same | D16 **3** | **format-dependent** |
| `TBUFFER_STORE_FORMAT_D16_XYZW` | same | D16 **4** | **format-dependent** |

---

## Flat instructions (`FLAT_*`)

*Operand pattern: **`VDST/DATA, ADDR, offset`** with `addr = CalcFlatAddr(ADDR, OFFSET)` (32-bit addr shown in §12.15.1; **64-bit** uses **ADDR** pair).*

### Flat — Loads and stores

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `FLAT_LOAD_UBYTE` | `VDATA, ADDR, offset` | **u8** → **u32** | **1** |
| `FLAT_LOAD_SBYTE` | same | **s8** → **s32** | **1** |
| `FLAT_LOAD_USHORT` | same | **u16** → **u32** | **2** |
| `FLAT_LOAD_SSHORT` | same | **s16** → **s32** | **2** |
| `FLAT_LOAD_DWORD` | same | Load **32-bit** | **4** |
| `FLAT_LOAD_DWORDX2` | same | Load **64-bit** | **8** |
| `FLAT_LOAD_DWORDX3` | same | Load **96-bit** | **12** |
| `FLAT_LOAD_DWORDX4` | same | Load **128-bit** | **16** |
| `FLAT_STORE_BYTE` | same | Store byte from VDATA | **1** |
| `FLAT_STORE_BYTE_D16_HI` | same | Store from **bits [23:16]** | **1** |
| `FLAT_STORE_SHORT` | same | Store **16-bit** | **2** |
| `FLAT_STORE_SHORT_D16_HI` | same | Store high **16-bit** | **2** |
| `FLAT_STORE_DWORD` | same | Store **32-bit** | **4** |
| `FLAT_STORE_DWORDX2` | same | Store **64-bit** | **8** |
| `FLAT_STORE_DWORDX3` | same | Store **96-bit** | **12** |
| `FLAT_STORE_DWORDX4` | same | Store **128-bit** | **16** |
| `FLAT_LOAD_UBYTE_D16` | same | **u8** → **16-bit** low | **1** |
| `FLAT_LOAD_UBYTE_D16_HI` | same | **u8** → **16-bit** high | **1** |
| `FLAT_LOAD_SBYTE_D16` | same | **s8** → **16-bit** low | **1** |
| `FLAT_LOAD_SBYTE_D16_HI` | same | **s8** → **16-bit** high | **1** |
| `FLAT_LOAD_SHORT_D16` | same | **16-bit** → low half | **2** |
| `FLAT_LOAD_SHORT_D16_HI` | same | **16-bit** → high half | **2** |

### Flat — Atomics

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `FLAT_ATOMIC_SWAP` | `VDATA, ADDR, offset` | **u32** exchange; pre-op if **SC0** | **4** |
| `FLAT_ATOMIC_CMPSWAP` | same | **u32** CAS (**RETURN_DATA[1]** unmodified per §12.15.1) | **4** |
| `FLAT_ATOMIC_ADD` | same | **u32** add | **4** |
| `FLAT_ATOMIC_SUB` | same | **u32** sub | **4** |
| `FLAT_ATOMIC_SMIN` | same | **s32** min | **4** |
| `FLAT_ATOMIC_UMIN` | same | **u32** min | **4** |
| `FLAT_ATOMIC_SMAX` | same | **s32** max | **4** |
| `FLAT_ATOMIC_UMAX` | same | **u32** max | **4** |
| `FLAT_ATOMIC_AND` | same | **u32** AND | **4** |
| `FLAT_ATOMIC_OR` | same | **u32** OR | **4** |
| `FLAT_ATOMIC_XOR` | same | **u32** XOR | **4** |
| `FLAT_ATOMIC_INC` | same | **u32** inc wrap | **4** |
| `FLAT_ATOMIC_DEC` | same | **u32** dec wrap | **4** |
| `FLAT_ATOMIC_ADD_F32` | same | **f32** add | **4** |
| `FLAT_ATOMIC_PK_ADD_F16` | same | **2×f16** add | **4** |
| `FLAT_ATOMIC_ADD_F64` | same | **f64** add | **8** |
| `FLAT_ATOMIC_MIN_F64` | same | **f64** min | **8** |
| `FLAT_ATOMIC_MAX_F64` | same | **f64** max | **8** |
| `FLAT_ATOMIC_PK_ADD_BF16` | same | **2×bf16** add | **4** |
| `FLAT_ATOMIC_SWAP_X2` | same | **u64** exchange | **8** |
| `FLAT_ATOMIC_CMPSWAP_X2` | same | **u64** CAS (**RETURN_DATA[2:3]** not modified) | **8** |
| `FLAT_ATOMIC_ADD_X2` | same | **u64** add | **8** |
| `FLAT_ATOMIC_SUB_X2` | same | **u64** sub | **8** |
| `FLAT_ATOMIC_SMIN_X2` | same | **s64** min | **8** |
| `FLAT_ATOMIC_UMIN_X2` | same | **u64** min | **8** |
| `FLAT_ATOMIC_SMAX_X2` | same | **s64** max | **8** |
| `FLAT_ATOMIC_UMAX_X2` | same | **u64** max | **8** |
| `FLAT_ATOMIC_AND_X2` | same | **u64** AND | **8** |
| `FLAT_ATOMIC_OR_X2` | same | **u64** OR | **8** |
| `FLAT_ATOMIC_XOR_X2` | same | **u64** XOR | **8** |
| `FLAT_ATOMIC_INC_X2` | same | **u64** inc wrap | **8** |
| `FLAT_ATOMIC_DEC_X2` | same | **u64** dec wrap | **8** |

---

## Scratch instructions (`SCRATCH_*`)

*Operand pattern: **`VDATA, ADDR, SADDR, offset`** with `addr = CalcScratchAddr(ADDR, SADDR, OFFSET)`.*

### Scratch — Loads and stores

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `SCRATCH_LOAD_UBYTE` | `VDATA, ADDR, SADDR, offset` | **u8** → **u32** | **1** |
| `SCRATCH_LOAD_SBYTE` | same | **s8** → **s32** | **1** |
| `SCRATCH_LOAD_USHORT` | same | **u16** → **u32** | **2** |
| `SCRATCH_LOAD_SSHORT` | same | **s16** → **s32** | **2** |
| `SCRATCH_LOAD_DWORD` | same | Load **32-bit** | **4** |
| `SCRATCH_LOAD_DWORDX2` | same | Load **64-bit** | **8** |
| `SCRATCH_LOAD_DWORDX3` | same | Load **96-bit** | **12** |
| `SCRATCH_LOAD_DWORDX4` | same | Load **128-bit** | **16** |
| `SCRATCH_STORE_BYTE` | same | Store byte | **1** |
| `SCRATCH_STORE_BYTE_D16_HI` | same | Store from **[23:16]** | **1** |
| `SCRATCH_STORE_SHORT` | same | Store **16-bit** | **2** |
| `SCRATCH_STORE_SHORT_D16_HI` | same | Store high **16-bit** | **2** |
| `SCRATCH_STORE_DWORD` | same | Store **32-bit** | **4** |
| `SCRATCH_STORE_DWORDX2` | same | Store **64-bit** | **8** |
| `SCRATCH_STORE_DWORDX3` | same | Store **96-bit** | **12** |
| `SCRATCH_STORE_DWORDX4` | same | Store **128-bit** | **16** |
| `SCRATCH_LOAD_UBYTE_D16` | same | **u8** → **16-bit** low | **1** |
| `SCRATCH_LOAD_UBYTE_D16_HI` | same | **u8** → **16-bit** high | **1** |
| `SCRATCH_LOAD_SBYTE_D16` | same | **s8** → **16-bit** low | **1** |
| `SCRATCH_LOAD_SBYTE_D16_HI` | same | **s8** → **16-bit** high | **1** |
| `SCRATCH_LOAD_SHORT_D16` | same | **16-bit** → low | **2** |
| `SCRATCH_LOAD_SHORT_D16_HI` | same | **16-bit** → high | **2** |

### Scratch — Load to LDS

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `SCRATCH_LOAD_LDS_UBYTE` | `VDATA/encoding, ADDR, SADDR, offset` + **LDS/M0** | **u8** extended → **LDS** | **1** (mem) |
| `SCRATCH_LOAD_LDS_SBYTE` | same | **s8** extended → **LDS** | **1** |
| `SCRATCH_LOAD_LDS_USHORT` | same | **u16** extended → **LDS** | **2** |
| `SCRATCH_LOAD_LDS_SSHORT` | same | **s16** extended → **LDS** | **2** |
| `SCRATCH_LOAD_LDS_DWORD` | same | **32-bit** → **LDS** | **4** |

*Full LDS addressing uses **M0** and **LDS=1** path (see Ch. 9.1.9 / §12.15 microcode).*

---

## Global instructions (`GLOBAL_*`)

*Operand pattern: **`VDATA, ADDR, SADDR, offset`** with `addr = CalcGlobalAddr(ADDR, SADDR, OFFSET)`; **SADDR=0x7F** disables SGPR base/offset contribution per Ch. 10.*

### Global — Loads and stores

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `GLOBAL_LOAD_UBYTE` | `VDATA, ADDR, SADDR, offset` | **u8** → **u32** | **1** |
| `GLOBAL_LOAD_SBYTE` | same | **s8** → **s32** | **1** |
| `GLOBAL_LOAD_USHORT` | same | **u16** → **u32** | **2** |
| `GLOBAL_LOAD_SSHORT` | same | **s16** → **s32** | **2** |
| `GLOBAL_LOAD_DWORD` | same | Load **32-bit** | **4** |
| `GLOBAL_LOAD_DWORDX2` | same | Load **64-bit** | **8** |
| `GLOBAL_LOAD_DWORDX3` | same | Load **96-bit** | **12** |
| `GLOBAL_LOAD_DWORDX4` | same | Load **128-bit** | **16** |
| `GLOBAL_STORE_BYTE` | same | Store byte | **1** |
| `GLOBAL_STORE_BYTE_D16_HI` | same | Store **[23:16]** | **1** |
| `GLOBAL_STORE_SHORT` | same | Store **16-bit** | **2** |
| `GLOBAL_STORE_SHORT_D16_HI` | same | Store high **16-bit** | **2** |
| `GLOBAL_STORE_DWORD` | same | Store **32-bit** | **4** |
| `GLOBAL_STORE_DWORDX2` | same | Store **64-bit** | **8** |
| `GLOBAL_STORE_DWORDX3` | same | Store **96-bit** | **12** |
| `GLOBAL_STORE_DWORDX4` | same | Store **128-bit** | **16** |
| `GLOBAL_LOAD_UBYTE_D16` | same | **u8** → **16-bit** low | **1** |
| `GLOBAL_LOAD_UBYTE_D16_HI` | same | **u8** → **16-bit** high | **1** |
| `GLOBAL_LOAD_SBYTE_D16` | same | **s8** → **16-bit** low | **1** |
| `GLOBAL_LOAD_SBYTE_D16_HI` | same | **s8** → **16-bit** high | **1** |
| `GLOBAL_LOAD_SHORT_D16` | same | **16-bit** → low | **2** |
| `GLOBAL_LOAD_SHORT_D16_HI` | same | **16-bit** → high | **2** |

### Global — Load to LDS

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `GLOBAL_LOAD_LDS_UBYTE` | `encoding, ADDR, SADDR, offset` + **LDS/M0** | **u8** → **LDS** | **1** |
| `GLOBAL_LOAD_LDS_SBYTE` | same | **s8** → **LDS** | **1** |
| `GLOBAL_LOAD_LDS_USHORT` | same | **u16** → **LDS** | **2** |
| `GLOBAL_LOAD_LDS_SSHORT` | same | **s16** → **LDS** | **2** |
| `GLOBAL_LOAD_LDS_DWORD` | same | **32-bit** → **LDS** | **4** |
| `GLOBAL_LOAD_LDS_DWORDX3` | same | **3 dwords** → **LDS** | **12** |
| `GLOBAL_LOAD_LDS_DWORDX4` | same | **4 dwords** → **LDS** | **16** |

### Global — Atomics

| Mnemonic | Syntax (operands) | Brief description | Data size (bytes) |
|----------|-------------------|-------------------|-------------------|
| `GLOBAL_ATOMIC_SWAP` | `VDATA, ADDR, SADDR, offset` | **u32** exchange | **4** |
| `GLOBAL_ATOMIC_CMPSWAP` | same | **u32** CAS | **4** |
| `GLOBAL_ATOMIC_ADD` | same | **u32** add | **4** |
| `GLOBAL_ATOMIC_SUB` | same | **u32** sub | **4** |
| `GLOBAL_ATOMIC_SMIN` | same | **s32** min | **4** |
| `GLOBAL_ATOMIC_UMIN` | same | **u32** min | **4** |
| `GLOBAL_ATOMIC_SMAX` | same | **s32** max | **4** |
| `GLOBAL_ATOMIC_UMAX` | same | **u32** max | **4** |
| `GLOBAL_ATOMIC_AND` | same | **u32** AND | **4** |
| `GLOBAL_ATOMIC_OR` | same | **u32** OR | **4** |
| `GLOBAL_ATOMIC_XOR` | same | **u32** XOR | **4** |
| `GLOBAL_ATOMIC_INC` | same | **u32** inc wrap | **4** |
| `GLOBAL_ATOMIC_DEC` | same | **u32** dec wrap | **4** |
| `GLOBAL_ATOMIC_ADD_F32` | same | **f32** add | **4** |
| `GLOBAL_ATOMIC_PK_ADD_F16` | same | **2×f16** add | **4** |
| `GLOBAL_ATOMIC_ADD_F64` | same | **f64** add | **8** |
| `GLOBAL_ATOMIC_MIN_F64` | same | **f64** min | **8** |
| `GLOBAL_ATOMIC_MAX_F64` | same | **f64** max | **8** |
| `GLOBAL_ATOMIC_PK_ADD_BF16` | same | **2×bf16** add | **4** |
| `GLOBAL_ATOMIC_SWAP_X2` | same | **u64** exchange | **8** |
| `GLOBAL_ATOMIC_CMPSWAP_X2` | same | **u64** CAS | **8** |
| `GLOBAL_ATOMIC_ADD_X2` | same | **u64** add | **8** |
| `GLOBAL_ATOMIC_SUB_X2` | same | **u64** sub | **8** |
| `GLOBAL_ATOMIC_SMIN_X2` | same | **s64** min | **8** |
| `GLOBAL_ATOMIC_UMIN_X2` | same | **u64** min | **8** |
| `GLOBAL_ATOMIC_SMAX_X2` | same | **s64** max | **8** |
| `GLOBAL_ATOMIC_UMAX_X2` | same | **u64** max | **8** |
| `GLOBAL_ATOMIC_AND_X2` | same | **u64** AND | **8** |
| `GLOBAL_ATOMIC_OR_X2` | same | **u64** OR | **8** |
| `GLOBAL_ATOMIC_XOR_X2` | same | **u64** XOR | **8** |
| `GLOBAL_ATOMIC_INC_X2` | same | **u64** inc wrap | **8** |
| `GLOBAL_ATOMIC_DEC_X2` | same | **u64** dec wrap | **8** |

*Float atomics: **SC[0]=0** (no return) required for flat path per Ch. 10; global follows same FP rules as buffer (see Ch. 9.2).*
