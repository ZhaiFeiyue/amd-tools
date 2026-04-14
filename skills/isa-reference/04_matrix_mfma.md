# CDNA4 Matrix Arithmetic (MFMA) Instructions (gfx950)

## Overview

The **matrix core** extends the CDNA shader VALU with **Machine Intelligence SIMD**. It uses a dedicated **accumulation VGPR (AccVGPR)** file in addition to architectural VGPRs. MFMA instructions move data to/from AccVGPRs via `V_ACCVGPR_READ` / `V_ACCVGPR_WRITE`; encoding uses an **ACC** field (low bit: A matrix bank, high bit: B matrix bank) and **ACC_CD** (C/D in Arch VGPR vs AccVGPR).

The hardware primitive is a **4×1 by 1×4 outer product** (dense or **2:4 sparse**). MFMA ops fuse **D = C + A×B** over one or more **blocks** `B` in the mnemonic (`_2B`, `_4B`, `_16B`, or single-block).

**Naming:** `V_MFMA_<D-type>_<M>x<N>x<K>[_<B>B]_<A-type>[_<B-type>]` — per block, **A** is **M×K**, **B** is **K×N**, **C/D** is **M×N** (all indices logical matrix shapes, not register indices).

**Global rules (dense MFMA, except F8F6F4 family where noted):**

- **Denorms:** Ignored from MODE for inputs/outputs; denorms preserved per opcode family (see §7.4).
- **Clamp:** Uses `FP16_OVFL` in MODE (F32 → ±MAX vs ±INF; I32 saturation).
- **Round:** RNE forced; ignores MODE round mode.
- **Exceptions:** Not supported (except DGEMM / F64 path may support exceptions — see ISA §7.4).
- **Exec mask:** Treated as all-1.
- **Alignment:** `src0` / `src1` / `vdst` (when VGPR) **even-aligned**; operands **contiguous**; first register must align to required span.
- **Scale:** No FP16/BF16/I8 MFMA scaling (per ISA).
- **CBSZ / ABID / BLGP (broadcast):** Control lane permutations for **A** (CBSZ+ABID) and **B** (BLGP) — see **Modifiers** below. **F64** repurposes fields (no A broadcast; **BLGP** = negation of A/B/C). **F8F6F4** repurposes **CBSZ** = A format, **BLGP** = B format (no broadcast semantics).

**Dependency / latency:** Matrix ops are **multi-pass**; partially written results may be visible — insert independent ops between issue and consumer (§7.6). **Passes** are documented per opcode in the ISA reference text (not identical to total latency). **Table 28** gives representative **cycle** counts per variant.

**Tools:** [AMD Matrix Instruction Calculator](https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator); [GPUOpen matrix cores notes](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-README/) (CDNA2-oriented but conceptually useful).

---

## Modifiers: CBSZ, ABID, BLGP

| Context | CBSZ | ABID | BLGP |
|--------|------|------|------|
| **Dense MFMA (default)** | Broadcast block size: `S = 64 / (1 << CBSZ)`; legal **CBSZ ≤ 4**; **ABID** selects which **S**-lane block supplies **A**; illegal if `ABID >= (1 << CBSZ)` | See CBSZ | **B** lane permutation / broadcast (Table 29): 0 none; 1–2 broadcast half-wave; 3 rotate; 4–7 broadcast 16-lane quarters |
| **F8F6F4** | **A** element format selector | **ABID[0]=1** required for **V_MFMA_SCALE_***; **ABID[0]=0** forces scale **1.0f** on non-SCALE ops | **B** element format selector |
| **F64 MFMA** | **Ignored** | **Ignored** | **BLGP[0]** negate **A**; **[1]** negate **B**; **[2]** negate **C** |
| **V_SMFMAC** | With **ABID**, selects **which index bits** in `src2` VGPR (not A broadcast) | Index subfield select | Not used for broadcast (per sparse §7.5) |

**F8F6F4 format encoding (CBSZ[2:0] / BLGP[2:0]):** `000` FP8 E4M3; `001` BF8 E5M2; `010` FP6 E2M3; `011` BF6 E3M2; `100` FP4 E2M1.

**SCALE family:** Block scale is **E8M0** exponent; **OP_SEL** on scale sources picks byte lane; scale applies in exponent path (`d_exp = … + scale_a + scale_b` per ISA).

---

## FP4 / FP6 / FP8 / BF8 MFMA Instructions

| Instruction | Input | Output | Registers (VGPR/ACC, contiguous) | Cycles | Notes |
|-------------|-------|--------|----------------------------------|--------|-------|
| `V_MFMA_F32_16X16X128_F8F6F4` | A **16×128**, B **128×16**; **A/B** each **FP8, BF8, FP6, BF6, or FP4** via **CBSZ** / **BLGP** | **16×16** **F32** | **A:** **4** (FP4), **6** (FP6/BF6), or **8** (FP8/BF8) VGPRs; **B:** same by format; **C/D:** **4** VGPRs (ISA §7.1.5; VOP3P also gives 8 regs total A+B at 8b sizing) | **16** if neither side FP8; **32** if **A** or **B** uses FP8/BF8 | **CBSZ**/**BLGP** = format; broadcast disabled (`BLGP==0` behavior for broadcast); **NEG[1:0]/ABS[1:0]=0**; **NEG[2]/ABS[2]** may modify **C**; scale format **E8M0**; **ABID[0]=0** → implicit scale 1.0 |
| `V_MFMA_F32_32X32X64_F8F6F4` | A **32×64**, B **64×32**; same format rules | **32×32** **F32** | **A/B:** **4 / 6 / 8** VGPRs by format; **C/D:** **16** VGPRs | **32** or **64** (higher if FP8/BF8 on **A** or **B**) | Same as above |
| `V_MFMA_SCALE_F32_16X16X128_F8F6F4` | Same tensor shapes as **16×16×128** F8F6F4; **block scale** K-block size **32** | **16×16** **F32** | Same matrix regs as non-SCALE counterpart; **4-DWORD** encoding: first pair **load-scale** (**SRC0** A scales, **SRC1** B scales, **ENCODING 0xCC35** in high bits); second **VOP3P** with MFMA opcode | **16** or **32** (F8 path longer) | **ABID[0] must be 1**; scale **does not** carry to non-SCALE MFMA; **OP_SEL** picks scale byte lane |
| `V_MFMA_SCALE_F32_32X32X64_F8F6F4` | Same as **32×32×64** F8F6F4 + **block scaling** | **32×32** **F32** | Same + scale dword pair | **32** or **64** | **ABID[0]=1** required |
| `V_MFMA_F32_16X16X32_BF8_BF8` | A **16×32** BF8, B **32×16** BF8 | **16×16** **F32** | **A:** 2; **B:** 2; **C/D:** 4 | **16** | **4 passes** (ISA text); **NEG[1:0]/ABS[1:0]=0** |
| `V_MFMA_F32_16X16X32_BF8_FP8` | A BF8, B FP8 (E4M3) | **16×16** **F32** | 2; 2; 4 | **16** | 4 passes |
| `V_MFMA_F32_16X16X32_FP8_BF8` | A FP8, B BF8 | **16×16** **F32** | 2; 2; 4 | **16** | 4 passes |
| `V_MFMA_F32_16X16X32_FP8_FP8` | A FP8, B FP8 | **16×16** **F32** | 2; 2; 4 | **16** | 4 passes |
| `V_MFMA_F32_32X32X16_BF8_BF8` | A **32×16** BF8, B **16×32** BF8 | **32×32** **F32** | 2; 2; 16 | **32** | **8 passes** |
| `V_MFMA_F32_32X32X16_BF8_FP8` | A BF8, B FP8 | **32×32** **F32** | 2; 2; 16 | **32** | 8 passes |
| `V_MFMA_F32_32X32X16_FP8_BF8` | A FP8, B BF8 | **32×32** **F32** | 2; 2; 16 | **32** | 8 passes |
| `V_MFMA_F32_32X32X16_FP8_FP8` | A FP8, B FP8 | **32×32** **F32** | 2; 2; 16 | **32** | 8 passes |

**SH_MEM_CONFIG[8]=1** required for correct BF8/FP8 behavior (ISA §7.3).

---

## FP16 / BF16 MFMA Instructions

| Instruction | Input | Output | Registers | Cycles | Notes |
|-------------|-------|--------|-----------|--------|-------|
| `V_MFMA_F32_32X32X4_2B_F16` | A **32×4** ×2 blocks, B **4×32** ×2 | **32×32** **F32** ×2 | 2; 2; 32 | **64** | **16 passes** |
| `V_MFMA_F32_16X16X4_4B_F16` | **16×4** ×4 / **4×16** ×4 | **16×16** **F32** ×4 | 2; 2; 16 | **32** | **8 passes** |
| `V_MFMA_F32_4X4X4_16B_F16` | **4×4** ×16 / **4×4** ×16 | **4×4** **F32** ×16 | 2; 2; 4 | **8** | **2 passes**; packed **16-bit** in halves of VGPR |
| `V_MFMA_F32_32X32X8_F16` | **32×8** / **8×32** | **32×32** **F32** | 2; 2; 16 | **32** | **8 passes** |
| `V_MFMA_F32_16X16X16_F16` | **16×16** / **16×16** | **16×16** **F32** | 2; 2; 4 | **16** | **4 passes** |
| `V_MFMA_F32_16X16X32_F16` | **16×32** / **32×16** | **16×16** **F32** | 2; 2; 4 | **16** | 4 passes |
| `V_MFMA_F32_32X32X16_F16` | **32×16** / **16×32** | **32×32** **F32** | 2; 2; 16 | **32** | 8 passes |
| `V_MFMA_F32_32X32X4_2B_BF16` | Same layout as F16 **2B** variant | **32×32** **F32** ×2 | 2; 2; 32 | **64** | 16 passes |
| `V_MFMA_F32_16X16X4_4B_BF16` | Same as F16 **4B** | **16×16** **F32** ×4 | 2; 2; 16 | **32** | 8 passes |
| `V_MFMA_F32_4X4X4_16B_BF16` | Same as F16 **16B** | **4×4** **F32** ×16 | 2; 2; 4 | **8** | 2 passes |
| `V_MFMA_F32_32X32X8_BF16` | **32×8** / **8×32** | **32×32** **F32** | 2; 2; 16 | **32** | 8 passes |
| `V_MFMA_F32_16X16X16_BF16` | **16×16** / **16×16** | **16×16** **F32** | 2; 2; 4 | **16** | 4 passes |
| `V_MFMA_F32_16X16X32_BF16` | **16×32** / **32×16** | **16×16** **F32** | 2; 2; 4 | **16** | 4 passes |
| `V_MFMA_F32_32X32X16_BF16` | **32×16** / **16×32** | **32×32** **F32** | 2; 2; 16 | **32** | 8 passes |

---

## FP32 MFMA Instructions

| Instruction | Input | Output | Registers | Cycles | Notes |
|-------------|-------|--------|-----------|--------|-------|
| `V_MFMA_F32_32X32X1_2B_F32` | A **32×1** ×2, B **1×32** ×2 | **32×32** **F32** ×2 | 1; 1; 32 | **64** | **16 passes**; example §7.1.3.1; **CBSZ=ABID=BLGP=0** in basic layout |
| `V_MFMA_F32_16X16X1_4B_F32` | **16×1** ×4 / **1×16** ×4 | **16×16** **F32** ×4 | 1; 1; 16 | **32** | **8 passes** |
| `V_MFMA_F32_4X4X1_16B_F32` | **4×1** ×16 / **1×4** ×16 | **4×4** **F32** ×16 | 1; 1; 4 | **8** | **2 passes** |
| `V_MFMA_F32_32X32X2_F32` | **32×2** / **2×32** | **32×32** **F32** | 1; 1; 16 | **64** | **16 passes** |
| `V_MFMA_F32_16X16X4_F32` | **16×4** / **4×16** | **16×16** **F32** | 1; 1; 4 | **32** | **8 passes** |

**F32 input MFMA** honors **MODE** denormal flags for **A/B**; **C/D** still do not flush per ISA §7.4.

---

## FP64 (DGEMM) MFMA Instructions

| Instruction | Input | Output | Registers | Cycles | Notes |
|-------------|-------|--------|-----------|--------|-------|
| `V_MFMA_F64_16X16X4_F64` | **16×4** / **4×16** **F64** | **16×16** **F64** | **2**; **2**; **8** (pairs for **F64**) | **64** | **16 passes**; **row-major packed** output (not 4×N tile); **CBSZ/ABID ignored**; **BLGP** = negates |
| `V_MFMA_F64_4X4X4_4B_F64` | **4×4** ×4 blocks | **4×4** **F64** ×4 | 2; 2; **2** | **32** | **4 passes**; **64** **F64** result elements across the wave (**2** VGPR pairs) |

---

## INT8 MFMA Instructions

| Instruction | Input | Output | Registers | Cycles | Notes |
|-------------|-------|--------|-----------|--------|-------|
| `V_MFMA_I32_32X32X4_2B_I8` | **32×4** ×2 / **4×32** ×2 **I8** | **32×32** **I32** ×2 | 1; 1; 32 | **64** | **16 passes**; **I8** products **sign-extended** per §7.4 |
| `V_MFMA_I32_16X16X4_4B_I8` | **16×4** ×4 / **4×16** ×4 | **16×16** **I32** ×4 | 1; 1; 16 | **32** | 8 passes |
| `V_MFMA_I32_4X4X4_16B_I8` | **4×4** ×16 | **4×4** **I32** ×16 | 1; 1; 4 | **8** | 2 passes |
| `V_MFMA_I32_32X32X16_I8` | **32×16** / **16×32** | **32×32** **I32** | 2; 2; 16 | **32** | 8 passes |
| `V_MFMA_I32_16X16X32_I8` | **16×32** / **32×16** | **16×16** **I32** | 2; 2; 4 | **16** | 4 passes |
| `V_MFMA_I32_16X16X64_I8` | **16×64** / **64×16** | **16×16** **I32** | 4; 4; 4 | **16** | Table 28 |
| `V_MFMA_I32_32X32X32_I8` | **32×32** / **32×32** | **32×32** **I32** | 4; 4; 16 | **32** | Table 28 |

---

## V_SMFMAC Instructions (4:2 sparse **A**, dense **B**)

**Semantics:** **D = D + Aₛₚₐᵣₛₑ × B**; only **A** is **4:2 structured sparse** (2 of 4 **K** elements zero); **index VGPR** in **src2** encodes which two positions are non-zero. **C** is **not** a separate operand — **VDST** is both **C** and **D**. **src0/src1/vdst** even-aligned; **src2** VGPR only (no even requirement). **ACC_CD** only affects **dest** bank, not **src2** interpretation. **CBSZ/ABID** select index bitfields (not MFMA A-broadcast). **Scale:** not supported on FP16/BF16/I8 SMFMAC (per ISA).

**Operand counts (per §7.5):** **A** sparse: **2** VGPRs/lane; **B** dense: **4** VGPRs/lane; **D** (accum): **16** VGPRs for **16×16** tile, **16** VGPRs for **32×32** tile (full **F32**/**I32** accum).

| Instruction | Sparse **A** / dense **B** | **D** accum type | Cycles | Notes |
|-------------|---------------------------|------------------|--------|-------|
| `V_SMFMAC_F32_16X16X32_F16` | **16×32**ₛ / **32×16** | **F32** | **16** | **4 passes**; **16-bit** index packing in **src2** |
| `V_SMFMAC_F32_32X32X16_F16` | **32×16**ₛ / **16×32** | **F32** | **32** | **8 passes** |
| `V_SMFMAC_F32_16X16X32_BF16` | same K-tile **BF16** | **F32** | **16** | 4 passes |
| `V_SMFMAC_F32_32X32X16_BF16` | | **F32** | **32** | 8 passes |
| `V_SMFMAC_F32_16X16X64_F16` | **16×64**ₛ / **64×16** | **F32** | **16** | 4 passes |
| `V_SMFMAC_F32_32X32X32_F16` | **32×32**ₛ / **32×32** | **F32** | **32** | 8 passes |
| `V_SMFMAC_F32_16X16X64_BF16` | **BF16** | **F32** | **16** | 4 passes |
| `V_SMFMAC_F32_32X32X32_BF16` | | **F32** | **32** | 8 passes |
| `V_SMFMAC_I32_16X16X64_I8` | **I8** sparse/dense | **I32** | **16** | 4 passes |
| `V_SMFMAC_I32_32X32X32_I8` | | **I32** | **32** | 8 passes |
| `V_SMFMAC_I32_16X16X128_I8` | **16×128**ₛ / **128×16** | **I32** | **16** | **CBSZ/ABID** ignored; single index set |
| `V_SMFMAC_I32_32X32X64_I8` | **32×64**ₛ / **64×32** | **I32** | **32** | Same |
| `V_SMFMAC_F32_16X16X64_BF8_BF8` | **BF8** / **BF8** | **F32** | **16** | 4 passes |
| `V_SMFMAC_F32_16X16X64_BF8_FP8` | **BF8** / **FP8** | **F32** | **16** | 4 passes |
| `V_SMFMAC_F32_16X16X64_FP8_BF8` | **FP8** / **BF8** | **F32** | **16** | 4 passes |
| `V_SMFMAC_F32_16X16X64_FP8_FP8` | **FP8** / **FP8** | **F32** | **16** | 4 passes |
| `V_SMFMAC_F32_32X32X32_BF8_BF8` | **32×32**ₛ / **32×32** | **F32** | **32** | 8 passes |
| `V_SMFMAC_F32_32X32X32_BF8_FP8` | | **F32** | **32** | 8 passes |
| `V_SMFMAC_F32_32X32X32_FP8_BF8` | | **F32** | **32** | 8 passes |
| `V_SMFMAC_F32_32X32X32_FP8_FP8` | | **F32** | **32** | 8 passes |
| `V_SMFMAC_F32_16X16X128_BF8_BF8` | **16×128**ₛ / **128×16** | **F32** | **16** | |
| `V_SMFMAC_F32_16X16X128_BF8_FP8` | | **F32** | **16** | |
| `V_SMFMAC_F32_16X16X128_FP8_BF8` | | **F32** | **16** | |
| `V_SMFMAC_F32_16X16X128_FP8_FP8` | | **F32** | **16** | |
| `V_SMFMAC_F32_32X32X64_BF8_BF8` | **32×64**ₛ / **64×32** | **F32** | **32** | |
| `V_SMFMAC_F32_32X32X64_BF8_FP8` | | **F32** | **32** | |
| `V_SMFMAC_F32_32X32X64_FP8_BF8` | | **F32** | **32** | |
| `V_SMFMAC_F32_32X32X64_FP8_FP8` | | **F32** | **32** | |

---

## Encoding / opcode index (VOP3P subop — reference)

Dense MFMA / related ops appear in ISA **§12.10** with subop numbers **45–46, 53–56, 64–66, 68–69, 72–74, 76–77, 80–82, 84–87, 93–97, 110–119**; **V_SMFMAC** **57–61, 67, 70–71, 75, 78–79, 83, 90–91, 98, 100, 102, 104, 106–108, 120–127**. **V_MFMA_SCALE_*** use the **4-dword** fused encoding described in **§7.2** (constant **0xD3AC** / **VOP3P** opcode field pattern per ISA).

---

*Source: AMD CDNA4 ISA document `cdna4_isa_full.txt` — Chapter 7 (§§7.1–7.6), Table 28–33, §12.10 VOP3P.*
