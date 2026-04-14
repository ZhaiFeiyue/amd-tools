# CDNA4 Scalar ALU Instructions (gfx950)

Source: CDNA4 ISA — sections 12.1–12.5 (`cdna4_isa_full.txt`). SOPK/SOP1/SOPC/SOPP formats may consume an optional **32-bit literal** immediately after the instruction word when required by the encoding or specific opcode (see per-instruction notes).

## SOP2 — Scalar Two-Operand Instructions

| Instruction | Syntax | Description |
|---|---|---|
| S_ADD_U32 | D, S0, S1 | Unsigned 32-bit add; `D = S0 + S1`; SCC = carry-out (unsigned overflow). |
| S_SUB_U32 | D, S0, S1 | Unsigned 32-bit subtract; `D = S0 - S1`; SCC = borrow/unsigned overflow. |
| S_ADD_I32 | D, S0, S1 | Signed 32-bit add; SCC = signed overflow. **Note:** Not suitable with `S_ADDC_U32` for 64-bit add chains. |
| S_SUB_I32 | D, S0, S1 | Signed 32-bit subtract; SCC = signed overflow. **Note:** Not suitable with `S_SUBB_U32` for 64-bit subtract chains. |
| S_ADDC_U32 | D, S0, S1 | Unsigned add with carry-in from SCC; SCC = new carry-out. |
| S_SUBB_U32 | D, S0, S1 | Unsigned subtract with borrow-in from SCC; SCC = new borrow-out. |
| S_MIN_I32 | D, S0, S1 | Signed minimum; SCC set iff S0 is selected (`S0 < S1`). |
| S_MIN_U32 | D, S0, S1 | Unsigned minimum; SCC set iff S0 is selected (`S0 < S1`). |
| S_MAX_I32 | D, S0, S1 | Signed maximum; SCC set iff S0 is selected (`S0 >= S1`). |
| S_MAX_U32 | D, S0, S1 | Unsigned maximum; SCC set iff S0 is selected (`S0 >= S1`). |
| S_CSELECT_B32 | D, S0, S1 | `D = SCC ? S0 : S1` (32-bit). |
| S_CSELECT_B64 | D, S0, S1 | `D = SCC ? S0 : S1` (64-bit). |
| S_AND_B32 | D, S0, S1 | Bitwise AND (32-bit); SCC iff result ≠ 0. |
| S_AND_B64 | D, S0, S1 | Bitwise AND (64-bit); SCC iff result ≠ 0. |
| S_OR_B32 | D, S0, S1 | Bitwise OR (32-bit); SCC iff result ≠ 0. |
| S_OR_B64 | D, S0, S1 | Bitwise OR (64-bit); SCC iff result ≠ 0. |
| S_XOR_B32 | D, S0, S1 | Bitwise XOR (32-bit); SCC iff result ≠ 0. |
| S_XOR_B64 | D, S0, S1 | Bitwise XOR (64-bit); SCC iff result ≠ 0. |
| S_ANDN2_B32 | D, S0, S1 | `D = S0 & ~S1` (32-bit); SCC iff result ≠ 0. |
| S_ANDN2_B64 | D, S0, S1 | `D = S0 & ~S1` (64-bit); SCC iff result ≠ 0. |
| S_ORN2_B32 | D, S0, S1 | `D = S0 OR (NOT S1)` (32-bit); SCC iff result ≠ 0. |
| S_ORN2_B64 | D, S0, S1 | `D = S0 OR (NOT S1)` (64-bit); SCC iff result ≠ 0. |
| S_NAND_B32 | D, S0, S1 | Bitwise NAND (32-bit); SCC iff result ≠ 0. |
| S_NAND_B64 | D, S0, S1 | Bitwise NAND (64-bit); SCC iff result ≠ 0. |
| S_NOR_B32 | D, S0, S1 | Bitwise NOR (32-bit); SCC iff result ≠ 0. |
| S_NOR_B64 | D, S0, S1 | Bitwise NOR (64-bit); SCC iff result ≠ 0. |
| S_XNOR_B32 | D, S0, S1 | Bitwise XNOR (32-bit); SCC iff result ≠ 0. |
| S_XNOR_B64 | D, S0, S1 | Bitwise XNOR (64-bit); SCC iff result ≠ 0. |
| S_LSHL_B32 | D, S0, S1 | Logical shift left; shift count = `S1[4:0]`; SCC iff result ≠ 0. |
| S_LSHL_B64 | D, S0, S1 | Logical shift left; shift count = `S1[5:0]`; SCC iff result ≠ 0. |
| S_LSHR_B32 | D, S0, S1 | Logical shift right; shift count = `S1[4:0]`; SCC iff result ≠ 0. |
| S_LSHR_B64 | D, S0, S1 | Logical shift right; shift count = `S1[5:0]`; SCC iff result ≠ 0. |
| S_ASHR_I32 | D, S0, S1 | Arithmetic shift right (sign-preserving); count = `S1[4:0]`; SCC iff result ≠ 0. |
| S_ASHR_I64 | D, S0, S1 | Arithmetic shift right (sign-preserving); count = `S1[5:0]`; SCC iff result ≠ 0. |
| S_BFM_B32 | D, S0, S1 | Bitfield mask: width=`S0[4:0]`, offset=`S1[4:0]` → `((1<<width)-1) << offset`. |
| S_BFM_B64 | D, S0, S1 | Bitfield mask (64-bit); width=`S0[5:0]`, offset=`S1[5:0]`. |
| S_MUL_I32 | D, S0, S1 | Signed 32×32 → 32-bit product (low half). |
| S_BFE_U32 | D, S0, S1 | Unsigned bitfield extract from S0; offset=`S1[4:0]`, width=`S1[22:16]`; SCC iff result ≠ 0. |
| S_BFE_I32 | D, S0, S1 | Signed bitfield extract from S0; offset/width as in `S_BFE_U32`; sign-extends field; SCC iff result ≠ 0. |
| S_BFE_U64 | D, S0, S1 | Unsigned bitfield extract (64-bit source); offset=`S1[5:0]`, width=`S1[22:16]`; SCC iff result ≠ 0. |
| S_BFE_I64 | D, S0, S1 | Signed bitfield extract (64-bit); SCC iff result ≠ 0. |
| S_CBRANCH_G_FORK | S0, S1 | Conditional branch using branch-stack. S0 = compare mask (VCC or any SGPR); S1 = 64-bit **byte** address of target. Pairs with `S_CBRANCH_JOIN`. |
| S_ABSDIFF_I32 | D, S0, S1 | Absolute difference abs(S0 − S1) in signed domain; SCC iff result ≠ 0. **Note:** `0x80000000` cases can yield negative-looking results per ISA examples. |
| S_MUL_HI_U32 | D, S0, S1 | High 32 bits of unsigned 32×32 multiply. |
| S_MUL_HI_I32 | D, S0, S1 | High 32 bits of signed 32×32 multiply. |
| S_LSHL1_ADD_U32 | D, S0, S1 | `(S0 << 1) + S1`; SCC on unsigned overflow of 32-bit sum. |
| S_LSHL2_ADD_U32 | D, S0, S1 | `(S0 << 2) + S1`; SCC on unsigned overflow. |
| S_LSHL3_ADD_U32 | D, S0, S1 | `(S0 << 3) + S1`; SCC on unsigned overflow. |
| S_LSHL4_ADD_U32 | D, S0, S1 | `(S0 << 4) + S1`; SCC on unsigned overflow. |
| S_PACK_LL_B32_B16 | D, S0, S1 | Pack: `D = { S1[15:0], S0[15:0] }` (16-bit halves). |
| S_PACK_LH_B32_B16 | D, S0, S1 | Pack: `D = { S1[31:16], S0[15:0] }`. |
| S_PACK_HH_B32_B16 | D, S0, S1 | Pack: `D = { S1[31:16], S0[31:16] }`. |

## SOPK — Scalar Immediate Instructions

| Instruction | Syntax | Description |
|---|---|---|
| S_MOVK_I32 | D, simm16 | Sign-extend 16-bit immediate to 32 bits; store in D. |
| S_CMOVK_I32 | D, simm16 | If SCC: `D = signext(simm16)`; else D unchanged. |
| S_CMPK_EQ_I32 | S0, simm16 | SCC = (`S0.i32 == signext(simm16)`). |
| S_CMPK_LG_I32 | S0, simm16 | SCC = (`S0.i32 != signext(simm16)`). |
| S_CMPK_GT_I32 | S0, simm16 | SCC = (`S0.i32 > signext(simm16)`). |
| S_CMPK_GE_I32 | S0, simm16 | SCC = (`S0.i32 >= signext(simm16)`). |
| S_CMPK_LT_I32 | S0, simm16 | SCC = (`S0.i32 < signext(simm16)`). |
| S_CMPK_LE_I32 | S0, simm16 | SCC = (`S0.i32 <= signext(simm16)`). |
| S_CMPK_EQ_U32 | S0, simm16 | SCC = (`S0.u32 == zeroext(simm16)`). |
| S_CMPK_LG_U32 | S0, simm16 | SCC = (`S0.u32 != zeroext(simm16)`). |
| S_CMPK_GT_U32 | S0, simm16 | SCC = (`S0.u32 > zeroext(simm16)`). |
| S_CMPK_GE_U32 | S0, simm16 | SCC = (`S0.u32 >= zeroext(simm16)`). |
| S_CMPK_LT_U32 | S0, simm16 | SCC = (`S0.u32 < zeroext(simm16)`). |
| S_CMPK_LE_U32 | S0, simm16 | SCC = (`S0.u32 <= zeroext(simm16)`). |
| S_ADDK_I32 | D, simm16 | `D += signext(simm16)`; SCC = signed overflow (RMW on D). |
| S_MULK_I32 | D, simm16 | `D *= signext(simm16)` (RMW on D). |
| S_CBRANCH_I_FORK | S0, simm16 | Branch-stack fork: S0 = mask; target = PC + signext(simm16)×4 + 4 (signed dword offset from next insn). See `S_CBRANCH_JOIN`. |
| S_GETREG_B32 | D, simm16 | Read hardware register subset: `hwRegId=simm[5:0]`, `offset=simm[10:6]`, `size=simm[15:11]+1` (1..32 bits) into D’s LSBs. |
| S_SETREG_B32 | simm16, S0 | Write masked subset of S0 into HW register (field layout as `S_GETREG_B32`); respects `HwRegWriteMask` and privilege. Side-effects possible. |
| S_SETREG_IMM32_B32 | simm16, imm32 | Like `S_SETREG_B32` but value from **following 32-bit literal**. |
| S_CALL_B64 | D, simm16 | `D = return address (PC+4)`; PC += signext(simm16)×4 + 4. Short call; **must be 4 bytes**. Long calls may use `S_SWAPPC_B64`. |

## SOP1 — Scalar Single-Operand Instructions

| Instruction | Syntax | Description |
|---|---|---|
| S_MOV_B32 | D, S0 | Move 32-bit scalar. |
| S_MOV_B64 | D, S0 | Move 64-bit scalar. |
| S_CMOV_B32 | D, S0 | If SCC: `D = S0` (32-bit). |
| S_CMOV_B64 | D, S0 | If SCC: `D = S0` (64-bit). |
| S_NOT_B32 | D, S0 | Bitwise NOT; SCC iff result ≠ 0. |
| S_NOT_B64 | D, S0 | Bitwise NOT (64-bit); SCC iff result ≠ 0. |
| S_WQM_B32 | D, S0 | Whole-quad mask from pixel mask S0 (expand quads); SCC iff result ≠ 0. |
| S_WQM_B64 | D, S0 | Whole-quad mask (64-bit); SCC iff result ≠ 0. |
| S_BREV_B32 | D, S0 | Reverse bits (32-bit). |
| S_BREV_B64 | D, S0 | Reverse bits (64-bit). |
| S_BCNT0_I32_B32 | D, S0 | Count zero bits in S0; SCC iff count ≠ 0. |
| S_BCNT0_I32_B64 | D, S0 | Count zero bits in 64-bit S0; SCC iff count ≠ 0. |
| S_BCNT1_I32_B32 | D, S0 | Count one bits (population) in S0; SCC iff count ≠ 0. |
| S_BCNT1_I32_B64 | D, S0 | Population count (64-bit); SCC iff count ≠ 0. |
| S_FF0_I32_B32 | D, S0 | Index of first 0 bit from LSB; −1 if none. |
| S_FF0_I32_B64 | D, S0 | First 0 bit from LSB (64-bit input); −1 if none. |
| S_FF1_I32_B32 | D, S0 | Index of first 1 bit from LSB; −1 if none. |
| S_FF1_I32_B64 | D, S0 | First 1 bit from LSB (64-bit); −1 if none. |
| S_FLBIT_I32_B32 | D, S0 | Leading zeros before first 1 (MSB search); −1 if no 1. |
| S_FLBIT_I32_B64 | D, S0 | Same for 64-bit input. |
| S_FLBIT_I32 | D, S0 | Leading bits equal to sign bit of S0; −1 if all bits same. |
| S_FLBIT_I32_I64 | D, S0 | Same as `S_FLBIT_I32` for 64-bit input. |
| S_SEXT_I32_I8 | D, S0 | Sign-extend byte to i32. |
| S_SEXT_I32_I16 | D, S0 | Sign-extend 16-bit to i32. |
| S_BITSET0_B32 | D, S0 | Clear bit `S0[4:0]` in D (D is dest; bit index from S0). |
| S_BITSET0_B64 | D, S0 | Clear bit `S0[5:0]` in D. |
| S_BITSET1_B32 | D, S0 | Set bit `S0[4:0]` in D. |
| S_BITSET1_B64 | D, S0 | Set bit `S0[5:0]` in D. |
| S_GETPC_B64 | D | `D = PC + 4` (byte address of next instruction). **Must be 4 bytes.** |
| S_SETPC_B64 | S0 | `PC = S0` (byte address of target instruction). |
| S_SWAPPC_B64 | D, S0 | `D = PC + 4`; `PC = S0`. **Must be 4 bytes.** |
| S_RFE_B64 | S0 | Return from exception: clear PRIV; `PC = S0`. **Trap handler only.** |
| S_AND_SAVEEXEC_B64 | D, S0 | Save old EXEC to D; `EXEC = S0 AND EXEC`; SCC iff new EXEC ≠ 0. |
| S_OR_SAVEEXEC_B64 | D, S0 | Save old EXEC to D; `EXEC = S0 OR EXEC`; SCC iff new EXEC ≠ 0. |
| S_XOR_SAVEEXEC_B64 | D, S0 | Save old EXEC to D; `EXEC = S0 XOR EXEC`; SCC iff new EXEC ≠ 0. |
| S_ANDN2_SAVEEXEC_B64 | D, S0 | Save old EXEC to D; `EXEC = S0 AND (NOT EXEC)`; SCC iff new EXEC ≠ 0. |
| S_ORN2_SAVEEXEC_B64 | D, S0 | Save old EXEC to D; `EXEC = S0 OR (NOT EXEC)`; SCC iff new EXEC ≠ 0. |
| S_NAND_SAVEEXEC_B64 | D, S0 | Save old EXEC to D; `EXEC = NOT (S0 AND EXEC)`; SCC iff new EXEC ≠ 0. |
| S_NOR_SAVEEXEC_B64 | D, S0 | Save old EXEC to D; `EXEC = NOT (S0 OR EXEC)`; SCC iff new EXEC ≠ 0. |
| S_XNOR_SAVEEXEC_B64 | D, S0 | `D = EXEC`; `EXEC = ~(S0 ^ EXEC)`; SCC iff new EXEC ≠ 0. |
| S_QUADMASK_B32 | D, S0 | Reduce pixel mask to 8-bit quad mask; SCC iff result ≠ 0. Inverse: `S_BITREPLICATE_B64_B32`. |
| S_QUADMASK_B64 | D, S0 | Reduce to 16-bit quad mask (64-bit lanes); SCC iff result ≠ 0. |
| S_MOVRELS_B32 | D, S0 | `D = SGPR[S0.raw + M0]`. Example: `s_mov_b32 m0, 10` then `s_movrels_b32 s5, s7` → s5←s17. |
| S_MOVRELS_B64 | D, S0 | 64-bit load from relative SGPR; **M0 and SRC0 index must be even**. |
| S_MOVRELD_B32 | D, S0 | `SGPR[D.raw + M0] = S0`. Example with M0=10, D=s5 → s15←S0. |
| S_MOVRELD_B64 | D, S0 | 64-bit relative store; **M0 and DST index must be even**. |
| S_CBRANCH_JOIN | S0 | Branch-stack join: S0 = saved CSP; reconciles EXEC/PC with stacked fork state (see `S_CBRANCH_G_FORK` / `S_CBRANCH_I_FORK`). |
| S_ABS_I32 | D, S0 | Absolute value; SCC iff result ≠ 0. **Note:** `0x80000000` maps to `0x80000000` (still “negative” in i32). |
| S_SET_GPR_IDX_IDX | S0 | `M0[7:0] = S0[7:0]` (vector GPR index byte). Related: `S_SET_GPR_IDX_ON`, `S_SET_GPR_IDX_OFF`, `S_SET_GPR_IDX_MODE`. |
| S_ANDN1_SAVEEXEC_B64 | D, S0 | Save old EXEC to D; `EXEC = (NOT S0) AND EXEC`; SCC iff new EXEC ≠ 0. |
| S_ORN1_SAVEEXEC_B64 | D, S0 | Save old EXEC to D; `EXEC = (NOT S0) OR EXEC`; SCC iff new EXEC ≠ 0. |
| S_ANDN1_WREXEC_B64 | D, S0 | `EXEC = ~S0 & EXEC`; `D = EXEC` (result, not old EXEC); SCC iff EXEC ≠ 0. Waterfall optimization. |
| S_ANDN2_WREXEC_B64 | D, S0 | `EXEC = S0 & ~EXEC`; `D = EXEC`; SCC iff EXEC ≠ 0. ISA documents replacing `s_andn2` + `s_mov exec` pair. |
| S_BITREPLICATE_B64_B32 | D, S0 | Each bit of 32-bit S0 duplicated to 64-bit D. Expands quad mask toward pixel mask; inverse of `S_QUADMASK_*`. |

**Encoding note:** `S_GETPC_B64`, `S_SWAPPC_B64` are fixed 4-byte encodings per ISA.

## SOPC — Scalar Comparison and Control Instructions

| Instruction | Syntax | Description |
|---|---|---|
| S_CMP_EQ_I32 | S0, S1 | SCC = (`S0.i32 == S1.i32`). **Note:** Same opcode bits as `S_CMP_EQ_U32` (symmetry in mnemonic set). |
| S_CMP_LG_I32 | S0, S1 | SCC = (`S0.i32 != S1.i32`). Same opcode pattern as `S_CMP_LG_U32`. |
| S_CMP_GT_I32 | S0, S1 | SCC = (`S0.i32 > S1.i32`). |
| S_CMP_GE_I32 | S0, S1 | SCC = (`S0.i32 >= S1.i32`). |
| S_CMP_LT_I32 | S0, S1 | SCC = (`S0.i32 < S1.i32`). |
| S_CMP_LE_I32 | S0, S1 | SCC = (`S0.i32 <= S1.i32`). |
| S_CMP_EQ_U32 | S0, S1 | SCC = (`S0.u32 == S1.u32`). |
| S_CMP_LG_U32 | S0, S1 | SCC = (`S0.u32 != S1.u32`). |
| S_CMP_GT_U32 | S0, S1 | SCC = (`S0.u32 > S1.u32`). |
| S_CMP_GE_U32 | S0, S1 | SCC = (`S0.u32 >= S1.u32`). |
| S_CMP_LT_U32 | S0, S1 | SCC = (`S0.u32 < S1.u32`). |
| S_CMP_LE_U32 | S0, S1 | SCC = (`S0.u32 <= S1.u32`). |
| S_BITCMP0_B32 | S0, S1 | SCC iff bit `S0[S1[4:0]]` is 0. |
| S_BITCMP1_B32 | S0, S1 | SCC iff bit `S0[S1[4:0]]` is 1. |
| S_BITCMP0_B64 | S0, S1 | SCC iff bit `S0[S1[5:0]]` is 0. |
| S_BITCMP1_B64 | S0, S1 | SCC iff bit `S0[S1[5:0]]` is 1. |
| S_SETVSKIP | S0, S1 | `VSKIP = S0[S1[4:0]]`; enables/disables VSKIP (no VOP*/MUBUF/MIMG/DS/FLAT issue). **Note:** VSKIPped mem ops don’t advance waitcnt; consider `S_WAITCNT 0` before enable if memory is outstanding. |
| S_SET_GPR_IDX_ON | S0, S1 | Enable GPR indexing: `GPR_IDX_EN=1`; `M0[7:0]=S0[7:0]`; `M0[15:12]=raw S1[3:0]` for VSRC0/1/2/VDST_REL flags. |
| S_CMP_EQ_U64 | S0, S1 | SCC = (`S0.u64 == S1.u64`). |
| S_CMP_LG_U64 | S0, S1 | SCC = (`S0.u64 != S1.u64`). |

## SOPP — Scalar Program Control Instructions

| Instruction | Syntax | Description |
|---|---|---|
| S_NOP | simm16 | No operation; delay by `simm16[3:0]` wait states (0 = next insn next clock; 0xF = 16 clocks). |
| S_ENDPGM | simm16 | End program; terminate wave. Hardware implicitly `S_WAITCNT 0` first. See `S_ENDPGM_SAVED` for context-switch. |
| S_BRANCH | simm16 | `PC = PC + signext(simm16)×4 + 4`. Short PC-relative branch. |
| S_WAKEUP | simm16 | Ping other waves in threadgroup to wake early from `S_SLEEP`; no-op if not in TG. |
| S_CBRANCH_SCC0 | simm16 | Branch if SCC==0 (same PC-relative encoding as `S_BRANCH`). |
| S_CBRANCH_SCC1 | simm16 | Branch if SCC==1. |
| S_CBRANCH_VCCZ | simm16 | Branch if VCCZ==1. |
| S_CBRANCH_VCCNZ | simm16 | Branch if VCCZ==0. |
| S_CBRANCH_EXECZ | simm16 | Branch if EXECZ==1. |
| S_CBRANCH_EXECNZ | simm16 | Branch if EXECZ==0. |
| S_BARRIER | simm16 | Threadgroup wave barrier; legal in trap handlers. Does not drain counters—protect mem with `S_WAITCNT` if needed. |
| S_SETKILL | simm16 | Kill wave if `simm16[0]==1` (debug / host-kill behavior). |
| S_WAITCNT | simm16 | Wait on VM/LGKM/export counters per packed `simm16` fields (vmcnt low/high, export, LGKM). |
| S_SETHALT | simm16 | Halt wave if `simm16[0]==1`, clear if 0; ignored while PRIV (in trap), may halt after return if still set. |
| S_SLEEP | simm16 | Sleep ~`64×(simm[6:0]-1)`..`64×simm[6:0]` clocks; 0 = no sleep. |
| S_SETPRIO | simm16 | User wave priority = `simm16[1:0]` (0 lowest, 3 highest). |
| S_SENDMSG | simm16 | Send message upstream; `simm16[9:0]` message type (see §12.5.1 in ISA). |
| S_SENDMSGHALT | simm16 | `S_SENDMSG` then halt wave. |
| S_TRAP | simm16 | Enter trap handler; `TrapID = simm16[7:0]` (TrapID 0 reserved for HW). PC in trap points at `S_TRAP`. Sets PRIV. |
| S_ICACHE_INV | simm16 | Invalidate L0 I$; need **16× `S_NOP`** or a branch after to flush internal buffers. |
| S_INCPERFLEVEL | simm16 | Increment perf counter `simm16[3:0]`. |
| S_DECPERFLEVEL | simm16 | Decrement perf counter `simm16[3:0]`. |
| S_TTRACEDATA | simm16 | Send **M0** as user data to thread trace. |
| S_CBRANCH_CDBGSYS | simm16 | Branch if `COND_DBG_SYS` set. |
| S_CBRANCH_CDBGUSER | simm16 | Branch if `COND_DBG_USER` set. |
| S_CBRANCH_CDBGSYS_OR_USER | simm16 | Branch if either debug flag set. |
| S_CBRANCH_CDBGSYS_AND_USER | simm16 | Branch if both debug flags set. |
| S_ENDPGM_SAVED | simm16 | Context-saved wave termination; implicit `S_WAITCNT 0`. |
| S_SET_GPR_IDX_OFF | simm16 | Disable GPR indexing (`GPR_IDX_EN=0`); does not change M0. |
| S_SET_GPR_IDX_MODE | simm16 | Set `M0[15:12]` from `simm16[3:0]` (VSRC0/1/2/VDST_REL). |

### S_SENDMSG immediate layout (ISA §12.5.1 excerpt)

Message types use `SIMM16` subfields; examples from the ISA table: **Interrupt** (1) — payload in M0[23:0]; **Save wave** (4); **Stall wave gen** (5); **Halt waves** (6); **Get doorbell ID** (10) — returns doorbell in EXEC, physical address bits [12:3]. `SIMM16[3:0]=0` is illegal for “none”.
