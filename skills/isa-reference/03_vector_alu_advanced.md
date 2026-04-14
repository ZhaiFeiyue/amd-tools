# CDNA4 Vector ALU — VOP3P, VOP3A, VOP3B (gfx950)

Source: CDNA4 ISA sections 12.10–12.11 (`docs/cdna4_isa_full.txt`, lines 8921–14612). VOP3P is **packed math** (two 16-bit or sub-word operations per instruction). VOP3A covers most VALU three-operand encodings; **VOP3B** is used for opcodes that need a **scalar destination** (e.g. `V_ADD_CO_U32` with carry-out SGPR).

Tables list mnemonics with one-line descriptions extracted from the ISA; see the PDF for full semantics, modifiers, and SDWA/DPP rules.

## VOP3P instructions

### Packed (V_PK_*) — FP16/FP32/int component-wise

| Instruction | Brief description |
|---|---|
| V_PK_MAD_I16 | Multiply two packed signed 16-bit integer inputs component-wise, add a packed signed 16-bit integer value from a third input component-wise, and store the result into a vector register. |
| V_PK_MUL_LO_U16 | Multiply two packed unsigned 16-bit integer inputs component-wise and store the low bits of each resulting component into a vector register. |
| V_PK_ADD_I16 | Add two packed signed 16-bit integer inputs component-wise and store the result into a vector register. |
| V_PK_SUB_I16 | Subtract the second packed signed 16-bit integer input from the first input component-wise and store the result into a vector register. |
| V_PK_LSHLREV_B16 | Given a packed shift count in the first vector input, calculate the component-wise logical shift left of the second packed vector input and store the result into a vector register. |
| V_PK_LSHRREV_B16 | Given a packed shift count in the first vector input, calculate the component-wise logical shift right of the second packed vector input and store the result into a vector register. |
| V_PK_ASHRREV_I16 | Given a packed shift count in the first vector input, calculate the component-wise arithmetic shift right (preserving sign bit) of the second packed vector input and store the result into a vector register. |
| V_PK_MAX_I16 | Select the component-wise maximum of two packed signed 16-bit integer inputs and store the selected values into a vector register. |
| V_PK_MIN_I16 | Select the component-wise minimum of two packed signed 16-bit integer inputs and store the selected values into a vector register. |
| V_PK_MAD_U16 | Multiply two packed unsigned 16-bit integer inputs component-wise, add a packed unsigned 16-bit integer value from a third input component-wise, and store the result into a vector register. |
| V_PK_ADD_U16 | Add two packed unsigned 16-bit integer inputs component-wise and store the result into a vector register. |
| V_PK_SUB_U16 | Subtract the second packed unsigned 16-bit integer input from the first input component-wise and store the result into a vector register. |
| V_PK_MAX_U16 | Select the component-wise maximum of two packed unsigned 16-bit integer inputs and store the selected values into a vector register. |
| V_PK_MIN_U16 | Select the component-wise minimum of two packed unsigned 16-bit integer inputs and store the selected values into a vector register. |
| V_PK_FMA_F16 | Multiply two packed half-precision float inputs component-wise and add a third input component-wise using fused multiply add, and store the result into a vector register. |
| V_PK_ADD_F16 | Add two packed half-precision float inputs component-wise and store the result into a vector register. |
| V_PK_MUL_F16 | Multiply two packed half-precision float inputs component-wise and store the result into a vector register. |
| V_PK_MIN_F16 | Select the component-wise minimum of two packed half-precision float inputs and store the result into a vector register. |
| V_PK_MAX_F16 | Select the component-wise maximum of two packed half-precision float inputs and store the result into a vector register. |
| V_PK_MINIMUM3_F16 | Select the component-wise IEEE minimum() of three half-precision float inputs and store the result into a vector register. |
| V_PK_MAXIMUM3_F16 | Select the component-wise IEEE maximum() of three half-precision float inputs and store the result into a vector register. |
| V_PK_FMA_F32 | Multiply two packed single-precision float inputs component-wise and add a third input component-wise using fused multiply add, and store the result into a vector register. |
| V_PK_MUL_F32 | Multiply two packed single-precision float inputs component-wise and store the result into a vector register. |
| V_PK_ADD_F32 | Add two packed single-precision float inputs component-wise and store the result into a vector register. |
| V_PK_MOV_B32 | Move data from two vector inputs into two vector registers. |

### Dot products (V_DOT*)

| Instruction | Brief description |
|---|---|
| V_DOT2_F32_BF16 | Calculate the dot product of BF16 float 2-vectors from the first and second inputs, convert the product to single-precision float format, add the third input and store the result into a vector register. |
| V_DOT2_F32_F16 | Compute the dot product of two packed 2-D half-precision float inputs in the single-precision float domain, add a single-precision float value from the third input and store the result into a vecto... |
| V_DOT2_I32_I16 | Compute the dot product of two packed 2-D signed 16-bit integer inputs in the signed 32-bit integer domain, add a signed 32-bit integer value from the third input and store the result into a vector... |
| V_DOT2_U32_U16 | Compute the dot product of two packed 2-D unsigned 16-bit integer inputs in the unsigned 32-bit integer domain, add an unsigned 32-bit integer value from the third input and store the result into a... |
| V_DOT4_I32_I8 | Compute the dot product of two packed 4-D signed 8-bit integer inputs in the signed 32-bit integer domain, add a signed 32-bit integer value from the third input and store the result into a vector ... |
| V_DOT4_U32_U8 | Compute the dot product of two packed 4-D unsigned 8-bit integer inputs in the unsigned 32-bit integer domain, add an unsigned 32-bit integer value from the third input and store the result into a ... |
| V_DOT8_I32_I4 | Compute the dot product of two packed 8-D signed 4-bit integer inputs in the signed 32-bit integer domain, add a signed 32-bit integer value from the third input and store the result into a vector ... |
| V_DOT8_U32_U4 | Compute the dot product of two packed 8-D unsigned 4-bit integer inputs in the unsigned 32-bit integer domain, add an unsigned 32-bit integer value from the third input and store the result into a ... |

### Mixed-precision FMA (V_MAD_MIX*)

| Instruction | Brief description |
|---|---|
| V_MAD_MIX_F32 | Multiply two inputs and add a third input where the inputs are a mix of half-precision float and single-precision float values. |
| V_MAD_MIXLO_F16 | Multiply two inputs and add a third input where the inputs are a mix of half-precision float and single-precision float values. |
| V_MAD_MIXHI_F16 | Multiply two inputs and add a third input where the inputs are a mix of half-precision float and single-precision float values. |

### MFMA / sparse MFMA (matrix)

| Instruction | Brief description |
|---|---|
| V_MFMA_F32_16X16X128_F8F6F4 | Multiply the 16x128 matrix in the first input by the 128x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_MFMA_F32_32X32X64_F8F6F4 | Multiply the 32x64 matrix in the first input by the 64x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. |
| V_MFMA_F32_16X16X32_BF16 | Multiply the 16x32 matrix in the first input by the 32x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_MFMA_I32_16X16X64_I8 | Multiply the 16x64 matrix in the first input by the 64x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_MFMA_F32_32X32X16_BF16 | Multiply the 32x16 matrix in the first input by the 16x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. |
| V_MFMA_I32_32X32X32_I8 | Multiply the 32x32 matrix in the first input by the 32x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. |
| V_SMFMAC_F32_16X16X64_BF16 | Multiply the 16x64 sparse matrix in the first input by the 64x16 matrix in the second input and accumulate the result into the 16x16 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_I32_16X16X128_I8 | Multiply the 16x128 sparse matrix in the first input by the 128x16 matrix in the second input and accumulate the result into the 16x16 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_F32_16X16X128_BF8_BF8 | Multiply the 16x128 sparse matrix in the first input by the 128x16 matrix in the second input and accumulate the result into the 16x16 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_F32_16X16X128_BF8_FP8 | Multiply the 16x128 sparse matrix in the first input by the 128x16 matrix in the second input and accumulate the result into the 16x16 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_F32_16X16X128_FP8_BF8 | Multiply the 16x128 sparse matrix in the first input by the 128x16 matrix in the second input and accumulate the result into the 16x16 matrix stored in the destination registers using fused multiply add. |
| V_MFMA_F32_32X32X1_2B_F32 | Multiply the 32x1 matrix in the first input by the 1x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. |
| V_MFMA_F32_16X16X1_4B_F32 | Multiply the 16x1 matrix in the first input by the 1x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_MFMA_F32_4X4X1_16B_F32 | Multiply the 4x1 matrix in the first input by the 1x4 matrix in the second input and add the 4x4 matrix in the third input using fused multiply add. |
| V_SMFMAC_F32_16X16X128_FP8_FP8 | Multiply the 16x128 sparse matrix in the first input by the 128x16 matrix in the second input and accumulate the result into the 16x16 matrix stored in the destination registers using fused multiply add. |
| V_MFMA_F32_32X32X2_F32 | Multiply the 32x2 matrix in the first input by the 2x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. |
| V_MFMA_F32_16X16X4_F32 | Multiply the 16x4 matrix in the first input by the 4x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_SMFMAC_F32_32X32X32_BF16 | Multiply the 32x32 sparse matrix in the first input by the 32x32 matrix in the second input and accumulate the result into the 32x32 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_I32_32X32X64_I8 | Multiply the 32x64 sparse matrix in the first input by the 64x32 matrix in the second input and accumulate the result into the 32x32 matrix stored in the destination registers using fused multiply add. |
| V_MFMA_F32_32X32X4_2B_F16 | Multiply the 32x4 matrix in the first input by the 4x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. |
| V_MFMA_F32_16X16X4_4B_F16 | Multiply the 16x4 matrix in the first input by the 4x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_MFMA_F32_4X4X4_16B_F16 | Multiply the 4x4 matrix in the first input by the 4x4 matrix in the second input and add the 4x4 matrix in the third input using fused multiply add. |
| V_SMFMAC_F32_32X32X64_BF8_BF8 | Multiply the 32x64 sparse matrix in the first input by the 64x32 matrix in the second input and accumulate the result into the 32x32 matrix stored in the destination registers using fused multiply add. |
| V_MFMA_F32_32X32X8_F16 | Multiply the 32x8 matrix in the first input by the 8x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. |
| V_MFMA_F32_16X16X16_F16 | Multiply the 16x16 matrix in the first input by the 16x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_SMFMAC_F32_32X32X64_BF8_FP8 | Multiply the 32x64 sparse matrix in the first input by the 64x32 matrix in the second input and accumulate the result into the 32x32 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_F32_32X32X64_FP8_BF8 | Multiply the 32x64 sparse matrix in the first input by the 64x32 matrix in the second input and accumulate the result into the 32x32 matrix stored in the destination registers using fused multiply add. |
| V_MFMA_I32_32X32X4_2B_I8 | Multiply the 32x4 matrix in the first input by the 4x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. |
| V_MFMA_I32_16X16X4_4B_I8 | Multiply the 16x4 matrix in the first input by the 4x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_MFMA_I32_4X4X4_16B_I8 | Multiply the 4x4 matrix in the first input by the 4x4 matrix in the second input and add the 4x4 matrix in the third input using fused multiply add. |
| V_SMFMAC_F32_32X32X64_FP8_FP8 | Multiply the 32x64 sparse matrix in the first input by the 64x32 matrix in the second input and accumulate the result into the 32x32 matrix stored in the destination registers using fused multiply add. |
| V_MFMA_F32_16X16X32_F16 | Multiply the 16x32 matrix in the first input by the 32x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_MFMA_F32_32X32X16_F16 | Multiply the 32x16 matrix in the first input by the 16x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. |
| V_MFMA_I32_32X32X16_I8 | Multiply the 32x16 matrix in the first input by the 16x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. |
| V_MFMA_I32_16X16X32_I8 | Multiply the 16x32 matrix in the first input by the 32x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_SMFMAC_F32_16X16X64_F16 | Multiply the 16x64 sparse matrix in the first input by the 64x16 matrix in the second input and accumulate the result into the 16x16 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_F32_32X32X32_F16 | Multiply the 32x32 sparse matrix in the first input by the 32x32 matrix in the second input and accumulate the result into the 32x32 matrix stored in the destination registers using fused multiply add. |
| V_MFMA_F32_32X32X4_2B_BF16 | Multiply the 32x4 matrix in the first input by the 4x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. |
| V_MFMA_F32_16X16X4_4B_BF16 | Multiply the 16x4 matrix in the first input by the 4x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_MFMA_F32_4X4X4_16B_BF16 | Multiply the 4x4 matrix in the first input by the 4x4 matrix in the second input and add the 4x4 matrix in the third input using fused multiply add. |
| V_MFMA_F32_32X32X8_BF16 | Multiply the 32x8 matrix in the first input by the 8x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. |
| V_MFMA_F32_16X16X16_BF16 | Multiply the 16x16 matrix in the first input by the 16x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_SMFMAC_F32_16X16X32_F16 | Multiply the 16x32 sparse matrix in the first input by the 32x16 matrix in the second input and accumulate the result into the 16x16 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_F32_32X32X16_F16 | Multiply the 32x16 sparse matrix in the first input by the 16x32 matrix in the second input and accumulate the result into the 32x32 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_F32_16X16X32_BF16 | Multiply the 16x32 sparse matrix in the first input by the 32x16 matrix in the second input and accumulate the result into the 16x16 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_F32_32X32X16_BF16 | Multiply the 32x16 sparse matrix in the first input by the 16x32 matrix in the second input and accumulate the result into the 32x32 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_I32_16X16X64_I8 | Multiply the 16x64 sparse matrix in the first input by the 64x16 matrix in the second input and accumulate the result into the 16x16 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_I32_32X32X32_I8 | Multiply the 32x32 sparse matrix in the first input by the 32x32 matrix in the second input and accumulate the result into the 32x32 matrix stored in the destination registers using fused multiply add. |
| V_MFMA_F64_16X16X4_F64 | Multiply the 16x4 matrix in the first input by the 4x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_MFMA_F64_4X4X4_4B_F64 | Multiply the 4x4 matrix in the first input by the 4x4 matrix in the second input and add the 4x4 matrix in the third input using fused multiply add. |
| V_MFMA_F32_16X16X32_BF8_BF8 | Multiply the 16x32 matrix in the first input by the 32x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_MFMA_F32_16X16X32_BF8_FP8 | Multiply the 16x32 matrix in the first input by the 32x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_MFMA_F32_16X16X32_FP8_BF8 | Multiply the 16x32 matrix in the first input by the 32x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_MFMA_F32_16X16X32_FP8_FP8 | Multiply the 16x32 matrix in the first input by the 32x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. |
| V_MFMA_F32_32X32X16_BF8_BF8 | Multiply the 32x16 matrix in the first input by the 16x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. |
| V_MFMA_F32_32X32X16_BF8_FP8 | Multiply the 32x16 matrix in the first input by the 16x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. |
| V_MFMA_F32_32X32X16_FP8_BF8 | Multiply the 32x16 matrix in the first input by the 16x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. |
| V_MFMA_F32_32X32X16_FP8_FP8 | Multiply the 32x16 matrix in the first input by the 16x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. |
| V_SMFMAC_F32_16X16X64_BF8_BF8 | Multiply the 16x64 sparse matrix in the first input by the 64x16 matrix in the second input and accumulate the result into the 16x16 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_F32_16X16X64_BF8_FP8 | Multiply the 16x64 sparse matrix in the first input by the 64x16 matrix in the second input and accumulate the result into the 16x16 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_F32_16X16X64_FP8_BF8 | Multiply the 16x64 sparse matrix in the first input by the 64x16 matrix in the second input and accumulate the result into the 16x16 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_F32_16X16X64_FP8_FP8 | Multiply the 16x64 sparse matrix in the first input by the 64x16 matrix in the second input and accumulate the result into the 16x16 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_F32_32X32X32_BF8_BF8 | Multiply the 32x32 sparse matrix in the first input by the 32x32 matrix in the second input and accumulate the result into the 32x32 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_F32_32X32X32_BF8_FP8 | Multiply the 32x32 sparse matrix in the first input by the 32x32 matrix in the second input and accumulate the result into the 32x32 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_F32_32X32X32_FP8_BF8 | Multiply the 32x32 sparse matrix in the first input by the 32x32 matrix in the second input and accumulate the result into the 32x32 matrix stored in the destination registers using fused multiply add. |
| V_SMFMAC_F32_32X32X32_FP8_FP8 | Multiply the 32x32 sparse matrix in the first input by the 32x32 matrix in the second input and accumulate the result into the 32x32 matrix stored in the destination registers using fused multiply add. |

### Accumulator VGPR moves

| Instruction | Brief description |
|---|---|
| V_ACCVGPR_READ | Move 32 bits of data from an accumulator vector register into an architectural vector register. |
| V_ACCVGPR_WRITE | Move 32 bits of data from an architectural vector register into an accumulator vector register. |

## VOP3A and VOP3B instructions

### Arithmetic and shifts

| Instruction | Brief description |
|---|---|
| V_ADD_F32 | Add two floating point inputs and store the result into a vector register. |
| V_SUB_F32 | Subtract the second floating point input from the first input and store the result into a vector register. |
| V_SUBREV_F32 | Subtract the first floating point input from the second input and store the result into a vector register. |
| V_FMAC_F64 | Multiply two floating point inputs and accumulate the result into the destination register using fused multiply add. |
| V_MUL_F32 | Multiply two floating point inputs and store the result into a vector register. |
| V_MUL_I32_I24 | Multiply two signed 24-bit integer inputs and store the result as a signed 32-bit integer into a vector register. |
| V_MUL_HI_I32_I24 | Multiply two signed 24-bit integer inputs and store the high 32 bits of the result as a signed 32-bit integer into a vector register. |
| V_MUL_U32_U24 | Multiply two unsigned 24-bit integer inputs and store the result as an unsigned 32-bit integer into a vector register. |
| V_MUL_HI_U32_U24 | Multiply two unsigned 24-bit integer inputs and store the high 32 bits of the result as an unsigned 32-bit integer into a vector register. |
| V_LSHRREV_B32 | Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store the result into a vector register. |
| V_ASHRREV_I32 | Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second vector input and store the result into a vector register. |
| V_LSHLREV_B32 | Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the result into a vector register. |
| V_ADD_CO_U32 | Add two unsigned 32-bit integer inputs, store the result into a vector register and store the carry-out mask into a scalar register. |
| V_SUB_CO_U32 | Subtract the second unsigned 32-bit integer input from the first input, store the result into a vector register and store the carry-out mask into a scalar register. |
| V_SUBREV_CO_U32 | Subtract the first unsigned 32-bit integer input from the second input, store the result into a vector register and store the carry-out mask into a scalar register. |
| V_ADDC_CO_U32 | Add two unsigned 32-bit integer inputs and a bit from a carry-in mask, store the result into a vector register and store the carry-out mask into a scalar register. |
| V_SUBB_CO_U32 | Subtract the second unsigned 32-bit integer input from the first input, subtract a bit from the carry-in mask, store the result into a vector register and store the carry-out mask into a scalar register. |
| V_SUBBREV_CO_U32 | Subtract the first unsigned 32-bit integer input from the second input, subtract a bit from the carry-in mask, store the result into a vector register and store the carry-out mask into a scalar register. |
| V_ADD_F16 | Add two floating point inputs and store the result into a vector register. |
| V_SUB_F16 | Subtract the second floating point input from the first input and store the result into a vector register. |
| V_SUBREV_F16 | Subtract the first floating point input from the second input and store the result into a vector register. |
| V_MUL_F16 | Multiply two floating point inputs and store the result into a vector register. |
| V_ADD_U16 | Add two unsigned 16-bit integer inputs and store the result into a vector register. |
| V_SUB_U16 | Subtract the second unsigned 16-bit integer input from the first input and store the result into a vector register. |
| V_SUBREV_U16 | Subtract the first unsigned 16-bit integer input from the second input and store the result into a vector register. |
| V_MUL_LO_U16 | Multiply two unsigned 16-bit integer inputs and store the low bits of the result into a vector register. |
| V_LSHLREV_B16 | Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the result into a vector register. |
| V_LSHRREV_B16 | Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store the result into a vector register. |
| V_ASHRREV_I16 | Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second vector input and store the result into a vector register. |
| V_ADD_U32 | Add two unsigned 32-bit integer inputs and store the result into a vector register. |
| V_SUB_U32 | Subtract the second unsigned 32-bit integer input from the first input and store the result into a vector register. |
| V_SUBREV_U32 | Subtract the first unsigned 32-bit integer input from the second input and store the result into a vector register. |
| V_FMAC_F32 | Multiply two floating point inputs and accumulate the result into the destination register using fused multiply add. |
| V_PK_FMAC_F16 | Multiply two packed half-precision float inputs component-wise and accumulate the result into the destination register using fused multiply add. |
| V_MAD_I32_I24 | Multiply two signed 24-bit integer inputs in the signed 32-bit integer domain, add a signed 32-bit integer value from a third input, and store the result as a signed 32-bit integer into a vector re... |
| V_MAD_U32_U24 | Multiply two unsigned 24-bit integer inputs in the unsigned 32-bit integer domain, add a unsigned 32-bit integer value from a third input, and store the result as an unsigned 32-bit integer into a ... |
| V_FMA_F32 | Multiply two single-precision float inputs and add a third input using fused multiply add, and store the result into a vector register. |
| V_FMA_F64 | Multiply two double-precision float inputs and add a third input using fused multiply add, and store the result into a vector register. |
| V_MAD_U64_U32 | Multiply two unsigned integer inputs, add a third unsigned integer input, store the result into a 64-bit vector register and store the overflow/carryout into a scalar mask register. |
| V_MAD_I64_I32 | Multiply two signed integer inputs, add a third signed integer input, store the result into a 64-bit vector register and store the overflow/carryout into a scalar mask register. |
| V_MAD_LEGACY_F16 | Multiply add of FP16 values. |
| V_MAD_LEGACY_U16 | Multiply add of unsigned short values. |
| V_MAD_LEGACY_I16 | Multiply add of signed short values. |
| V_FMA_LEGACY_F16 | Fused half precision multiply add. |
| V_MAD_U32_U16 | Multiply two unsigned 16-bit integer inputs in the unsigned 32-bit integer domain, add an unsigned 32-bit integer value from a third input, and store the result as an unsigned 32-bit integer into a... |
| V_MAD_I32_I16 | Multiply two signed 16-bit integer inputs in the signed 32-bit integer domain, add a signed 32-bit integer value from a third input, and store the result as a signed 32-bit integer into a vector re... |
| V_LSHL_ADD_U32 | Given a shift count in the second input, calculate the logical shift left of the first input, then add the third input to the intermediate result, then store the final result into a vector register. |
| V_ADD_LSHL_U32 | Add the first two integer inputs, then given a shift count in the third input, calculate the logical shift left of the intermediate result, then store the final result into a vector register. |
| V_ADD3_U32 | Add three unsigned inputs and store the result into a vector register. |
| V_LSHL_OR_B32 | Given a shift count in the second input, calculate the logical shift left of the first input, then calculate the bitwise OR of the intermediate result and the third input, then store the final resu... |
| V_MAD_F16 | Multiply two half-precision float inputs and add a third input, and store the result into a vector register. |
| V_MAD_U16 | Multiply two unsigned 16-bit integer inputs, add an unsigned 16-bit integer value from a third input, and store the result into a vector register. |
| V_MAD_I16 | Multiply two signed 16-bit integer inputs, add a signed 16-bit integer value from a third input, and store the result into a vector register. |
| V_FMA_F16 | Multiply two half-precision float inputs and add a third input using fused multiply add, and store the result into a vector register. |
| V_LSHL_ADD_U64 | Given a shift count in the second input, calculate the logical shift left of the first input, then add the third input to the intermediate result, then store the final result into a vector register. |
| V_ASHR_PK_I8_I32 | Given two signed 32-bit integers and a shift count, calculate the arithmetic shift right (preserving sign bit) of the two integers, saturate the two results in the signed 8-bit interval [-128, 127], pack the bytes and store the result into a vector register. |
| V_ASHR_PK_U8_I32 | Given two signed 32-bit integers and a shift count, calculate the arithmetic shift right (preserving sign bit) of the two integers, saturate the two results in the unsigned 8-bit interval [0, 255], pack the bytes and store the result into a vector register. |
| V_ADD_F64 | Add two floating point inputs and store the result into a vector register. |
| V_MUL_F64 | Multiply two floating point inputs and store the result into a vector register. |
| V_MUL_LO_U32 | Multiply two unsigned 32-bit integer inputs and store the result into a vector register. |
| V_MUL_HI_U32 | Multiply two unsigned 32-bit integer inputs and store the high 32 bits of the result into a vector register. |
| V_MUL_HI_I32 | Multiply two signed 32-bit integer inputs and store the high 32 bits of the result into a vector register. |
| V_LSHLREV_B64 | Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the result into a vector register. |
| V_LSHRREV_B64 | Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store the result into a vector register. |
| V_ASHRREV_I64 | Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second vector input and store the result into a vector register. |
| V_ADD_I32 | Add two signed 32-bit integer inputs and store the result into a vector register. |
| V_SUB_I32 | Subtract the second signed 32-bit integer input from the first input and store the result into a vector register. |
| V_ADD_I16 | Add two signed 16-bit integer inputs and store the result into a vector register. |
| V_SUB_I16 | Subtract the second signed 16-bit integer input from the first input and store the result into a vector register. |
| V_MUL_LEGACY_F32 | Multiply two floating point inputs and store the result into a vector register. |

### Compares and class

| Instruction | Brief description |
|---|---|
| V_CMP_CLASS_F32 | Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a single-precision float, and set the per-lane condition code to the result. |
| V_CMPX_CLASS_F32 | Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a single-precision float, and set the per-lane condition code to the result. |
| V_CMP_CLASS_F64 | Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a double-precision float, and set the per-lane condition code to the result. |
| V_CMPX_CLASS_F64 | Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a double-precision float, and set the per-lane condition code to the result. |
| V_CMP_CLASS_F16 | Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a half-precision float, and set the per-lane condition code to the result. |
| V_CMPX_CLASS_F16 | Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a half-precision float, and set the per-lane condition code to the result. |
| V_CMP_F_F16 | Set the per-lane condition code to 0. |
| V_CMP_LT_F16 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_F16 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_F16 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_F16 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_LG_F16 | Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. |
| V_CMP_GE_F16 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_O_F16 | Set the per-lane condition code to 1 iff the first input is orderable to the second input. |
| V_CMP_U_F16 | Set the per-lane condition code to 1 iff the first input is not orderable to the second input. |
| V_CMP_NGE_F16 | Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. |
| V_CMP_NLG_F16 | Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. |
| V_CMP_NGT_F16 | Set the per-lane condition code to 1 iff the first input is not greater than the second input. |
| V_CMP_NLE_F16 | Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. |
| V_CMP_NEQ_F16 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_NLT_F16 | Set the per-lane condition code to 1 iff the first input is not less than the second input. |
| V_CMP_TRU_F16 | Set the per-lane condition code to 1. |
| V_CMPX_F_F16 | Set the per-lane condition code to 0. |
| V_CMPX_LT_F16 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_F16 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_F16 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_F16 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_LG_F16 | Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. |
| V_CMPX_GE_F16 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_O_F16 | Set the per-lane condition code to 1 iff the first input is orderable to the second input. |
| V_CMPX_U_F16 | Set the per-lane condition code to 1 iff the first input is not orderable to the second input. |
| V_CMPX_NGE_F16 | Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. |
| V_CMPX_NLG_F16 | Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. |
| V_CMPX_NGT_F16 | Set the per-lane condition code to 1 iff the first input is not greater than the second input. |
| V_CMPX_NLE_F16 | Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. |
| V_CMPX_NEQ_F16 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_NLT_F16 | Set the per-lane condition code to 1 iff the first input is not less than the second input. |
| V_CMPX_TRU_F16 | Set the per-lane condition code to 1. |
| V_CMP_F_F32 | Set the per-lane condition code to 0. |
| V_CMP_LT_F32 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_F32 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_F32 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_F32 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_LG_F32 | Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. |
| V_CMP_GE_F32 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_O_F32 | Set the per-lane condition code to 1 iff the first input is orderable to the second input. |
| V_CMP_U_F32 | Set the per-lane condition code to 1 iff the first input is not orderable to the second input. |
| V_CMP_NGE_F32 | Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. |
| V_CMP_NLG_F32 | Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. |
| V_CMP_NGT_F32 | Set the per-lane condition code to 1 iff the first input is not greater than the second input. |
| V_CMP_NLE_F32 | Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. |
| V_CMP_NEQ_F32 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_NLT_F32 | Set the per-lane condition code to 1 iff the first input is not less than the second input. |
| V_CMP_TRU_F32 | Set the per-lane condition code to 1. |
| V_CMPX_F_F32 | Set the per-lane condition code to 0. |
| V_CMPX_LT_F32 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_F32 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_F32 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_F32 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_LG_F32 | Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. |
| V_CMPX_GE_F32 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_O_F32 | Set the per-lane condition code to 1 iff the first input is orderable to the second input. |
| V_CMPX_U_F32 | Set the per-lane condition code to 1 iff the first input is not orderable to the second input. |
| V_CMPX_NGE_F32 | Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. |
| V_CMPX_NLG_F32 | Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. |
| V_CMPX_NGT_F32 | Set the per-lane condition code to 1 iff the first input is not greater than the second input. |
| V_CMPX_NLE_F32 | Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. |
| V_CMPX_NEQ_F32 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_NLT_F32 | Set the per-lane condition code to 1 iff the first input is not less than the second input. |
| V_CMPX_TRU_F32 | Set the per-lane condition code to 1. |
| V_CMP_F_F64 | Set the per-lane condition code to 0. |
| V_CMP_LT_F64 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_F64 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_F64 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_F64 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_LG_F64 | Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. |
| V_CMP_GE_F64 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_O_F64 | Set the per-lane condition code to 1 iff the first input is orderable to the second input. |
| V_CMP_U_F64 | Set the per-lane condition code to 1 iff the first input is not orderable to the second input. |
| V_CMP_NGE_F64 | Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. |
| V_CMP_NLG_F64 | Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. |
| V_CMP_NGT_F64 | Set the per-lane condition code to 1 iff the first input is not greater than the second input. |
| V_CMP_NLE_F64 | Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. |
| V_CMP_NEQ_F64 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_NLT_F64 | Set the per-lane condition code to 1 iff the first input is not less than the second input. |
| V_CMP_TRU_F64 | Set the per-lane condition code to 1. |
| V_CMPX_F_F64 | Set the per-lane condition code to 0. |
| V_CMPX_LT_F64 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_F64 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_F64 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_F64 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_LG_F64 | Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. |
| V_CMPX_GE_F64 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_O_F64 | Set the per-lane condition code to 1 iff the first input is orderable to the second input. |
| V_CMPX_U_F64 | Set the per-lane condition code to 1 iff the first input is not orderable to the second input. |
| V_CMPX_NGE_F64 | Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. |
| V_CMPX_NLG_F64 | Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. |
| V_CMPX_NGT_F64 | Set the per-lane condition code to 1 iff the first input is not greater than the second input. |
| V_CMPX_NLE_F64 | Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. |
| V_CMPX_NEQ_F64 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_NLT_F64 | Set the per-lane condition code to 1 iff the first input is not less than the second input. |
| V_CMPX_TRU_F64 | Set the per-lane condition code to 1. |
| V_CMP_F_I16 | Set the per-lane condition code to 0. |
| V_CMP_LT_I16 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_I16 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_I16 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_I16 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_NE_I16 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_GE_I16 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_T_I16 | Set the per-lane condition code to 1. |
| V_CMP_F_U16 | Set the per-lane condition code to 0. |
| V_CMP_LT_U16 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_U16 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_U16 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_U16 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_NE_U16 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_GE_U16 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_T_U16 | Set the per-lane condition code to 1. |
| V_CMPX_F_I16 | Set the per-lane condition code to 0. |
| V_CMPX_LT_I16 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_I16 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_I16 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_I16 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_NE_I16 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_GE_I16 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_T_I16 | Set the per-lane condition code to 1. |
| V_CMPX_F_U16 | Set the per-lane condition code to 0. |
| V_CMPX_LT_U16 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_U16 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_U16 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_U16 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_NE_U16 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_GE_U16 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_T_U16 | Set the per-lane condition code to 1. |
| V_CMP_F_I32 | Set the per-lane condition code to 0. |
| V_CMP_LT_I32 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_I32 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_I32 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_I32 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_NE_I32 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_GE_I32 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_T_I32 | Set the per-lane condition code to 1. |
| V_CMP_F_U32 | Set the per-lane condition code to 0. |
| V_CMP_LT_U32 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_U32 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_U32 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_U32 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_NE_U32 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_GE_U32 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_T_U32 | Set the per-lane condition code to 1. |
| V_CMPX_F_I32 | Set the per-lane condition code to 0. |
| V_CMPX_LT_I32 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_I32 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_I32 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_I32 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_NE_I32 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_GE_I32 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_T_I32 | Set the per-lane condition code to 1. |
| V_CMPX_F_U32 | Set the per-lane condition code to 0. |
| V_CMPX_LT_U32 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_U32 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_U32 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_U32 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_NE_U32 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_GE_U32 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_T_U32 | Set the per-lane condition code to 1. |
| V_CMP_F_I64 | Set the per-lane condition code to 0. |
| V_CMP_LT_I64 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_I64 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_I64 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_I64 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_NE_I64 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_GE_I64 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_T_I64 | Set the per-lane condition code to 1. |
| V_CMP_F_U64 | Set the per-lane condition code to 0. |
| V_CMP_LT_U64 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_U64 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_U64 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_U64 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_NE_U64 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_GE_U64 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_T_U64 | Set the per-lane condition code to 1. |
| V_CMPX_F_I64 | Set the per-lane condition code to 0. |
| V_CMPX_LT_I64 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_I64 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_I64 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_I64 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_NE_I64 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_GE_I64 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_T_I64 | Set the per-lane condition code to 1. |
| V_CMPX_F_U64 | Set the per-lane condition code to 0. |
| V_CMPX_LT_U64 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_U64 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_U64 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_U64 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_NE_U64 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_GE_U64 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_T_U64 | Set the per-lane condition code to 1. |

### Min / max / med3

| Instruction | Brief description |
|---|---|
| V_MIN_F32 | Select the minimum of two single-precision float inputs and store the result into a vector register. |
| V_MAX_F32 | Select the maximum of two single-precision float inputs and store the result into a vector register. |
| V_MIN_I32 | Select the minimum of two signed 32-bit integer inputs and store the selected value into a vector register. |
| V_MAX_I32 | Select the maximum of two signed 32-bit integer inputs and store the selected value into a vector register. |
| V_MIN_U32 | Select the minimum of two unsigned 32-bit integer inputs and store the selected value into a vector register. |
| V_MAX_U32 | Select the maximum of two unsigned 32-bit integer inputs and store the selected value into a vector register. |
| V_MAX_F16 | Select the maximum of two half-precision float inputs and store the result into a vector register. |
| V_MIN_F16 | Select the minimum of two half-precision float inputs and store the result into a vector register. |
| V_MAX_U16 | Select the maximum of two unsigned 16-bit integer inputs and store the selected value into a vector register. |
| V_MAX_I16 | Select the maximum of two signed 16-bit integer inputs and store the selected value into a vector register. |
| V_MIN_U16 | Select the minimum of two unsigned 16-bit integer inputs and store the selected value into a vector register. |
| V_MIN_I16 | Select the minimum of two signed 16-bit integer inputs and store the selected value into a vector register. |
| V_MIN3_F32 | Select the minimum of three single-precision float inputs and store the selected value into a vector register. |
| V_MIN3_I32 | Select the minimum of three signed 32-bit integer inputs and store the selected value into a vector register. |
| V_MIN3_U32 | Select the minimum of three unsigned 32-bit integer inputs and store the selected value into a vector register. |
| V_MAX3_F32 | Select the maximum of three single-precision float inputs and store the selected value into a vector register. |
| V_MAX3_I32 | Select the maximum of three signed 32-bit integer inputs and store the selected value into a vector register. |
| V_MAX3_U32 | Select the maximum of three unsigned 32-bit integer inputs and store the selected value into a vector register. |
| V_MED3_F32 | Select the median of three single-precision float values and store the selected value into a vector register. |
| V_MED3_I32 | Select the median of three signed 32-bit integer values and store the selected value into a vector register. |
| V_MED3_U32 | Select the median of three unsigned 32-bit integer values and store the selected value into a vector register. |
| V_MIN3_F16 | Select the minimum of three half-precision float inputs and store the selected value into a vector register. |
| V_MIN3_I16 | Select the minimum of three signed 16-bit integer inputs and store the selected value into a vector register. |
| V_MIN3_U16 | Select the minimum of three unsigned 16-bit integer inputs and store the selected value into a vector register. |
| V_MAX3_F16 | Select the maximum of three half-precision float inputs and store the selected value into a vector register. |
| V_MAX3_I16 | Select the maximum of three signed 16-bit integer inputs and store the selected value into a vector register. |
| V_MAX3_U16 | Select the maximum of three unsigned 16-bit integer inputs and store the selected value into a vector register. |
| V_MED3_F16 | Select the median of three half-precision float values and store the selected value into a vector register. |
| V_MED3_I16 | Select the median of three signed 16-bit integer values and store the selected value into a vector register. |
| V_MED3_U16 | Select the median of three unsigned 16-bit integer values and store the selected value into a vector register. |
| V_MIN_F64 | Select the minimum of two double-precision float inputs and store the result into a vector register. |
| V_MAX_F64 | Select the maximum of two double-precision float inputs and store the result into a vector register. |
| V_MINIMUM3_F32 | Select the IEEE minimum() of three single-precision float inputs and store the result into a vector register. |
| V_MAXIMUM3_F32 | Select the IEEE maximum() of three single-precision float inputs and store the result into a vector register. |

### Conversions and rounding

| Instruction | Brief description |
|---|---|
| V_CVT_I32_F64 | Convert from a double-precision float input to a signed 32-bit integer value and store the result into a vector register. |
| V_CVT_F64_I32 | Convert from a signed 32-bit integer input to a double-precision float value and store the result into a vector register. |
| V_CVT_F32_I32 | Convert from a signed 32-bit integer input to a single-precision float value and store the result into a vector register. |
| V_CVT_F32_U32 | Convert from an unsigned 32-bit integer input to a single-precision float value and store the result into a vector register. |
| V_CVT_U32_F32 | Convert from a single-precision float input to an unsigned 32-bit integer value and store the result into a vector register. |
| V_CVT_I32_F32 | Convert from a single-precision float input to a signed 32-bit integer value and store the result into a vector register. |
| V_CVT_F16_F32 | Convert from a single-precision float input to a half-precision float value and store the result into a vector register. |
| V_CVT_F32_F16 | Convert from a half-precision float input to a single-precision float value and store the result into a vector register. |
| V_CVT_RPI_I32_F32 | Convert from a single-precision float input to a signed 32-bit integer value using round to nearest integer semantics (ignore the default rounding mode) and store the result into a vector register. |
| V_CVT_FLR_I32_F32 | Convert from a single-precision float input to a signed 32-bit integer value using round-down semantics (ignore the default rounding mode) and store the result into a vector register. |
| V_CVT_OFF_F32_I4 | Convert from a signed 4-bit integer input to a single-precision float value using an offset table and store the result into a vector register. |
| V_CVT_F32_F64 | Convert from a double-precision float input to a single-precision float value and store the result into a vector register. |
| V_CVT_F64_F32 | Convert from a single-precision float input to a double-precision float value and store the result into a vector register. |
| V_CVT_F32_UBYTE0 | Convert an unsigned byte in byte 0 of the input to a single-precision float value and store the result into a vector register. |
| V_CVT_F32_UBYTE1 | Convert an unsigned byte in byte 1 of the input to a single-precision float value and store the result into a vector register. |
| V_CVT_F32_UBYTE2 | Convert an unsigned byte in byte 2 of the input to a single-precision float value and store the result into a vector register. |
| V_CVT_F32_UBYTE3 | Convert an unsigned byte in byte 3 of the input to a single-precision float value and store the result into a vector register. |
| V_CVT_U32_F64 | Convert from a double-precision float input to an unsigned 32-bit integer value and store the result into a vector register. |
| V_CVT_F64_U32 | Convert from an unsigned 32-bit integer input to a double-precision float value and store the result into a vector register. |
| V_TRUNC_F64 | Compute the integer part of a double-precision float input using round toward zero semantics and store the result in floating point format into a vector register. |
| V_CEIL_F64 | Round the double-precision float input up to next integer and store the result in floating point format into a vector register. |
| V_RNDNE_F64 | Round the double-precision float input to the nearest even integer and store the result in floating point format into a vector register. |
| V_FLOOR_F64 | Round the double-precision float input down to previous integer and store the result in floating point format into a vector register. |
| V_FRACT_F32 | Compute the fractional portion of a single-precision float input and store the result in floating point format into a vector register. |
| V_TRUNC_F32 | Compute the integer part of a single-precision float input using round toward zero semantics and store the result in floating point format into a vector register. |
| V_CEIL_F32 | Round the single-precision float input up to next integer and store the result in floating point format into a vector register. |
| V_RNDNE_F32 | Round the single-precision float input to the nearest even integer and store the result in floating point format into a vector register. |
| V_FLOOR_F32 | Round the single-precision float input down to previous integer and store the result in floating point format into a vector register. |
| V_FRACT_F64 | Compute the fractional portion of a double-precision float input and store the result in floating point format into a vector register. |
| V_CVT_F16_U16 | Convert from an unsigned 16-bit integer input to a half-precision float value and store the result into a vector register. |
| V_CVT_F16_I16 | Convert from a signed 16-bit integer input to a half-precision float value and store the result into a vector register. |
| V_CVT_U16_F16 | Convert from a half-precision float input to an unsigned 16-bit integer value and store the result into a vector register. |
| V_CVT_I16_F16 | Convert from a half-precision float input to a signed 16-bit integer value and store the result into a vector register. |
| V_FLOOR_F16 | Round the half-precision float input down to previous integer and store the result in floating point format into a vector register. |
| V_CEIL_F16 | Round the half-precision float input up to next integer and store the result in floating point format into a vector register. |
| V_TRUNC_F16 | Compute the integer part of a half-precision float input using round toward zero semantics and store the result in floating point format into a vector register. |
| V_RNDNE_F16 | Round the half-precision float input to the nearest even integer and store the result in floating point format into a vector register. |
| V_FRACT_F16 | Compute the fractional portion of a half-precision float input and store the result in floating point format into a vector register. |
| V_CVT_NORM_I16_F16 | Convert from a half-precision float input to a signed normalized short and store the result into a vector register. |
| V_CVT_NORM_U16_F16 | Convert from a half-precision float input to an unsigned normalized short and store the result into a vector register. |
| V_CVT_F32_FP8 | Convert from an FP8 float input to a single-precision float value and store the result into a vector register. |
| V_CVT_F32_BF8 | Convert from a BF8 float input to a single-precision float value and store the result into a vector register. |
| V_CVT_PK_F32_FP8 | Convert from a packed 2-component FP8 float input to a packed single-precision float value and store the result into a vector register. |
| V_CVT_PK_F32_BF8 | Convert from a packed 2-component BF8 float input to a packed single-precision float value and store the result into a vector register. |
| V_CVT_F32_BF16 | Convert from a BF16 float input to a single-precision float value and store the result into a vector register. |
| V_CVT_PK_U8_F32 | Convert a single-precision float value from the first input to an unsigned 8-bit integer value and pack the result into one byte of the third input using the second input as a byte select. |
| V_CVT_PKACCUM_U8_F32 | Convert a single-precision float value in the first input to an unsigned 8-bit integer value and store the result into one byte of the destination register using the second input as a byte select. |
| V_CVT_SCALEF32_PK_FP8_F32 | Scale two single-precision float inputs using the exponent provided by the third single-precision float input, then convert the values to a packed FP8 float value with round toward nearest even semantics. |
| V_CVT_SCALEF32_PK_BF8_F32 | Scale two single-precision float inputs using the exponent provided by the third single-precision float input, then convert the values to a packed BF8 float value with round toward nearest even semantics. |
| V_CVT_SCALEF32_SR_FP8_F32 | Scale a single-precision float input using the exponent provided by the third single-precision float input, then convert the values to an FP8 float value with stochastic rounding using seed data from the second input. |
| V_CVT_SCALEF32_SR_BF8_F32 | Scale a single-precision float input using the exponent provided by the third single-precision float input, then convert the values to a BF8 float value with stochastic rounding using seed data from the second input. |
| V_CVT_SCALEF32_PK_F32_FP8 | Convert from a packed 2-component FP8 float input to a packed single-precision float value, then scale the packed values using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_PK_F32_BF8 | Convert from a packed 2-component BF8 float input to a packed single-precision float value, then scale the packed values using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_F32_FP8 | Convert from an FP8 float input to a single-precision float value, then scale the value using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_F32_BF8 | Convert from a BF8 float input to a single-precision float value, then scale the value using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_PK_FP4_F32 | Scale two single-precision float inputs using the exponent provided by the third single-precision float input, then convert the values to a packed FP4 float value with round toward nearest even semantics. |
| V_CVT_SCALEF32_SR_PK_FP4_F32 | Scale a packed 2-component single-precision float input using the exponent provided by the third single-precision float input, then convert the values to a packed FP4 float value with stochastic rounding using seed data from the second input. |
| V_CVT_SCALEF32_PK_F32_FP4 | Convert from a packed 2-component FP4 float input to a packed single-precision float value, then scale the packed values using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_PK_FP8_F16 | Scale a packed 2-component half-precision float input using the exponent provided by the second single-precision float input, then convert the values to a packed FP8 float value with round toward nearest even semantics. |
| V_CVT_SCALEF32_PK_BF8_F16 | Scale a packed 2-component half-precision float input using the exponent provided by the second single-precision float input, then convert the values to a packed BF8 float value with round toward nearest even semantics. |
| V_CVT_SCALEF32_SR_FP8_F16 | Scale a half-precision float input using the exponent provided by the third single-precision float input, then convert the values to an FP8 float value with stochastic rounding using seed data from the second input. |
| V_CVT_SCALEF32_SR_BF8_F16 | Scale a half-precision float input using the exponent provided by the third single-precision float input, then convert the values to a BF8 float value with stochastic rounding using seed data from the second input. |
| V_CVT_SCALEF32_PK_FP8_BF16 | Scale a packed 2-component BF16 float input using the exponent provided by the second single-precision float input, then convert the values to a packed FP8 float value with round toward nearest even semantics. |
| V_CVT_SCALEF32_PK_BF8_BF16 | Scale a packed 2-component BF16 float input using the exponent provided by the second single-precision float input, then convert the values to a packed BF8 float value with round toward nearest even semantics. |
| V_CVT_SCALEF32_SR_FP8_BF16 | Scale a BF16 float input using the exponent provided by the third single-precision float input, then convert the values to an FP8 float value with stochastic rounding using seed data from the second input. |
| V_CVT_SCALEF32_SR_BF8_BF16 | Scale a BF16 float input using the exponent provided by the third single-precision float input, then convert the values to a BF8 float value with stochastic rounding using seed data from the second input. |
| V_CVT_SCALEF32_PK_F16_FP8 | Convert from a packed 2-component FP8 float input to a packed half-precision float value, then scale the packed values using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_PK_F16_BF8 | Convert from a packed 2-component BF8 float input to a packed half-precision float value, then scale the packed values using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_F16_FP8 | Convert from an FP8 float input to a half-precision float value, then scale the value using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_F16_BF8 | Convert from a BF8 float input to a half-precision float value, then scale the value using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_PK_FP4_F16 | Scale a packed 2-component half-precision float input using the exponent provided by the second single-precision float input, then convert the values to a packed FP4 float value with round toward nearest even semantics. |
| V_CVT_SCALEF32_PK_FP4_BF16 | Scale a packed 2-component BF16 float input using the exponent provided by the second single-precision float input, then convert the values to a packed FP4 float value with round toward nearest even semantics. |
| V_CVT_SCALEF32_SR_PK_FP4_F16 | Scale a packed 2-component half-precision float input using the exponent provided by the third single-precision float input, then convert the values to a packed FP4 float value with stochastic rounding using seed data from the second input. |
| V_CVT_SCALEF32_SR_PK_FP4_BF16 | Scale a packed 2-component BF16 float input using the exponent provided by the third single-precision float input, then convert the values to a packed FP4 float value with stochastic rounding using seed data from the second input. |
| V_CVT_SCALEF32_PK_F16_FP4 | Convert from a packed 2-component FP4 float input to a packed half-precision float value, then scale the packed values using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_PK_BF16_FP4 | Convert from a packed 2-component FP4 float input to a packed BF16 float value, then scale the packed values using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_2XPK16_FP6_F32 | Scale packed 16-component single-precision float vectors from two source inputs using the exponent provided by the third single-precision float input, then convert the values to a packed 32-component FP6 float value. |
| V_CVT_SCALEF32_2XPK16_BF6_F32 | Scale packed 16-component single-precision float vectors from two source inputs using the exponent provided by the third single-precision float input, then convert the values to a packed 32-component BF6 float value. |
| V_CVT_SCALEF32_SR_PK32_FP6_F32 | Scale a packed 32-component single-precision float input using the exponent provided by the third single-precision float input, then convert the values to a packed 32-component FP6 float value with stochastic rounding using seed data from the second input. |
| V_CVT_SCALEF32_SR_PK32_BF6_F32 | Scale a packed 32-component single-precision float input using the exponent provided by the third single-precision float input, then convert the values to a packed 32-component BF6 float value with stochastic rounding using seed data from the second input. |
| V_CVT_SCALEF32_PK32_F32_FP6 | Convert from a packed 32-component FP6 float input to a packed single-precision float value, then scale the packed values using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_PK32_F32_BF6 | Convert from a packed 32-component BF6 float input to a packed single-precision float value, then scale the packed values using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_PK32_FP6_F16 | Scale a packed 32-component half-precision float input using the exponent provided by the second single-precision float input, then convert the values to a packed 32-component FP6 float value. |
| V_CVT_SCALEF32_PK32_FP6_BF16 | Scale a packed 32-component BF16 float input using the exponent provided by the second single-precision float input, then convert the values to a packed 32-component FP6 float value. |
| V_CVT_SCALEF32_PK32_BF6_F16 | Scale a packed 32-component half-precision float input using the exponent provided by the second single-precision float input, then convert the values to a packed 32-component BF6 float value. |
| V_CVT_SCALEF32_PK32_BF6_BF16 | Scale a packed 32-component BF16 float input using the exponent provided by the second single-precision float input, then convert the values to a packed 32-component BF6 float value. |
| V_CVT_SCALEF32_SR_PK32_FP6_F16 | Scale a packed 32-component half-precision float input using the exponent provided by the third single-precision float input, then convert the values to a packed 32-component FP6 float value with stochastic rounding using seed data from the second input. |
| V_CVT_SCALEF32_SR_PK32_FP6_BF16 | Scale a packed 32-component BF16 float input using the exponent provided by the third single-precision float input, then convert the values to a packed 32-component FP6 float value with stochastic rounding using seed data from the second input. |
| V_CVT_SCALEF32_SR_PK32_BF6_F16 | Scale a packed 32-component half-precision float input using the exponent provided by the third single-precision float input, then convert the values to a packed 32-component BF6 float value with stochastic rounding using seed data from the second input. |
| V_CVT_SCALEF32_SR_PK32_BF6_BF16 | Scale a packed 32-component BF16 float input using the exponent provided by the third single-precision float input, then convert the values to a packed 32-component BF6 float value with stochastic rounding using seed data from the second input. |
| V_CVT_SCALEF32_PK32_F16_FP6 | Convert from a packed 32-component FP6 float input to a packed half-precision float value, then scale the packed values using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_PK32_BF16_FP6 | Convert from a packed 32-component FP6 float input to a packed BF16 float value, then scale the packed values using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_PK32_F16_BF6 | Convert from a packed 32-component BF6 float input to a packed half-precision float value, then scale the packed values using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_PK32_BF16_BF6 | Convert from a packed 32-component BF6 float input to a packed BF16 float value, then scale the packed values using the exponent provided by the second single-precision float input. |
| V_CVT_PK_F16_F32 | Convert from two single-precision float inputs to a packed half-precision value and store the result into a vector register. |
| V_CVT_PK_BF16_F32 | Convert from two single-precision float inputs to a packed BF16 value and store the result into a vector register. |
| V_CVT_SCALEF32_PK_BF16_FP8 | Convert from a packed 2-component FP8 float input to a packed BF16 float value, then scale the packed values using the exponent provided by the second single-precision float input. |
| V_CVT_SCALEF32_PK_BF16_BF8 | Convert from a packed 2-component BF8 float input to a packed BF16 float value, then scale the packed values using the exponent provided by the second single-precision float input. |
| V_CVT_PKNORM_I16_F32 | Convert from two single-precision float inputs to a packed signed normalized short and store the result into a vector register. |
| V_CVT_PKNORM_U16_F32 | Convert from two single-precision float inputs to a packed unsigned normalized short and store the result into a vector register. |
| V_CVT_PKRTZ_F16_F32 | Convert two single-precision float inputs to a packed half-precision float value using round toward zero semantics (ignore the current rounding mode), and store the result into a vector register. |
| V_CVT_PK_U16_U32 | Convert from two unsigned 32-bit integer inputs to a packed unsigned 16-bit integer value and store the result into a vector register. |
| V_CVT_PK_I16_I32 | Convert from two signed 32-bit integer inputs to a packed signed 16-bit integer value and store the result into a vector register. |
| V_CVT_PKNORM_I16_F16 | Convert from two half-precision float inputs to a packed signed normalized short and store the result into a vector register. |
| V_CVT_PKNORM_U16_F16 | Convert from two half-precision float inputs to a packed unsigned normalized short and store the result into a vector register. |
| V_CVT_PK_FP8_F32 | Convert from two single-precision float inputs to a packed FP8 float value with round to nearest even semantics and store the result into 16 bits of a vector register using OPSEL. |
| V_CVT_PK_BF8_F32 | Convert from two single-precision float inputs to a packed BF8 float value with round to nearest even semantics and store the result into 16 bits of a vector register using OPSEL. |
| V_CVT_SR_FP8_F32 | Convert from a single-precision float input to an FP8 value with stochastic rounding using seed data from the second input. |
| V_CVT_SR_BF8_F32 | Convert from a single-precision float input to a BF8 value with stochastic rounding using seed data from the second input. |
| V_CVT_SR_F16_F32 | Convert from a single-precision float input to a half-precision value with stochastic rounding using seed data from the second input. |
| V_CVT_SR_BF16_F32 | Convert from a single-precision float input to a BF16 value with stochastic rounding using seed data from the second input. |

### Bitwise and bitfield

| Instruction | Brief description |
|---|---|
| V_NOT_B32 | Calculate bitwise negation on a vector input and store the result into a vector register. |
| V_BFREV_B32 | Reverse the order of bits in a vector input and store the result into a vector register. |
| V_FFBH_U32 | Count the number of leading "0" bits before the first "1" in a vector input and store the result into a vector register. |
| V_FFBL_B32 | Count the number of trailing "0" bits before the first "1" in a vector input and store the result into a vector register. |
| V_FFBH_I32 | Count the number of leading bits that are the same as the sign bit of a vector input and store the result into a vector register. |
| V_AND_B32 | Calculate bitwise AND on two vector inputs and store the result into a vector register. |
| V_OR_B32 | Calculate bitwise OR on two vector inputs and store the result into a vector register. |
| V_XOR_B32 | Calculate bitwise XOR on two vector inputs and store the result into a vector register. |
| V_XNOR_B32 | Calculate bitwise XNOR on two vector inputs and store the result into a vector register. |
| V_BFE_U32 | Extract an unsigned bitfield from the first input using field offset from the second input and size from the third input, then store the result into a vector register. |
| V_BFE_I32 | Extract a signed bitfield from the first input using field offset from the second input and size from the third input, then store the result into a vector register. |
| V_BFI_B32 | Overwrite a bitfield in the third input with a bitfield from the second input using a mask from the first input, then store the result into a vector register. |
| V_AND_OR_B32 | Calculate bitwise AND on the first two vector inputs, then compute the bitwise OR of the intermediate result and the third vector input, then store the final result into a vector register. |
| V_OR3_B32 | Calculate the bitwise OR of three vector inputs and store the result into a vector register. |
| V_BFM_B32 | Calculate a bitfield mask given a field offset and size and store the result into a vector register. |

### Division, rcp, sqrt

| Instruction | Brief description |
|---|---|
| V_RCP_F32 | Calculate the reciprocal of the single-precision float input using IEEE rules and store the result into a vector register. |
| V_RCP_IFLAG_F32 | Calculate the reciprocal of the vector float input in a manner suitable for integer division and store the result into a vector register. |
| V_RCP_F64 | Calculate the reciprocal of the double-precision float input using IEEE rules and store the result into a vector register. |
| V_SQRT_F32 | Calculate the square root of the single-precision float input using IEEE rules and store the result into a vector register. |
| V_SQRT_F64 | Calculate the square root of the double-precision float input using IEEE rules and store the result into a vector register. |
| V_RCP_F16 | Calculate the reciprocal of the half-precision float input using IEEE rules and store the result into a vector register. |
| V_SQRT_F16 | Calculate the square root of the half-precision float input using IEEE rules and store the result into a vector register. |
| V_DIV_FIXUP_F32 | Given a single-precision float quotient in the first input, a denominator in the second input and a numerator in the third input, detect and apply corner cases related to division, including divide by zero, NaN inputs and overflow, and modify the quotient accordingly. |
| V_DIV_FIXUP_F64 | Given a double-precision float quotient in the first input, a denominator in the second input and a numerator in the third input, detect and apply corner cases related to division, including divide by zero, NaN inputs and overflow, and modify the quotient accordingly. |
| V_DIV_SCALE_F32 | Given a single-precision float value to scale in the first input, a denominator in the second input and a numerator in the third input, scale the first input for division if required to avoid subnormal terms appearing during application of the Newton-Raphson correction method. |
| V_DIV_SCALE_F64 | Given a double-precision float value to scale in the first input, a denominator in the second input and a numerator in the third input, scale the first input for division if required to avoid subnormal terms appearing during application of the Newton-Raphson correction method. |
| V_DIV_FMAS_F32 | Multiply two single-precision float inputs and add a third input using fused multiply add, then scale the exponent of the result by a fixed factor if the vector condition code is set. |
| V_DIV_FMAS_F64 | Multiply two double-precision float inputs and add a third input using fused multiply add, then scale the exponent of the result by a fixed factor if the vector condition code is set. |
| V_DIV_FIXUP_LEGACY_F16 | Half precision division fixup. |
| V_DIV_FIXUP_F16 | Given a half-precision float quotient in the first input, a denominator in the second input and a numerator in the third input, detect and apply corner cases related to division, including divide by zero, NaN inputs and overflow, and modify the quotient accordingly. |

### Lane / scalar moves

| Instruction | Brief description |
|---|---|
| V_READFIRSTLANE_B32 | Read the scalar value in the lowest active lane of the input vector register and store it into a scalar register. |
| V_CNDMASK_B32 | Copy data from one of two inputs based on the per-lane condition code and store the result into a vector register. |
| V_READLANE_B32 | Read the scalar value in the specified lane of the first input where the lane select is in the second input. |
| V_WRITELANE_B32 | Write the scalar value in the first input into the specified lane of a vector register where the lane select is in the second input. |

### Permute / align / lerp

| Instruction | Brief description |
|---|---|
| V_PERMLANE16_SWAP_B32 | Swap data between two vector registers. |
| V_PERMLANE32_SWAP_B32 | Swap data between two vector registers. |
| V_LERP_U8 | Average two 4-D vectors stored as packed bytes in the first two inputs with rounding control provided by the third input, then store the result into a vector register. |
| V_ALIGNBIT_B32 | Align a 64-bit value encoded in the first two inputs to a bit position specified in the third input, then store the result into a 32-bit vector register. |
| V_ALIGNBYTE_B32 | Align a 64-bit value encoded in the first two inputs to a byte position specified in the third input, then store the result into a 32-bit vector register. |
| V_PERM_B32 | Permute a 64-bit value constructed from two vector inputs (most significant bits come from the first input) using a per-lane selector from the third input. |
| V_BCNT_U32_B32 | Count the number of "1" bits in the vector input and store the result into a vector register. |
| V_MBCNT_LO_U32_B32 | For each lane 0 <= N < 32, examine the N least significant bits of the first input and count how many of those bits are "1". |
| V_MBCNT_HI_U32_B32 | For each lane 32 <= N < 64, examine the N least significant bits of the first input and count how many of those bits are "1". |

### Other VOP3A/B

| Instruction | Brief description |
|---|---|
| V_NOP | Do nothing. |
| V_MOV_B32 | Move 32-bit data from a vector input into a vector register. |
| V_EXP_F32 | Calculate 2 raised to the power of the single-precision float input and store the result into a vector register. |
| V_LOG_F32 | Calculate the base 2 logarithm of the single-precision float input and store the result into a vector register. |
| V_RSQ_F32 | Calculate the reciprocal of the square root of the single-precision float input using IEEE rules and store the result into a vector register. |
| V_RSQ_F64 | Calculate the reciprocal of the square root of the double-precision float input using IEEE rules and store the result into a vector register. |
| V_SIN_F32 | Calculate the trigonometric sine of a single-precision float value using IEEE rules and store the result into a vector register. |
| V_COS_F32 | Calculate the trigonometric cosine of a single-precision float value using IEEE rules and store the result into a vector register. |
| V_FREXP_EXP_I32_F64 | Extract the exponent of a double-precision float input and store the result as a signed 32-bit integer into a vector register. |
| V_FREXP_MANT_F64 | Extract the binary significand, or mantissa, of a double-precision float input and store the result as a double-precision float into a vector register. |
| V_FREXP_EXP_I32_F32 | Extract the exponent of a single-precision float input and store the result as a signed 32-bit integer into a vector register. |
| V_FREXP_MANT_F32 | Extract the binary significand, or mantissa, of a single-precision float input and store the result as a single-precision float into a vector register. |
| V_CLREXCP | Clear this wave's exception state in the vector ALU. |
| V_MOV_B64 | Move data from a 64-bit vector input into a vector register. |
| V_RSQ_F16 | Calculate the reciprocal of the square root of the half-precision float input using IEEE rules and store the result into a vector register. |
| V_LOG_F16 | Calculate the base 2 logarithm of the half-precision float input and store the result into a vector register. |
| V_EXP_F16 | Calculate 2 raised to the power of the half-precision float input and store the result into a vector register. |
| V_FREXP_MANT_F16 | Extract the binary significand, or mantissa, of a half-precision float input and store the result as a half-precision float into a vector register. |
| V_FREXP_EXP_I16_F16 | Extract the exponent of a half-precision float input and store the result as a signed 16-bit integer into a vector register. |
| V_SIN_F16 | Calculate the trigonometric sine of a half-precision float value using IEEE rules and store the result into a vector register. |
| V_COS_F16 | Calculate the trigonometric cosine of a half-precision float value using IEEE rules and store the result into a vector register. |
| V_SAT_PK_U8_I16 | Given 2 signed 16-bit integer inputs, saturate each input over an unsigned 8-bit integer range, pack the resulting values into a packed 16-bit value and store the result into a vector register. |
| V_SWAP_B32 | Swap the values in two vector registers. |
| V_ACCVGPR_MOV_B32 | Move data from one accumulator register to another accumulator register. |
| V_PRNG_B32 | Generate a pseudorandom number using an LFSR (linear feedback shift register) seeded with the vector input, then store the result into a vector register. |
| V_DOT2C_F32_BF16 | Compute the dot product of two packed 2-D BF16 float inputs in the single-precision float domain and accumulate with the single-precision float value in the destination register. |
| V_MAC_F16 | Multiply two floating point inputs and accumulate the result into the destination register. |
| V_LDEXP_F16 | Multiply the first input, a floating point value, by an integral power of 2 specified in the second input, a signed integer value, and store the floating point result into a vector register. |
| V_DOT2C_F32_F16 | Compute the dot product of two packed 2-D half-precision float inputs in the single-precision float domain and accumulate with the single-precision float value in the destination register. |
| V_DOT2C_I32_I16 | Compute the dot product of two packed 2-D signed 16-bit integer inputs in the signed 32-bit integer domain and accumulate with the signed 32-bit integer value in the destination register. |
| V_DOT4C_I32_I8 | Compute the dot product of two packed 4-D signed 8-bit integer inputs in the signed 32-bit integer domain and accumulate with the signed 32-bit integer value in the destination register. |
| V_DOT8C_I32_I4 | Compute the dot product of two packed 8-D signed 4-bit integer inputs in the signed 32-bit integer domain and accumulate with the signed 32-bit integer value in the destination register. |
| V_CUBEID_F32 | Compute the cubemap face ID of a 3D coordinate specified as three single-precision float inputs. |
| V_CUBESC_F32 | Compute the cubemap S coordinate of a 3D coordinate specified as three single-precision float inputs. |
| V_CUBETC_F32 | Compute the cubemap T coordinate of a 3D coordinate specified as three single-precision float inputs. |
| V_CUBEMA_F32 | Compute the cubemap major axis of a 3D coordinate specified as three single-precision float inputs. |
| V_SAD_U8 | Calculate the sum of absolute differences of elements in two packed 4-component unsigned 8-bit integer inputs, add an unsigned 32-bit integer value from the third input and store the result into a vector register. |
| V_SAD_HI_U8 | Calculate the sum of absolute differences of elements in two packed 4-component unsigned 8-bit integer inputs, shift the sum left by 16 bits, add an unsigned 32-bit integer value from the third input and store the result into a vector register. |
| V_SAD_U16 | Calculate the sum of absolute differences of elements in two packed 2-component unsigned 16-bit integer inputs, add an unsigned 32-bit integer value from the third input and store the result into a vector register. |
| V_SAD_U32 | Calculate the absolute difference of two unsigned 32-bit integer inputs, add an unsigned 32-bit integer value from the third input and store the result into a vector register. |
| V_MSAD_U8 | Calculate the sum of absolute differences of elements in two packed 4-component unsigned 8-bit integer inputs, except that elements where the second input (known as the reference input) is zero are not included in the sum. |
| V_QSAD_PK_U16_U8 | Perform the V_SAD_U8 operation four times using different slices of the first array, all entries of the second array and each entry of the third array. |
| V_MQSAD_PK_U16_U8 | Perform the V_MSAD_U8 operation four times using different slices of the first array, all entries of the second array and each entry of the third array. |
| V_MQSAD_U32_U8 | Perform the V_MSAD_U8 operation four times using different slices of the first array, all entries of the second array and each entry of the third array. |
| V_XAD_U32 | Calculate bitwise XOR of the first two vector inputs, then add the third vector input to the intermediate result, then store the final result into a vector register. |
| V_BITOP3_B16 | Calculate the generic bitwise operation of three 16-bit vector inputs using a truth table encoded in the instruction and store the result into a vector register. |
| V_BITOP3_B32 | Calculate the generic bitwise operation of three 32-bit vector inputs using a truth table encoded in the instruction and store the result into a vector register. |
| V_LDEXP_F64 | Multiply the first input, a floating point value, by an integral power of 2 specified in the second input, a signed integer value, and store the floating point result into a vector register. |
| V_LDEXP_F32 | Multiply the first input, a floating point value, by an integral power of 2 specified in the second input, a signed integer value, and store the floating point result into a vector register. |
| V_TRIG_PREOP_F64 | Look up a 53-bit segment of 2/PI using an integer segment select in the second input. |
| V_PACK_B32_F16 | Pack two half-precision float values into a single 32-bit value and store the result into a vector register. |
