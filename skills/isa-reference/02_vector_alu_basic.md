# CDNA4 Vector ALU — VOP2, VOP1, VOPC (gfx950)

Source: CDNA4 ISA sections 12.7–12.9 (`docs/cdna4_isa_full.txt`, lines 6267–8920). VOP2/VOP1 may use a **32-bit literal**, **DPP**, or **SDWA** immediately after the instruction word when supported. VOPC compares write per-lane results to **VCC** (or a scalar pair in VOP3A); Instructions with prefix `V_CMPX_` also update **EXEC**.

Operand names follow AMDGPU assembly style: `vdst`, `vsrc0`, `vsrc1` (SGPR or literal allowed per encoding rules).

### VOP2 / VOP1 as VOP3

- VOP2 in VOP3: opcode = VOP2 opcode + `0x100` (no literal; extra modifiers).
- VOP1 in VOP3: opcode = VOP1 opcode + `0x140`.
- VOPC in VOP3A: opcode = VOPC opcode + `0x000` (see §12.9.1).

## VOP2 — two vector sources

| Instruction | Syntax | Description |
|---|---|---|
| V_CNDMASK_B32 | vdst, vsrc0, vsrc1 | Copy data from one of two inputs based on the per-lane condition code and store the result into a vector register. |
| V_ADD_F32 | vdst, vsrc0, vsrc1 | Add two floating point inputs and store the result into a vector register. |
| V_SUB_F32 | vdst, vsrc0, vsrc1 | Subtract the second floating point input from the first input and store the result into a vector register. |
| V_SUBREV_F32 | vdst, vsrc0, vsrc1 | Subtract the first floating point input from the second input and store the result into a vector register. |
| V_FMAC_F64 | vdst, vsrc0, vsrc1 | Multiply two floating point inputs and accumulate the result into the destination register using fused multiply add. |
| V_MUL_F32 | vdst, vsrc0, vsrc1 | Multiply two floating point inputs and store the result into a vector register. |
| V_MUL_I32_I24 | vdst, vsrc0, vsrc1 | Multiply two signed 24-bit integer inputs and store the result as a signed 32-bit integer into a vector register. |
| V_MUL_HI_I32_I24 | vdst, vsrc0, vsrc1 | Multiply two signed 24-bit integer inputs and store the high 32 bits of the result as a signed 32-bit integer into a vector register. |
| V_MUL_U32_U24 | vdst, vsrc0, vsrc1 | Multiply two unsigned 24-bit integer inputs and store the result as an unsigned 32-bit integer into a vector register. |
| V_MUL_HI_U32_U24 | vdst, vsrc0, vsrc1 | Multiply two unsigned 24-bit integer inputs and store the high 32 bits of the result as an unsigned 32-bit integer into a vector register. |
| V_MIN_F32 | vdst, vsrc0, vsrc1 | Select the minimum of two single-precision float inputs and store the result into a vector register. |
| V_MAX_F32 | vdst, vsrc0, vsrc1 | Select the maximum of two single-precision float inputs and store the result into a vector register. |
| V_MIN_I32 | vdst, vsrc0, vsrc1 | Select the minimum of two signed 32-bit integer inputs and store the selected value into a vector register. |
| V_MAX_I32 | vdst, vsrc0, vsrc1 | Select the maximum of two signed 32-bit integer inputs and store the selected value into a vector register. |
| V_MIN_U32 | vdst, vsrc0, vsrc1 | Select the minimum of two unsigned 32-bit integer inputs and store the selected value into a vector register. |
| V_MAX_U32 | vdst, vsrc0, vsrc1 | Select the maximum of two unsigned 32-bit integer inputs and store the selected value into a vector register. |
| V_LSHRREV_B32 | vdst, vsrc0, vsrc1 | Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store the result into a vector register. |
| V_ASHRREV_I32 | vdst, vsrc0, vsrc1 | Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second vector input and store the result into a vector register. |
| V_LSHLREV_B32 | vdst, vsrc0, vsrc1 | Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the result into a vector register. |
| V_AND_B32 | vdst, vsrc0, vsrc1 | Calculate bitwise AND on two vector inputs and store the result into a vector register. |
| V_OR_B32 | vdst, vsrc0, vsrc1 | Calculate bitwise OR on two vector inputs and store the result into a vector register. |
| V_XOR_B32 | vdst, vsrc0, vsrc1 | Calculate bitwise XOR on two vector inputs and store the result into a vector register. |
| V_DOT2C_F32_BF16 | vdst, vsrc0, vsrc1 | Compute the dot product of two packed 2-D BF16 float inputs in the single-precision float domain and accumulate with the single-precision float value in the destination register. |
| V_FMAMK_F32 | vdst, vsrc0, vsrc1 (+ literal) | Multiply a single-precision float input with a literal constant and add a second single-precision float input using fused multiply add, and store the result into a vector register. |
| V_FMAAK_F32 | vdst, vsrc0, vsrc1 (+ literal) | Multiply two single-precision float inputs and add a literal constant using fused multiply add, and store the result into a vector register. |
| V_ADD_CO_U32 | vdst, vsrc0, vsrc1 | Add two unsigned 32-bit integer inputs, store the result into a vector register and store the carry-out mask into a scalar register. |
| V_SUB_CO_U32 | vdst, vsrc0, vsrc1 | Subtract the second unsigned 32-bit integer input from the first input, store the result into a vector register and store the carry-out mask into a scalar register. |
| V_SUBREV_CO_U32 | vdst, vsrc0, vsrc1 | Subtract the first unsigned 32-bit integer input from the second input, store the result into a vector register and store the carry-out mask into a scalar register. |
| V_ADDC_CO_U32 | vdst, vsrc0, vsrc1 | Add two unsigned 32-bit integer inputs and a bit from a carry-in mask, store the result into a vector register and store the carry-out mask into a scalar register. |
| V_SUBB_CO_U32 | vdst, vsrc0, vsrc1 | Subtract the second unsigned 32-bit integer input from the first input, subtract a bit from the carry-in mask, store the result into a vector register and store the carry-out mask into a scalar register. |
| V_SUBBREV_CO_U32 | vdst, vsrc0, vsrc1 | Subtract the first unsigned 32-bit integer input from the second input, subtract a bit from the carry-in mask, store the result into a vector register and store the carry-out mask into a scalar register. |
| V_ADD_F16 | vdst, vsrc0, vsrc1 | Add two floating point inputs and store the result into a vector register. |
| V_SUB_F16 | vdst, vsrc0, vsrc1 | Subtract the second floating point input from the first input and store the result into a vector register. |
| V_SUBREV_F16 | vdst, vsrc0, vsrc1 | Subtract the first floating point input from the second input and store the result into a vector register. |
| V_MUL_F16 | vdst, vsrc0, vsrc1 | Multiply two floating point inputs and store the result into a vector register. |
| V_MAC_F16 | vdst, vsrc0, vsrc1 | Multiply two floating point inputs and accumulate the result into the destination register. |
| V_MADMK_F16 | vdst, vsrc0, vsrc1 (+ literal) | Multiply a floating point input with a literal constant and add a second floating point input, and store the result into a vector register. |
| V_MADAK_F16 | vdst, vsrc0, vsrc1 (+ literal) | Multiply two floating point inputs and add a literal constant, and store the result into a vector register. |
| V_ADD_U16 | vdst, vsrc0, vsrc1 | Add two unsigned 16-bit integer inputs and store the result into a vector register. |
| V_SUB_U16 | vdst, vsrc0, vsrc1 | Subtract the second unsigned 16-bit integer input from the first input and store the result into a vector register. |
| V_SUBREV_U16 | vdst, vsrc0, vsrc1 | Subtract the first unsigned 16-bit integer input from the second input and store the result into a vector register. |
| V_MUL_LO_U16 | vdst, vsrc0, vsrc1 | Multiply two unsigned 16-bit integer inputs and store the low bits of the result into a vector register. |
| V_LSHLREV_B16 | vdst, vsrc0, vsrc1 | Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the result into a vector register. |
| V_LSHRREV_B16 | vdst, vsrc0, vsrc1 | Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store the result into a vector register. |
| V_ASHRREV_I16 | vdst, vsrc0, vsrc1 | Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second vector input and store the result into a vector register. |
| V_MAX_F16 | vdst, vsrc0, vsrc1 | Select the maximum of two half-precision float inputs and store the result into a vector register. |
| V_MIN_F16 | vdst, vsrc0, vsrc1 | Select the minimum of two half-precision float inputs and store the result into a vector register. |
| V_MAX_U16 | vdst, vsrc0, vsrc1 | Select the maximum of two unsigned 16-bit integer inputs and store the selected value into a vector register. |
| V_MAX_I16 | vdst, vsrc0, vsrc1 | Select the maximum of two signed 16-bit integer inputs and store the selected value into a vector register. |
| V_MIN_U16 | vdst, vsrc0, vsrc1 | Select the minimum of two unsigned 16-bit integer inputs and store the selected value into a vector register. |
| V_MIN_I16 | vdst, vsrc0, vsrc1 | Select the minimum of two signed 16-bit integer inputs and store the selected value into a vector register. |
| V_LDEXP_F16 | vdst, vsrc0, vsrc1 | Multiply the first input, a floating point value, by an integral power of 2 specified in the second input, a signed integer value, and store the floating point result into a vector register. |
| V_ADD_U32 | vdst, vsrc0, vsrc1 | Add two unsigned 32-bit integer inputs and store the result into a vector register. |
| V_SUB_U32 | vdst, vsrc0, vsrc1 | Subtract the second unsigned 32-bit integer input from the first input and store the result into a vector register. |
| V_SUBREV_U32 | vdst, vsrc0, vsrc1 | Subtract the first unsigned 32-bit integer input from the second input and store the result into a vector register. |
| V_DOT2C_F32_F16 | vdst, vsrc0, vsrc1 | Compute the dot product of two packed 2-D half-precision float inputs in the single-precision float domain and accumulate with the single-precision float value in the destination register. |
| V_DOT2C_I32_I16 | vdst, vsrc0, vsrc1 | Compute the dot product of two packed 2-D signed 16-bit integer inputs in the signed 32-bit integer domain and accumulate with the signed 32-bit integer value in the destination register. |
| V_DOT4C_I32_I8 | vdst, vsrc0, vsrc1 | Compute the dot product of two packed 4-D signed 8-bit integer inputs in the signed 32-bit integer domain and accumulate with the signed 32-bit integer value in the destination register. |
| V_DOT8C_I32_I4 | vdst, vsrc0, vsrc1 | Compute the dot product of two packed 8-D signed 4-bit integer inputs in the signed 32-bit integer domain and accumulate with the signed 32-bit integer value in the destination register. |
| V_FMAC_F32 | vdst, vsrc0, vsrc1 | Multiply two floating point inputs and accumulate the result into the destination register using fused multiply add. |
| V_PK_FMAC_F16 | vdst, vsrc0, vsrc1 | Multiply two packed half-precision float inputs component-wise and accumulate the result into the destination register using fused multiply add. |
| V_XNOR_B32 | vdst, vsrc0, vsrc1 | Calculate bitwise XNOR on two vector inputs and store the result into a vector register. |
## VOP1 — one vector source

| Instruction | Syntax | Description |
|---|---|---|
| V_NOP | (none) | Do nothing. |
| V_MOV_B32 | vdst, vsrc0 | Move 32-bit data from a vector input into a vector register. |
| V_READFIRSTLANE_B32 | vdst, vsrc0 | Read the scalar value in the lowest active lane of the input vector register and store it into a scalar register. |
| V_CVT_I32_F64 | vdst, vsrc0 | Convert from a double-precision float input to a signed 32-bit integer value and store the result into a vector register. |
| V_CVT_F64_I32 | vdst, vsrc0 | Convert from a signed 32-bit integer input to a double-precision float value and store the result into a vector register. |
| V_CVT_F32_I32 | vdst, vsrc0 | Convert from a signed 32-bit integer input to a single-precision float value and store the result into a vector register. |
| V_CVT_F32_U32 | vdst, vsrc0 | Convert from an unsigned 32-bit integer input to a single-precision float value and store the result into a vector register. |
| V_CVT_U32_F32 | vdst, vsrc0 | Convert from a single-precision float input to an unsigned 32-bit integer value and store the result into a vector register. |
| V_CVT_I32_F32 | vdst, vsrc0 | Convert from a single-precision float input to a signed 32-bit integer value and store the result into a vector register. |
| V_CVT_F16_F32 | vdst, vsrc0 | Convert from a single-precision float input to a half-precision float value and store the result into a vector register. |
| V_CVT_F32_F16 | vdst, vsrc0 | Convert from a half-precision float input to a single-precision float value and store the result into a vector register. |
| V_CVT_RPI_I32_F32 | vdst, vsrc0 | Convert from a single-precision float input to a signed 32-bit integer value using round to nearest integer semantics (ignore the default rounding mode) and store the result into a vector register. |
| V_CVT_FLR_I32_F32 | vdst, vsrc0 | Convert from a single-precision float input to a signed 32-bit integer value using round-down semantics (ignore the default rounding mode) and store the result into a vector register. |
| V_CVT_OFF_F32_I4 | vdst, vsrc0 | Convert from a signed 4-bit integer input to a single-precision float value using an offset table and store the result into a vector register. |
| V_CVT_F32_F64 | vdst, vsrc0 | Convert from a double-precision float input to a single-precision float value and store the result into a vector register. |
| V_CVT_F64_F32 | vdst, vsrc0 | Convert from a single-precision float input to a double-precision float value and store the result into a vector register. |
| V_CVT_F32_UBYTE0 | vdst, vsrc0 | Convert an unsigned byte in byte 0 of the input to a single-precision float value and store the result into a vector register. |
| V_CVT_F32_UBYTE1 | vdst, vsrc0 | Convert an unsigned byte in byte 1 of the input to a single-precision float value and store the result into a vector register. |
| V_CVT_F32_UBYTE2 | vdst, vsrc0 | Convert an unsigned byte in byte 2 of the input to a single-precision float value and store the result into a vector register. |
| V_CVT_F32_UBYTE3 | vdst, vsrc0 | Convert an unsigned byte in byte 3 of the input to a single-precision float value and store the result into a vector register. |
| V_CVT_U32_F64 | vdst, vsrc0 | Convert from a double-precision float input to an unsigned 32-bit integer value and store the result into a vector register. |
| V_CVT_F64_U32 | vdst, vsrc0 | Convert from an unsigned 32-bit integer input to a double-precision float value and store the result into a vector register. |
| V_TRUNC_F64 | vdst, vsrc0 | Compute the integer part of a double-precision float input using round toward zero semantics and store the result in floating point format into a vector register. |
| V_CEIL_F64 | vdst, vsrc0 | Round the double-precision float input up to next integer and store the result in floating point format into a vector register. |
| V_RNDNE_F64 | vdst, vsrc0 | Round the double-precision float input to the nearest even integer and store the result in floating point format into a vector register. |
| V_FLOOR_F64 | vdst, vsrc0 | Round the double-precision float input down to previous integer and store the result in floating point format into a vector register. |
| V_FRACT_F32 | vdst, vsrc0 | Compute the fractional portion of a single-precision float input and store the result in floating point format into a vector register. |
| V_TRUNC_F32 | vdst, vsrc0 | Compute the integer part of a single-precision float input using round toward zero semantics and store the result in floating point format into a vector register. |
| V_CEIL_F32 | vdst, vsrc0 | Round the single-precision float input up to next integer and store the result in floating point format into a vector register. |
| V_RNDNE_F32 | vdst, vsrc0 | Round the single-precision float input to the nearest even integer and store the result in floating point format into a vector register. |
| V_FLOOR_F32 | vdst, vsrc0 | Round the single-precision float input down to previous integer and store the result in floating point format into a vector register. |
| V_EXP_F32 | vdst, vsrc0 | Calculate 2 raised to the power of the single-precision float input and store the result into a vector register. |
| V_LOG_F32 | vdst, vsrc0 | Calculate the base 2 logarithm of the single-precision float input and store the result into a vector register. |
| V_RCP_F32 | vdst, vsrc0 | Calculate the reciprocal of the single-precision float input using IEEE rules and store the result into a vector register. |
| V_RCP_IFLAG_F32 | vdst, vsrc0 | Calculate the reciprocal of the vector float input in a manner suitable for integer division and store the result into a vector register. |
| V_RSQ_F32 | vdst, vsrc0 | Calculate the reciprocal of the square root of the single-precision float input using IEEE rules and store the result into a vector register. |
| V_RCP_F64 | vdst, vsrc0 | Calculate the reciprocal of the double-precision float input using IEEE rules and store the result into a vector register. |
| V_RSQ_F64 | vdst, vsrc0 | Calculate the reciprocal of the square root of the double-precision float input using IEEE rules and store the result into a vector register. |
| V_SQRT_F32 | vdst, vsrc0 | Calculate the square root of the single-precision float input using IEEE rules and store the result into a vector register. |
| V_SQRT_F64 | vdst, vsrc0 | Calculate the square root of the double-precision float input using IEEE rules and store the result into a vector register. |
| V_SIN_F32 | vdst, vsrc0 | Calculate the trigonometric sine of a single-precision float value using IEEE rules and store the result into a vector register. |
| V_COS_F32 | vdst, vsrc0 | Calculate the trigonometric cosine of a single-precision float value using IEEE rules and store the result into a vector register. |
| V_NOT_B32 | vdst, vsrc0 | Calculate bitwise negation on a vector input and store the result into a vector register. |
| V_BFREV_B32 | vdst, vsrc0 | Reverse the order of bits in a vector input and store the result into a vector register. |
| V_FFBH_U32 | vdst, vsrc0 | Count the number of leading "0" bits before the first "1" in a vector input and store the result into a vector register. |
| V_FFBL_B32 | vdst, vsrc0 | Count the number of trailing "0" bits before the first "1" in a vector input and store the result into a vector register. |
| V_FFBH_I32 | vdst, vsrc0 | Count the number of leading bits that are the same as the sign bit of a vector input and store the result into a vector register. |
| V_FREXP_EXP_I32_F64 | vdst, vsrc0 | Extract the exponent of a double-precision float input and store the result as a signed 32-bit integer into a vector register. |
| V_FREXP_MANT_F64 | vdst, vsrc0 | Extract the binary significand, or mantissa, of a double-precision float input and store the result as a double-precision float into a vector register. |
| V_FRACT_F64 | vdst, vsrc0 | Compute the fractional portion of a double-precision float input and store the result in floating point format into a vector register. |
| V_FREXP_EXP_I32_F32 | vdst, vsrc0 | Extract the exponent of a single-precision float input and store the result as a signed 32-bit integer into a vector register. |
| V_FREXP_MANT_F32 | vdst, vsrc0 | Extract the binary significand, or mantissa, of a single-precision float input and store the result as a single-precision float into a vector register. |
| V_CLREXCP | (no VGPR) | Clear this wave's exception state in the vector ALU. |
| V_MOV_B64 | vdst, vsrc0 | Move data from a 64-bit vector input into a vector register. |
| V_CVT_F16_U16 | vdst, vsrc0 | Convert from an unsigned 16-bit integer input to a half-precision float value and store the result into a vector register. |
| V_CVT_F16_I16 | vdst, vsrc0 | Convert from a signed 16-bit integer input to a half-precision float value and store the result into a vector register. |
| V_CVT_U16_F16 | vdst, vsrc0 | Convert from a half-precision float input to an unsigned 16-bit integer value and store the result into a vector register. |
| V_CVT_I16_F16 | vdst, vsrc0 | Convert from a half-precision float input to a signed 16-bit integer value and store the result into a vector register. |
| V_RCP_F16 | vdst, vsrc0 | Calculate the reciprocal of the half-precision float input using IEEE rules and store the result into a vector register. |
| V_SQRT_F16 | vdst, vsrc0 | Calculate the square root of the half-precision float input using IEEE rules and store the result into a vector register. |
| V_RSQ_F16 | vdst, vsrc0 | Calculate the reciprocal of the square root of the half-precision float input using IEEE rules and store the result into a vector register. |
| V_LOG_F16 | vdst, vsrc0 | Calculate the base 2 logarithm of the half-precision float input and store the result into a vector register. |
| V_EXP_F16 | vdst, vsrc0 | Calculate 2 raised to the power of the half-precision float input and store the result into a vector register. |
| V_FREXP_MANT_F16 | vdst, vsrc0 | Extract the binary significand, or mantissa, of a half-precision float input and store the result as a half-precision float into a vector register. |
| V_FREXP_EXP_I16_F16 | vdst, vsrc0 | Extract the exponent of a half-precision float input and store the result as a signed 16-bit integer into a vector register. |
| V_FLOOR_F16 | vdst, vsrc0 | Round the half-precision float input down to previous integer and store the result in floating point format into a vector register. |
| V_CEIL_F16 | vdst, vsrc0 | Round the half-precision float input up to next integer and store the result in floating point format into a vector register. |
| V_TRUNC_F16 | vdst, vsrc0 | Compute the integer part of a half-precision float input using round toward zero semantics and store the result in floating point format into a vector register. |
| V_RNDNE_F16 | vdst, vsrc0 | Round the half-precision float input to the nearest even integer and store the result in floating point format into a vector register. |
| V_FRACT_F16 | vdst, vsrc0 | Compute the fractional portion of a half-precision float input and store the result in floating point format into a vector register. |
| V_SIN_F16 | vdst, vsrc0 | Calculate the trigonometric sine of a half-precision float value using IEEE rules and store the result into a vector register. |
| V_COS_F16 | vdst, vsrc0 | Calculate the trigonometric cosine of a half-precision float value using IEEE rules and store the result into a vector register. |
| V_CVT_NORM_I16_F16 | vdst, vsrc0 | Convert from a half-precision float input to a signed normalized short and store the result into a vector register. |
| V_CVT_NORM_U16_F16 | vdst, vsrc0 | Convert from a half-precision float input to an unsigned normalized short and store the result into a vector register. |
| V_SAT_PK_U8_I16 | vdst, vsrc0 | Given 2 signed 16-bit integer inputs, saturate each input over an unsigned 8-bit integer range, pack the resulting values into a packed 16-bit value and store the result into a vector register. |
| V_SWAP_B32 | vdst, vsrc0 | Swap the values in two vector registers. |
| V_ACCVGPR_MOV_B32 | vdst, vsrc0 | Move data from one accumulator register to another accumulator register. |
| V_CVT_F32_FP8 | vdst, vsrc0 | Convert from an FP8 float input to a single-precision float value and store the result into a vector register. |
| V_CVT_F32_BF8 | vdst, vsrc0 | Convert from a BF8 float input to a single-precision float value and store the result into a vector register. |
| V_CVT_PK_F32_FP8 | vdst, vsrc0 | Convert from a packed 2-component FP8 float input to a packed single-precision float value and store the result into a vector register. |
| V_CVT_PK_F32_BF8 | vdst, vsrc0 | Convert from a packed 2-component BF8 float input to a packed single-precision float value and store the result into a vector register. |
| V_PRNG_B32 | vdst, vsrc0 | Generate a pseudorandom number using an LFSR (linear feedback shift register) seeded with the vector input, then store the result into a vector register. |
| V_PERMLANE16_SWAP_B32 | vdst, vsrc0 | Swap data between two vector registers. |
| V_PERMLANE32_SWAP_B32 | vdst, vsrc0 | Swap data between two vector registers. |
| V_CVT_F32_BF16 | vdst, vsrc0 | Convert from a BF16 float input to a single-precision float value and store the result into a vector register. |
## VOPC — compares

| Instruction | Syntax | Description |
|---|---|---|
| V_CMP_CLASS_F32 | vcc, vsrc0, vsrc1 | Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a single-precision float, and set the per-lane condition code to the result. |
| V_CMPX_CLASS_F32 | vcc & exec, vsrc0, vsrc1 | Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a single-precision float, and set the per-lane condition code to the result. |
| V_CMP_CLASS_F64 | vcc, vsrc0, vsrc1 | Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a double-precision float, and set the per-lane condition code to the result. |
| V_CMPX_CLASS_F64 | vcc & exec, vsrc0, vsrc1 | Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a double-precision float, and set the per-lane condition code to the result. |
| V_CMP_CLASS_F16 | vcc, vsrc0, vsrc1 | Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a half-precision float, and set the per-lane condition code to the result. |
| V_CMPX_CLASS_F16 | vcc & exec, vsrc0, vsrc1 | Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a half-precision float, and set the per-lane condition code to the result. |
| V_CMP_F_F16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMP_LT_F16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_F16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_F16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_F16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_LG_F16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. |
| V_CMP_GE_F16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_O_F16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is orderable to the second input. |
| V_CMP_U_F16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not orderable to the second input. |
| V_CMP_NGE_F16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. |
| V_CMP_NLG_F16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. |
| V_CMP_NGT_F16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not greater than the second input. |
| V_CMP_NLE_F16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. |
| V_CMP_NEQ_F16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_NLT_F16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than the second input. |
| V_CMP_TRU_F16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMPX_F_F16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMPX_LT_F16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_F16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_F16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_F16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_LG_F16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. |
| V_CMPX_GE_F16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_O_F16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is orderable to the second input. |
| V_CMPX_U_F16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not orderable to the second input. |
| V_CMPX_NGE_F16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. |
| V_CMPX_NLG_F16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. |
| V_CMPX_NGT_F16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not greater than the second input. |
| V_CMPX_NLE_F16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. |
| V_CMPX_NEQ_F16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_NLT_F16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than the second input. |
| V_CMPX_TRU_F16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMP_F_F32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMP_LT_F32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_F32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_F32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_F32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_LG_F32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. |
| V_CMP_GE_F32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_O_F32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is orderable to the second input. |
| V_CMP_U_F32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not orderable to the second input. |
| V_CMP_NGE_F32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. |
| V_CMP_NLG_F32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. |
| V_CMP_NGT_F32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not greater than the second input. |
| V_CMP_NLE_F32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. |
| V_CMP_NEQ_F32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_NLT_F32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than the second input. |
| V_CMP_TRU_F32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMPX_F_F32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMPX_LT_F32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_F32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_F32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_F32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_LG_F32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. |
| V_CMPX_GE_F32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_O_F32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is orderable to the second input. |
| V_CMPX_U_F32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not orderable to the second input. |
| V_CMPX_NGE_F32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. |
| V_CMPX_NLG_F32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. |
| V_CMPX_NGT_F32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not greater than the second input. |
| V_CMPX_NLE_F32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. |
| V_CMPX_NEQ_F32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_NLT_F32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than the second input. |
| V_CMPX_TRU_F32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMP_F_F64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMP_LT_F64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_F64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_F64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_F64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_LG_F64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. |
| V_CMP_GE_F64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_O_F64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is orderable to the second input. |
| V_CMP_U_F64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not orderable to the second input. |
| V_CMP_NGE_F64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. |
| V_CMP_NLG_F64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. |
| V_CMP_NGT_F64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not greater than the second input. |
| V_CMP_NLE_F64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. |
| V_CMP_NEQ_F64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_NLT_F64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than the second input. |
| V_CMP_TRU_F64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMPX_F_F64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMPX_LT_F64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_F64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_F64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_F64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_LG_F64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. |
| V_CMPX_GE_F64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_O_F64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is orderable to the second input. |
| V_CMPX_U_F64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not orderable to the second input. |
| V_CMPX_NGE_F64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. |
| V_CMPX_NLG_F64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. |
| V_CMPX_NGT_F64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not greater than the second input. |
| V_CMPX_NLE_F64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. |
| V_CMPX_NEQ_F64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_NLT_F64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not less than the second input. |
| V_CMPX_TRU_F64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMP_F_I16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMP_LT_I16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_I16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_I16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_I16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_NE_I16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_GE_I16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_T_I16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMP_F_U16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMP_LT_U16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_U16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_U16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_U16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_NE_U16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_GE_U16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_T_U16 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMPX_F_I16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMPX_LT_I16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_I16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_I16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_I16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_NE_I16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_GE_I16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_T_I16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMPX_F_U16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMPX_LT_U16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_U16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_U16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_U16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_NE_U16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_GE_U16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_T_U16 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMP_F_I32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMP_LT_I32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_I32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_I32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_I32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_NE_I32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_GE_I32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_T_I32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMP_F_U32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMP_LT_U32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_U32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_U32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_U32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_NE_U32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_GE_U32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_T_U32 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMPX_F_I32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMPX_LT_I32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_I32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_I32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_I32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_NE_I32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_GE_I32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_T_I32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMPX_F_U32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMPX_LT_U32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_U32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_U32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_U32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_NE_U32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_GE_U32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_T_U32 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMP_F_I64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMP_LT_I64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_I64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_I64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_I64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_NE_I64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_GE_I64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_T_I64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMP_F_U64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMP_LT_U64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMP_EQ_U64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMP_LE_U64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMP_GT_U64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMP_NE_U64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMP_GE_U64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMP_T_U64 | vcc, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMPX_F_I64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMPX_LT_I64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_I64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_I64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_I64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_NE_I64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_GE_I64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_T_I64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
| V_CMPX_F_U64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 0. |
| V_CMPX_LT_U64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than the second input. |
| V_CMPX_EQ_U64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is equal to the second input. |
| V_CMPX_LE_U64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. |
| V_CMPX_GT_U64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than the second input. |
| V_CMPX_NE_U64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is not equal to the second input. |
| V_CMPX_GE_U64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. |
| V_CMPX_T_U64 | vcc & exec, vsrc0, vsrc1 | Set the per-lane condition code to 1. |
