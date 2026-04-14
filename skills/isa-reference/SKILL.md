---
name: isa-reference
description: AMD CDNA4 (gfx950) ISA instruction set reference for MI355X. Contains all scalar, vector, matrix (MFMA), memory, and LDS instructions. Use when writing AMD GPU assembly kernels, understanding disassembled .co code objects, or asking about AMDGPU instructions.
---

# CDNA4 ISA Reference (gfx950 / MI355X)

## 指令分类

| 文件 | 内容 |
|------|------|
| [01_scalar_alu.md](01_scalar_alu.md) | SOP2, SOPK, SOP1, SOPC, SOPP |
| [02_vector_alu_basic.md](02_vector_alu_basic.md) | VOP2, VOP1, VOPC |
| [03_vector_alu_advanced.md](03_vector_alu_advanced.md) | VOP3P, VOP3A/VOP3B |
| [04_matrix_mfma.md](04_matrix_mfma.md) | MFMA/SMFMAC 矩阵指令 |
| [05_scalar_memory.md](05_scalar_memory.md) | SMEM |
| [06_vector_memory.md](06_vector_memory.md) | MUBUF, MTBUF, Flat, Global |
| [07_lds_data_share.md](07_lds_data_share.md) | DS_READ, DS_WRITE, DS_ATOMIC |

## FP4 GEMM 核心指令

| 用途 | 指令 |
|------|------|
| FP4 矩阵乘 | `V_MFMA_SCALE_F32_16X16X128_F8F6F4` (cbsz=4,blgp=4) |
| 全局内存读 | `BUFFER_LOAD_DWORDX4`, `GLOBAL_LOAD_DWORDX4` |
| LDS 读写 | `DS_READ_B128`, `DS_WRITE_B128` |
| 控制 | `S_WAITCNT`, `S_BARRIER`, `S_ENDPGM` |

## 汇编工具链

```bash
/opt/rocm/llvm/bin/llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx950 -filetype=obj kernel.s -o kernel.o
/opt/rocm/llvm/bin/ld.lld -shared kernel.o -o kernel.co
/opt/rocm/llvm/bin/llvm-objdump -d kernel.co
```

## gfx950 架构要点

| 特性 | 值 |
|------|-----|
| Wavefront | 64 threads |
| CU (MI355X) | 256 |
| VGPR / CU | 512 |
| LDS / CU | 64 KB |
| MFMA FP4 peak | ~2.6 PFLOPS |
