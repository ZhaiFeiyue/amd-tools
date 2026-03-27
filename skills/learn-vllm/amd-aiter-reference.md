# AMD AITER (AI Tensor Engine for ROCm) in vLLM

Source: https://rocmdocs.amd.com/en/develop/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
Blog: https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
AITER repo: https://github.com/ROCm/aiter

## What is AITER

AITER (AI Tensor Engine for ROCm) is AMD's centralized repository for high-performance AI operators.
Provides C++ and Python APIs with kernels from Triton, CK (Composable Kernel), and ASM implementations.
Targets AMD Instinct MI300X, MI325X, MI350X, MI355X GPUs (CDNA 3 architecture).

## Environment Variables

### Master Switch

```bash
export VLLM_ROCM_USE_AITER=1   # Enable all AITER optimizations (recommended)
export VLLM_ROCM_USE_AITER=0   # Disable entirely, use vLLM Triton kernels (default)
```

When `VLLM_ROCM_USE_AITER=1`, all sub-flags default to `True`. You typically only need the master switch.

### AITER Component Flags

| Environment Variable | Default (when AITER on) | Description |
|---|---|---|
| `VLLM_ROCM_USE_AITER` | `0` (off) | **Master switch**. All other flags require this to be `1`. |
| `VLLM_ROCM_USE_AITER_LINEAR` | `True` | AITER quantization + GEMM for linear layers (`tgemm` kernel) |
| `VLLM_ROCM_USE_AITER_MOE` | `True` | AITER fused MoE kernels (`ck_moe`, `asm_moe`, `fmoe_fp8_block_scaled`) |
| `VLLM_ROCM_USE_AITER_RMSNORM` | `True` | AITER RMSNorm kernels (`rmsnorm2d_fwd_with_add`) |
| `VLLM_ROCM_USE_AITER_MLA` | `True` | AITER Multi-head Latent Attention (DeepSeek-V2/V3/R1) |
| `VLLM_ROCM_USE_AITER_MHA` | `True` | AITER Multi-Head Attention (standard transformers) |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | `False` | AITER unified attention kernel (for GPT-OSS) |
| `VLLM_ROCM_USE_AITER_FP8BMM` | `True` | AITER FP8 batched matmul (MLA models like DeepSeek) |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `True` | Skinny-GEMM kernel for small batch sizes |
| `VLLM_ROCM_FP8_PADDING` | `True` | Pad FP8 weight tensors for memory locality |
| `VLLM_ROCM_MOE_PADDING` | `True` | Pad MoE weight tensors for memory access |
| `VLLM_ROCM_CUSTOM_PAGED_ATTN` | `True` | Custom paged-attention decode kernel |

### Other ROCm Performance Variables

| Variable | Description |
|---|---|
| `HIP_FORCE_DEV_KERNARG=1` | Improve kernel launch performance (set by default in Docker) |
| `TORCH_BLAS_PREFER_HIPBLASLT=1` | Prefer hipBLASLt for GEMM |
| `NCCL_MIN_NCHANNELS=112` | Increase RCCL channels (multi-GPU only) |
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` | Quick Reduce quantization: `NONE`, `FP`, `INT8`, `INT6`, `INT4` |
| `VLLM_V1_USE_PREFILL_DECODE_ATTENTION` | Enable Triton Prefill-Decode split attention |

## Attention Backends on ROCm (7 total)

### Backend Selection (auto)

```
VLLM_ROCM_USE_AITER=1 →
  MLA model (DeepSeek-V3/R1)? → AITER MLA (auto)
  Standard model (Llama, Qwen)? → AITER MHA (auto)

VLLM_ROCM_USE_AITER=0 →
  MLA model? → vLLM Triton MLA
  Standard model? → vLLM Triton Unified Attention
```

### Backend Details

| Backend | Prefill Kernel | Decode Kernel | Model Type | How to Enable |
|---|---|---|---|---|
| **AITER MHA** | AITER flash_attn (3-path routing) | AITER pa_fwd_asm | Standard (Llama, Mistral, Qwen) | `VLLM_ROCM_USE_AITER=1` (auto) |
| **AITER MLA** | AITER flash_attn | AITER mla_decode_fwd (assembly) | MLA (DeepSeek, Kimi-K2) | `VLLM_ROCM_USE_AITER=1` + `--block-size 1` (auto) |
| **AITER Triton MLA** | AITER Triton MHA | AITER assembly | MLA | Manual config |
| **TRITON_MLA** | vLLM Triton | vLLM Triton | MLA | `VLLM_ROCM_USE_AITER=0` (auto for MLA) |
| **vLLM Triton Unified** | Triton unified kernel | Triton unified kernel | Any | Default (AITER off) |
| **AITER Triton Unified** | AITER Triton unified | AITER Triton unified | GPT-OSS | `AITER=1, MHA=0, UNIFIED=1` |
| **Triton Prefill-Decode** | context_attention_fwd | paged_attention / fallback | Any | `PREFILL_DECODE_ATTENTION=1` |

### AITER MHA: 3-Path Routing (ROCM_AITER_FA)

The key innovation: routes each request type to a specialized kernel instead of one-size-fits-all.

- **Prefill path**: `flash_attn_varlen_func` — compute-bound, maximizes ALU utilization
- **Extend path**: Chunked attention with LSE merging — handles 100K+ contexts
- **Decode path**: `pa_fwd_asm` — memory-bound, optimized for bandwidth

Performance: 1.2-4.4x higher throughput vs other backends.

Batch reordering: Requests reordered to `[decode > extend > prefill]` for contiguous memory access.

KV Cache layout (preshuffled for AMD CDNA):
```
k_cache: [num_blocks, num_heads, head_dim // x, block_size, x]
v_cache: [num_blocks, num_heads, block_size // x, head_dim, x]
```

### CUDA Graph Compatibility

| Backend | CUDA Graph Support |
|---|---|
| Triton Unified / Prefill-Decode | Full (prefill + decode) |
| AITER MHA, AITER MLA | Uniform batches only |
| vLLM Triton MLA | PIECEWISE required |

## AITER MoE Kernels

### Available Kernels

| Kernel | Quantization | Use Case |
|---|---|---|
| `ck_moe` | BF16/FP16 (unquantized) | Mixtral-style MoE |
| `asm_moe` | Dynamic per-tensor FP8 | FP8 quantized MoE |
| `fmoe_fp8_block_scaled` | Block-scaled FP8 | DeepSeek-V3 style block FP8 |
| MXFP4 MoE (Triton) | MXFP4 (4-bit float) | MI350/MI355X only (gfx950) |

### Applicable Models

- Mixtral-8x7B / 8x22B
- Llama-4-Scout / Maverick
- DeepSeek-V2 / V3 / R1
- Qwen1.5-MoE / Qwen2-MoE / Qwen2.5-MoE
- Kimi-K2 / K2.5

### Performance Gains

| Model | Improvement |
|---|---|
| DeepSeek-V3 (block FP8) | 8-26.7% throughput, 41% TTFT speedup |
| Mixtral-8x7B-FP8 | up to 75% improvement |

## AITER MLA Requirements

For MLA models (DeepSeek-V2/V3/R1, Kimi-K2):

- **Must** set `--block-size 1` (vLLM errors without it)
- `VLLM_ROCM_USE_AITER_MLA=1` (default when AITER on)
- Persistent MLA kernel: 1.21-1.47x throughput uplift on DeepSeek-R1

```bash
VLLM_ROCM_USE_AITER=1 vllm serve deepseek-ai/DeepSeek-R1 \
  --block-size 1 \
  --tensor-parallel-size 8
```

### Kimi-K2.5 on AITER

Kimi-K2.5 uses MLA (64 heads) but AITER MLA has head-count constraints with certain TP values:
- TP=4: 64/4=16 heads/GPU — **AITER MLA may not support** (set `VLLM_ROCM_USE_AITER=0`)
- TP=8: 64/8=8 heads/GPU — may work depending on AITER version
- Use `TRITON_MLA` backend as fallback: `VLLM_ROCM_USE_AITER=0`

## FP8 / FP4 Quantization on ROCm

### Supported Quantization Methods

| Method | Precision | Memory Reduction | Best Use Case |
|---|---|---|---|
| **FP8 (W8A8)** | 8-bit float | 2x (50%) | Production, balanced speed/accuracy |
| **PTPC-FP8** | 8-bit float (per-token per-channel) | 2x (50%) | Higher accuracy than standard FP8 |
| **MXFP4** | 4-bit float (microscaling) | 4x (75%) | MI350/MI355X only, max compression |
| **AWQ** | 4-bit int (W4A16) | 4x (75%) | Memory-constrained |
| **GPTQ** | 4/8-bit int | 2-4x | Pre-quantized models |
| **FP8 KV-cache** | 8-bit float KV | KV cache 50% | All workloads |
| **Quark (AMD)** | FP8/MXFP4 | 2-4x | AMD pre-quantized models |
| **compressed-tensors** | W8A8 INT8/FP8 | 2x | LLM Compressor models |

### On-the-fly Quantization

```bash
# FP8 per-tensor (simplest)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --quantization fp8 --dtype auto --tensor-parallel-size 4

# PTPC-FP8 (recommended, better accuracy)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --quantization ptpc_fp8 --dtype auto --tensor-parallel-size 4
```

### Pre-quantized Models (AMD Quark)

```bash
# FP8 W8A8
vllm serve amd/Llama-3.1-8B-Instruct-FP8-KV --dtype auto

# MXFP4 (MI350/MI355X only)
vllm serve amd/Llama-3.3-70B-Instruct-MXFP4-Preview --dtype auto -tp 1

# DeepSeek-R1 MXFP4
vllm serve amd/DeepSeek-R1-0528-MXFP4-Preview --dtype auto -tp 8 --block-size 1
```

Available pre-quantized models on HuggingFace (`amd/` namespace):
- Llama-3.1-{8B,70B,405B}-Instruct-FP8-KV
- Mixtral-{8x7B,8x22B}-Instruct-v0.1-FP8-KV
- Llama-3.3-70B-Instruct-MXFP4-Preview (MI350/MI355)
- Llama-3.1-405B-Instruct-MXFP4-Preview (MI350/MI355)
- DeepSeek-R1-0528-MXFP4-Preview (MI350/MI355)

### AITER FP8 Kernels

| Kernel | Description | Flag |
|---|---|---|
| AITER Linear (FP8) | FP8 quantized GEMM for linear layers | `VLLM_ROCM_USE_AITER_LINEAR` |
| AITER FP8BMM | FP8 batched matmul for MLA models | `VLLM_ROCM_USE_AITER_FP8BMM` |
| asm_moe (FP8) | Assembly MoE kernel for dynamic per-tensor FP8 | `VLLM_ROCM_USE_AITER_MOE` |
| fmoe_fp8_block_scaled | Block-scaled FP8 MoE (DeepSeek style) | `VLLM_ROCM_USE_AITER_MOE` |
| FP8 Paged Attention | FP8 MFMA for paged attention with query scaling | via AITER pa_fwd_asm |

### FP4 (MXFP4) on MI350/MI355X

MXFP4 (Microscaling FP4) is supported on gfx950 (MI350/MI355X) only:
- Uses hardware MXFP4 matrix cores
- 4x memory reduction vs BF16
- MXFP4 MoE kernel via Triton (in AITER)
- Requires Quark-quantized models or compatible format

## Quick Reference: Common Deployment Recipes

### Standard Dense Model (Llama, Qwen, Mistral)

```bash
VLLM_ROCM_USE_AITER=1 \
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 4 --dtype auto
```

### MoE Model with FP8 (Mixtral)

```bash
VLLM_ROCM_USE_AITER=1 \
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --quantization fp8 --dtype auto
```

### MLA + MoE Model (DeepSeek-V3/R1)

```bash
VLLM_ROCM_USE_AITER=1 \
vllm serve deepseek-ai/DeepSeek-R1 \
  --block-size 1 --tensor-parallel-size 8
```

### Kimi-K2.5 on MI355X (AITER off, TRITON_MLA)

```bash
VLLM_ROCM_USE_AITER=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
RCCL_MSCCL_ENABLE=0 \
vllm serve moonshotai/Kimi-K2.5 \
  --tensor-parallel-size 4 --trust-remote-code \
  --max-model-len 32768 --gpu-memory-utilization 0.85
```

### DeepSeek-R1 MXFP4 on MI355X

```bash
VLLM_ROCM_USE_AITER=1 \
vllm serve amd/DeepSeek-R1-0528-MXFP4-Preview \
  --block-size 1 --tensor-parallel-size 8 --dtype auto
```

## Verify Active Backend

```bash
VLLM_ROCM_USE_AITER=1 vllm serve <model> 2>&1 | grep -i attention
```

Expected log messages:
- `Using Aiter Flash Attention backend on V1 engine.` → AITER MHA
- `Using AITER MLA backend on V1 engine.` → AITER MLA
- `Using Triton MLA backend on V1 engine.` → vLLM Triton MLA
- `Using Triton Attention backend on V1 engine.` → vLLM Triton Unified
