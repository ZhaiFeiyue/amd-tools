---
name: learn-vllm
description: >-
  Guide for learning and working with vLLM inference engine. Covers parallelism strategies
  (DP, TP, CP, SP, PCP, DCP, EP, PP), serving configuration, installation, benchmarking,
  and model support including Kimi-K2.5. Use when the user asks about vLLM, LLM serving,
  distributed inference, tensor parallelism, pipeline parallelism, expert parallelism,
  or deploying models with vLLM.
---

# Learn vLLM

## What is vLLM

vLLM is a high-throughput LLM inference and serving engine. Core technologies:
- **PagedAttention**: Efficient KV cache memory management (inspired by OS virtual memory)
- **Continuous batching**: Dynamic request scheduling without padding
- **CUDA/HIP graph**: Fast model execution with graph capture
- **Quantization**: GPTQ, AWQ, AutoRound, INT4, INT8, FP8
- **Speculative decoding**: Draft model + target model for faster generation

## Key Resources

| Resource | URL |
|----------|-----|
| GitHub | https://github.com/vllm-project/vllm |
| Documentation | https://docs.vllm.ai/en/latest/ |
| Installation (GPU) | https://docs.vllm.ai/en/latest/getting_started/installation/gpu/ |
| Installation (ROCm) | https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#amd-rocm |
| Issues | https://github.com/vllm-project/vllm/issues |
| Pull Requests | https://github.com/vllm-project/vllm/pulls |
| Discussions | https://github.com/vllm-project/vllm/discussions |
| Blog | https://blog.vllm.ai/ |
| Supported Models | https://docs.vllm.ai/en/latest/models/supported_models.html |

## Installation

```bash
# Quick install (NVIDIA CUDA)
pip install vllm

# From source
git clone https://github.com/vllm-project/vllm.git
cd vllm && pip install -e .

# Docker (ROCm)
docker pull rocm/vllm-dev:nightly_main_<date>

# Docker (NVIDIA)
docker pull vllm/vllm-openai:latest
```

Requirements: Python 3.9+, CUDA 12.x / ROCm 6.x, PyTorch 2.x

## Parallelism Strategies

vLLM supports 8 parallelism strategies. For detailed configuration, see [parallelism-reference.md](parallelism-reference.md).

| Abbreviation | Full Name | Parameter | When to Use |
|---|---|---|---|
| **TP** | Tensor Parallelism | `--tensor-parallel-size N` | Model too large for 1 GPU, single-node |
| **PP** | Pipeline Parallelism | `--pipeline-parallel-size N` | Model too large for single node |
| **DP** | Data Parallelism | `--data-parallel-size N` | Increase throughput with replicas |
| **EP** | Expert Parallelism | `--enable-expert-parallel` | MoE models, distribute experts across GPUs |
| **CP** | Context Parallelism | prefill context parallel | Long sequence prefill across GPUs |
| **DCP** | Decode Context Parallel | decode context parallel flag | Shard KV cache across GPUs in decode |
| **PCP** | Prefill Context Parallel | prefill context parallel | Split prefill computation across GPUs |
| **SP** | Sequence Parallelism | sequence parallel config | Parallelize along sequence dimension |

### Quick Decision Guide

```
Single GPU sufficient?
  → No parallelism needed

Model fits on one node?
  → Use TP (set tensor_parallel_size = num_gpus)

Model requires multiple nodes?
  → Use TP + PP (TP = gpus_per_node, PP = num_nodes)

MoE model (e.g., Mixtral, DeepSeek, Kimi-K2.5)?
  → Use EP (--enable-expert-parallel)
  → EP_SIZE = TP_SIZE × DP_SIZE

Need higher throughput?
  → Add DP (--data-parallel-size N)

Long context (>32k tokens)?
  → Consider CP/PCP for prefill, DCP for decode
```

## Serving

### Start OpenAI-Compatible Server

```bash
vllm serve <model> \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code
```

### Client Usage

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model>","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

### Offline Inference (Python)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-8B-Instruct", tensor_parallel_size=2)
params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["What is AI?"], params)
```

### Key Server Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--tensor-parallel-size` | Number of GPUs for TP | 1 |
| `--pipeline-parallel-size` | Number of stages for PP | 1 |
| `--data-parallel-size` | Number of DP replicas | 1 |
| `--max-model-len` | Maximum sequence length | model default |
| `--gpu-memory-utilization` | Fraction of GPU memory to use | 0.9 |
| `--quantization` | Quantization method (gptq, awq, fp8, etc.) | None |
| `--enable-prefix-caching` | Enable automatic prefix caching | False |
| `--enable-expert-parallel` | Enable EP for MoE models | False |
| `--kv-transfer-config` | KV cache transfer config (PD disagg) | None |
| `--trust-remote-code` | Allow custom model code | False |
| `--profiler-config` | Torch profiler configuration | None |

## Benchmarking

### Run Benchmarks

```bash
# Serving benchmark
vllm bench serve \
  --model <model> --host 0.0.0.0 --port 8000 \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 128 \
  --num-prompts 20 --max-concurrency 8 \
  --request-rate inf --trust-remote-code

# With profiling
vllm bench serve ... --profile
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| TTFT | Time to first token (prefill latency) |
| ITL | Inter-token latency (decode latency per token) |
| Output tok/s | Output generation throughput |
| Total tok/s | Total tokens processed per second |
| P99 latency | 99th percentile tail latency |

## Code Architecture

For detailed code structure and Kimi-K2.5 specifics, see [code-architecture.md](code-architecture.md).

```
vllm/
├── entrypoints/          # API servers (OpenAI-compatible, CLI)
│   ├── openai/           # OpenAI API implementation
│   ├── cli/              # vllm serve, vllm bench commands
│   └── serve/            # Serving utilities
├── engine/               # Core engine (scheduling, batching)
├── model_executor/       # Model execution layer
│   ├── models/           # Model implementations (Llama, Kimi, DeepSeek...)
│   ├── layers/           # Attention, MoE, quantization layers
│   │   ├── fused_moe/    # Fused MoE kernel
│   │   └── quantization/ # GPTQ, AWQ, FP8 implementations
│   └── model_loader/     # Weight loading
├── distributed/          # Distributed communication
│   ├── kv_transfer/      # KV cache transfer (NixlConnector, P2P)
│   ├── elastic_ep/       # Elastic expert parallelism
│   └── eplb/             # Expert parallel load balancing
├── v1/                   # V1 engine (newer architecture)
│   ├── core/             # Core scheduler
│   ├── worker/           # GPU worker
│   └── engine/           # V1 engine
├── config.py             # All configuration classes
└── attention/            # Attention backends (FlashAttention, FlashInfer, Triton)
```

## Model Support

For the complete model list with architecture classes, see [supported-models.md](supported-models.md).

### Text Generation (100+ architectures)

| Category | Key Models |
|----------|------------|
| Dense LLMs | Llama-3, Qwen2/3, Gemma-2/3, Mistral, Phi-3, ChatGLM/GLM-4, InternLM2, Falcon, OLMo, GPT-NeoX |
| MoE LLMs | DeepSeek-V2/V3/V3.2, **Kimi-K2**, Mixtral, Qwen2-MoE/Qwen3-MoE, Llama-4, DBRX, Arctic, Grok-1, MiniMax, HunYuan, Ernie-4.5-MoE |
| SSM / Hybrid | Mamba/Mamba2, Jamba, Falcon-H1, Zamba2, Bamba |

### Multimodal (50+ architectures)

| Category | Key Models |
|----------|------------|
| Vision (Image) | **Kimi-K2.5**, **Kimi-VL**, Qwen2-VL/Qwen2.5-VL/Qwen3-VL, Gemma3, Llama-4, InternVL, LLaVA/LLaVA-NeXT, Phi-3V/Phi-4MM, Pixtral, Mistral-3, DeepSeek-VL2, GLM-4V, MiniCPM-V, Molmo |
| Audio | **Kimi-Audio**, Qwen2-Audio, Ultravox, Whisper |
| Omni (多模态) | Qwen2.5-Omni, Qwen3-Omni-MoE, MiniCPM-O, Phi-4MM |

### Embedding & Retrieval

| Category | Key Models |
|----------|------------|
| Text Embedding | BERT, RoBERTa, GTE, NomicBERT, ModernBERT, E5-Mistral, BGE-M3, GritLM |
| Multimodal Embedding | CLIP, SigLIP, ColPali, ColQwen3 |
| Reward / Classification | Qwen2 Reward, BERT Classification |

### Architecture Types

- **Dense**: All parameters activated per token (e.g., Llama-3 70B)
- **MoE**: Sparse activation (e.g., Kimi-K2.5: 1T total / 32B activated, 384 experts)
- **MLA**: Multi-head Latent Attention, compresses KV cache (DeepSeek-V2/V3, Kimi-K2/K2.5)
- **Hybrid**: Transformer + SSM layers (Jamba, Falcon-H1, Zamba2)

Full list: https://docs.vllm.ai/en/latest/models/supported_models.html

## AMD AITER (AI Tensor Engine for ROCm)

For the full AITER reference (all flags, backends, FP8/FP4), see [amd-aiter-reference.md](amd-aiter-reference.md).

### Quick Start

```bash
# Enable all AITER optimizations (recommended for MI300X/MI355X)
export VLLM_ROCM_USE_AITER=1
vllm serve <model>
```

### Key Flags

| Flag | Default | Description |
|---|---|---|
| `VLLM_ROCM_USE_AITER` | `0` | **Master switch** — all others require this = `1` |
| `VLLM_ROCM_USE_AITER_LINEAR` | `True` | FP8 GEMM for linear layers (`tgemm`) |
| `VLLM_ROCM_USE_AITER_MOE` | `True` | Fused MoE: `ck_moe` (BF16), `asm_moe` (FP8), `fmoe_fp8_block_scaled` |
| `VLLM_ROCM_USE_AITER_RMSNORM` | `True` | Fused RMSNorm kernel |
| `VLLM_ROCM_USE_AITER_MLA` | `True` | MLA attention (DeepSeek, Kimi-K2). Requires `--block-size 1` |
| `VLLM_ROCM_USE_AITER_MHA` | `True` | MHA attention (Llama, Qwen, Mistral). 3-path routing: 1.2-4.4x speedup |
| `VLLM_ROCM_USE_AITER_FP8BMM` | `True` | FP8 batched matmul for MLA models |

### Attention Backend Auto-Selection

```
AITER=1 + MLA model → AITER MLA (requires --block-size 1)
AITER=1 + standard model → AITER MHA (3-path: prefill/extend/decode)
AITER=0 + MLA model → vLLM Triton MLA
AITER=0 + standard model → vLLM Triton Unified
```

### FP8 / FP4 Quantization on ROCm

| Method | Precision | Memory Reduction | Command |
|---|---|---|---|
| FP8 on-the-fly | W8A8 | 2x | `--quantization fp8` |
| PTPC-FP8 | Per-token per-channel | 2x | `--quantization ptpc_fp8` |
| MXFP4 (MI350/MI355X only) | 4-bit float | 4x | Pre-quantized models (`amd/` on HF) |
| AWQ | W4A16 | 4x | `--quantization awq` |
| GPTQ | W4A16 / W8A16 | 2-4x | `--quantization gptq` |

### Kimi-K2.5 AITER Compatibility

AITER MLA has head-count constraints with Kimi-K2.5 (64 MLA heads):
- TP=4 → 16 heads/GPU: **AITER incompatible**, use `VLLM_ROCM_USE_AITER=0` (TRITON_MLA)
- TP=8 → 8 heads/GPU: may work (version-dependent)

## Additional Resources

- For parallelism deep dive, see [parallelism-reference.md](parallelism-reference.md)
- For code architecture and Kimi-K2.5 details, see [code-architecture.md](code-architecture.md)
- For local benchmark results, see [benchmark-results.md](benchmark-results.md)
- For complete model support list, see [supported-models.md](supported-models.md)
- For AMD AITER full reference (attention backends, MoE kernels, FP8/FP4), see [amd-aiter-reference.md](amd-aiter-reference.md)
