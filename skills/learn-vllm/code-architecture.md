# vLLM Code Architecture & Kimi-K2.5

## Repository Structure

```
vllm/
├── entrypoints/              # User-facing entry points
│   ├── openai/               # OpenAI-compatible API server
│   │   ├── api_server.py     # FastAPI server
│   │   ├── completion/       # Completions endpoint
│   │   └── generate/         # Generation logic
│   ├── cli/                  # CLI commands (vllm serve, vllm bench)
│   └── serve/                # Server utilities (disagg, elastic_ep, rlhf)
│
├── engine/                   # Core engine
│   └── (legacy v0 engine)    # Original engine implementation
│
├── v1/                       # V1 engine (current default)
│   ├── engine/               # Async engine, request handling
│   ├── core/                 # Scheduler, KV cache management
│   ├── worker/               # GPU worker processes
│   │   └── gpu/              # GPU-specific worker (model_runner, sampling)
│   ├── executor/             # Process/Ray executor
│   ├── spec_decode/          # Speculative decoding
│   └── attention/            # V1 attention ops
│
├── model_executor/           # Model execution layer
│   ├── models/               # Model implementations
│   │   ├── llama.py          # Llama family
│   │   ├── kimi_k2.py        # Kimi-K2 (MoE + MLA)
│   │   ├── kimi_k25.py       # Kimi-K2.5 (multimodal MoE)
│   │   ├── deepseek_v2.py    # DeepSeek-V2
│   │   ├── deepseek_v3.py    # DeepSeek-V3
│   │   ├── qwen2.py          # Qwen2
│   │   └── ...               # 100+ model implementations
│   ├── layers/               # Reusable layers
│   │   ├── attention/        # Attention implementations
│   │   ├── fused_moe/        # Fused MoE kernel (critical for MoE perf)
│   │   │   ├── router/       # Expert routing
│   │   │   ├── runner/       # MoE execution
│   │   │   └── oracle/       # Expert selection oracle
│   │   ├── quantization/     # Quantization methods
│   │   │   ├── gptq.py
│   │   │   ├── awq.py
│   │   │   ├── fp8.py
│   │   │   └── compressed_tensors/
│   │   ├── rotary_embedding.py
│   │   └── linear.py         # Column/Row parallel linear
│   ├── model_loader/         # Weight loading from HF/safetensors
│   └── offloader/            # CPU/disk offloading
│
├── distributed/              # Distributed communication
│   ├── parallel_state.py     # Process group management
│   ├── communication_op.py   # All-reduce, broadcast ops
│   ├── kv_transfer/          # KV cache transfer for PD disagg
│   │   └── kv_connector/     # NixlConnector, P2PConnector, etc.
│   ├── elastic_ep/           # Elastic expert parallelism
│   └── eplb/                 # Expert parallel load balancing
│
├── attention/                # Attention backends
│   ├── backends/             # FlashAttention, FlashInfer, Triton, CK
│   └── selector.py           # Backend auto-selection
│
├── config.py                 # All config classes (ParallelConfig, ModelConfig, etc.)
├── sampling_params.py        # Sampling parameters
└── transformers_utils/       # HuggingFace integration utilities
    ├── configs/              # Custom model configs (kimi_k25, etc.)
    └── processors/           # Custom multimodal processors
```

## Kimi-K2.5 Implementation

### Model Architecture
- **Type**: Multimodal MoE with MLA (Multi-head Latent Attention)
- **Parameters**: 1T total, 32B activated per token
- **Experts**: 384 experts in MoE layers
- **Attention**: MLA (Multi-head Latent Attention) with 64 heads
- **Modalities**: Text + Image + Video-chunks

### Key Files
- `vllm/model_executor/models/kimi_k25.py` — `KimiK25ForConditionalGeneration`
- `vllm/model_executor/models/kimi_k2.py` — Base MoE text model
- `vllm/transformers_utils/configs/kimi_k25/` — Custom config
- `vllm/transformers_utils/processors/` — `KimiK25MultiModalProcessor`

### Serving Kimi-K2.5

```bash
# NVIDIA (TP=8)
vllm serve moonshotai/Kimi-K2.5 -tp 8 \
    --mm-encoder-tp-mode data \
    --compilation_config.pass_config.fuse_allreduce_rms true \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --trust-remote-code

# AMD MI355X (TP=4, requires special env vars)
VLLM_ROCM_USE_AITER=0 \
VLLM_USE_TRITON_FLASH_ATTN=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
RCCL_MSCCL_ENABLE=0 \
vllm serve moonshotai/Kimi-K2.5 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85
```

### AMD MI355X Specific Constraints

| Config | Value | Reason |
|--------|-------|--------|
| `VLLM_ROCM_USE_AITER` | `0` | AITER MLA incompatible with TP=4 head count (64/4=16) |
| `VLLM_USE_TRITON_FLASH_ATTN` | `0` | Vision encoder needs CK attention |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | ROCm requires spawn, not fork |
| `--tensor-parallel-size` | `4` | TP=4 gives 16 heads/GPU (TP=8 gives 8, not supported) |
| `--block-size` | **DO NOT set to 1** | TRITON_MLA doesn't support block_size=1 |
| Model load time | ~50 min | 64 safetensors shards, ~55s/shard, 144.63 GiB total |

## Request Flow

```
Client Request (HTTP)
  → FastAPI Server (entrypoints/openai/)
    → AsyncEngine (v1/engine/)
      → Scheduler (v1/core/) — decides which requests to process
        → ModelRunner (v1/worker/gpu/) — prepares input tensors
          → Model Forward Pass (model_executor/models/)
            → Attention Layer → Attention Backend (FlashAttention/Triton/CK)
            → MoE Layer → Fused MoE Kernel (model_executor/layers/fused_moe/)
            → Communication → All-Reduce (distributed/)
          → Sampler (v1/sample/) — generates next token
        → KV Cache Update (v1/core/)
      → Detokenize + Stream Response
  → Client
```

## Attention Backends

| Backend | Hardware | Notes |
|---------|----------|-------|
| FlashAttention-2 | NVIDIA | Default for most models |
| FlashInfer | NVIDIA | Alternative, good for MLA |
| Triton Flash Attention | NVIDIA/AMD | Cross-platform |
| TRITON_MLA | AMD | MLA-specific, used by Kimi-K2.5 on ROCm |
| CK (Composable Kernel) | AMD | AMD-native attention |
| AITER | AMD | AMD AI Engine for Transformers |

## MoE Kernel Details

The fused MoE kernel (`model_executor/layers/fused_moe/`) is critical for performance:
- Routes tokens to top-K experts via gating network
- Fuses expert selection + GEMM into single kernel
- Supports GPTQ/AWQ quantized experts (INT4)
- On Kimi-K2.5: `fused_moe_kernel_gptq_awq` takes 53% of GPU time
