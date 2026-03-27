# vLLM Parallelism Deep Dive

## Tensor Parallelism (TP)

Splits model weight tensors across GPUs within a single node. Uses Megatron-LM's tensor parallel algorithm.

- Each GPU holds a shard of every layer's weight matrices
- Requires all-reduce communication after each layer
- Best for single-node multi-GPU setups

```bash
# 4-GPU tensor parallelism
vllm serve meta-llama/Llama-3-70B --tensor-parallel-size 4
```

**Constraints**: `tensor_parallel_size` must evenly divide the number of attention heads. For MLA models (like Kimi-K2.5 with 64 heads), TP=4 gives 16 heads/GPU (works), TP=8 gives 8 heads/GPU (may not work with some attention backends).

**Code path**: `vllm/distributed/` handles process groups, `vllm/model_executor/layers/` applies column/row parallel linear layers.

## Pipeline Parallelism (PP)

Splits model layers across nodes. Each node processes a subset of transformer layers.

```bash
# 2-node, 8 GPUs per node
vllm serve <model> --tensor-parallel-size 8 --pipeline-parallel-size 2
```

**Multi-node setup** requires Ray cluster:
```bash
# Head node
ray start --head
# Worker nodes
ray start --address=<head-ip>:6379
# Then launch vllm on head node with PP config
```

**Code path**: `vllm/v1/executor/` manages pipeline stage assignment, `vllm/distributed/` handles inter-node P2P communication.

## Data Parallelism (DP)

Replicates the model (or attention weights in EP mode) across multiple GPU groups. Each replica handles different requests independently.

```bash
vllm serve <model> --data-parallel-size 2 --tensor-parallel-size 4
# Total GPUs = DP × TP = 8
```

When combined with EP: attention weights are replicated across DP ranks, while expert weights are distributed.

**Code path**: Multiple vLLM engine instances, each handling a subset of incoming requests.

## Expert Parallelism (EP)

For Mixture-of-Experts (MoE) models. Distributes expert FFN blocks across GPUs.

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel
# EP_SIZE = TP × DP = 8
```

**Expert Parallel Load Balancing (EPLB)**:
```bash
# Advanced config via EPLBConfig
--eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":0,"policy":"default"}'
```

**Elastic EP**: Dynamic expert scaling at runtime, useful for varying load patterns.

**Code path**: `vllm/model_executor/layers/fused_moe/` implements fused MoE kernels, `vllm/distributed/eplb/` handles expert load balancing, `vllm/distributed/elastic_ep/` supports elastic scaling.

## Context Parallelism (CP)

Addresses long-context scenarios by distributing context processing across GPUs.

### Prefill Context Parallel (PCP)

Splits long prefill requests across N GPUs:
- **Strategy 1**: Partial query with full K/V — for moderate lengths
- **Strategy 2**: Partial Q/K/V with ring-attention — for very long sequences

Reduces prefill latency for long-context workloads.

### Decode Context Parallel (DCP)

Shards KV cache across GPUs along the sequence dimension during decode:
- Reduces per-GPU KV cache memory
- Size bounded by `[1, tp_size / num_kv_heads]`
- Larger DCP = less memory per GPU, more communication

**Code path**: Context parallel implementations in attention backends, KV cache management in `vllm/v1/core/`.

## Sequence Parallelism (SP)

Parallelizes computation along the sequence dimension. Complementary to TP — while TP parallelizes weight computation, SP parallelizes activation computation (LayerNorm, Dropout) along the sequence axis.

**Code path**: `vllm/compilation/sequence_parallelism/`

## Prefill-Decode Disaggregation (PD Separation)

Not a traditional parallelism but a deployment pattern. Runs separate vLLM instances for prefill and decode, connected via KV cache transfer.

```bash
# Prefill instance
vllm serve <model> --tensor-parallel-size 4 --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

# Decode instance
vllm serve <model> --tensor-parallel-size 4 --port 8200 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

KV connectors: `NixlConnector` (NVIDIA/AMD), `P2PConnector`, `MooncakeConnector`

## Parallelism Combination Matrix

| Scenario | TP | PP | DP | EP | Notes |
|----------|----|----|----|----|-------|
| Single GPU | 1 | 1 | 1 | - | Simplest setup |
| Single node, dense model | N | 1 | 1 | - | N = num GPUs |
| Multi-node, dense model | N | M | 1 | - | N GPUs/node, M nodes |
| MoE, single node | 1 | 1 | N | Yes | EP_SIZE = N |
| MoE, multi-node | T | 1 | D | Yes | EP = T × D |
| High throughput | T | 1 | D | - | D replicas of TP=T model |
| Long context | T | 1 | 1 | - | + CP/PCP/DCP |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` (required for ROCm) or `fork` |
| `VLLM_ROCM_USE_AITER` | Set `0` if AITER incompatible with TP config |
| `VLLM_USE_TRITON_FLASH_ATTN` | Set `0` for vision encoder compatibility |
| `RCCL_MSCCL_ENABLE` | Set `0` to avoid RCCL issues on some workloads |
| `VLLM_RPC_TIMEOUT` | RPC timeout in ms (increase for large models) |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | Side channel port for NixlConnector |
| `UCX_NET_DEVICES` | Network devices for UCX transport |
| `HIP_VISIBLE_DEVICES` | Visible AMD GPUs (like CUDA_VISIBLE_DEVICES) |
| `CUDA_VISIBLE_DEVICES` | Visible NVIDIA GPUs |
