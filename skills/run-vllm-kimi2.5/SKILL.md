---
name: run-vllm-kimi25
description: >-
  Deploy, benchmark, and profile Kimi K2.5 (1T MoE) on vLLM with AMD MI355X/MI325X ROCm GPUs.
  Covers functional testing, serving benchmark, PyTorch profiler traces for prefill/decode,
  and disaggregated PD serving (prefill 4 GPU + decode 4 GPU on single node).
  Use when setting up vLLM for Kimi K2.5, running inference benchmarks on AMD GPUs,
  profiling prefill/decode phases, or deploying disaggregated prefill-decode.
---

# 在 AMD ROCm GPU 上运行 vLLM Kimi K2.5

## 前置条件

- AMD MI355X (gfx950) 或 MI325X (gfx942), 至少 4 卡 (TP=4)
- Docker image: `rocm/vllm-dev:nightly_main_20260318` (或更新的 nightly)
- 模型权重: `moonshotai/Kimi-K2.5` (HuggingFace 或本地路径, ~400GB, 64 safetensors shards)
- 模型路径示例: `/apps/data/models/moonshotai/Kimi-K2.5`

## Kimi K2.5 模型要点

- 1T 总参数, 32B 每 token 激活 (MoE 架构, 基于 DeepSeek-V3)
- 61 层 transformer: Layer 0 是 dense (intermediate=18432), Layer 1-60 是 MoE (384 experts, top-8, moe_intermediate=2048, 1 shared expert)
- MLA attention: 64 heads, kv_lora_rank=512, q_lora_rank=1536
- 默认 INT4 group 量化 (compressed-tensors), 只量化 routed expert 权重
- 模型加载约 50 分钟 (144.63 GiB)

## AMD ROCm 关键配置 (已验证, 必须遵守)

| 项目 | 值 | 原因 |
|---|---|---|
| `VLLM_ROCM_USE_AITER=0` | **必须** | AITER MLA 不兼容 TP=4 (64 heads / 4 = 16 heads/GPU, AITER 要求特定 head 数) |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | **必须** | ROCm 平台需要 spawn 而非 fork |
| `RCCL_MSCCL_ENABLE=0` | 推荐 | 避免某些 workload 下 RCCL 问题 |
| `VLLM_USE_TRITON_FLASH_ATTN=0` | 推荐 | Vision encoder 需要 (env var 名在新版本可能有 warning 但不影响) |
| `--tensor-parallel-size 4` | **必须** | MLA 64 heads, TP=4 得 16 heads/GPU (可用); TP=8 得 8 heads/GPU (不支持) |
| `--trust-remote-code` | **必须** | 模型使用自定义代码 (KimiK25ForConditionalGeneration) |
| `--block-size 1` | **禁止使用** | TRITON_MLA 不支持 block_size=1, 会报 ValueError |

## Docker 通用参数模板

所有 `docker run` 命令共用以下 ROCm 参数:

```bash
docker run -d --name <CONTAINER_NAME> \
  --network=host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /apps:/apps -v /tmp:/tmp \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e RCCL_MSCCL_ENABLE=0 \
  -e HIP_VISIBLE_DEVICES=<GPU_IDS> \
  <IMAGE> \
  vllm serve <MODEL_PATH> \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --host 0.0.0.0 --port <PORT> \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85
```

变量说明:
- `<IMAGE>`: `rocm/vllm-dev:nightly_main_20260318`
- `<MODEL_PATH>`: `/apps/data/models/moonshotai/Kimi-K2.5` 或 `moonshotai/Kimi-K2.5`
- `<GPU_IDS>`: `0,1,2,3` (前4卡) 或 `4,5,6,7` (后4卡) 或 `0,1,2,3,4,5,6,7` (全部)

## 工作流

### Step 1: 功能验证

启动 server:
```bash
docker run -d --name vllm_kimi_test \
  --network=host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /apps:/apps \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e RCCL_MSCCL_ENABLE=0 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  rocm/vllm-dev:nightly_main_20260318 \
  vllm serve /apps/data/models/moonshotai/Kimi-K2.5 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85
```

等待就绪 (~50 分钟), 看到 `Application startup complete`:
```bash
docker logs -f vllm_kimi_test
```

验证推理:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/apps/data/models/moonshotai/Kimi-K2.5","messages":[{"role":"user","content":"What is 2+3?"}],"max_tokens":32}'
```

### Step 2: Benchmark

在 server 容器内执行:
```bash
docker exec vllm_kimi_test vllm bench serve \
  --model /apps/data/models/moonshotai/Kimi-K2.5 \
  --host 0.0.0.0 --port 8000 \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --num-prompts 20 --max-concurrency <CONC> \
  --request-rate inf --trust-remote-code
```

按 `<CONC>` = 1, 8, 32, 64 分别执行, 收集 TTFT / ITL / throughput。

### Step 3: Prefill Profiling (GPU 0-3, port 8100)

停掉 Step 1 的容器后启动:
```bash
docker rm -f vllm_kimi_test
mkdir -p /tmp/vllm_kimi_prefill_profile

docker run -d --name vllm_kimi_prefill \
  --network=host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /apps:/apps -v /tmp:/tmp \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e RCCL_MSCCL_ENABLE=0 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_RPC_TIMEOUT=1800000 \
  rocm/vllm-dev:nightly_main_20260318 \
  vllm serve /apps/data/models/moonshotai/Kimi-K2.5 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8100 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/vllm_kimi_prefill_profile","torch_profiler_with_stack":true,"torch_profiler_with_flops":true,"torch_profiler_use_gzip":true}'
```

Server ready 后:
```bash
# Warmup
docker exec vllm_kimi_prefill vllm bench serve \
  --model /apps/data/models/moonshotai/Kimi-K2.5 \
  --host 0.0.0.0 --port 8100 \
  --dataset-name random --random-input-len 2048 --random-output-len 8 \
  --num-prompts 8 --max-concurrency 4 --request-rate inf --trust-remote-code

# Profile (prefill-dominated: 长输入, 短输出)
docker exec vllm_kimi_prefill vllm bench serve \
  --model /apps/data/models/moonshotai/Kimi-K2.5 \
  --host 0.0.0.0 --port 8100 \
  --dataset-name random --random-input-len 2048 --random-output-len 4 \
  --num-prompts 4 --max-concurrency 1 --request-rate inf --trust-remote-code --profile
```

Trace 输出在 `/tmp/vllm_kimi_prefill_profile/`, 用 https://ui.perfetto.dev/ 查看。

### Step 4: Decode Profiling (GPU 4-7, port 8200)

```bash
docker rm -f vllm_kimi_prefill
mkdir -p /tmp/vllm_kimi_decode_profile

docker run -d --name vllm_kimi_decode \
  --network=host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /apps:/apps -v /tmp:/tmp \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e RCCL_MSCCL_ENABLE=0 \
  -e HIP_VISIBLE_DEVICES=4,5,6,7 \
  -e VLLM_RPC_TIMEOUT=1800000 \
  rocm/vllm-dev:nightly_main_20260318 \
  vllm serve /apps/data/models/moonshotai/Kimi-K2.5 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8200 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/vllm_kimi_decode_profile","torch_profiler_with_stack":true,"torch_profiler_with_flops":true,"torch_profiler_use_gzip":true}'
```

Server ready 后:
```bash
# Warmup
docker exec vllm_kimi_decode vllm bench serve \
  --model /apps/data/models/moonshotai/Kimi-K2.5 \
  --host 0.0.0.0 --port 8200 \
  --dataset-name random --random-input-len 128 --random-output-len 64 \
  --num-prompts 8 --max-concurrency 4 --request-rate inf --trust-remote-code

# Profile (decode-dominated: 短输入, 中等输出)
docker exec vllm_kimi_decode vllm bench serve \
  --model /apps/data/models/moonshotai/Kimi-K2.5 \
  --host 0.0.0.0 --port 8200 \
  --dataset-name random --random-input-len 128 --random-output-len 64 \
  --num-prompts 4 --max-concurrency 1 --request-rate inf --trust-remote-code --profile
```

### Step 5: PD 分离部署 (Prefill GPU 0-3 + Decode GPU 4-7)

同时启动两个 vLLM 实例, NixlConnector (ROCm 上用 RIXL) 传输 KV cache:

```bash
docker rm -f vllm_kimi_decode

# Prefill 实例 (GPU 0-3, port 8100)
docker run -d --name vllm_kimi_pd_prefill \
  --network=host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /apps:/apps -v /tmp:/tmp \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e RCCL_MSCCL_ENABLE=0 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
  -e VLLM_RPC_TIMEOUT=1800000 \
  -e UCX_NET_DEVICES=all \
  rocm/vllm-dev:nightly_main_20260318 \
  vllm serve /apps/data/models/moonshotai/Kimi-K2.5 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8100 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}'

# Decode 实例 (GPU 4-7, port 8200)
docker run -d --name vllm_kimi_pd_decode \
  --network=host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /apps:/apps -v /tmp:/tmp \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e RCCL_MSCCL_ENABLE=0 \
  -e HIP_VISIBLE_DEVICES=4,5,6,7 \
  -e VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
  -e VLLM_RPC_TIMEOUT=1800000 \
  -e UCX_NET_DEVICES=all \
  rocm/vllm-dev:nightly_main_20260318 \
  vllm serve /apps/data/models/moonshotai/Kimi-K2.5 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8200 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}'
```

PD 的 NixlConnector 需要注意:
- 两个实例必须使用不同的 `VLLM_NIXL_SIDE_CHANNEL_PORT` (5600 / 5601)
- `UCX_NET_DEVICES=all` 确保 UCX 传输发现所有网卡
- `kv_role` 设为 `kv_both` (角色由上层 proxy 决定)

### Step 6: 清理

```bash
docker rm -f vllm_kimi_test vllm_kimi_prefill vllm_kimi_decode
docker rm -f vllm_kimi_pd_prefill vllm_kimi_pd_decode
```

## Profiling 分析方法

### 区分 Prefill vs Decode Step

| 信号 | Prefill | Decode |
|------|---------|--------|
| HIP graph | 无 graph (或 piecewise) | 有 graph (hipGraphLaunch) |
| Step 时长 | 长 (与 input_len 成正比) | 短 (1 token/step) |
| GEMM M 维度 | 大 (= input_len) | 小 (= batch_size) |
| Attention kernel | flash_attn varlen / ck_tile | _fwd_grouped_kernel_stage1/2 |

### Kimi K2.5 每个 Step 的结构

```
embedding (token_id -> vector)
  Layer 0 (dense):  self_attn -> dense_mlp(gate_up+down) -> layer_norm
  Layer 1-60 (MoE): self_attn -> gatherTopK(routing) -> fused_moe_kernel_gptq_awq -> shared_expert -> layer_norm
                     allreduce (TP 通信, 在 attn 和 MoE 之后各一次)
lm_head (hidden -> logits)
```

### 已验证的 Kernel 分布 (TP4, MI355X)

| Kernel | % GPU Time | 类别 |
|--------|-----------|------|
| fused_moe_kernel_gptq_awq | 53.25% | MoE expert GEMM (INT4) |
| _fwd_grouped_kernel_stage1 | 13.31% | MLA decode attention |
| gatherTopK | 3.74% | MoE routing |
| cross_device_reduce | 3.50% | TP allreduce |
| wvSplitK | 3.31% | GEMM |
| flash_attn varlen | 3.21% | MLA prefill attention |

## 已知问题

1. **`--block-size 1` 会报错**: TRITON_MLA 不支持, 使用默认值即可
2. **DecodeBenchConnector 挂起**: `kv_connector=DecodeBenchConnector` 在当前 build 导致 engine hang, 不要使用
3. **Trace 文件过大**: output_len > 256 时 trace 可达 GB 级, 建议 decode profiling 用 output_len=64
4. **FP8 KV cache 不支持**: MLA 架构不支持 `--kv-cache-dtype fp8`, 不要使用
5. **TP=8 不支持**: MLA 64 heads / TP=8 = 8 heads/GPU, AITER 和 TRITON_MLA 都不支持该配置
