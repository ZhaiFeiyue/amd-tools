---
name: vllm-amd-profile-test
description: >-
  Profile and benchmark vLLM serving Kimi K2.5 (1T MoE) on AMD MI355X/MI325X ROCm GPUs.
  Covers PyTorch profiler traces for prefill/decode phases, RPD low-overhead profiling,
  serving benchmarks at various concurrencies, and trace analysis with Perfetto.
  Use when profiling vLLM inference on AMD GPUs, analyzing kernel-level performance,
  or comparing prefill vs decode characteristics for Kimi K2.5.
---

# Profiling vLLM Kimi K2.5 on AMD ROCm GPUs

本 skill 提供在 AMD MI355X/MI325X 上对 vLLM 服务 Kimi K2.5 进行 profiling 的完整流程。
覆盖两种 profiling 方法: PyTorch Profiler (vLLM 内置) 和 RPD (低开销, AMD 原生)。

## 前置条件

- AMD MI355X (gfx950) 或 MI325X (gfx942), 至少 4 卡 (TP=4)
- Docker image: `rocm/vllm-dev:nightly_main_20260318` (或更新的 nightly)
- 模型权重: `/apps/data/models/moonshotai/Kimi-K2.5` (~400GB, 64 safetensors shards)
- 模型加载约 50 分钟 (144.63 GiB)

## AMD ROCm 必须配置

| 环境变量 / 参数 | 值 | 原因 |
|---|---|---|
| `VLLM_ROCM_USE_AITER=0` | **必须** | AITER MLA 不兼容 TP=4 |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | **必须** | ROCm 需要 spawn |
| `RCCL_MSCCL_ENABLE=0` | 推荐 | 避免 RCCL 问题 |
| `VLLM_USE_TRITON_FLASH_ATTN=0` | 推荐 | Vision encoder 需要 |
| `--tensor-parallel-size 4` | **必须** | MLA 64 heads, TP=4 → 16 heads/GPU |
| `--trust-remote-code` | **必须** | KimiK25ForConditionalGeneration |

## 脚本说明

本目录提供以下脚本:

| 脚本 | 用途 |
|------|------|
| `server_prefill_profile.sh` | 启动 prefill profiling server (PyTorch profiler, GPU 0-3, port 8100) |
| `server_decode_profile.sh` | 启动 decode profiling server (PyTorch profiler, GPU 4-7, port 8200) |
| `bench_warmup.sh` | Warmup benchmark (在 profiling 前运行) |
| `bench_profile.sh` | 触发 profiling 的 benchmark 运行 |
| `bench_sweep.sh` | 多并发 benchmark sweep (concurrency 1, 8, 32, 64) |
| `cleanup.sh` | 清理所有容器和临时文件 |

## 工作流

### 方法一: PyTorch Profiler (vLLM 内置, 推荐)

vLLM 内置 `--profiler-config` 支持, 无需修改源码。

#### Step 1: Prefill Profiling

```bash
# 启动 server
./server_prefill_profile.sh

# 等待 ~50 分钟, 看到 "Application startup complete"
docker logs -f vllm_kimi_prefill_prof

# Warmup
./bench_warmup.sh 8100 prefill

# 执行 profiling
./bench_profile.sh 8100 prefill

# Trace 输出: /tmp/vllm_kimi_prefill_profile/
```

#### Step 2: Decode Profiling

```bash
# 清理 prefill 容器
docker rm -f vllm_kimi_prefill_prof

# 启动 decode server
./server_decode_profile.sh

# Warmup + Profile
./bench_warmup.sh 8200 decode
./bench_profile.sh 8200 decode

# Trace 输出: /tmp/vllm_kimi_decode_profile/
```

#### Step 3: 查看 Trace

用 https://ui.perfetto.dev/ 打开 `/tmp/vllm_kimi_*_profile/*.json.gz` 文件。
大 trace 使用 Perfetto streaming mode (支持到 ~8GB)。

### 方法二: RPD Profiler (低开销, GPU kernel 级)

RPD 是 AMD 原生的低开销 profiler, 捕获 HIP kernel 和 RCCL 通信。
需要在容器内安装 RPD 并用 `loadTracer.sh` 包装 vLLM 启动。

详细步骤参见 [PROFILING.md](PROFILING.md)。

## Benchmark Sweep

不带 profiling 的纯性能 benchmark:

```bash
# 先启动一个普通 server (不带 --profiler-config)
docker run -d --name vllm_kimi_bench \
  --network=host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /apps:/apps \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e RCCL_MSCCL_ENABLE=0 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  rocm/vllm-dev:nightly_main_20260318 \
  vllm serve /apps/data/models/moonshotai/Kimi-K2.5 \
    --tensor-parallel-size 4 --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32768 --gpu-memory-utilization 0.85

# 等待就绪后, 跑 sweep
./bench_sweep.sh 8000
```

## Profiling 分析

### Prefill vs Decode 区分

| 信号 | Prefill | Decode |
|------|---------|--------|
| HIP graph | 无 graph | 有 graph (hipGraphLaunch) |
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

## 清理

```bash
./cleanup.sh
```

## 已知问题

1. `--block-size 1` 会报错: TRITON_MLA 不支持, 使用默认值
2. Trace 文件过大: output_len > 256 时 trace 达 GB 级, decode profiling 用 output_len=64
3. FP8 KV cache 不支持: MLA 架构不支持 `--kv-cache-dtype fp8`
4. TP=8 不支持: MLA 64 heads / TP=8 = 8 heads/GPU, 不支持该配置
5. RPD 和 PyTorch Profiler 不要同时使用, 会互相干扰
