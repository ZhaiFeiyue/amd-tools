---
name: sglang-amd-profile-test
description: >-
  Test SGLang profiling on AMD MI355X GPUs with DSR1-FP4 model.
  Covers TP, TEP, and DEP parallelism configs with CUDA graph enabled,
  PyTorch profiler traces for prefill (EXTEND) and decode phases.
  Use when testing SGLang profile functionality on AMD ROCm GPUs,
  verifying trace output contains device kernels, or running profile smoke tests.
---

# SGLang AMD Profile Test

在 AMD MI355X 上测试 SGLang 的 profiling 功能, 验证 TP / TEP / DEP 三种并行模式下
profile trace 能正确捕获 GPU kernel 事件, 环境不崩溃。

## Prerequisites

- AMD MI355X (gfx950), 8 卡
- Docker image: `rocm/pytorch-private:sglang-0.5.9-rocm700-mi35x-mori-0327`
- 模型: `/apps/data/models/DSR1-FP4`

## Configurable Variables

在新环境中只需修改以下变量:

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NODE` | `smci355-ccs-aus-n09-25` | 目标节点 |
| `IMAGE` | `rocm/pytorch-private:sglang-0.5.9-rocm700-mi35x-mori-0327` | Docker image |
| `MODEL` | `/apps/data/models/DSR1-FP4` | 模型路径 |
| `CONTAINER` | `zhai_profile_test` | 容器名 |
| `PORT` | `30000` | Server 端口 |
| `NIC` | `enp81s0f1` | RDMA 网卡 (DEP 用) |

## Test Matrix

| # | Config | Model | 并行参数 |
|---|--------|-------|----------|
| 1 | **TP** | DSR1-FP4 | `--tp-size 8` |
| 2 | **TEP** | DSR1-FP4 | `--tp-size 8 --ep-size 8 --ep-dispatch-algorithm fake` |
| 3 | **DEP** | DSR1-FP4 | `--tp-size 8 --ep-size 8 --dp-size 8 --enable-dp-attention` |

所有测试共用: `--cuda-graph-bs 1 --trust-remote-code --host 0.0.0.0 --port 30000`

## Parallelism Configs

| Config | 说明 |
|--------|------|
| **TP** (Tensor Parallel) | 纯 TP, 每个 GPU 持有模型的 1/8 |
| **TEP** (Tensor + Expert Parallel) | TP + EP, expert 分布在多卡上 |
| **DEP** (Data + Expert Parallel) | DP + EP + DP Attention, 每个 DP rank 处理不同请求 |

## Step 0: Launch Container

```bash
docker run -d --name ${CONTAINER} \
  --network=host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /apps:/apps -v /tmp:/tmp \
  ${IMAGE} sleep infinity
```

## Step 0.5: Environment Check (必须)

```bash
# 检查 GPU 内存占用 (确保没有其他容器占用)
docker exec ${CONTAINER} rocm-smi --showmeminfo vram

# 检查 aiter 导入 (mori-0327 image 的 editable install 有问题)
docker exec ${CONTAINER} bash -c '
  export PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/aiter/aiter/jit/utils:$PYTHONPATH
  python3 -c "import aiter; print(f\"aiter OK: {len(dir(aiter))} attrs\")"
'
```

如果 GPU VRAM 被大量占用, 先清理其他容器。
如果 aiter 导入失败, 后续所有命令必须带上 PYTHONPATH。

## Step 1: Launch Server (模板)

```bash
docker exec -d ${CONTAINER} bash -c '
  export PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/aiter/aiter/jit/utils:/sgl-workspace/sglang/python:$PYTHONPATH
  export SGLANG_TORCH_PROFILER_DIR=/tmp/sglang_profile_test/${TEST_NAME}

  python3 -m sglang.launch_server \
    --model-path ${MODEL} \
    --tp-size 8 \
    ${PARALLEL_ARGS} \
    --cuda-graph-bs 1 \
    --trust-remote-code \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.80 \
    --chunked-prefill-size 4096
'
```

等待 server ready:
```bash
docker exec ${CONTAINER} bash -c 'while ! curl -s http://localhost:30000/health | grep -q "200\|ok"; do sleep 10; done'
```

## Step 2: Warmup

```bash
docker exec ${CONTAINER} bash -c '
  export PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/aiter/aiter/jit/utils:/sgl-workspace/sglang/python:$PYTHONPATH
  python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --dataset-name random \
    --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 4 --request-rate 1
'
```

## Step 3: Profile

```bash
docker exec ${CONTAINER} bash -c '
  export PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/aiter/aiter/jit/utils:/sgl-workspace/sglang/python:$PYTHONPATH
  python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --dataset-name random \
    --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 1 --request-rate 1 \
    --profile --profile-num-steps 16 --profile-by-stage
'
```

## Step 4: Validate

```bash
# Health check
curl -s http://localhost:30000/health

# Trace 文件检查
ls -lh /tmp/sglang_profile_test/${TEST_NAME}/

# 验证 trace 包含 device kernel 事件
python3 -c "
import json, gzip, glob
files = glob.glob('/tmp/sglang_profile_test/${TEST_NAME}/**/*.trace.json.gz', recursive=True)
print(f'Trace files: {len(files)}')
for f in files[:2]:
    with gzip.open(f) as g:
        data = json.load(g)
    events = data if isinstance(data, list) else data.get('traceEvents', [])
    kernels = [e for e in events if e.get('cat') == 'device']
    print(f'{f}: {len(kernels)} device kernels / {len(events)} total events')
"
```

## Step 5: Cleanup Server

```bash
docker exec ${CONTAINER} pkill -f sglang || true
sleep 5
```

---

## All 3 Tests (完整展开命令)

### Test 1: TP + DSR1-FP4

```bash
# Launch server
docker exec -d ${CONTAINER} bash -c '
  export PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/aiter/aiter/jit/utils:/sgl-workspace/sglang/python:$PYTHONPATH
  export SGLANG_TORCH_PROFILER_DIR=/tmp/sglang_profile_test/tp_dsr1fp4
  mkdir -p $SGLANG_TORCH_PROFILER_DIR

  python3 -m sglang.launch_server \
    --model-path /apps/data/models/DSR1-FP4 \
    --tp-size 8 \
    --cuda-graph-bs 1 \
    --trust-remote-code \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.80 \
    --chunked-prefill-size 4096
'

# Warmup (after server ready)
docker exec ${CONTAINER} bash -c '
  export PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/aiter/aiter/jit/utils:/sgl-workspace/sglang/python:$PYTHONPATH
  python3 -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --dataset-name random --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 4 --request-rate 1
'

# Profile
docker exec ${CONTAINER} bash -c '
  export PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/aiter/aiter/jit/utils:/sgl-workspace/sglang/python:$PYTHONPATH
  python3 -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --dataset-name random --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 1 --request-rate 1 \
    --profile --profile-num-steps 16 --profile-by-stage
'
```

### Test 2: TEP + DSR1-FP4

```bash
docker exec ${CONTAINER} pkill -f sglang || true; sleep 5

docker exec -d ${CONTAINER} bash -c '
  export PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/aiter/aiter/jit/utils:/sgl-workspace/sglang/python:$PYTHONPATH
  export SGLANG_TORCH_PROFILER_DIR=/tmp/sglang_profile_test/tep_dsr1fp4
  mkdir -p $SGLANG_TORCH_PROFILER_DIR

  python3 -m sglang.launch_server \
    --model-path /apps/data/models/DSR1-FP4 \
    --tp-size 8 --ep-size 8 --ep-dispatch-algorithm fake \
    --cuda-graph-bs 1 \
    --trust-remote-code \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.80 \
    --chunked-prefill-size 4096
'

# Warmup (after server ready)
docker exec ${CONTAINER} bash -c '
  export PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/aiter/aiter/jit/utils:/sgl-workspace/sglang/python:$PYTHONPATH
  python3 -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --dataset-name random --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 4 --request-rate 1
'

# Profile
docker exec ${CONTAINER} bash -c '
  export PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/aiter/aiter/jit/utils:/sgl-workspace/sglang/python:$PYTHONPATH
  python3 -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --dataset-name random --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 1 --request-rate 1 \
    --profile --profile-num-steps 16 --profile-by-stage
'
```

### Test 3: DEP + DSR1-FP4

```bash
docker exec ${CONTAINER} pkill -f sglang || true; sleep 5

docker exec -d ${CONTAINER} bash -c '
  export PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/aiter/aiter/jit/utils:/sgl-workspace/sglang/python:$PYTHONPATH
  export SGLANG_TORCH_PROFILER_DIR=/tmp/sglang_profile_test/dep_dsr1fp4
  mkdir -p $SGLANG_TORCH_PROFILER_DIR

  python3 -m sglang.launch_server \
    --model-path /apps/data/models/DSR1-FP4 \
    --tp-size 8 --ep-size 8 --dp-size 8 --enable-dp-attention \
    --cuda-graph-bs 1 \
    --max-running-requests 8 \
    --trust-remote-code \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.80 \
    --chunked-prefill-size 4096
'

# Warmup (after server ready, DEP startup is slow ~855s)
docker exec ${CONTAINER} bash -c '
  export PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/aiter/aiter/jit/utils:/sgl-workspace/sglang/python:$PYTHONPATH
  python3 -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --dataset-name random --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 4 --request-rate 1
'

# Profile
docker exec ${CONTAINER} bash -c '
  export PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/aiter/aiter/jit/utils:/sgl-workspace/sglang/python:$PYTHONPATH
  python3 -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --dataset-name random --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 1 --request-rate 1 \
    --profile --profile-num-steps 16 --profile-by-stage
'
```

## Verified Test Results

| # | Config | Kernel Events | Trace Files | Health | 结果 |
|---|--------|---------------|-------------|--------|------|
| 1 | TP | 22,817 | 16 (8 TP x 2 stages) | 200 | **PASS** |
| 2 | TEP | 26,529 | 16 (8 TP x 2 stages) | 200 | **PASS** |
| 3 | DEP | 31,585 | 3 (TP-6/7 ranks only) | 200 | **PASS** |

Trace 文件格式: `{test_name}-{timestamp}-TP-{rank}[-EP-{rank}][-DP-{rank}]-{EXTEND|DECODE}.trace.json.gz`

## Pass/Fail Criteria

1. Server 启动成功 ("The server is fired up and ready to roll!")
2. Warmup 请求正常完成 (无 timeout/error)
3. Profile 请求完成后生成 `.trace.json.gz` 文件
4. Trace 文件包含 device kernel 事件 (cat="device", 数量 > 0)
5. Health endpoint 返回 200 (server 没崩溃)

## Troubleshooting

### aiter 导入失败

**症状:** `ModuleNotFoundError: No module named 'cpp_extension'` 或 `aiter` import 返回空模块

**原因:** mori-0327 image 的 editable install finder 未正确加载, `aiter` 解析到错误路径

**修复:** 所有命令前设置:
```bash
export PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/aiter/aiter/jit/utils:$PYTHONPATH
```

### bench_serving 找不到 sglang.benchmark.datasets

**症状:** `ModuleNotFoundError: No module named 'sglang.benchmark'`

**修复:** 加上 sglang 源码路径:
```bash
export PYTHONPATH=/sgl-workspace/sglang/python:$PYTHONPATH
```

### GPU OOM

**症状:** `OutOfMemoryError` 或 `mem-fraction-static` 被自动调低

**修复:** 先用 `rocm-smi --showmeminfo vram` 检查是否有其他容器占用 GPU 内存, 清理后重试

### DEP capture_bs=[0] AssertionError

**症状:** `AssertionError: capture_bs=[0]` 在 DEP 模式启动时

**原因:** `--max-running-requests 1` 被 8 个 DP rank 均分后变成 0

**修复:** 设 `--max-running-requests 8` (>= dp-size)

### DEP trace 文件名不同

**注意:** DEP 模式下 trace 文件名包含 DP/EP rank (如 `TP-6-DP-6-EP-6`), 不是从 TP-0 开始。
验证时需要用 glob `*.trace.json.gz` 而非硬编码 `*TP-0*`。

### DEP 启动慢

DEP 模式 server 启动约 855s (vs TP 的 ~60s), 需要耐心等待。

## How SGLang Profiling Works

1. Server 启动时设 `SGLANG_TORCH_PROFILER_DIR` 环境变量, scheduler 初始化 `torch.profiler.profile()`
2. `bench_serving --profile` 发送 `POST /start_profile` 开始采集
3. 执行 benchmark 请求期间采集 GPU kernel 和 (可选) CPU activity
4. `POST /stop_profile` 停止采集, 导出 Chrome Trace JSON (.trace.json.gz) 到指定目录
