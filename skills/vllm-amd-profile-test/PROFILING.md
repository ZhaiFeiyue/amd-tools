## Profiling vLLM Kimi K2.5 Infer System with AMD GPUs

This document describes profiling techniques, code details, and step-by-step instructions for
profiling vLLM serving Kimi K2.5 (1T MoE model) on AMD Instinct GPUs (MI355X/MI325X).

Two primary methods are covered:
- [PyTorch Profiler](#profiling-with-pytorch-profiler-vllm-builtin) (vLLM built-in, recommended)
- [RPD Profiler](#profiling-with-rpd-profiler-low-overhead-amd-native) (low-overhead, AMD native)

---

### Profiling with PyTorch Profiler (vLLM Built-in)

vLLM natively supports PyTorch Profiler via `--profiler-config` and `--profile` flag in `vllm bench serve`.
No source code patches are needed. This is the recommended method for most use cases.

#### How It Works

1. `--profiler-config` on the server side sets up `torch.profiler.profile()` with specified activities
   (CUDA/HIP kernels, call stacks, FLOPs counting).
2. `--profile` on the `vllm bench serve` client side triggers `/start_profile` and `/stop_profile`
   HTTP endpoints around the benchmark run.
3. Trace files (Chrome Trace format, `.json.gz`) are written to the configured directory.

#### profiler-config JSON Schema

```json
{
  "profiler": "torch",
  "torch_profiler_dir": "/tmp/vllm_profile",
  "torch_profiler_with_stack": true,
  "torch_profiler_with_flops": true,
  "torch_profiler_use_gzip": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `profiler` | string | Must be `"torch"` |
| `torch_profiler_dir` | string | Output directory for trace files |
| `torch_profiler_with_stack` | bool | Record Python call stacks |
| `torch_profiler_with_flops` | bool | Estimate FLOPs for matrix operations |
| `torch_profiler_use_gzip` | bool | Gzip compress trace files |

#### Step-by-Step: Prefill Profiling (PyTorch)

1. Start the profiling-enabled server on GPU 0-3:

```bash
mkdir -p /tmp/vllm_kimi_prefill_profile

docker run -d --name vllm_kimi_prefill_prof \
  --network=host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /apps:/apps -v /tmp:/tmp \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e RCCL_MSCCL_ENABLE=0 \
  -e VLLM_USE_TRITON_FLASH_ATTN=0 \
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

2. Wait for server ready (~50 minutes for Kimi K2.5 model loading):

```bash
docker logs -f vllm_kimi_prefill_prof
# Wait for: "Application startup complete"
```

3. Warmup the server (important for HIP graph compilation):

```bash
docker exec vllm_kimi_prefill_prof vllm bench serve \
  --model /apps/data/models/moonshotai/Kimi-K2.5 \
  --host 0.0.0.0 --port 8100 \
  --dataset-name random --random-input-len 2048 --random-output-len 8 \
  --num-prompts 8 --max-concurrency 4 --request-rate inf --trust-remote-code
```

4. Run profiling with `--profile` flag (prefill-dominated: long input, short output):

```bash
docker exec vllm_kimi_prefill_prof vllm bench serve \
  --model /apps/data/models/moonshotai/Kimi-K2.5 \
  --host 0.0.0.0 --port 8100 \
  --dataset-name random --random-input-len 2048 --random-output-len 4 \
  --num-prompts 4 --max-concurrency 1 --request-rate inf --trust-remote-code \
  --profile
```

5. Trace output in `/tmp/vllm_kimi_prefill_profile/`.

#### Step-by-Step: Decode Profiling (PyTorch)

Same procedure but with short input, medium output, on GPU 4-7:

```bash
mkdir -p /tmp/vllm_kimi_decode_profile

docker run -d --name vllm_kimi_decode_prof \
  --network=host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /apps:/apps -v /tmp:/tmp \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e RCCL_MSCCL_ENABLE=0 \
  -e VLLM_USE_TRITON_FLASH_ATTN=0 \
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

Warmup:
```bash
docker exec vllm_kimi_decode_prof vllm bench serve \
  --model /apps/data/models/moonshotai/Kimi-K2.5 \
  --host 0.0.0.0 --port 8200 \
  --dataset-name random --random-input-len 128 --random-output-len 64 \
  --num-prompts 8 --max-concurrency 4 --request-rate inf --trust-remote-code
```

Profile:
```bash
docker exec vllm_kimi_decode_prof vllm bench serve \
  --model /apps/data/models/moonshotai/Kimi-K2.5 \
  --host 0.0.0.0 --port 8200 \
  --dataset-name random --random-input-len 128 --random-output-len 64 \
  --num-prompts 4 --max-concurrency 1 --request-rate inf --trust-remote-code \
  --profile
```

#### Common Notes (PyTorch Profiler)

- **Warmup is critical**: The first few requests trigger HIP graph compilation. Always warmup before profiling.
- **Concurrency 1**: Use `--max-concurrency 1` during profiling to get clean, non-overlapping traces.
- **Trace size**: Keep `--random-output-len` small for prefill profiling (4-8), medium for decode (64).
  Output > 256 can produce multi-GB traces.
- **Gzip**: Always use `torch_profiler_use_gzip: true` to compress traces.

---

### Profiling with RPD Profiler (Low-Overhead, AMD Native)

[RPD (ROCm Profile Data)](https://github.com/ROCm/rocmProfileData) is a low-overhead profiler
that captures HIP kernel launches, RCCL communication, and optionally Python activities.
It stores data in SQLite format (`.rpd`) and converts to Chrome Trace JSON for Perfetto.

#### When to Use RPD vs PyTorch Profiler

| Feature | PyTorch Profiler | RPD |
|---------|-----------------|-----|
| Setup complexity | None (built-in) | Install + loadTracer.sh |
| Overhead | Medium | Low |
| GPU kernel capture | Yes (via CUDA/HIP activity) | Yes (via roctracer) |
| Python call stack | Yes (with_stack=true) | Yes (setPythonTrace) |
| FLOP estimation | Yes (with_flops=true) | No |
| Multi-rank control | All ranks | Selectable (e.g., rank 0,1 only) |
| Output format | Chrome Trace JSON | SQLite → Chrome Trace JSON |
| Trace size control | Limited | Better (fewer ranks logged) |

#### Step 1: Install RPD Inside Container

```bash
docker exec -it vllm_kimi_prefill_prof bash

apt update && apt install -y sqlite3 libsqlite3-dev libfmt-dev

git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData
cd rocmProfileData
make && make install
cd rocpd_python && python setup.py install && cd ..
cd rpd_tracer && make clean && make install && python setup.py install && cd ..
```

#### Step 2: loadTracer.sh

Copy the included `loadTracer.sh` script to the container. This script:
- Creates the RPD SQLite database
- Sets `LD_PRELOAD` to inject the RPD tracer library
- Wraps the actual vLLM process

```bash
#!/bin/bash
OUTPUT_FILE="trace.rpd"

if [ "$1" = "-o" ] ; then
  OUTPUT_FILE=$2
  shift
  shift
fi

if [ -e ${OUTPUT_FILE} ] ; then
  rm ${OUTPUT_FILE}
fi

python3 -m rocpd.schema --create ${OUTPUT_FILE}
if [ $? != 0 ] ; then
  echo "Error: Could not create rpd file."
  exit
fi

export RPDT_FILENAME=${OUTPUT_FILE}
export RPDT_AUTOSTART=0
LD_PRELOAD=librocm-smi_64:librpd_tracer.so "$@"
```

#### Step 3: Launch vLLM with RPD

Instead of using the normal docker run command, enter the container interactively and launch
vLLM wrapped with `loadTracer.sh`:

```bash
docker run -it --name vllm_kimi_rpd \
  --network=host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /apps:/apps -v /tmp:/tmp \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e RCCL_MSCCL_ENABLE=0 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_RPC_TIMEOUT=1800000 \
  rocm/vllm-dev:nightly_main_20260318 bash
```

Inside the container:
```bash
# Install RPD (see Step 1)
# Copy loadTracer.sh

./loadTracer.sh vllm serve /apps/data/models/moonshotai/Kimi-K2.5 \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8100 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85
```

#### Step 4: Trigger Profiling via HTTP

From another terminal (or `docker exec`):

```bash
# Start profiling
curl http://localhost:8100/start_profile -H "Content-Type: application/json"

# Run benchmark
docker exec vllm_kimi_rpd vllm bench serve \
  --model /apps/data/models/moonshotai/Kimi-K2.5 \
  --host 0.0.0.0 --port 8100 \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --num-prompts 20 --max-concurrency 8 --request-rate inf --trust-remote-code

# Stop profiling
curl http://localhost:8100/stop_profile -H "Content-Type: application/json"
```

#### Step 5: Convert RPD to Chrome Trace

```bash
# Inside container, after stopping the server:
sqlite3 trace.rpd ".mode csv" ".header on" ".output trace.csv" "select * from top;" ".output stdout"
python3 ./rocmProfileData/tools/rpd2tracing.py trace.rpd trace.json
```

#### Common Notes (RPD)

- Do NOT use RPD and PyTorch Profiler simultaneously — they interfere with each other.
- For multi-rank profiling, RPD allows logging only specific ranks (e.g., rank 0 and 1)
  to keep trace size manageable. Even Perfetto streaming mode tops out at ~8GB JSON.
- `RPDT_AUTOSTART=0` means profiling starts paused; use `/start_profile` to begin.

---

### Trace Visualization

1. Open https://ui.perfetto.dev/
2. For small traces (< 2GB): drag-and-drop the `.json.gz` file
3. For large traces (2-8GB): use [Perfetto streaming mode](https://perfetto.dev/docs/visualization/large-traces):
   - Click "Open trace file" → select file
   - Enable "Use streaming mode" if prompted
4. Navigate the timeline:
   - `W/S` = zoom in/out
   - `A/D` = pan left/right
   - Click a kernel slice to see duration, call stack, arguments

### Key Metrics to Look For

| Metric | Where to Find | What It Tells You |
|--------|---------------|-------------------|
| `fused_moe_kernel_gptq_awq` duration | Kernel slices | MoE expert GEMM (53% of GPU time) |
| `_fwd_grouped_kernel_stage1/2` | Kernel slices | MLA decode attention (13%) |
| `cross_device_reduce` | Kernel slices | TP allreduce overhead (3.5%) |
| Gap between kernels | Timeline view | CPU-side scheduling overhead |
| `hipGraphLaunch` presence | Kernel names | Decode steps use HIP graph |
| `flash_attn_varlen_*` | Kernel names | Prefill attention |

### Comparing Prefill vs Decode Traces

Open both `/tmp/vllm_kimi_prefill_profile/` and `/tmp/vllm_kimi_decode_profile/` traces
side by side in separate Perfetto tabs:

| Aspect | Prefill Trace | Decode Trace |
|--------|---------------|--------------|
| Step duration | Long (~10-100ms per step, scales with input_len) | Short (~2-5ms per step) |
| Graph capture | No HIP graph (dynamic shapes) | HIP graph (hipGraphLaunch wraps all kernels) |
| Attention | `flash_attn_varlen_fwd` / CK tile kernels | `_fwd_grouped_kernel_stage1/2` |
| GEMM M dimension | Large (= input_len, e.g., 2048) | Small (= batch_size, e.g., 1-4) |
| MoE kernel | Larger M dimension, potentially different tiling | Small M, repeated per decode step |
| allreduce | Larger payload (more data per step) | Smaller payload, more frequent |

---

### Advanced: rocprofv3 (Hardware Counters)

For hardware-level analysis (cache hit rates, memory bandwidth, occupancy), use `rocprofv3`:

```bash
# Inside container with rocprofv3 available:
rocprofv3 -i pmc_input.txt -d ./rocprof_output -- \
  python3 -c "import torch; ..."  # or wrap specific kernel benchmark
```

See `/apps/feiyue/cache-bw-profiler/` for MI355X cache bandwidth profiling scripts.

---

### References

- [vLLM Profiling Docs](https://docs.vllm.ai/en/latest/)
- [RPD (ROCm Profile Data)](https://github.com/ROCm/rocmProfileData)
- [Perfetto Trace Viewer](https://ui.perfetto.dev/)
- [Perfetto Large Traces](https://perfetto.dev/docs/visualization/large-traces)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/)
