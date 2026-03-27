#!/bin/bash
set -euo pipefail

IMAGE="${VLLM_IMAGE:-rocm/vllm-dev:nightly_main_20260318}"
MODEL="${MODEL_PATH:-/apps/data/models/moonshotai/Kimi-K2.5}"
CONTAINER="vllm_kimi_prefill_prof"
PORT=8100
GPUS="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
PROFILE_DIR="/tmp/vllm_kimi_prefill_profile"

mkdir -p "$PROFILE_DIR"

docker rm -f "$CONTAINER" 2>/dev/null || true

echo "Starting vLLM prefill profiling server on GPU [$GPUS], port $PORT"
echo "Profile output: $PROFILE_DIR"

docker run -d --name "$CONTAINER" \
  --network=host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /apps:/apps -v /tmp:/tmp \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e RCCL_MSCCL_ENABLE=0 \
  -e VLLM_USE_TRITON_FLASH_ATTN=0 \
  -e HIP_VISIBLE_DEVICES="$GPUS" \
  -e VLLM_RPC_TIMEOUT=1800000 \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_with_stack\":true,\"torch_profiler_with_flops\":true,\"torch_profiler_use_gzip\":true}"

echo "Container '$CONTAINER' started."
echo "Monitor with: docker logs -f $CONTAINER"
echo "Wait for 'Application startup complete' (~50 min)"
