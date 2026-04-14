#!/bin/bash
###############################################################################
# SGLang PD Disaggregation - Best Performance Config
#
# 1P1D, DeepSeek-R1-0528-MXFP4-th
# Prefill: DP4 TP4 EP4 (8 GPU)
# Decode:  DP8 TP8 EP8 (8 GPU), BS=160
#
# Optimizations applied:
#   - decoded_texts redundancy elimination (+36% TPS on detokenizer bottleneck)
#   - pickle protocol >= 5 for IPC
#   - SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=128
#   - millisecond logging timestamps
#
# Usage:
#   ./run_best_config.sh <PREFILL_NODE> <DECODE_NODE> [NUM_PROMPTS] [CONCURRENCY]
#
# Example:
#   ./run_best_config.sh smci355-ccs-aus-n08-29 smci355-ccs-aus-n08-33 10240 2048
###############################################################################
set -uo pipefail

PREFILL_NODE="${1:?Usage: $0 <PREFILL_NODE> <DECODE_NODE> [NUM_PROMPTS] [CONCURRENCY]}"
DECODE_NODE="${2:?Usage: $0 <PREFILL_NODE> <DECODE_NODE> [NUM_PROMPTS] [CONCURRENCY]}"
NUM_PROMPTS="${3:-10240}"
CONCURRENCY="${4:-2048}"

IMAGE="rocm/pytorch-private:sglang-0.5.9-rocm720-mi35x-mori-0327"
MODEL="/apps/data/models/DeepSeek-R1-0528-MXFP4-th"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${SCRIPT_DIR}/best_config_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT}"

echo "============================================================"
echo "  SGLang PD Best Config"
echo "  Prefill: ${PREFILL_NODE}"
echo "  Decode:  ${DECODE_NODE}"
echo "  Prompts: ${NUM_PROMPTS}, Concurrency: ${CONCURRENCY}"
echo "  Output:  ${OUT}"
echo "  $(date -u)"
echo "============================================================"

###############################################################################
# Step 1: Fresh containers
###############################################################################
echo "[1/7] Creating fresh containers..."
for NODE_CONTAINER in "${PREFILL_NODE}:pd_spike_prefill" "${DECODE_NODE}:pd_spike_decode"; do
    NODE="${NODE_CONTAINER%%:*}"
    CONTAINER="${NODE_CONTAINER##*:}"
    ssh ${NODE} "docker rm -f ${CONTAINER} 2>/dev/null; \
        docker run -d --name ${CONTAINER} \
        --shm-size=128g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network=host --group-add=video --ipc=host \
        --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband \
        -v /apps:/apps \
        ${IMAGE} bash -c 'sleep infinity'" 2>/dev/null
done
sleep 2

###############################################################################
# Step 2: Apply patches
###############################################################################
echo "[2/7] Applying patches (dt_fix + pickle5 + ms_logging + prealloc)..."

for NODE_CONTAINER in "${PREFILL_NODE}:pd_spike_prefill" "${DECODE_NODE}:pd_spike_decode"; do
    NODE="${NODE_CONTAINER%%:*}"
    CONTAINER="${NODE_CONTAINER##*:}"
    for PATCH in decoded_texts_patch.py msgpack_ipc_patch.py ms_logging_patch.py; do
        scp ${SCRIPT_DIR}/${PATCH} ${NODE}:/tmp/ 2>/dev/null
        ssh ${NODE} "docker cp /tmp/${PATCH} ${CONTAINER}:/tmp/ && \
            docker exec ${CONTAINER} python3 /tmp/${PATCH}" 2>/dev/null
    done
done

# Prealloc PR on decode
ssh ${DECODE_NODE} "docker exec pd_spike_decode python3 -c '
path = \"/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py\"
with open(path) as f: code = f.read()
old = \"pre_alloc_size = max_num_reqs * 2 if max_num_reqs <= 32 else 0\"
new = \"\"\"pre_alloc_size = int(__import__(\"os\").environ.get(\"SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS\", \"0\"))
                pre_alloc_size = (max_num_reqs * 2 if max_num_reqs <= 32 else pre_alloc_size)\"\"\"
code = code.replace(old, new)
with open(path, \"w\") as f: f.write(code)
print(\"Prealloc patched!\")
'" 2>/dev/null

###############################################################################
# Step 3: Launch servers
###############################################################################
echo "[3/7] Launching servers..."

# Common env vars
ENV_COMMON='PYTHONPATH=/sgl-workspace/aiter:${PYTHONPATH} NCCL_SOCKET_IFNAME=enp81s0f1 NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 GLOO_SOCKET_IFNAME=enp81s0f1 SGLANG_USE_AITER=1 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600 MORI_SHMEM_MODE=ISOLATION SGLANG_MORI_FP8_DISP=False SGLANG_MORI_FP4_DISP=True SGLANG_MORI_FP8_COMB=True MORI_EP_LAUNCH_CONFIG_MODE=MANUAL MORI_IO_QP_MAX_SEND_WR=16384 MORI_IO_QP_MAX_CQE=32768 MORI_IO_QP_MAX_SGE=4 MORI_MAX_DISPATCH_TOKENS_PREFILL=12288 MORI_MAX_DISPATCH_TOKENS_DECODE=160 SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=320 MORI_APP_LOG_LEVEL=INFO MORI_RDMA_SL=3 MORI_RDMA_TC=96 SGLANG_MORI_NEXTN_FP4_DISP=False SGLANG_ENABLE_SPEC_V2=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1'

# Prefill
ssh ${PREFILL_NODE} "docker exec -d pd_spike_prefill bash -c '${ENV_COMMON} SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=12288 python3 -m sglang.launch_server --model-path ${MODEL} --disaggregation-mode prefill --disaggregation-transfer-backend mori --disaggregation-ib-device ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 --host 0.0.0.0 --port 8000 --trust-remote-code --tp-size 4 --ep-size 4 --dp-size 4 --decode-log-interval 1 --watchdog-timeout 3600 --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter --moe-a2a-backend mori --deepep-mode normal --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head --mem-fraction-static 0.8 --max-running-requests 1280 --chunked-prefill-size 49152 --disable-radix-cache 2>&1 | stdbuf -oL tee ${OUT}/prefill.log'"

# Decode
ssh ${DECODE_NODE} "docker exec -d pd_spike_decode bash -c '${ENV_COMMON} SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=160 SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=128 python3 -m sglang.launch_server --model-path ${MODEL} --disaggregation-mode decode --disaggregation-transfer-backend mori --disaggregation-ib-device ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 --host 0.0.0.0 --port 8000 --trust-remote-code --tp-size 8 --ep-size 8 --dp-size 8 --decode-log-interval 1 --watchdog-timeout 3600 --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter --moe-a2a-backend mori --deepep-mode normal --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head --mem-fraction-static 0.85 --max-running-requests 1280 --context-length 10240 --cuda-graph-bs $(seq 1 160 | tr "\n" " ") --prefill-round-robin-balance 2>&1 | stdbuf -oL tee ${OUT}/decode.log'"

###############################################################################
# Step 4: Wait for servers
###############################################################################
echo "[4/7] Waiting for servers..."
for i in $(seq 1 120); do
    sleep 10
    PH=$(ssh ${PREFILL_NODE} "docker exec pd_spike_prefill curl -s -o /dev/null -w '%{http_code}' http://0.0.0.0:8000/health 2>/dev/null" 2>/dev/null || echo "000")
    DH=$(ssh ${DECODE_NODE} "docker exec pd_spike_decode curl -s -o /dev/null -w '%{http_code}' http://0.0.0.0:8000/health 2>/dev/null" 2>/dev/null || echo "000")
    [ $((i % 6)) -eq 0 ] && echo "  [$((i*10))s] prefill=${PH} decode=${DH}"
    if [ "${PH}" = "200" ] && [ "${DH}" = "200" ]; then
        echo "  Both servers ready at $((i*10))s"
        break
    fi
done

###############################################################################
# Step 5: Router
###############################################################################
echo "[5/7] Starting router..."
PREFILL_IP=$(ssh ${PREFILL_NODE} "docker exec pd_spike_prefill hostname -I | tr ' ' '\n' | grep '^10\.235\.192' | head -1")
DECODE_IP=$(ssh ${DECODE_NODE} "docker exec pd_spike_decode hostname -I | tr ' ' '\n' | grep '^10\.235\.192' | head -1")

ssh ${PREFILL_NODE} "docker exec -d pd_spike_prefill bash -c 'python3 -m sglang_router.launch_router \
  --pd-disaggregation --port 30000 \
  --policy random --prefill-policy round_robin --decode-policy round_robin \
  --prefill http://${PREFILL_IP}:8000 --decode http://${DECODE_IP}:8000 \
  2>&1 | stdbuf -oL tee ${OUT}/router.log'"
sleep 3

# Smoke test
SMOKE=$(ssh ${PREFILL_NODE} "docker exec pd_spike_prefill curl -s --max-time 120 \
  http://0.0.0.0:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\":\"${MODEL}\",\"prompt\":\"Hello\",\"max_tokens\":3,\"temperature\":0}'" 2>&1)
echo "  Smoke: $(echo ${SMOKE} | head -c 60)"

###############################################################################
# Step 6: Benchmark
###############################################################################
echo "[6/7] Setting up benchmark..."
ssh ${PREFILL_NODE} "docker exec pd_spike_prefill bash -c '\
  cd /tmp && rm -rf bench_serving && \
  git clone -q https://github.com/ZhaiFeiyue/bench_serving.git && \
  cd bench_serving && git checkout -q fix_tokenizer && echo OK'" 2>/dev/null

# Warmup
echo "  Warmup (1024 prompts)..."
ssh ${PREFILL_NODE} "docker exec pd_spike_prefill bash -c '\
  cd /tmp/bench_serving && python3 benchmark_serving.py \
  --model ${MODEL} --backend openai --base-url http://0.0.0.0:30000 \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.8 \
  --num-prompts 1024 --max-concurrency ${CONCURRENCY} --request-rate inf --ignore-eos \
  --num-warmups 0 --disable-tqdm --percentile-metrics ttft,tpot,itl,e2el 2>&1 | tail -3'"
sleep 3

# Real test
echo "  Real test (${NUM_PROMPTS} prompts, concurrency=${CONCURRENCY})..."
ssh ${PREFILL_NODE} "docker exec pd_spike_prefill bash -c '\
  cd /tmp/bench_serving && python3 benchmark_serving.py \
  --model ${MODEL} --backend openai --base-url http://0.0.0.0:30000 \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.8 \
  --num-prompts ${NUM_PROMPTS} --max-concurrency ${CONCURRENCY} --request-rate inf --ignore-eos \
  --num-warmups 0 --disable-tqdm --percentile-metrics ttft,tpot,itl,e2el 2>&1'" > ${OUT}/bench.log

###############################################################################
# Step 7: Results
###############################################################################
echo ""
echo "[7/7] Collecting results..."

# Collect logs
ssh ${DECODE_NODE} "cat ${OUT}/decode.log" > ${OUT}/decode_collected.log 2>/dev/null
ssh ${PREFILL_NODE} "cat ${OUT}/prefill.log" > ${OUT}/prefill_collected.log 2>/dev/null

echo ""
echo "============================================================"
echo "  RESULTS"
echo "============================================================"
grep -E "Successful|Benchmark duration|Output token|Total Token|Request throughput|TPOT|ITL|TTFT|E2EL" ${OUT}/bench.log
echo ""
echo "============================================================"
echo "  Output: ${OUT}"
echo "  $(date -u)"
echo "============================================================"
