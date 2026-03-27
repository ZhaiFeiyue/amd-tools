#!/bin/bash
set -euo pipefail

PORT="${1:-8000}"
MODEL="${MODEL_PATH:-/apps/data/models/moonshotai/Kimi-K2.5}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-20}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

CONTAINER=$(docker ps --format '{{.Names}}' | grep -E 'vllm_kimi' | head -1)
if [ -z "$CONTAINER" ]; then
  echo "Error: No running vllm_kimi container found"
  exit 1
fi

echo "=== Benchmark Sweep ==="
echo "Container: $CONTAINER | Port: $PORT"
echo "input=$INPUT_LEN output=$OUTPUT_LEN prompts=$NUM_PROMPTS"
echo ""

for CONC in 1 8 32 64; do
  LOGFILE="vllm_bench_conc${CONC}_${TIMESTAMP}.log"
  echo "--- Concurrency $CONC ---"

  docker exec "$CONTAINER" vllm bench serve \
    --model "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" \
    --request-rate inf --trust-remote-code 2>&1 | tee "$LOGFILE"

  echo ""
done

echo "=== Sweep complete ==="
