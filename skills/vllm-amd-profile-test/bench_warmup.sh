#!/bin/bash
set -euo pipefail

PORT="${1:-8100}"
MODE="${2:-prefill}"
MODEL="${MODEL_PATH:-/apps/data/models/moonshotai/Kimi-K2.5}"

if [ "$MODE" = "prefill" ]; then
  INPUT_LEN=2048
  OUTPUT_LEN=8
  NUM_PROMPTS=8
  CONCURRENCY=4
elif [ "$MODE" = "decode" ]; then
  INPUT_LEN=128
  OUTPUT_LEN=64
  NUM_PROMPTS=8
  CONCURRENCY=4
else
  echo "Usage: $0 <port> <prefill|decode>"
  exit 1
fi

CONTAINER=$(docker ps --format '{{.Names}}' | grep -E 'vllm_kimi.*(prof|bench)' | head -1)
if [ -z "$CONTAINER" ]; then
  echo "Error: No running vllm_kimi container found"
  exit 1
fi

echo "=== Warmup ($MODE mode) on port $PORT ==="
echo "Container: $CONTAINER | input=$INPUT_LEN output=$OUTPUT_LEN prompts=$NUM_PROMPTS concurrency=$CONCURRENCY"

docker exec "$CONTAINER" vllm bench serve \
  --model "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONCURRENCY" \
  --request-rate inf --trust-remote-code

echo "=== Warmup complete ==="
