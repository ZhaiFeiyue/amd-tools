#!/bin/bash
set -euo pipefail

PORT="${1:-8100}"
MODE="${2:-prefill}"
MODEL="${MODEL_PATH:-/apps/data/models/moonshotai/Kimi-K2.5}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ "$MODE" = "prefill" ]; then
  INPUT_LEN=2048
  OUTPUT_LEN=4
  NUM_PROMPTS=4
  CONCURRENCY=1
  PROFILE_DIR="/tmp/vllm_kimi_prefill_profile"
elif [ "$MODE" = "decode" ]; then
  INPUT_LEN=128
  OUTPUT_LEN=64
  NUM_PROMPTS=4
  CONCURRENCY=1
  PROFILE_DIR="/tmp/vllm_kimi_decode_profile"
else
  echo "Usage: $0 <port> <prefill|decode>"
  exit 1
fi

CONTAINER=$(docker ps --format '{{.Names}}' | grep -E 'vllm_kimi.*(prof|bench)' | head -1)
if [ -z "$CONTAINER" ]; then
  echo "Error: No running vllm_kimi container found"
  exit 1
fi

LOGFILE="vllm_${MODE}_profile_${TIMESTAMP}.log"

echo "=== Profiling ($MODE mode) on port $PORT ==="
echo "Container: $CONTAINER | input=$INPUT_LEN output=$OUTPUT_LEN prompts=$NUM_PROMPTS"
echo "Log: $LOGFILE"

docker exec "$CONTAINER" vllm bench serve \
  --model "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONCURRENCY" \
  --request-rate inf --trust-remote-code \
  --profile 2>&1 | tee "$LOGFILE"

echo "=== Profiling complete ==="
echo "Trace files: $PROFILE_DIR/"
echo "View with: https://ui.perfetto.dev/"
ls -lh "$PROFILE_DIR/" 2>/dev/null || echo "(check inside container)"
