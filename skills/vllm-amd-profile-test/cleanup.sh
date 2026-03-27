#!/bin/bash

echo "Cleaning up vLLM Kimi K2.5 profiling containers..."

docker rm -f vllm_kimi_prefill_prof 2>/dev/null && echo "  Removed vllm_kimi_prefill_prof" || true
docker rm -f vllm_kimi_decode_prof 2>/dev/null && echo "  Removed vllm_kimi_decode_prof" || true
docker rm -f vllm_kimi_bench 2>/dev/null && echo "  Removed vllm_kimi_bench" || true

echo ""
echo "Profile data locations (not deleted):"
echo "  /tmp/vllm_kimi_prefill_profile/"
echo "  /tmp/vllm_kimi_decode_profile/"
echo ""
echo "To also remove profile data: rm -rf /tmp/vllm_kimi_*_profile/"
