#!/bin/bash
set -ex

python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:8000 \
  --dataset-name random \
  --num-prompts 128 \
  --random-input 1024 \
  --random-output 1024