#!/bin/bash

set -ex

python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 10 --request-rate 10 --dataset-name random --random-input-len 128 --random-output-len 1024 --random-range-ratio 1 | tee tp8dp2ep16.log
