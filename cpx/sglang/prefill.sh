#!/bin/bash
se -ex

HIP_VISIBLE_DEVICES=56,58,60,62 python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-32B \
  --disaggregation-mode prefill \
  --host 0.0.0.0 \
  --port 30002 \
  --tp-size 4 \
  --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 \
  --trust-remote-code 
