#!/bin/bash
set -ex

BIN=`dirname ${0}`
BIN=`cd ${BIN}; pwd`
ROOT=${BIN}

SGL_W_DIR=/sgl-workspace
SGL_DIR=${SGL_W_DIR}/sglang

cd ${SGL_DIR}
git fetch upstream
git checkout upstream/dpeptp_skip -b dpeptp_skip

export RCCL_MSCCL_ENABLE=0
export GPU_FORCE_BLIT_COPY_SIZE=64
export HIP_FORCE_DEV_KERNARG=1
export HSA_NO_SCRATCH_RECLAIM=1
export DEBUG_HIP_BLOCK_SYNC=1024
export CK_MOE=1
export DEBUG_HIP_BLOCK_SYN=1024
export GLOO_SOCKET_IFNAME=ens50f0
export NCCL_SOCKET_IFNAME=ens50f0
export NCCL_ALGO=Tree

TP=8
DP=8
EP=8
NNODE=1
NRANK=0

python3 -m sglang.launch_server  --disable-cuda-graph --model-path /apps/data/models/DSV3/ --tp ${TP} --dist-init-addr 10.235.192.60:30001 --nnodes ${NNODE} --node-rank ${NRANK} --trust-remote-code --ep-size ${EP} --enable-ep-moe --chunked-prefill-size 130172 --moe-dense-tp-size 1 --enable-dp-attention --dp-size ${DP}
