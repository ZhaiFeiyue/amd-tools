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

GPUS=8
TP=${GPUS}
DP=${GPUS}
EP=${GPUS}
NNODES=1
NRANK=0
DIST=localhost
DIST_PORT=30001

while getopts "n:r:h:" opt; do
       case "${opt}" in
       n)
              NNODES=${OPTARG}
              ;;
       r)
              NRANK=${OPTARG}
              ;;
       h)
              DIST=${OPTARG}:${DIST_PORT}
              ;;
       :)
              echo "missing -${opt}"
              exit 1
              ;;
       ?)
              echo "invalid opt"
              exit 2
    esac
done


python3 -m sglang.launch_server  --disable-cuda-graph --model-path /apps/data/models/DSV3/ --tp ${TP} --dist-init-addr ${DIST} --nnodes ${NNODES} --node-rank ${NRANK} --trust-remote-code --ep-size ${EP} --enable-ep-moe --chunked-prefill-size 130172 --moe-dense-tp-size 1 --enable-dp-attention --dp-size ${DP}
