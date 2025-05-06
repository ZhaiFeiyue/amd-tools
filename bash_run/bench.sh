#!/bin/bash
set -ex


BIN=`dirname ${0}`
BIN=`cd ${BIN}; pwd`

EP=16
TP=1
DP=16

BASE=Tree.IL16OL1024_PP${TP}.${DP}.${EP}

mkdir -p ${BASE}

#CON="16 32 64 128 256 512 1024 2048 4096 8192 16384"
CON="16"
COMBINATIONS=("16/1024")
for combo in "${COMBINATIONS[@]}"; do
   IFS="/" read -r isl osl <<< "$combo"
   for con in $CON; do
	
	LOG=${BASE}/${con}
	p_con=$(($con * 2))
       if [ "$p_con" -lt 16 ]; then
           p_con=16
       fi
       echo "[RUNNING] prompts $prompts isl $isl osl $osl con $con"
       python3 -m sglang.bench_serving \
       --backend sglang \
       --host 127.0.0.1 \
       --port 30000 \
       --dataset-name generated-shared-prefix \
       --gsp-system-prompt-len 0 \
       --gsp-question-len $isl \
       --gsp-output-len $osl \
       --gsp-num-groups 1 \
       --gsp-prompts-per-group $p_con\
       --random-range-ratio 1 \
       --max-concurrency $con --output-file ${LOG}.json \
       2>&1 | tee -a ${LOG}.log
    done
done
