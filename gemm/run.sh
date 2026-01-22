#!/bin/bash
set -e

XCDS_PER_GPU=8
GPUS=8
max_exponent=6

for ((n=4; n<=max_exponent; n++)); do
    result=$(echo "scale=0; 2^$n" | bc)
    DEVICES=`python gen.py ${GPUS} ${XCDS_PER_GPU} ${n}`
    echo ${DEVICES}
    #CUDA_VISIBLE_DEVICES=${DEVICES} HSA_ENABLE_SDMA=0 python torch_gemm_multi.py 2>&1 | tee tmp.log
    #tops=`cat tmp.log | grep "TOPS" | awk -F'\ ' '{print $2}'`
    #echo "${result} = ${tops}" >> results_xcd${XCDS_PER_GPU}.log
done




