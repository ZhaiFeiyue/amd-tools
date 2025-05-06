#!/bin/bash
set -ex

BIN=`dirname ${0}`
BIN=`cd ${BIN}; pwd`
ROOT=${BIN}

DOCKER_IMAGE=sabreshao/sglang_bcm:042304
NAME=bench_0
NNODES=1
NRANK=0
DIST=localhost
while getopts "d:n:m:h:r:" opt; do
       case "${opt}" in
       d)
              DOCKER_IMAGE=${OPTARG}
              ;;
       n)
              NAME=${OPTARG}
              ;;
       m)
              NNODES=${OPTARG}
              ;;
       h)
              DIST=${OPTARG}
              ;;
       r)
              NRANK=${OPTARG}
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

sudo docker run --name ${NAME} --rm --network=host --group-add=video  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device /dev/kfd  --device /dev/dri -v /apps:/apps -v ${BIN}:/bench --privileged --shm-size=128G --ulimit memlock=-1 --ulimit stack=67108864 ${DOCKER_IMAGE} /bench/launch.sh -n ${NNODES} -r ${NRANK} -h ${DIST} 2>&1 | tee ${NAME}.log &
