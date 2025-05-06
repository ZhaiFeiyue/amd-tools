#!/bin/bash
set -ex

BIN=`dirname ${0}`
BIN=`cd ${BIN}; pwd`
ROOT=${BIN}

DOCKER_IMAGE=sabreshao/sglang_bcm:042304
NAME=bench

while getopts "c:" opt; do
    case "${opt}" in
            c)
            CONF=${OPTARG}
                   ;;
            i)
                   IMG=${OPTARG}
                  ;;
       :)
            echo "missing -${opt}"
                  exit 1
            ;;
?)
       echo "invalid opt"
              exit 2
																												                    ;;
																														        esac
																														done


sudo docker run --name ${BENCH} --rm --network=host --group-add=video  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device /dev/kfd  --device /dev/dri -v /apps:/apps -v ${BIN}:/bench --privileged --shm-size=128G --ulimit memlock=-1 --ulimit stack=67108864 ${DOCKER_IMAGE} /bench/launch.sh
