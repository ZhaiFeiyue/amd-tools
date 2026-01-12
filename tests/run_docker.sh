#!/bin/bash
NAME="merge_deepep"
NAME="rtp_llm"
DOCKER_IMAGE=zhai_deepep:latest
DOCKER_IMAGE=registry.cn-hangzhou.aliyuncs.com/havenask/rtp_llm:rocm6302
docker run -it --name ${NAME} --network=host --group-add=video  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device /dev/kfd  --device /dev/dri -v /home/feiyzhai/code:/code --privileged --shm-size=128G --ulimit memlock=-1 --ulimit stack=67108864 ${DOCKER_IMAGE} bash 
