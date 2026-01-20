#!/bin/bash

set -x

sudo docker rm zhai_vllm_debug
sudo docker run --name zhai_vllm_debug -it  --network=host  --group-add=video \
	        --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
		 --device /dev/kfd  --device /dev/dri \
		-v /raid/models/HF:/hf  -v /home/feiyzhai/code:/workspace/code \
		rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0 bash
		# registry-sc-harbor.amd.com/framework/compute-rocm-rel-7.1:34_ubuntu.04_py3.13_pytorch_release-2.9_62316079 bash

