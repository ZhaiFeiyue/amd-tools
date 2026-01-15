#!/bin/bash

set -x

sudo docker rm zhai_vllm_profile
sudo docker run --name zhai_vllm_profile -it  --network=host  --group-add=video \
	        --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
		        --device /dev/kfd  --device /dev/dri \
			        -v /raid/models/HF:/hf  -v /home/feiyzhai/code:/workspace/code \
				registry-sc-harbor.amd.com/framework/compute-rocm-rel-7.1:34_ubuntu24.04_py3.13_pytorch_release-2.9_62316079 bash
				# rocm/aigmodels-private:lip_dev_sglang_aiter_epmoe bash

				#rocm/ali-private:v0.4.4.post3-rocm630-deepep-aiter-0603 bash

				#rocm/ali-private:sglang_ep_052600 bash
				#rocm/ali-private:sglang_ep_052701 bash
				#rocm/aigmodels-private:experimental_950_5_1_qfvllm_aiter bash
				       #sabreshao/sglang:950_051600  bash

