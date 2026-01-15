#!/bin/bash

set -ex
sudo docker run -it \
	  --privileged \
	    --network=host \
	      --group-add=video \
	        --ipc=host \
		  --cap-add=SYS_PTRACE \
		    --security-opt seccomp=unconfined \
		      --device /dev/kfd \
		        --device /dev/dri \
			  --name zhai_vllm_omni -v /raid/models/HF:/hf  -v /home/feiyzhai/code:/workspace/code \
			    rocm/vllm-dev:vllm-omni-12012025 bash
