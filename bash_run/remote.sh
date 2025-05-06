#!/bin/bash

if [ $? -ne 0 ]; then
    echo "Usage: $0 [-c|-i] [arguments]"
    exit 1
fi

BIN=`dirname ${0}`
BIN=`cd ${BIN}; pwd`
ROOT=${BIN}
CONF_DIR=${ROOT}/config

TMP_DIR=/tmp
TARGET_DIR=${TMP_DIR}/bench_dir
CONF=${ROOT}/config.1
IMG=sabreshao/sglang_bcm:042304
DIST=localhost
CLEAN=0

while getopts "c:i:r" opt; do
	case "${opt}" in
		c)
			CONF=${OPTARG}
			;;
		i)
			IMG=${OPTARG}
			;;
		r)
			CLEAN=1
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

if [[ ${CLEAN} == 1 ]];then
	echo "clean dockers"
	while read line
	do
		host=`echo $line | cut -d " " -f 1`
		user=`echo $line | cut -d " " -f 2`
		rank=`echo $line | cut -d " " -f 3`

		echo "clean env for ${host}"
		ssh ${user}@${host} "rm -rf ${TARGET_DIR}"
		ssh ${user}@${host} "mkdir -p ${TARGET_DIR}"
		DOCKER_NAME=bench_${rank}

		ssh ${user}@${host} "sudo docker stop ${DOCKER_NAME}"
	done <  ${CONF}
	exit 0
fi

NNODES=`cat ${CONF} | wc -l`

while read line
do
	host=`echo $line | cut -d " " -f 1`
	user=`echo $line | cut -d " " -f 2`
	rank=`echo $line | cut -d " " -f 3`
	if [[ ${rank} == 0 ]];then
		DIST=${host}
	fi
done < ${CONF}

# setup
while read line
do
	host=`echo $line | cut -d " " -f 1`
	user=`echo $line | cut -d " " -f 2`
	rank=`echo $line | cut -d " " -f 3`

	echo "setup env for ${host}"
	ssh ${user}@${host} "rm -rf ${TARGET_DIR}"
	ssh ${user}@${host} "mkdir -p ${TARGET_DIR}"

	scp -r ${ROOT}/run.sh ${user}@${host}:${TARGET_DIR}
	scp -r ${ROOT}/launch.sh ${user}@${host}:${TARGET_DIR}

	DOCKER_NAME=bench_${rank}
	ssh -fn ${user}@${host} "${TARGET_DIR}/run.sh -m ${NNODES} -r ${rank} -h ${DIST} -n ${DOCKER_NAME} -d ${IMG} > /dev/null 2>&1 &"

done <  ${CONF}

