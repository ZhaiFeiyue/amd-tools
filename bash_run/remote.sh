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
DIST_IP=localhost

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

NNODES=`cat ${CONF} | wc -l`

while read line
do
	host=`echo $line | cut -d " " -f 1`
	user=`echo $line | cut -d " " -f 2`
	rank=`echo $line | cut -d " " -f 3`
	if [[ ${rank} == 0 ]];then
		DIST_IP=${host}
	fi
done < ${CONF}

echo ${DIST_IP}
exit

# setup
while read line
do
	host=`echo $line | cut -d " " -f 1`
	user=`echo $line | cut -d " " -f 2`
	rank=`echo $line | cut -d " " -f 3`

	echo "setup env for ${host}"
	ssh ${user}@${host} <<EOF
	set -ex
	rm -rf ${TARGET_DIR}
	mkdir -p ${TARGET_DIR}
EOF
	scp -r ${ROOT}/run.sh ${user}@${host}:${TARGET_DIR}
	scp -r ${ROOT}/launch.sh ${user}@${host}:${TARGET_DIR}

	ssh ${user}@${host} <<EOF
	set -ex
	${TARGET_DIR}/run.sh ${NNODES} ${rank} ${DIST_IP}
EOF

done <  ${CONF}

