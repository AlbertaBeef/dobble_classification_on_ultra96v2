#!/bin/sh

# Copyright 2022 Avnet Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Mario Bergeron, Avnet Inc
#
# Based on: https://github.com/Xilinx/Vitis-AI-Tutorials/tree/master/Design_Tutorials/08-tf2_flow
#   Original Author: Mark Harvey, Xilinx Inc

# activate the python virtual environment
conda activate vitis-ai-tensorflow2
pip install tensorflow_addons

# folders
export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}

export NETNAME=dobble

# list of GPUs to use - modify as required for your system
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="0"


# convert dataset to TFRecords - this only needs to be run once
python -u images_to_tfrec.py 2>&1 | tee ${LOG}/tfrec.log


# training
python -u train.py 2>&1 | tee ${LOG}/train.log


# quantize & evaluate
python -u quantize.py --evaluate 2>&1 | tee ${LOG}/quantize.log

# compile/target for selected board

#export TARGET=zcu102
#source compile.sh ${TARGET} ${NETNAME}
#python -u target.py -m ${BUILD}/compiled_${TARGET}/${NETNAME}.xmodel -t ${BUILD}/target_${TARGET} 2>&1 | tee ${LOG}/target_${TARGET}.log

#export TARGET=zcu104
#source compile.sh ${TARGET} ${NETNAME}
#python -u target.py -m ${BUILD}/compiled_${TARGET}/${NETNAME}.xmodel -t ${BUILD}/target_${TARGET} 2>&1 | tee ${LOG}/target_${TARGET}.log

# compile/target for selected DPU architecture

export TARGET=b4096
source compile.sh ${TARGET} ${NETNAME}
python -u target.py -m ${BUILD}/compiled_${TARGET}/${NETNAME}.xmodel -t ${BUILD}/target_${TARGET} 2>&1 | tee ${LOG}/target_${TARGET}.log

export TARGET=b3136
source compile.sh ${TARGET} ${NETNAME}
python -u target.py -m ${BUILD}/compiled_${TARGET}/${NETNAME}.xmodel -t ${BUILD}/target_${TARGET} 2>&1 | tee ${LOG}/target_${TARGET}.log

export TARGET=b2304
source compile.sh ${TARGET} ${NETNAME}
python -u target.py -m ${BUILD}/compiled_${TARGET}/${NETNAME}.xmodel -t ${BUILD}/target_${TARGET} 2>&1 | tee ${LOG}/target_${TARGET}.log

export TARGET=b1152
source compile.sh ${TARGET} ${NETNAME}
python -u target.py -m ${BUILD}/compiled_${TARGET}/${NETNAME}.xmodel -t ${BUILD}/target_${TARGET} 2>&1 | tee ${LOG}/target_${TARGET}.log

export TARGET=b512
source compile.sh ${TARGET} ${NETNAME}
python -u target.py -m ${BUILD}/compiled_${TARGET}/${NETNAME}.xmodel -t ${BUILD}/target_${TARGET} 2>&1 | tee ${LOG}/target_${TARGET}.log

export TARGET=b128
source compile.sh ${TARGET} ${NETNAME}
python -u target.py -m ${BUILD}/compiled_${TARGET}/${NETNAME}.xmodel -t ${BUILD}/target_${TARGET} 2>&1 | tee ${LOG}/target_${TARGET}.log

