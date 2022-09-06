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

TARGET=$1
NETNAME=$2

if [ $1 = zcu102 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json
      TARGET=zcu102
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZCU102.."
      echo "-----------------------------------------"
elif [ $1 = zcu104 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
      TARGET=zcu104
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZCU104.."
      echo "-----------------------------------------"
elif [ $1 = vck190 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json
      TARGET=vck190
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR VCK190.."
      echo "-----------------------------------------"
elif [ $1 = u50 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U50/arch.json
      TARGET=u50
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ALVEO U50.."
      echo "-----------------------------------------"
elif [ $1 = b128 ]; then
      ARCH=./arch/B128_LR/arch.json
      TARGET=b128
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR B128 ..."
      echo "-----------------------------------------"
elif [ $1 = b512 ]; then
      ARCH=./arch/B512_LR/arch.json
      TARGET=b512
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR B512 ..."
      echo "-----------------------------------------"
elif [ $1 = b1152 ]; then
      ARCH=./arch/B1152_LR/arch.json
      TARGET=b1152
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR B1152 ..."
      echo "-----------------------------------------"
elif [ $1 = b2304 ]; then
      ARCH=./arch/B2304_LR/arch.json
      TARGET=b2304
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR B2304 ..."
      echo "-----------------------------------------"
elif [ $1 = b3136 ]; then
      ARCH=./arch/B3136_LR/arch.json
      TARGET=b3136
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR B3136 ..."
      echo "-----------------------------------------"
elif [ $1 = b4096 ]; then
      ARCH=./arch/B4096_LR/arch.json
      TARGET=b4096
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR B4096 ..."
      echo "-----------------------------------------"
else
      #echo  "Target not found. Valid choices are: zcu102, zcu104, vck190, u50 ..exiting"
      echo  "Target not found. Valid choices are: b512, b1152, b2304, b3136, b4096 ..exiting"
      exit 1
fi

compile() {
      vai_c_tensorflow2 \
            --model           build/quant_model/q_model.h5 \
            --arch            ${ARCH} \
            --output_dir      build/compiled_${TARGET} \
            --net_name        ${NETNAME}
}


compile 2>&1 | tee build/logs/compile_$TARGET.log


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"



