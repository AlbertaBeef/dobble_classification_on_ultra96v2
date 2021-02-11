#!/bin/bash


# Copyright 2020 Xilinx Inc.
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


# evaluate graph with test dataset
eval_graph() {
  dir_name=$1
  graph=$2
  python eval_graph.py \
    --graph        $dir_name/$graph \
    --input_node   ${INPUT_NODE} \
    --output_node  ${OUTPUT_NODE} \
    --batchsize    100
}

echo "-----------------------------------------"
echo "EVALUATING THE QUANTIZED GRAPH.."
echo "-----------------------------------------"

eval_graph ${QUANT} quantize_eval_model.pb 2>&1 | tee ${LOG}/${EVAL_Q_LOG}


echo "-----------------------------------------"
echo "EVALUATION COMPLETED"
echo "-----------------------------------------"

