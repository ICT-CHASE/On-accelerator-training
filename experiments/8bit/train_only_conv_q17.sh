#!/bin/bash
GPU_ID=$1
cur_date=`date +%Y-%m-%d-%H-%M-%S`
cur_dir=./experiments/8bit
log_file_name="$cur_dir/log/only_conv_q17-${cur_date}.log"
solver="$cur_dir/solver_only_conv_q17.prototxt"
weights="./experiments/bn_test/res18_only_conv.caffemodel"
./build/tools/caffe train \
    -weights $weights \
    -solver $solver \
    -gpu $GPU_ID 2>&1 | tee -a ${log_file_name}
