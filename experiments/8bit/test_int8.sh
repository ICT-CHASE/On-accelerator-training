#!/bin/bash
GPU_ID=$1
cur_date=`date +%Y-%m-%d-%H-%M-%S`
cur_dir=./experiments/8bit
log_file_name="$cur_dir/log/test_int8-${cur_date}.log"
model="$cur_dir/res18_only_conv_q17.prototxt"
weights="$cur_dir/models/res18_only_conv_q17_smaller_lr_iter_1000.caffemodel"
iterations=100
./build/tools/caffe test \
    -model $model \
    -weights $weights \
    -iterations $iterations \
    -gpu $GPU_ID 2>&1 | tee -a ${log_file_name}
