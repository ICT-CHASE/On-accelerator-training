#! /bin/bash
gpu=$1
model=./experiments/8bit/res18_only_conv_q17.prototxt
weights=./experiments/8bit/models/res18_only_conv_q17_iter_1000.caffemodel
output_dir=./experiments/stat/res18_int8
python ./experiments/vis_weight.py --gpu_id $gpu --model $model \
--weights $weights \
--output_dir $output_dir