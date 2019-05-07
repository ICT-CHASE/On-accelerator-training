#! /bin/bash
gpu=$1
model=./experiments/8bit/res18_only_conv_q17.prototxt
weights=./experiments/8bit/models/res18_only_conv_q17_smaller_lr_iter_10.caffemodel
output_dir=./0201_simulator_res18_nearest_without_saturation
python ./experiments/output_weight_for_simulator.py \
--model $model --weights $weights --gpu_id $gpu \
--output_dir $output_dir