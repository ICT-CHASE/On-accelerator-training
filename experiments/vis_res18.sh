#! /bin/bash
gpu=$1
model=./experiments/maxpool/res18_only_conv_no_avg_pool_fp32.prototxt
weights=./experiments/maxpool/res18_only_conv_no_avg_pool.caffemodel
output_dir=./experiments/stat/test
python ./experiments/vis_weight.py --gpu_id $gpu --model $model \
--weights $weights \
--output_dir $output_dir