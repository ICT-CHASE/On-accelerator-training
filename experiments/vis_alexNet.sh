#! /bin/bash
gpu=1
model=./accelerator/train_val_13.prototxt
weights=./accelerator/accelerator__train_iter_20000.caffemodel
output_dir=./experiments/stat/test
python ./experiments/vis_weight.py --gpu_id $gpu --model $model \
--weights $weights \
--output_dir $output_dir
