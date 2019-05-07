#!/bin/bash
model=../alexnet_imagenet/deploy.prototxt
weights=../alexnet_imagenet/models/alexnet_iter_100000.caffemodel
python vis_weight.py $model $weights