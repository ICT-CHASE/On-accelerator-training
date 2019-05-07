# -*- coding=utf8 -*-
""" Merge BatchNorm and Scale into conv layer
"""
from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
## for debug
import ipdb

## try to import caffe
#  = os.path.dirname(os.path.realpath(__file__))
# caffe_cur_pathpath = os.path.join(cur_path, os.path.pardir, 'python')
# print(caffe_path)
caffe_path = '/home/xiaomeifeng/caffe/python'
sys.path.append(caffe_path)

try:
    import caffe
except ImportError:
    print(caffe_path)
    print('cannot import caffe module')
    raise ImportError

def build_net(model_file, weight_file=None):
    """ Build net from model file and weight
    """
    if not os.path.exists(model_file):
        raise ValueError('cannot find model file: {}'.format(model_file))
    if not weight_file is None and not os.path.exists(weight_file):
        raise ValueError('cannot find weight file: {}'.format(weight_file))

    if not weight_file is None:
        net = caffe.Net(model_file, weight_file, caffe.TEST)
    else:
        net = caffe.Net(model_file, caffe.TEST)
    return net

def get_layer_type(layer_des):
    """ Get layer type from layer description
    """
    for line in layer_des:
        line = line.strip()
        if line.find('type') != -1:
            begin = line.find('\"')
            end = line.rfind('\"')
            return line[begin+1 : end]
    return None

def add_bias(layer_des):
    """ Add bias to conv
    """
    for i, line in enumerate(layer_des):
        if line.find('bias_term') != -1:
            layer_des[i] = line.replace('false', 'true')
    return layer_des

def remove_bn_scale_from_prototxt(model_file, output_file):
    """ Remove bn from prototxt
    """
    with open(model_file, 'r') as f:
        lines = f.readlines()
    net_description = lines[0]
    print('net description: {}'.format(net_description))
    ## split by `layer`
    lines = lines[1:]
    i, j = 0, 0
    layers = []
    while i < len(lines):
        line = lines[i]
        if line.find('layer') != -1:
            layers.append(lines[j:i])
            j = i
        i += 1
    layers.append(lines[j:i])
    print('there are {} layers'.format(len(layers)))
    
    ## remove bn layers and scale layers
    removed_layers = [l for l in layers if get_layer_type(l) != 'BatchNorm']
    removed_layers = [l for l in removed_layers if get_layer_type(l) != 'Scale']
    
    ## add bias term
    for i, l in enumerate(removed_layers):
        if get_layer_type(l) == 'Convolution':
            removed_layers[i] = add_bias(l)

    with open(output_file, 'w') as f:
        f.write(net_description)
        for l in removed_layers:
            for line in l:
                f.write(line)

    print('output file: {}'.format(output_file))

def plot_and_save(data, prefix, name):
    """ plot histogram of data and save it to disk with given prefix and name
    """
    plt.figure()
    plt.hist(data)
    plt.title(name)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(prefix + name + '.png')
    plt.close()


def to_nearest_pow_of_2(x):
    """ give the nearest value y = 2^m -> x, where m is an interger
    """
    neg = x < 0
    m = round(np.log2(np.abs(x)))

    return -2**m if neg else 2**m
    
def to_nearest_rounding(x, il, fl):
    """ Nearest Rounding of given x
    """
    scale = 2**fl
    scaled_x = x * scale
    int_x = np.floor(scaled_x)
    fraction_part = scaled_x - int_x
    if fraction_part > 0.5:
        int_x = np.ceil(scaled_x)
    max_val = 2**(il+fl-1)
    min_val = -2**(il+fl-1)

    if int_x >= max_val:
        int_x = max_val - 1
    elif int_x < min_val:
        int_x = min_val
    
    return float(int_x) / scale
    
def merge_bn_weight(net, merged_net):
    # get all params' name
    source_params = net.params.keys()
    target_params = merged_net.params.keys()
    # all layers of target net
    layers = list(merged_net._layer_names)

    def get_bn_name(conv_name):
        """ Get bn name given conv layer name
        """
        scale_name = get_scale_name(conv_name)
        return scale_name.replace('scale', 'bn')
    
    def get_scale_name(conv_name):
        """ Get scale name given conv layer name
        """
        if conv_name.find('res') == -1:
            # the first conv layer
            return 'scale_' + conv_name
        else:
            return conv_name.replace('res', 'scale')

    for i, param in enumerate(target_params):
        print('{}th param, name: {}'.format(i+1, param))
        lidx = layers.index(param)
        layer = merged_net.layers[lidx]
        
        if layer.type == 'Convolution':
            param_list = merged_net.params[param]
            has_bias = len(param_list) > 1
            if not has_bias:
                raise ValueError('must have bias term')

            print('has bias? {}'.format('YES' if has_bias else 'No'))

            weight = param_list[0].data
            source_weight = net.params[param][0].data
            assert weight.shape == source_weight.shape

            bias = param_list[1].data

            ## get scale name and bn name
            scale_name = get_scale_name(param)
            bn_name = get_bn_name(param)

            scale_param_list = net.params[scale_name]
            gamma = scale_param_list[0].data
            beta = scale_param_list[1].data

            bn_param_list = net.params[bn_name]
            mean = bn_param_list[0].data
            var = bn_param_list[1].data
            scale_factor = bn_param_list[2].data[0]

            mean = mean / scale_factor
            var = var / scale_factor + 1E-5
            std = np.sqrt(var)

            new_gamma = gamma / std
            new_beta = beta - new_gamma * mean

            assert new_gamma.shape[0] == weight.shape[0]
            for c in range(weight.shape[0]):
                weight[c] = source_weight[c] * new_gamma[c]
                bias[c] = new_beta[c]


        elif layer.type == 'InnerProduct':
            param_list = merged_net.params[param]
            has_bias = len(param_list) > 1
            print('has bias? {}'.format('YES' if has_bias else 'NO'))
            
            weight = param_list[0].data
            source_weight = net.params[param][0].data
            assert weight.shape == source_weight.shape

            param_list[0].data[:] = source_weight.copy()
            if has_bias:
                source_bias = net.params[param][1].data
                param_list[1].data[:] = source_bias.copy()

            diff = np.abs(weight - source_weight)
            max_val = np.max(diff)
            print('max diff of ip weight is {}'.format(max_val))
            
            # plot_and_save(param_list[0].data, './experiments/bn_test/fig/{}-'.format(param), 'gamma2.jpg')
            # plot_and_save(param_list[1].data, './experiments/bn_test/fig/{}-'.format(param), 'beta2.jpg')  
        print('=' * 30)


## build pretrained model with FP32
#model_file = './experiments/resnet-18-imagenet-scale/train_val.prototxt'
#weight_file = '/home/zhaoxiandong/Models/resnet-18-scale-to-1/resnet-18_iter_650000.caffemodel'

#output_file = './experiments/bn_test/res18_only_conv_fp32.prototxt'
#output_weight_file = './experiments/bn_test/res18_only_conv.caffemodel'

model_file = './experiments/resnet-18-imagenet-scale/train_val_no_avgpool.prototxt'
weight_file = './experiments/resnet-18-imagenet-scale/models/res18_max_pool_iter_450000.caffemodel'

output_file = './experiments/maxpool/res18_only_conv_no_avg_pool_fp32.prototxt'
output_weight_file = './experiments/maxpool/res18_only_conv_no_avg_pool.caffemodel'
remove_bn_scale_from_prototxt(model_file, output_file)

caffe.set_mode_gpu()
caffe.set_device(1)
## get bn params
net = build_net(model_file, weight_file)
merged_net = build_net(output_file)

merge_bn_weight(net, merged_net)

merged_net.save(output_weight_file)

## test 
test_net = build_net(output_file, output_weight_file)

output1 = test_net.forward()
print(output1)

output2 = net.forward()
print(output2)