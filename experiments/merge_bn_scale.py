# -*- coding=utf8 -*-
""" Merge BatchNorm and Scale into one layer
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

def add_lr_to_scale_layer(layer_des):
    """ add lr = 0 to scale layer
    """
    layer_des = [l for l in layer_des if len(l.strip()) != 0]
    
    param = """    param { 
        lr_mult: 0 
        decay_mult: 0
    }
    """
    ret = layer_des[:-1]  # remove the last `}`
    ret += [param, param, '}\n\n']
    return ret

def remove_bn_from_prototxt(model_file, output_file):
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
    
    removed_layers = [l for l in layers if get_layer_type(l) != 'BatchNorm']
    ## set lr of bias param of scale layer to 0.0    
    for i, l in enumerate(removed_layers):
        if get_layer_type(l) == 'Scale':
            removed_layers[i] = add_lr_to_scale_layer(l)

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

    def get_bn_name(scale_name):
        """ Get bn name given scale layer name
        """
        return scale_name.replace('scale', 'bn')

    for i, param in enumerate(target_params):
        print('{}th param, name: {}'.format(i+1, param))
        lidx = layers.index(param)
        layer = merged_net.layers[lidx]
        
        if layer.type == 'Convolution':
            param_list = merged_net.params[param]
            has_bias = len(param_list) > 1
            print('has bias? {}'.format('YES' if has_bias else 'No'))

            weight = param_list[0].data
            source_weight = net.params[param][0].data
            assert weight.shape == source_weight.shape
            diff = np.abs(weight - source_weight)
            max_val = np.max(diff)
            print('max diff of conv weight is: {}'.format(max_val))
        elif layer.type == 'InnerProduct':
            param_list = merged_net.params[param]
            has_bias = len(param_list) > 1
            print('has bias? {}'.format('YES' if has_bias else 'NO'))
            
            weight = param_list[0].data
            source_weight = net.params[param][0].data
            assert weight.shape == source_weight.shape

            diff = np.abs(weight - source_weight)
            max_val = np.max(diff)
            print('max diff of ip weight is {}'.format(max_val))

        elif layer.type == 'Scale':
            ## key layer!
            source_param_list = net.params[param]
            gamma = source_param_list[0].data
            beta = source_param_list[1].data
            
            bn_param_name = get_bn_name(param)
            bn_param_list = net.params[bn_param_name]

            mean = bn_param_list[0].data
            var = bn_param_list[1].data
            scale_factor = bn_param_list[2].data[0]

            mean = mean / scale_factor
            var = var / scale_factor + 1E-5

            std = np.sqrt(var)

            assert gamma.shape == std.shape
            assert beta.shape == gamma.shape
            assert beta.shape == mean.shape

            new_gamma = gamma / std
            new_beta = beta - new_gamma * mean 
   
            param_list = merged_net.params[param]
            # print(param_list[0].data.shape, new_gamma.shape)
            assert param_list[0].data.shape == new_gamma.shape
            assert param_list[1].data.shape == new_beta.shape

            new_gamma = np.array([to_nearest_pow_of_2(x) for x in new_gamma]).reshape(param_list[0].data.shape)

            il = 1
            fl = 7
            new_beta = np.array([to_nearest_rounding(x, il, fl) for x in new_beta]).reshape(param_list[1].data.shape)

            param_list[0].data[:] = new_gamma
            param_list[1].data[:] = new_beta
            
            # plot_and_save(param_list[0].data, './experiments/bn_test/fig/{}-'.format(param), 'gamma2.jpg')
            # plot_and_save(param_list[1].data, './experiments/bn_test/fig/{}-'.format(param), 'beta2.jpg')

        
        print('=' * 30)


## build pretrained model with FP32
model_file = './experiments/resnet-18-imagenet-scale/train_val.prototxt'
weight_file = '/home/zhaoxiandong/Models/resnet-18-scale-to-1/resnet-18_iter_650000.caffemodel'

output_file = './experiments/bn_test/remove_bn_res18_fp32.prototxt'
output_weight_file = './experiments/bn_test/remove_bn_pow2.caffemodel'

remove_bn_from_prototxt(model_file, output_file)

## get bn params
net = build_net(model_file, weight_file)
merged_net = build_net(output_file, weight_file)

merge_bn_weight(net, merged_net)

merged_net.save(output_weight_file)

## test 
test_net = build_net(output_file, output_weight_file)

output1 = test_net.forward()
print(output1)

output2 = net.forward()
print(output2)