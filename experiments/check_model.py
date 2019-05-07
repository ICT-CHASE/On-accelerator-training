# -*- coding=utf8 -*-
""" Check int8 model
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
# caffe_path = '/home/xiaomeifeng/caffe/python'
caffe_path = '/home/xiaomeifeng/extra/caffe-round/python'
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

def data_round_nearest(x, q, qmax):
    scaled_x = x * q
    int_x = np.floor(scaled_x)
    residual = scaled_x - int_x
    if residual > 0.5:
        int_x = np.ceil(scaled_x)
    if int_x >= qmax:
        int_x = qmax - 1
    if int_x < -qmax:
        int_x = -qmax
    return int_x / q

def weight_round_nearest(weight, il, fl):
    q = 2**fl
    qmax = 2**(il+fl-1)
    return np.array([data_round_nearest(x, q, qmax) for x in weight.reshape(-1)]).reshape(weight.shape)
        
def round_weight(net):
    param_names = net.params.keys()
    # all layers of target net
    layers = list(net._layer_names)

    for i, param in enumerate(param_names):
        lidx = layers.index(param)
        layer = net.layers[lidx]
        if layer.type == 'ConvolutionFloat':
            print('it is a conv float layer')
            il, fl = 1, 7
            conv_params = net.params[param]
            weight = conv_params[0].data
            weight[:] = weight_round_nearest(weight, il, fl)
            if len(conv_params) > 1:
                bias = conv_params[1].data
                bias[:] = weight_round_nearest(bias, il, fl)
                
        elif layer.type == 'InnerProductFloat':
            print('it is a ip float layer')
            il, fl = 2, 6
            fc_params = net.params[param]
            weight = fc_params[0].data
            weight[:] = weight_round_nearest(weight, il, fl)

            if len(fc_params) > 1:
                bias = fc_params[1].data
                bias[:] = weight_round_nearest(bias, il, fl)

        else:
            raise ValueError('unknown layer: {}'.format(layer.type))

caffe.set_mode_gpu()
caffe.set_device(2)

model_file = './experiments/bn_test/res18_only_conv_q17.prototxt'
weight_file = './experiments/bn_test/models/res18_only_conv_q17_iter_1000.caffemodel'
#model_file = './experiments/bn_test/res18_only_conv_fp32.prototxt'
#weight_file = './experiments/bn_test/res18_only_conv.caffemodel'

output_weight_file = './experiments/bn_test/models/res18_only_conv_q17_2.caffemodel'

net = build_net(model_file, weight_file)
for i in range(10):
    output1 = net.forward()
    print(output1)

# round_weight(net)

# net.save(output_weight_file)
# #ipdb.set_trace()
# output2 = net.forward()
# print(output2)

