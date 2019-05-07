# -*- coding=utf8 -*-
""" Output model weight
"""
from __future__ import print_function
import os
# turn off verbose logging info of caffe
os.environ['GLOG_minloglevel'] = '1'
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
# for debug
# import ipdb

# try to import caffe
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


def parse_arg():
    parser = argparse.ArgumentParser(
        description='Output weight and blob for simulator')
    # here are the basic configurations
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='gpu id to do forward computing. Default: CPU')
    parser.add_argument('--model', type=str, default='',
                        help='model file(.prototxt)')
    parser.add_argument('--weights', type=str, default='',
                        help='weights file(.caffemodel)')
    parser.add_argument('--output_dir', type=str, default='',
                        help='output directory of histograms of weight and blobs')
    return parser


def get_layer_type(layer_des):
    """ Get layer type from layer description
    """
    for line in layer_des:
        line = line.strip()
        if line.find('type') != -1:
            begin = line.find('\"')
            end = line.rfind('\"')
            return line[begin + 1: end]
    return None


def get_layer_phase(layer_des):
    """ Get layer phase
    """
    for line in layer_des:
        line = line.strip()
        if line.find('phase') != -1:
            if line.find('TRAIN') != -1:
                return 'TRAIN'
            elif line.find('TEST') != -1:
                return 'TEST'
    return ['TRAIN', 'TEST']


def change_batch_size(layer_des):
    for i, line in enumerate(layer_des):
        if line.find('batch_size') != -1:
            layer_des[i] = 'batch_size: 1'
    return layer_des


def change_batchsize_to_one(model_file, output_file):
    """ Change batchsize of TEST phase to 1
    """
    with open(model_file, 'r') as f:
        lines = f.readlines()
    net_description = lines[0]
    print('net description: {}'.format(net_description))
    # split by `layer`
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

    # change batch size of TEST
    for i, layer in enumerate(layers):
        if get_layer_type(layer) == 'Data' and get_layer_phase(layer) == 'TEST':
            layers[i] = change_batch_size(layer)

    with open(output_file, 'w') as f:
        f.write(net_description)
        for l in layers:
            for line in l:
                f.write(line)

    print('output file: {}'.format(output_file))


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


def write_param_to_file(output_dir, param_list, param_name):
    def write_weight(filename, weight, bias):
        if bias:
            data = ' '.join(['{:.15f}'.format(x) for x in weight])
            with open(filename, 'w') as f:
                f.write(data)
                f.write('\n')
            return

        # reshape weight to `Cout x (Cin x K x K)` for conv
        # and  `Cout x Cin` for fc
        weight = weight.reshape((weight.shape[0], -1))
        with open(filename, 'w') as f:
            for i in range(weight.shape[0]):
                data = ' '.join(['{:.8f}'.format(x) for x in weight[i]])
                f.write(data)
                f.write('\n')
    filename = os.path.join(output_dir, param_name + '-weight.txt')
    weight = param_list[0].data
    write_weight(filename, weight, False)

    if len(param_list) > 1:
        filename = os.path.join(output_dir, param_name + '-bias.txt')
        bias = param_list[1].data
        write_weight(filename, bias, True)


def output_weight_to_file(net, output_dir):
    """ output weight as txt format to `output_dir`
    """
    param_names = net.params.keys()
    layers = list(net._layer_names)

    for i, name in enumerate(param_names):
        print('param index: {}, name: {}'.format(i, name))
        lidx = layers.index(name)
        layer = net.layers[lidx]
        if layer.type == 'ConvolutionFloat':
            write_param_to_file(output_dir, net.params[name], name)

        elif layer.type == 'InnerProductFloat':
            write_param_to_file(output_dir, net.params[name], name)
        else:
            raise ValueError('unknown layer: {}'.format(layer.type))


def has_sorted(l):
    """ for an iterable sequence, test it is already sorted or not
    """
    return all(l[i] <= l[i + 1] for i in xrange(len(l) - 1))


def get_blob_data(net, names):
    """ Get blob data when do forward computing
    """
    layer_names = list(net._layer_names)
    idxes = [layer_names.index(name) for name in names]

    if not has_sorted(idxes):
        raise ValueError(
            'you should given the name in order by forward computing')

    blobs = {}
    for i, name in enumerate(names):
        print('{}th blob, name: {}'.format(i, name))
        if i == 0:
            start = None
        else:
            prev_layer_idx = idxes[i - 1]
            start = layer_names[prev_layer_idx + 1]

        print('start = {}, end = {}'.format(start, name))

        output = net.forward(start=start, end=name)
        if name != 'data' and len(output.keys()) != 1:
            raise ValueError(
                'name: {}, multi-output: {}'.format(name, len(output.keys())))
        blob = output[output.keys()[0]].copy()
        print('blob shape: {}'.format(blob.shape))
        blobs[name] = blob
    return blobs


def output_blob_to_file(net, output_dir, blob_names):
    """ output blob into file
    """
    blobs = get_blob_data(net, blob_names)

    for name in blob_names:
        blob = blobs[name]
        assert blob.shape[0] == 1

        filename = os.path.join(output_dir, 'blob-{}.txt'.format(name))
        print('name: {}, blob shape: {}'.format(name, blob.shape))
        if name.find('fc') != -1:
            blob = blob.reshape(-1)
            with open(filename, 'w') as f:
                data = ' '.join(['{:.8f}'.format(x) for x in blob])
                f.write(data)
                f.write('\n')
            continue
        if name == 'data':
            assert blob.shape[1] == 3
        blob = blob.reshape((-1, blob.shape[3]))
        with open(filename, 'w') as f:
            for i in range(blob.shape[0]):
                data = ' '.join(['{:.8f}'.format(x) for x in blob[i]])
                f.write(data)
                f.write('\n')


if __name__ == '__main__':
    # get arguments
    args = parse_arg().parse_args()
    if args.gpu_id < 0:
        raise ValueError('caffe round cannot run without gpu')

    print('GPU ID: {}'.format(args.gpu_id))
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    output_dir = args.output_dir
    if len(output_dir) == 0:
        raise ValueError('output dir not given')
    if not os.path.exists(output_dir):
        print('given output dir: {} not exist. mkdir,'.format(output_dir))
        os.makedirs(output_dir)
    else:
        print('output directory: {}'.format(output_dir))

    model, weights = args.model, args.weights
    if len(model) == 0:
        raise ValueError('model not given')
    if len(weights) == 0:
        raise ValueError('weights not given')

    caffe_net = build_net(model, weights)
    print('model: {}, weight: {}, build net done'.format(model, weights))

    param_names = caffe_net.params.keys()

    layer_names = list(caffe_net._layer_names)
    blob_names = [x for x in layer_names if x.find('acc') == -1
                  and x.find('split') == -1
                  and x.find('label') == -1
                  and x.find('loss') == -1]
    print('=' * 30)
    print('param names:')
    print('\t'.join(param_names))
    print('=' * 30)
    print('blob names:')
    print('\t'.join(blob_names))
    print('=' * 30)

    modified_model_file = './tmp.prototxt'
    change_batchsize_to_one(model, modified_model_file)

    net = build_net(modified_model_file, weights)
    output_weight_to_file(net, output_dir)

    output_blob_to_file(net, output_dir, blob_names)
