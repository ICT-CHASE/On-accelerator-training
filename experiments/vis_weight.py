#-×- coding=utf-8 -*-
""" weight stat
"""
from __future__ import print_function
import os
# turn off verbose logging info of caffe
os.environ['GLOG_minloglevel'] = '2'
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
## for debug
# import ipdb

## try to import caffe
#  = os.path.dirname(os.path.realpath(__file__))
# caffe_cur_pathpath = os.path.join(cur_path, os.path.pardir, 'python')
# print(caffe_path)
caffe_path = '/home/xingkouzi/caffe-round/python'
sys.path.append(caffe_path)

try:
    import caffe
except ImportError:
    print(caffe_path)
    print('cannot import caffe module')
    raise ImportError

def parse_arg():
    parser = argparse.ArgumentParser(description='Visualization of CaffeNet weight and output of each layer')
    # here are the basic configurations
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='gpu id to do forward computing')
    parser.add_argument('--model', type=str, default='',
                        help='model file(.prototxt)')
    parser.add_argument('--weights', type=str, default='',
                        help='weights file(.caffemodel)')
    parser.add_argument('--output_dir', type=str, default='',
                        help='output directory of histograms of weight and blobs')
    parser.add_argument('--param_names', nargs='*', type=str,
                        help='param names to visualize. If not given, visualize all params')
    parser.add_argument('--blob_names', nargs='*', type=str,
                        help='blob names to visualize, should be given by order. If not given, visualize all blobs')
    return parser

def build_net(model_file, weight_file):
    """ Build net from model file and weight
    """
    if not os.path.exists(model_file):
        raise ValueError('cannot find model file: {}'.format(model_file))
    if not os.path.exists(weight_file):
        raise ValueError('cannot find weight file: {}'.format(weight_file))

    net = caffe.Net(model_file, weight_file, caffe.TEST)
    return net


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


def invaild_param_name_error(name, availables):
    """ Reture `ValueError` when use invalid name for extracting params
    """
    msg = 'cannot find param with name: {}, availables are {}'.format(
        name, ','.join(availables))
    return ValueError(msg)


def extract_weight(net, param_names):
    """ Extract param weights from net given name
    """
    # get all params
    params = net.params.keys()
    # get all layers of the net
    # we will use `l.type`
    layers = list(net._layer_names)
    weights = {name: None for name in param_names}
    for name in param_names:
        print('param name: {}'.format(name))
        if name not in params:
            raise invaild_param_name_error(name, params)
        layer = net.layers[layers.index(name)]
        param = net.params[name]
        if layer.type.find('Convolution') != -1:
            # this is a conv param
            d = {'type': layer.type, 'weight': param[0].data.copy()}
            if len(param) > 1:
                d['bias'] = param[1].data
        elif layer.type.find('InnerProduct') != -1:
            d = {'type': layer.type, 'weight': param[0].data.copy()}
            if len(param) > 1:
                d['bias'] = param[1].data
        elif layer.type == 'BatchNorm':
            # ipdb.set_trace()
            d = {'type': layer.type, 'mean': param[0].data.copy(),
                 'std': param[1].data.copy(), 'moving_fraction': param[2].data.copy()}
        elif layer.type == 'Scale':
            d = {'type': layer.type, 'gamma': param[0].data.copy(),
                 'beta': param[1].data.copy()}
        else:
            raise ValueError('unsupport type: {}'.format(layer.type))
        weights[name] = d
    return weights

def has_sorted(l):
    """ for an iterable sequence, test it is already sorted or not
    """
    return all(l[i] <= l[i+1] for i in xrange(len(l)-1))

def get_blob_data(net, names, iterations=100):
    """ Get blob data when do forward computing
    """
    layer_names = list(net._layer_names)
    idxes = [layer_names.index(name) for name in names]

    if not has_sorted(idxes):
        raise ValueError(
            'you should given the name in order by forward computing')

    blobs = {name: list() for name in names}
    for iter in range(iterations):
        print('iteration {:04d}'.format(iter))
        
        for i, name in enumerate(names):
            if i == 0:
                start = None
            else:
                prev_layer_idx = idxes[i - 1]
                start = layer_names[prev_layer_idx + 1]

            output = net.forward(start=start, end=name)
            if name != 'data' and len(output.keys()) != 1:
                raise ValueError(
                    'name: {}, multi-output: {}'.format(name, len(output.keys())))
            blob = output[output.keys()[0]].copy()
            blobs[name].append(blob)
    return blobs


if __name__ == '__main__':
    args = parse_arg().parse_args()
    if args.gpu_id == -1:
        print('Warning: GPU ID is not given, use cpu.')
        caffe.set_mode_cpu()
    else:
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

    if args.param_names is not None and len(args.param_names) != 0:
        param_names = args.param_names
    else:
        # get all param names of net
        param_names = caffe_net.params.keys()

    if args.blob_names is not None and len(args.blob_names) != 0:
        blob_names = args.blob_names
    else:
        # get all blob names of net
        layer_names = list(caffe_net._layer_names)
        blob_names = [x for x in layer_names if x.find('acc') == -1 
                                            and x.find('split') == -1 
                                            and x.find('label') == -1
                                            and x.find('loss') == -1
                                            and x.find('relu') == -1]

    print('=' * 30)
    print('param names:')
    print('\t'.join(param_names))
    print('=' * 30)
    print('blob names:')
    print('\t'.join(blob_names))
    print('=' * 30)

    ## for param names
    param_weights = extract_weight(caffe_net, param_names)
    # ipdb.set_trace()
    for name, value in param_weights.items():
        if value['type'] == 'BatchNorm':   
            mean = value['mean'].reshape(-1)
            plot_and_save(mean, os.path.join(output_dir, 'param-'), name + '-mean')
            std = value['std'].reshape(-1)
            plot_and_save(std, os.path.join(output_dir, 'param-'), name + '-std')
            mf = value['moving_fraction'].reshape(-1)[0]
            print('bn name: {}, moving fraction: {}'.format(name, mf))
        elif value['type'] == 'Scale':
            gamma = value['gamma'].reshape(-1)
            plot_and_save(gamma, os.path.join(output_dir, 'param-'), name + '-gamma')
            beta = value['beta'].reshape(-1)
            plot_and_save(beta, os.path.join(output_dir, 'param-'), name + '-beta')
        else:
            weight = value['weight'].reshape(-1)
            plot_and_save(weight, os.path.join(output_dir, 'param-'), name + '-weight')
            if 'bias' in value:
                bias = value['bias'].reshape(-1)
                plot_and_save(bias, os.path.join(output_dir, 'param-'), name + '-bias')   

    ## for blob names
    blob_datas = get_blob_data(caffe_net, blob_names, 10)

    for name, value_list in blob_datas.items():
        print('blob name: {}'.format(name))
        data = np.concatenate(value_list).reshape(-1)
        name = name.replace('/', '-')
        plot_and_save(data,  os.path.join(output_dir, 'blob-'), name)
