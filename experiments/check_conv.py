import os
import numpy as np

#N, C, H, W = 1, 3, 224, 224

#Cout, K = 64, 7
N, C, H, W = 1, 3, 224, 224

Cout, K = 64, 7

pad, stride = 3, 2
#N, C, H, W = 1, 3, 6, 4

#Cout, K = 4, 3

# out: N Count, Hnew, Wnew
HH, WW = 112, 112


row, col = 6782, 51
index = row * WW + col

tw = index % WW
th = (index // WW) % HH 
tc = (index // WW) // HH

print(tw, th, tc)

def read_data(file):
    """ Read data
    """
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]

    r = len(lines)
    c = len(lines[0].split())
    print r, c
    data = np.empty(shape=(r, c))
    assert r * c == N * C * W * H
    for i in range(r):
        data[i] = [float(x) for x in lines[i].split()]
    return data.reshape((N, C, H, W))


def read_weight(file):
    """ Read weight
    """
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    r = len(lines)
    c = len(lines[0].split())
    assert r * c == Cout * C * K * K
    data = np.empty(shape=(r, c))
    for i in range(r):
        data[i] = [float(x) for x in lines[i].split()]
    return data.reshape((Cout, C, K, K))


def read_bias(file):
    """ Read weight
    """
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    r = len(lines)
    c = len(lines[0].split())
    assert r == 1 and c == Cout
    data = np.array([float(x) for x in lines[0].split()])
    return data


def read_out(file):
    """ Read out
    """
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    r = len(lines)
    c = len(lines[0].split())
    data = np.empty(shape=(r, c))
    assert r * c == N * Cout * WW * HH
    for i in range(r):
        data[i] = [float(x) for x in lines[i].split()]
    return data.reshape((N, Cout, HH, WW))


def quantize(x, q, qmax, p):
    xx = x * q
    ix = np.floor(xx)
    res = xx - ix
    if res > p:
        ix = np.ceil(xx)
    if ix >= qmax:
        ix = qmax - 1
    if ix < -qmax:
        ix = -qmax
    return ix / q


def quantize_arr(x, q, qmax):
    tx = x.reshape(-1)
    prob = np.random.rand(*tx.shape)
    for i in range(tx.shape[0]):
        tx[i] = quantize(tx[i], q, qmax, prob[i])
    return tx.reshape(x.shape)


def my_dot(x, w, flag=False):
    fx = x.reshape(-1)
    fw = w.reshape(-1)
    sum = 0
    n = fx.shape[0]
    il_weight, il_input = 1, 1
    cmax = 2**(il_weight + il_input - 1)
    for i in range(n):
        sum += fx[i] * fw[i]
        if flag:
            print('{} x {}, sum = {}'.format(fx[i], fw[i], sum))
        if sum > cmax:
            sum = cmax
            if flag:
                print('xxx, sum = {}'.format(sum))
        if sum < -cmax:
            sum = -cmax
            if flag:
                print('ooo, sum = {}'.format(sum))
    return sum


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.
    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride
    out = np.zeros((N, F, H_out, W_out))

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)),
                   mode='constant', constant_values=0)
    
    
    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i * stride:i *
                              stride + HH, j * stride:j * stride + WW]
            for k in range(F):
                if i == th and j == tw and k == tc:
                    flag = True
                else:
                    flag = False
                out[:, k, i, j] = my_dot(x_pad_masked, w[k, :, :, :], flag)
                # out[:, k , i, j] = np.sum(x_pad_masked * w[k, :, :, :], axis=(1,2,3))

    out = out + (b)[None, :, None, None]
    out = quantize_arr(out, 2**6, 2**7)

    cache = (x, w, b, conv_param)
    return out, cache


# data_file = './simulator/blob-data.txt'
# weight_file = './simulator/conv1-weight.txt'
# bias_file = './simulator/conv1-bias.txt'
# out_file = './simulator/blob-conv1.txt'

data_file = './simulator_res18/blob-data.txt'
weight_file = './simulator_res18/conv1-weight.txt'
bias_file = './simulator_res18/conv1-bias.txt'
out_file = './simulator_res18/blob-conv1.txt'


data = read_data(data_file)
w = read_weight(weight_file)
b = read_bias(bias_file)
out = read_out(out_file)

print data.shape
print w.shape
print b.shape
print out.shape

data = quantize_arr(data, 2**7, 2**7)

# conv_param = {'stride': 2, 'pad': 3}
conv_param = {'stride': stride, 'pad': pad}


# conv_param = {'stride': 2, 'pad': 0}
check, _ = conv_forward_naive(data, w, b, conv_param)

print check.shape
assert check.shape == out.shape

err = np.abs(check - out)
print 'mean err: ', np.mean(err)
print 'max err: ', np.max(err)

check = check.reshape((-1, 112))
out = out.reshape(check.shape)
print(check[6782, 51])
print(out[6782, 51])