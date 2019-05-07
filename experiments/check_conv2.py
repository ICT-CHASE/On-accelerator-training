import os
import numpy as np

# N, C, H, W = 1, 3, 224, 224

# Cout, K = 64, 7

N, C, H, W = 1, 3, 6, 4

Cout, K = 4, 3

# out: N Count, Hnew, Wnew
HH, WW = 2, 1

def read_data(file):
  """ Read data
  """
  with open(file, 'r') as f:
    lines = f.readlines()
  lines = [l.strip() for l in lines]
  r = len(lines)
  c = len(lines[0].split())
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


def conv_simple_naive(w, x, b):
  """
  A naive method to calculate the convolution of x and w, b
  Input:
  - x: Input data after padding of shape (C, N)
  - w: Filter weights of shape (F, C)
  - b: Biase, shape: (F, 1)
  Returns the result of matrix multiply w*x + b
  """
  il = 5
  fl = 20
  cmax = 2 ** (il + il - 1)

  f, c = w.shape
  n = x.shape[-1]

  res = np.empty(shape=(f, n))

  for i in range(f):
    for j in range(n):
      # res[i, j] = dot(w[i,:], x[:, j])
      sum = 0.
      for k in range(c):
        sum += w[i,k] * x[k,j]
        if sum > cmax:
          sum = cmax
        if sum < -cmax:
          sum = -cmax
      res[i, j] = sum
  
  prob = np.random.rand(*res.shape)
  for i in range(f):
    bi = b[i, 0]
    for j in range(n):
      res[i,j] += bi
      res[i,j] = quantize(res[i,j], 2**fl, 2**(il+fl-1), prob[i,j])
  
  return res

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
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  stride, pad = conv_param['stride'], conv_param['pad']
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride
  out = np.zeros((N , F , H_out, W_out))

  x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
  for i in range(H_out):
      for j in range(W_out):
          x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
          for k in range(F):
              out[:, k , i, j] = np.sum(x_pad_masked * w[k, :, :, :], axis=(1,2,3))
          #out[:, : , i, j] = np.sum(x_pad_masked * w[:, :, :, :], axis=(1,2,3))
          
  #for k in range(F):
      #out[:, k, :, :] = out[:, k, :, :] + b[k]
  out = out + (b)[None, :, None, None]
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


# def conv_forward_naive(x, w, b, conv_param):
#   """
#   A naive implementation of the forward pass for a convolutional layer.
#   The input consists of N data points, each with C channels, height H and width
#   W. We convolve each input with F different filters, where each filter spans
#   all C channels and has height HH and width HH.
#   Input:
#   - x: Input data of shape (N, C, H, W)
#   - w: Filter weights of shape (F, C, HH, WW)
#   - b: Biases, of shape (F,)
#   - conv_param: A dictionary with the following keys:
#     - 'stride': The number of pixels between adjacent receptive fields in the
#       horizontal and vertical directions.
#     - 'pad': The number of pixels that will be used to zero-pad the input.
#   Returns a tuple of:
#   - out: Output data, of shape (N, F, H', W') where H' and W' are given by
#     H' = 1 + (H + 2 * pad - HH) / stride
#     W' = 1 + (W + 2 * pad - WW) / stride
#   - cache: (x, w, b, conv_param)
#   """
#   out = None
#   #############################################################################
#   # TODO: Implement the convolutional forward pass.                           #
#   # Hint: you can use the function np.pad for padding.                        #
#   #############################################################################

#   stride = conv_param['stride']
#   pad = conv_param['pad']
#   (N, C, H, W) = x.shape
#   (F, CC, HH, WW) = w.shape
#   assert C == CC
#   H_ = 1 + (H + 2 * pad - HH) / stride
#   W_ = 1 + (W + 2 * pad - WW) / stride
#   out_dim = (N, F, H_, W_)
#   out = np.zeros(out_dim, dtype = x.dtype)
#   # padding the input
#   x_pad = np.lib.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), \
#   'constant', constant_values = 0)
#   # [delta_H, HH-delta_H + center_y) is the desired x which
#   # will convolute with the weight
#   delta_H = (HH - 1) / 2   
#   delta_W = (WW - 1) / 2

#   x_conv = np.zeros((HH * WW * C, H_ * W_))
#   # for each input image
#   for n in xrange(N):
#     cnt_win = 0   # the number of windows that have done convolution
#     for i in xrange(H_):
#       for j in xrange(W_):
#         center = (i * stride + pad, j * stride + pad)
#         x_conv[:, cnt_win] = x_pad[n, :, \
#         center[0] - delta_H: HH - delta_H + center[0], \
#         center[1] - delta_W: WW - delta_W + center[1]].\
#         reshape((-1, 1)).squeeze()
#         cnt_win += 1
#     # res_conv = np.dot(w.reshape(F, -1), x_conv) + b.reshape(F, -1)  
#     res_conv = conv_simple_naive(w.reshape(F, -1), x_conv, b.reshape(F, -1))
#     out[n, :, :, :] = res_conv.reshape((F, H_, W_)) 
  
#   #############################################################################
#   #                             END OF YOUR CODE                              #
#   #############################################################################
#   # Add an assertion  
#   assert out_dim == out.shape
#   cache = (x, w, b, conv_param)
#   return out, cache

# data_file = './simulator/blob-data.txt'
# weight_file = './simulator/conv1-weight.txt'
# bias_file = './simulator/conv1-bias.txt'
# out_file = './simulator/blob-conv1.txt'

data_file = "./simulator/test-data.txt"
weight_file = "./simulator/test-w.txt"
bias_file = "./simulator/test-bias.txt"
out_file = "./simulator/test-out.txt"

data = read_data(data_file)
w = read_weight(weight_file)
b = read_bias(bias_file)
out = read_out(out_file)

print data.shape
print w.shape
print b.shape
print out.shape

conv_param = {'stride': 2, 'pad': 0}
check, _ = conv_forward_naive(data, w, b, conv_param)

print check.shape
assert check.shape == out.shape

err = np.abs(check - out)
print 'mean err: ', np.mean(err)
print 'max err: ', np.max(err)

with open('./simulator/check-conv1.txt', 'w') as f:
    reshaped = check.reshape((-1, check.shape[-1]))
    r, c = reshaped.shape
    for i in range(r):
        d = [str(x) for x in reshaped[i]]
        f.write(' '.join(d))
        f.write('\n')
