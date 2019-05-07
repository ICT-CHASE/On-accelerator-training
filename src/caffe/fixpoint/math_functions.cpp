// dummy
#include "caffe/fixpoint/math_functions.hpp"
#include "caffe/common.hpp"
#include <cmath>   // floor, ceil
#include <cstdlib> // rand
#include <iostream>

namespace caffe {
template <typename Dtype>
Dtype StochasticRounding(Dtype data, int q, int qmax) {
  Dtype fp_float = data * q;
  int fp_int = floor(fp_float);
  // 获取小数部分
  Dtype residual = fp_float - fp_int;
  CHECK_GE(residual, 0) << "residual part should be greater than 0";
  CHECK_LE(residual, 1) << "residual part should be less than 1";

  Dtype p = rand() / static_cast<Dtype>(RAND_MAX);
  // 依概率 p round 到\floor(x) + \episilon
  fp_int = residual < p ? fp_int : ceil(fp_float);
  // 饱和
  if (fp_int < -qmax)
    fp_int = -qmax;
  if (fp_int > qmax - 1)
    fp_int = qmax - 1;

  return static_cast<Dtype>(fp_int) / q;
}

template float StochasticRounding<float>(float data, int q, int qmax);

template <typename Dtype> 
Dtype RoundToNearest(Dtype data, int q, int qmax) {
  Dtype fp_float = data * q;
  int fp_int = floor(fp_float);
  Dtype residual = fp_float - fp_int;
  CHECK_GE(residual, 0) << "residual part should be greater than 0";
  CHECK_LE(residual, 1) << "residual part should be less than 1";

  fp_int = residual < 0.5 ? fp_int : ceil(fp_float);
  if (fp_int < -qmax)
    fp_int = -qmax;
  if (fp_int > qmax - 1)
    fp_int = qmax - 1;
  return static_cast<Dtype>(fp_int) / q;
}

template float RoundToNearest(float data, int q, int qmax);

} // namespace caffe