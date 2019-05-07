#ifndef CAFFE_FIXPOINT_MATHFUNCTIONS_HPP_
#define CAFFE_FIXPOINT_MATHFUNCTIONS_HPP_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

typedef unsigned long long seed_type;

template <typename Dtype>
Dtype StochasticRounding(Dtype data, int q, int qmax);

template <typename Dtype>
Dtype RoundToNearest(Dtype data, int q, int qmax);

// 一个naive的cublas transpose实现
void naive_cublas_transpose(const float* A, float* C, const int rowsA,
                            const int colsA);

// 为了方便理解接口而写的gemm，没有做定点化，被gpu_gemm_XX调用
void naive_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                const int rowsA, const int colsA, const int rowsB,
                const int colsB, const int rowsC, const int colsC,
                const float alpha, const float* A, const float* B,
                const float beta, float* C, const float cmax);

// 浮点的gemm，与cublas对照，for debugging
void gpu_gemm_float(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                    const int M, const int N, const int K, const float alpha,
                    const float* A, const float* B, const float beta, float* C);

// 定点的gemv
void gpu_gemv_fp(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float* A, const float* x,
                 const float beta, float* y, const int ymax);
// 定点的gemm     
// = A/B定点化 --> naive_gemm --> C的定点化   
void gpu_gemm_fp(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 const int M, const int N, const int K, const float alpha,
                 const float* A, const float* B, const float beta, float* C,
                 const int cmax);

void gpu_axpy_fp(const int N, const float alpha, const float* X, float* Y);

template <typename Dtype>
void Quantize(const int N, const Dtype* input, Dtype* output, const Dtype* prob,
              const QCodeConfig& config);
}  // namespace caffe

#endif