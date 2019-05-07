#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>
#include <limits.h>
#include <cmath>
#include <ctime>
#include "caffe/common.hpp"
#include "caffe/fixpoint/math_functions.hpp"

//#define SR   // stochastic rounding
namespace caffe {
template <typename Dtype>
inline void thres(Dtype* data, int min, int max) {
  if (*data < min) *data = min;
  if (*data > max) *data = max;
}

inline int get_q(const QCodeConfig& config) { return 1 << config.fl(); }

inline float get_qmax(const QCodeConfig& config) {
  return 1LL << (config.il() + config.fl() - 1);
}

// 将浮点数随机round为定点表示。
// 依概率round到ceil(x)或者floor(x)
// q --> 2^FL
// qmax --> 2^{IL+FL-1}
template <typename Dtype>
__device__ Dtype StochasticRoundingGPU(Dtype data, int q, float qmax, Dtype p) {
  Dtype fp_float = data * q;
  Dtype fp_int = floor(fp_float);
  // 获取小数部分
  Dtype residual = fp_float - fp_int;
  // 依概率 p round 到\floor(x) + \episilon
  // if (residual > p) fp_int = ceil(fp_float);

  if (residual > 0.5) fp_int = ceil(fp_float);
  // 饱和
  if (fp_int < -qmax) fp_int = -qmax;
  if (fp_int >= qmax) fp_int = qmax - 1;

  return fp_int / q;
}

// round to nearest
template <typename Dtype>
__device__ Dtype RoundToNearestGPU(Dtype data, int q, float qmax) {
  return StochasticRoundingGPU(data, q, qmax, 0.5);
}

// 对数组 data[0...N-1]做随机rounding
template <typename Dtype>
__global__ void StochasticRoundingKernel(const Dtype* input, Dtype* output,
                                         int N, int q, float qmax,
                                         const Dtype* prob) {
  CUDA_KERNEL_LOOP(idx, N) {
    output[idx] = StochasticRoundingGPU(input[idx], q, qmax, prob[idx]);
  }
}

template <typename Dtype>
void Quantize(const int N, const Dtype* input, Dtype* output, const Dtype* prob,
              const QCodeConfig& config) {
  StochasticRoundingKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      input, output, N, get_q(config), get_qmax(config), prob);
}

template void Quantize<float>(const int N, const float* input, float* output,
                              const float* prob, const QCodeConfig& config);

template void Quantize<double>(const int N, const double* input, double* output,
                               const double* prob, const QCodeConfig& config);

// fix point版本的 C = alpha * A * B + beta * C
template <typename Dtype>
__global__ void saturated_mul_kernel(const Dtype* A, const Dtype* B, Dtype* C,
                                     const int rowc, const int colc,
                                     const int rowa, const int cola,
                                     const int rowb, const int colb,
                                     const Dtype alpha, const Dtype beta,
                                     const Dtype cmax,  // cmax = qmax / 2^fl
                                     bool transA, bool transB,
                                     const int nthreads) {
  // for each thread, compute
  // dot(A[row,:], B[:, col] * alpha) + beta * C[row, col]
  CUDA_KERNEL_LOOP(idx, nthreads) {
    // r * colsc + c = idx
    int r = idx / colc;
    int c = idx - colc * r;

    int ca = transA ? rowa : cola;

    Dtype sum = 0.;
    int ia, ib;
    for (int i = 0; i < ca; ++i) {
      if (!transA) {
        ia = r * cola + i;  // A[r, i]
      } else {
        ia = i * cola + r;  // AT[r, i] --> A[i, r]
      }
      if (!transB) {
        ib = i * colb + c;  // B[i, c]
      } else {
        ib = c * colb + i;  // BT[i, c] --> B[c, i]
      }
      sum += A[ia] * B[ib];
      if (sum < -cmax) {
        sum = -cmax;
      } else if (sum > cmax) {
        sum = cmax;
      }
    }
    C[idx] = alpha * sum + beta * C[idx];
  }
}

// fix point版本的 y = alpha * A * x + beta * y
// 其中A是一个矩阵，x和y是向量

void naive_cublas_transpose(const float* A, float* C, const int rowsA,
                            const int colsA) {
  const float alpha_temp = 1.;
  const float beta_temp = 0.;
  CUBLAS_CHECK(cublasSgeam(Caffe::cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                           rowsA, colsA, &alpha_temp, A, colsA, &beta_temp, C,
                           rowsA, C, rowsA));
}

void naive_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                const int rowsA, const int colsA, const int rowsB,
                const int colsB, const int rowsC, const int colsC,
                const float alpha, const float* A, const float* B,
                const float beta, float* C, const float cmax) {
  const int nthreads = rowsC * colsC;
  bool transA = TransA == CblasTrans;
  bool transB = TransB == CblasTrans;
  saturated_mul_kernel<<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      A, B, C, rowsC, colsC, rowsA, colsA, rowsB, colsB, alpha, beta, cmax,
      transA, transB, nthreads);
  CUDA_POST_KERNEL_CHECK;
}

void naive_gemv(const CBLAS_TRANSPOSE TransA, const int rowsA, const int colsA,
                const float alpha, const float* A, const float* X,
                const float beta, float* y, const float ymax) {
  const int nthreads = rowsA;
  bool transA = TransA == CblasTrans;
  bool transB = false;
  const int rowsC = transA ? colsA : rowsA;
  saturated_mul_kernel<<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      A, X, y, rowsC, 1, rowsA, colsA, colsA, 1, alpha, beta, ymax, transA,
      transB, nthreads);
  CUDA_POST_KERNEL_CHECK;
}

// gemm 的接口参数意义：
// A,B,C 是输入矩阵（一维数组格式）
// CblasRowMajor :数据是行主序的（二维数据也是用一维数组储存的）
// TransA, TransB：是否要对A和B做转置操作（CblasTrans CblasNoTrans）
// M： op(A)、C 的行数
// N： op(B)、C 的列数
// K： op(A) 的列数， op(B) 的行数
// lda： A的列数（不做转置）行数（做转置）
// ldb： B的列数（不做转置）行数（做转置）

void gpu_gemm_float(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                    const int M, const int N, const int K, const float alpha,
                    const float* A, const float* B, const float beta,
                    float* C) {
  const float cmax = FLT_MAX;
  const int rowsA = TransA == CblasNoTrans ? M : K;
  const int colsA = TransA == CblasNoTrans ? K : M;
  const int rowsB = TransB == CblasNoTrans ? K : N;
  const int colsB = TransB == CblasNoTrans ? N : K;

  const int rowsC = M;
  const int colsC = N;

  naive_gemm(TransA, TransB, rowsA, colsA, rowsB, colsB, rowsC, colsC, alpha, A,
             B, beta, C, cmax);
}

template <typename Dtype>
__global__ void AxpyKernel(const int N, const Dtype* x, Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(idx, N) { y[idx] += x[idx] * alpha; }
}

void gpu_axpy_fp(const int N, const float alpha, const float* X, float* Y) {
  AxpyKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, X, alpha, Y);
  CUDA_POST_KERNEL_CHECK;
}

void gpu_gemv_fp(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float* A, const float* x,
                 const float beta, float* y, const int ymax) {
  const int rowsA = TransA == CblasNoTrans ? M : N;
  const int colsA = TransA == CblasNoTrans ? N : M;

  naive_gemv(TransA, rowsA, colsA, alpha, A, x, beta, y, ymax);
}

//　定点化的gemm
void gpu_gemm_fp(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 const int M, const int N, const int K, const float alpha,
                 const float* A, const float* B, const float beta, float* C,
                 const int cmax) {
  const int rowsA = TransA == CblasNoTrans ? M : K;
  const int colsA = TransA == CblasNoTrans ? K : M;
  const int rowsB = TransB == CblasNoTrans ? K : N;
  const int colsB = TransB == CblasNoTrans ? N : K;

  const int rowsC = M;
  const int colsC = N;

  naive_gemm(TransA, TransB, rowsA, colsA, rowsB, colsB, rowsC, colsC, alpha, A,
             B, beta, C, cmax);
}
}  // namespace caffe