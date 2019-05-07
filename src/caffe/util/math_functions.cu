//#include <math_functions.h> // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <string.h>
#include <ctime>

#define N_MAX 7
#define ERROR_LENGTH 0
//#define PRINT_Q
typedef int QCODE_TYPE;
int file_index = 0;

namespace caffe {

// random float kernal
__device__ void randomGenerator(float *data, unsigned long long seed) {
  curandState state;
  curand_init(seed, 0, 0, &state);
  *data = curand_uniform(&state);
}

// random double kernal
__device__ void randomGenerator(double *data, unsigned long long seed) {
  curandState state;
  curand_init(seed, 0, 0, &state);
  *data = abs(curand_uniform_double(&state));
}

// the Q_code funtion with stochastic rounding
__device__ float Q_code(
    float data, int q2, int qmax,
    unsigned long long seed) {            // q2 = pow(2,q)  qmax = pow(2,max)
  float integer_data = floor(data * q2);  // the integer part of A[tid] while q2
  float float_data =
      integer_data /
      q2;  // the float format of the integer part of A[tid] while q2
  float rand_data;
  float p;
  if (integer_data >= qmax - 1) {
    return float(qmax - 1) / q2;
  } else if (integer_data <= -qmax) {
    return float(-(qmax)) / q2;
  } else {
    // rand_data= float(rand()) / float(RAND_MAX + 1.0);
    randomGenerator(&rand_data, seed);
    p = (data - float_data) * q2;
    return (rand_data >= p ? float_data : 1.0 / q2 + float_data);
  }
}

__device__ double Q_code(double data, int q2, int qmax,
                         unsigned long long seed) {  // q2 = pow(2,q)
  double integer_data =
      floor(data * q2);  // the integer part of A[tid] while q2
  double float_data =
      integer_data /
      q2;  // the float format of the integer part of A[tid] while q2
  double rand_data;
  double p;
  if (integer_data >= qmax - 1) {
    return double(qmax - 1) / q2;
  } else if (integer_data <= -qmax) {
    return double(-(qmax)) / q2;
  } else {
    // rand_data= float(rand()) / float(RAND_MAX + 1.0);
    randomGenerator(&rand_data, seed);
    p = (data - float_data) * q2;
    return (rand_data >= p ? float_data : 1.0 / q2 + float_data);
  }
}

// kernal funtion, defined by __global__
__global__ void q_kernel(float *A, int q2, int N, int qmax) {
  int tid =
      threadIdx.x + blockIdx.x * blockDim.x;  // threadIdx the x-Index of thread
  unsigned long long seed = tid;
  float data;
  while (tid < N) {
    data = A[tid];
    A[tid] = Q_code(data, q2, qmax, seed);
    tid += blockDim.x *
           gridDim.x;  // blockDim the Dim of block, gridDim the Dim of grid
  }
}

__global__ void q_kernel(double *A, int q2, int N, int qmax) {
  int tid =
      threadIdx.x + blockIdx.x * blockDim.x;  // threadIdx the x-Index of thread
  unsigned long long seed = tid;
  double data;
  while (tid < N) {
    data = A[tid];
    A[tid] = Q_code(data, q2, qmax, seed);
    tid += blockDim.x *
           gridDim.x;  // blockDim the Dim of block, gridDim the Dim of grid
  }
}

// kernel fuction, used to caculate alpha*A*B+beta*C in fix point format
__global__ void mul_kernel(
    int cola, int rowc, int colc, const float alpha, float *A, float *B,
    const float beta, float *C, float aq2, float bq2, float sumq2, int qmaxab,
    int qmaxc,
    float
        cq2) {  // mul_kernel<<<block_c,thread_c>>>(cola,rowc,colc, alpha,
                // dev_a, dev_bT, beta,
                // C,pow(2,aq),pow(2,bq),pow(2,aq+bq),pow(2,7),pow(2,15),pow(2,cq));
  int sum = 0;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned long long seed = col;
  int a, b;
  int indexa, indexb, indexc;
  int cur_sum = 0;

  if (row < rowc && col < colc) {  // the thread(row,col) caculate the output
                                   // C(row,col)= A(row,:)*B(:,col)
    for (int i = 0; i < cola; ++i) {
      indexa = row * cola + i;
      indexb = i * colc + col;
      indexc = row * colc + col;
      a = Q_code(A[indexa], aq2, qmaxab, seed) * aq2;
      b = Q_code(B[indexb], bq2, qmaxab, seed) * bq2;
      cur_sum = a * b;
      sum += cur_sum;
      if (sum > qmaxc - 1) {
        sum = qmaxc - 1;
      } else if (sum < -qmaxc) {
        sum = -qmaxc;
      }
    }
    C[indexc] = Q_code(float(sum) / (aq2 * bq2), cq2, qmaxab, seed);
  }
}

/*
 mul_kernel<<<block_c, thread_c>>>(
      cola, rowc, colc, alpha, dev_a, dev_bT, beta, C, pow(2, aq), pow(2, bq),
      pow(2, aq + bq), pow(2, N_MAX), pow(2, (2 * N_MAX) + 1), pow(2, cq));
*/

// kernel fuction, used to caculate alpha*A*B+beta*C in fix point format
__global__ void mul_kernel(
    int cola, int rowc, int colc, const double alpha, double *A, double *B,
    const double beta, double *C, double aq2, double bq2, double sumq2,
    int qmaxab, int qmaxc,
    double
        cq2) {  // mul_kernel<<<block_c,thread_c>>>(cola,rowc,colc, alpha,
                // dev_a, dev_bT, beta,
                // C,pow(2,aq),pow(2,bq),pow(2,aq+bq),pow(2,7),pow(2,15),pow(2,cq));
  int sum = 0;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned long long seed = col;

  int a, b;
  int indexa, indexb, indexc;
  int cur_sum = 0;

  if (row < rowc && col < colc) {  // the thread(row,col) caculate the output
                                   // C(row,col)= A(row,:)*B(:,col)
    for (int i = 0; i < cola; ++i) {
      indexa = row * cola + i;
      indexb = i * colc + col;
      indexc = row * colc + col;
      a = Q_code(A[indexa], aq2, qmaxab, seed) * aq2;
      b = Q_code(B[indexb], bq2, qmaxab, seed) * bq2;
      cur_sum = a * b;

      sum += cur_sum;
      if (sum > qmaxc - 1) {
        sum = qmaxc - 1;
      } else if (sum < -qmaxc) {
        sum = -qmaxc;
      }
    }
    C[indexc] = Q_code(double(sum) / (aq2 * bq2), cq2, qmaxab, seed);
  }
}

template <>
void Qcode<float>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                  const int M, const int N, const int K, const float alpha,
                  const float *A, const float *B, const float beta, float *C,
                  float aq, float bq, float cq, bool isfc) {
  // size of A, B, C
  int rowa, cola;
  int rowb, colb;
  int rowc, colc;

  // max of A/B
  int qmax = pow(2, 7);  // qmax=2^7

  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;  // the length of row
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N
                               : CUBLAS_OP_T;  // whether to transpose or not
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  cola = lda;
  colb = ldb;
  colc = N;

  rowa = (TransA == CblasNoTrans) ? M : K;
  rowb = (TransB == CblasNoTrans) ? K : N;
  rowc = M;

  float *dev_a = 0;
  float *dev_b = 0;
  float *dev_bT = 0;

  // define block and thread
  dim3 block_c((colc + 32) / 32, (rowc + 32) / 32);
  dim3 thread_c(32, 32);
  dim3 block_b((colb + 32) / 32, (rowb + 32) / 32);
  dim3 thread_b(32, 32);

  // the params be used to transpose B
  float const alphab(1.0);
  float const betab(0.0);
  int colbT;

  // Allocate GPU buffers for three vectors (two input, one output)
  cudaError_t cudaStatus =
      cudaMalloc((void **)&dev_a, rowa * cola * sizeof(float));

  cudaStatus = cudaMalloc((void **)&dev_b, rowb * colb * sizeof(float));

  cudaStatus = cudaMalloc((void **)&dev_bT, rowb * colb * sizeof(float));


  // copy from const A to dev_a
  cublasScopy(Caffe::cublas_handle(), rowa * cola, A, 1, dev_a, 1);

  // if isfc = 1, transpose B
  if (isfc == true) {
    cublasScopy(Caffe::cublas_handle(), rowb * colb, B, 1, dev_b, 1);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, rowb, colb, &alphab, dev_b,
                colb, &betab, dev_b, rowb, dev_bT, rowb);
    cublasDestroy(handle);
  } else {
    cublasScopy(Caffe::cublas_handle(), rowb * colb, B, 1, dev_bT, 1);
  }

  // A data conversion
  if (aq >= 0) {
    q_kernel<<<(rowa * cola + 1024) / 1024, 1024>>>(dev_a, pow(2, aq),
                                                    rowa * cola, pow(2, N_MAX));
  }

  // B data conversion
  if (bq >= 0) {
    q_kernel<<<(rowb * colb + 1024) / 1024, 1024>>>(dev_bT, pow(2, bq),
                                                    rowb * colb, pow(2, N_MAX));
  }
  // caculate C
  mul_kernel<<<block_c, thread_c>>>(
      cola, rowc, colc, alpha, dev_a, dev_bT, beta, C, pow(2, aq), pow(2, bq),
      pow(2, aq + bq), pow(2, N_MAX), pow(2, (2 * N_MAX) + 1), pow(2, cq));

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_bT);
}

template <>
void Qcode<double>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                   const int M, const int N, const int K, const double alpha,
                   const double *A, const double *B, const double beta,
                   double *C, float aq, float bq, float cq, bool isfc) {
  int rowa, cola;
  int rowb, colb;
  int rowc, colc;

  int qmax = pow(2, 7);  // qmax=2^7

  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;  // the length of row
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N
                               : CUBLAS_OP_T;  // whether to transpose or not
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  cola = lda;
  colb = ldb;
  colc = N;

  rowa = (TransA == CblasNoTrans) ? M : K;
  rowb = (TransB == CblasNoTrans) ? K : N;
  rowc = M;

  double *dev_a = 0;
  double *dev_b = 0;
  double *dev_bT = 0;

  // define block and thread
  dim3 block_c((colc + 32) / 32, (rowc + 32) / 32);
  dim3 thread_c(32, 32);
  dim3 block_b((colb + 32) / 32, (rowb + 32) / 32);
  dim3 thread_b(32, 32);

  // the params be used to transpose B
  double const alphab(1.0);
  double const betab(0.0);
  int colbT;

  // Allocate GPU buffers for three vectors (two input, one output)
  cudaError_t cudaStatus =
      cudaMalloc((void **)&dev_a, rowa * cola * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc1 failed!");
    goto Error;
  }

  cudaStatus = cudaMalloc((void **)&dev_b, rowb * colb * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc2 failed!");
    goto Error;
  }

  cudaStatus = cudaMalloc((void **)&dev_bT, rowb * colb * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc2 failed!");
    goto Error;
  }

  // copy from const A to dev_a
  cublasDcopy(Caffe::cublas_handle(), rowa * cola, A, 1, dev_a, 1);

  // if isfc = 1, transpose B
  if (isfc == true) {
    cublasDcopy(Caffe::cublas_handle(), rowb * colb, B, 1, dev_b, 1);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, rowb, colb, &alphab, dev_b,
                colb, &betab, dev_b, rowb, dev_bT, rowb);
    cublasDestroy(handle);
  } else {
    cublasDcopy(Caffe::cublas_handle(), rowb * colb, B, 1, dev_bT, 1);
  }

  // A data conversion
  if (aq >= 0) {
    q_kernel<<<(rowa * cola + 1024) / 1024, 1024>>>(dev_a, pow(2, aq),
                                                    rowa * cola, pow(2, N_MAX));
  }
  // B data conversion
  if (bq >= 0) {
    q_kernel<<<(rowb * colb + 1024) / 1024, 1024>>>(dev_bT, pow(2, bq),
                                                    rowb * colb, pow(2, N_MAX));
  }
  // caculate C
  mul_kernel<<<block_c, thread_c>>>(
      cola, rowc, colc, alpha, dev_a, dev_bT, beta, C, pow(2, aq), pow(2, bq),
      pow(2, aq + bq), pow(2, N_MAX), pow(2, (2 * N_MAX) + 1), pow(2, cq));

#ifdef PRINT_Q
  // print out
  // memroy copy from GUP to CPU（cudaMencpy）
  cudaStatus = cudaMemcpy(QA, (void *)dev_a, rowa * cola * sizeof(double),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy1 failed!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(QB, (void *)dev_bT, rowb * colb * sizeof(double),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy2 failed!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(QC, (void *)C, rowc * colc * sizeof(double),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy2 failed!");
    goto Error;
  }

  file_index++;
  strstream << "-" << file_index << "-" << rowa << "x" << cola << ".txt";
  str = strstream.str();
  fA = fA + str;
  strstream.str("");
  data_file.open(fA.c_str(), ios::app);

  if (data_file.is_open()) {
    std::cout << "mathfunction.cpp:" << fA << " open successful!!\n";
  } else {
    std::cout << "mathfunction.cpp:" << fA << " open failed!!\n";
    exit(0);
  }
  // float data;
  for (int ai = 0; ai < rowa; ai++) {
    for (int aj = 0; aj < cola; aj++) {
      strstream.precision(8);
      strstream << QA[ai * cola + aj] << " ";
    }
    strstream << std::endl;
    str.clear();
    str = strstream.str();
    data_file.write(str.c_str(), str.length());
    strstream.str("");
  }
  data_file.close();

  if (isfc == true) {
    colbT = colb;
    colb = rowb;
    rowb = colbT;
  }

  strstream << "-" << file_index << "-" << rowb << "x" << colb << ".txt";
  str = strstream.str();
  fB = fB + str;
  strstream.str("");
  data_file.open(fB.c_str(), ios::app);
  if (data_file.is_open()) {
    std::cout << "mathfunction.cpp:" << fB << " open successful!!\n";
  } else {
    std::cout << "mathfunction.cpp:" << fB << " open failed!!\n";
    exit(0);
  }
  // float data;
  for (int bi = 0; bi < rowb; bi++) {
    for (int bj = 0; bj < colb; bj++) {
      strstream.precision(8);
      strstream << QB[bi * colb + bj] << " ";
    }
    strstream << std::endl;
    str.clear();
    str = strstream.str();
    data_file.write(str.c_str(), str.length());
    strstream.str("");
  }
  data_file.close();

  strstream << "-" << file_index << "-" << rowc << "x" << colc << ".txt";
  str = strstream.str();
  fC = fC + str;
  strstream.str("");
  data_file.open(fC.c_str(), ios::app);
  if (data_file.is_open()) {
    std::cout << "mathfunction.cpp:" << fC << " open successful!!\n";
  } else {
    std::cout << "mathfunction.cpp:" << fC << " open failed!!\n";
    exit(0);
  }
  // float data;
  for (int ci = 0; ci < rowc; ci++) {
    for (int cj = 0; cj < colc; cj++) {
      strstream.precision(8);
      strstream << QC[ci * colc + cj] << " ";
    }
    strstream << std::endl;
    str.clear();
    str = strstream.str();
    data_file.write(str.c_str(), str.length());
    strstream.str("");
  }
  data_file.close();
#endif

Error:
#ifdef PRINT_Q
  delete[] QC;  // delete
  delete[] QB;
  delete[] QA;
#endif  // release the memory on GPU
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_bT);
}
// gemm 的接口参数意义：
// A,B,C 是输入矩阵（一维数组格式）
// CblasRowMajor :数据是行主序的（二维数据也是用一维数组储存的）
// TransA, TransB：是否要对A和B做转置操作（CblasTrans CblasNoTrans）
// M： op(A)、C 的行数
// N： op(B)、C 的列数
// K： op(A) 的列数， op(B) 的行数
// lda ： A的列数（不做转置）行数（做转置）
// ldb： B的列数（不做转置）行数（做转置）
template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
                           const CBLAS_TRANSPOSE TransB, const int M,
                           const int N, const int K, const float alpha,
                           const float *A, const float *B, const float beta,
                           float *C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  // 我们想要计算 a * A * B + b * C
  // 但是blas默认是列优先存储，所以实际
  // 按照行优先的观点来看，blas看到的实际是AT和BT.
  // 所以我们实际要计算的是 CT = aBT*AT + b*CT
  // 所以你可以看到下式，transB 和 transA 调换了位置
  // B实际上成了第一个操作数，A变成了第二个操作数
  // 第3个参数（从0开始索引）是指number of rows of matrix op(A) and C.
  // 也就是BT的行数，也就是B的列数！
  // 第4个参数是指number of columns of matrix op(B) and C.
  // 也就是AT的列数，也就是A的行数！
  // 第5个参数是指number of columns of op(A) and rows of op(B).
  // 也就是BT的列数，也就是B的行数！

  // 后面又是lda和ldb。这个东西指的是leading dimension。也就是第一个维度的大小。
  // 或者是跨越的stride。
  // > The leading dimension always refers to the length of the first
  //   dimension of the array

  // 我们先考虑行优先的情况（因为倒一下就是列优先）。
  // 这时候lda就是A的列数。
  // 所以列优先的情况，就是A的行数。
  // 如果A转置了呢？那就是A的列数啦～
  // 所以 lda = A 转置？ colA : rowA
  // 从上面的讨论我们已经知道了，colA就是K，rowA就是M
  // 所以 lda = A 转置？ K : M;
  // 注意哦！我们的Ａ在内存里面已经是转置之后的了，所以
  // A转置这个条件实际对应的是　transA == NoTrans

  // 参考1：https://devtalk.nvidia.com/default/topic/766364/gpu-accelerated-libraries/row-major-matrix-to-row-major-matrix-multiplication-in-cublas/
  // 参考２:https://stackoverflow.com/questions/16376804/clarification-of-the-leading-dimension-in-cublas-when-transposing
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA, N, M, K,
                           &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
                            const CBLAS_TRANSPOSE TransB, const int M,
                            const int N, const int K, const double alpha,
                            const double *A, const double *B, const double beta,
                            double *C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA, N, M, K,
                           &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
                           const int N, const float alpha, const float *A,
                           const float *x, const float beta, float *y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha, A, N,
                           x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
                            const int N, const double alpha, const double *A,
                            const double *x, const double beta, double *y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha, A, N,
                           x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float *X,
                           float *Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double *X,
                            double *Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X,
                           cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float *X,
                            const float beta, float *Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double *X,
                             const double beta, double *Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float *x, const float *y,
                          float *out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double *x, const double *y,
                           double *out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float *x, float *y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double *x, double *y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float *y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double *y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = alpha; }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype *Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype>
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int *Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float *Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double *Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] += alpha; }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float *Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float>
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double *Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double>
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype *a, const Dtype *b,
                           Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = a[index] + b[index]; }
}

template <>
void caffe_gpu_add<float>(const int N, const float *a, const float *b,
                          float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float>
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double *a, const double *b,
                           double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double>
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype *a, const Dtype *b,
                           Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = a[index] - b[index]; }
}

template <>
void caffe_gpu_sub<float>(const int N, const float *a, const float *b,
                          float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float>
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double *a, const double *b,
                           double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double>
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype *a, const Dtype *b,
                           Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = a[index] * b[index]; }
}

template <>
void caffe_gpu_mul<float>(const int N, const float *a, const float *b,
                          float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float>
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double *a, const double *b,
                           double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double>
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype *a, const Dtype *b,
                           Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = a[index] / b[index]; }
}

template <>
void caffe_gpu_div<float>(const int N, const float *a, const float *b,
                          float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float>
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double *a, const double *b,
                           double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double>
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype *a, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = abs(a[index]); }
}

template <>
void caffe_gpu_abs<float>(const int N, const float *a, float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double *a, double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
}

template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype *a, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = exp(a[index]); }
}

template <>
void caffe_gpu_exp<float>(const int N, const float *a, float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double *a, double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype *a, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = log(a[index]); }
}

template <>
void caffe_gpu_log<float>(const int N, const float *a, float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double *a, double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype *a, const Dtype alpha,
                            Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = pow(a[index], alpha); }
}

template <>
void caffe_gpu_powx<float>(const int N, const float *a, const float alpha,
                           float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float>
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double *a, const double alpha,
                            double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double>
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype *a, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = sqrt(a[index]); }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float *a, float *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double *a, double *y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index]) -
                                                       (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int *r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float *r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double *r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float *r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double *r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
