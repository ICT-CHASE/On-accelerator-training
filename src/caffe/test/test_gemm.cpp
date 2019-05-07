#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/fixpoint/math_functions.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

const int ra = 5;
const int ca = 2;

const int rb = ca;
const int cb = 3;

void init_blob(Blob<float>* b) {
  float* data = b->mutable_cpu_data();
  for (int i = 0; i < b->count(); ++i) {
    data[i] = i;
  }
}

class GEMMTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    vector<int> shape(2, 1);
    shape[0] = ra;
    shape[1] = ca;
    ba = new Blob<float>(shape);
    shape[0] = rb;
    shape[1] = cb;
    bb = new Blob<float>(shape);

    init_blob(ba);
    init_blob(bb);
  }

  virtual void TearDown() {
    delete ba;
    delete bb;
  }

  Blob<float>* ba;
  Blob<float>* bb;
};

// copy from src to dst, but do transpose
template <typename Dtype>
void transpose(const Dtype* src, Dtype* dst, const int rows, const int cols) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const int src_offset = r * cols + c;
      const int dst_offset = c * rows + r;
      dst[dst_offset] = src[src_offset];
    }
  }
}

// create empty blob of desired shape (row, col)
template <typename Dtype>
Blob<Dtype>* create_empty_blob(const int rows, const int cols) {
  vector<int> shape(2, rows);
  shape[1] = cols;
  return new Blob<Dtype>(shape);
}

void print_blob(const Blob<float>* b) {
  int rows = b->shape(0);
  int cols = b->shape(1);
  const float* ptr = b->cpu_data();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << *ptr++ << "\t";
    }
    std::cout << std::endl;
  }
}

// ----------------- basic operations test ------------

// 1. transpose op
TEST_F(GEMMTest, transpose) {
  // temp is A^T
  Blob<float> temp;
  vector<int> shape(2, ba->shape(0));
  shape[0] = ba->shape(1);
  temp.Reshape(shape);

  //   std::cout << "a blob: \n";
  //   print_blob(ba);
  //   std::cout << "----------\n";

  transpose(ba->cpu_data(), temp.mutable_cpu_data(), ra, ca);

  //   std::cout << "temp blob: \n";
  //   print_blob(&temp);
  //   std::cout << "----------\n";

  Blob<float>* bc = create_empty_blob<float>(ca, ra);
  naive_cublas_transpose(ba->gpu_data(), bc->mutable_gpu_data(), ra, ca);

  //   std::cout << "c blob: \n";
  //   print_blob(bc);
  //   std::cout << "----------\n";

  for (int i = 0; i < temp.count(); ++i) {
    EXPECT_NEAR(bc->cpu_data()[i], temp.cpu_data()[i], 1E-2);
  }
  delete bc;
}

template <typename Dtype>
void mat_mul(const Dtype* A, const Dtype* B, Dtype* C, const int rowsC,
             const int colsC, const int rowsA, const int colsA, const int rowsB,
             const int colsB, const Dtype alpha, const Dtype beta, int qmax,
             bool transA, bool transB) {
  int ca = transA ? rowsA : colsA;

  int ia = 0;
  int ib = 0;
  for (int r = 0; r < rowsC; ++r) {
    for (int c = 0; c < colsC; ++c) {
      // compute C[r,c]=\alpha*dot(A[r,:], B[:,c])+\beta * C[r,c]
      Dtype dot = 0.;
      for (int i = 0; i < ca; ++i) {
        if (!transA) {
          ia = r * colsA + i;  // A[r, i]
        } else {
          ia = i * colsA + r;  // AT[r, i] --> A[i, r]
        }
        if (!transB) {
          ib = i * colsB + c;  // B[i, c]
        } else {
          ib = c * colsB + i;  // BT[i, c] --> B[c, i]
        }
        dot += A[ia] * B[ib];
        if (dot < -qmax)
          dot = -qmax;
        else if (dot > qmax - 1)
          dot = qmax - 1;
      }
      int ic = r * colsC + c;
      C[ic] = alpha * dot + beta * C[ic];
    }
  }
}

// naive gemm without transpose
TEST_F(GEMMTest, gemm_no_transpose) {
  const int rc = ra;
  const int cc = cb;

  Blob<float>* check = create_empty_blob<float>(rc, cc);

  const float alpha = 1.;
  const float beta = 0.;

  const int qmax = 1 << 16;

  CBLAS_TRANSPOSE TransA = CblasNoTrans;
  CBLAS_TRANSPOSE TransB = CblasNoTrans;

  mat_mul(ba->cpu_data(), bb->cpu_data(), check->mutable_cpu_data(), rc, cc, ra,
          ca, rb, cb, alpha, beta, qmax, false, false);

  Blob<float>* bc = create_empty_blob<float>(rc, cc);

  naive_gemm(TransA, TransB, ra, ca, rb, cb, rc, cc, alpha, ba->gpu_data(),
             bb->gpu_data(), beta, bc->mutable_gpu_data(), qmax);
  for (int i = 0; i < check->count(); ++i) {
    EXPECT_NEAR(bc->cpu_data()[i], check->cpu_data()[i], 1E-2);
  }

  delete check;
  delete bc;
}

// naive gemm with tranpose of A
TEST_F(GEMMTest, gemm_transposeA) {
  // create A^T
  Blob<float>* ta = create_empty_blob<float>(ca, ra);
  init_blob(ta);

  const int rc = ra;
  const int cc = cb;

  Blob<float>* check = create_empty_blob<float>(rc, cc);

  const float alpha = 1.;
  const float beta = 0.;

  const int qmax = 1 << 16;

  CBLAS_TRANSPOSE TransA = CblasTrans;
  CBLAS_TRANSPOSE TransB = CblasNoTrans;

  mat_mul(ta->cpu_data(), bb->cpu_data(), check->mutable_cpu_data(), rc, cc, ca,
          ra, rb, cb, alpha, beta, qmax, TransA == CblasTrans,
          TransB == CblasTrans);

  Blob<float>* bc = create_empty_blob<float>(rc, cc);

  naive_gemm(TransA, TransB, ca, ra, rb, cb, rc, cc, alpha, ta->gpu_data(),
             bb->gpu_data(), beta, bc->mutable_gpu_data(), qmax);
  for (int i = 0; i < check->count(); ++i) {
    EXPECT_NEAR(bc->cpu_data()[i], check->cpu_data()[i], 1E-2);
  }
  delete ta;
  delete check;
  delete bc;
}

// ----------------------------- GEMM TEST --------------

// gemm without transpose
TEST_F(GEMMTest, gemm_float) {
  const int rc = ra;
  const int cc = cb;
  Blob<float>* check = create_empty_blob<float>(rc, cc);

  const float alpha = 1.;
  const float beta = 0.;

  CBLAS_TRANSPOSE TransA = CblasNoTrans;
  CBLAS_TRANSPOSE TransB = CblasNoTrans;

  caffe_gpu_gemm(TransA, TransB, ra, cc, ca, alpha, ba->gpu_data(),
                 bb->gpu_data(), beta, check->mutable_gpu_data());

  Blob<float>* bc = create_empty_blob<float>(rc, cc);

  gpu_gemm_float(TransA, TransB, ra, cc, ca, alpha, ba->gpu_data(),
                 bb->gpu_data(), beta, bc->mutable_gpu_data());

  for (int i = 0; i < bc->count(); ++i) {
    EXPECT_NEAR(bc->cpu_data()[i], check->cpu_data()[i], 1E-2);
  }

  delete check;
  delete bc;
}

// gemm with tanspose of A
TEST_F(GEMMTest, gemm_float_transpose_a) {
  Blob<float>* ta = create_empty_blob<float>(ca, ra);
  init_blob(ta);

  const int rc = ra;
  const int cc = cb;
  Blob<float>* check = create_empty_blob<float>(rc, cc);

  const float alpha = 1.;
  const float beta = 0.;

  CBLAS_TRANSPOSE TransA = CblasTrans;
  CBLAS_TRANSPOSE TransB = CblasNoTrans;

  // std::cout << "ta blob: \n";
  // print_blob(ta);
  // std::cout << "----------\n";

  // std::cout << "b blob: \n";
  // print_blob(bb);
  // std::cout << "----------\n";

  caffe_gpu_gemm(TransA, TransB, ra, cc, ca, alpha, ta->gpu_data(),
                 bb->gpu_data(), beta, check->mutable_gpu_data());

  // std::cout << "check blob: \n";
  // print_blob(check);
  // std::cout << "----------\n";

  Blob<float>* bc = create_empty_blob<float>(rc, cc);

  gpu_gemm_float(TransA, TransB, ra, cc, ca, alpha, ta->gpu_data(),
                 bb->gpu_data(), beta, bc->mutable_gpu_data());

  // std::cout << "c blob: \n";
  // print_blob(bc);
  // std::cout << "----------\n";

  for (int i = 0; i < bc->count(); ++i) {
    EXPECT_NEAR(bc->cpu_data()[i], check->cpu_data()[i], 1E-2);
  }

  delete check;
  delete ta;
  delete bc;
}

// gemm with tanspose of A
TEST_F(GEMMTest, gemm_float_transpose_b) {
  Blob<float>* tb = create_empty_blob<float>(cb, rb);
  init_blob(tb);

  const int rc = ra;
  const int cc = cb;
  Blob<float>* check = create_empty_blob<float>(rc, cc);

  const float alpha = 1.;
  const float beta = 0.;

  CBLAS_TRANSPOSE TransA = CblasNoTrans;
  CBLAS_TRANSPOSE TransB = CblasTrans;

  caffe_gpu_gemm(TransA, TransB, ra, cc, ca, alpha, ba->gpu_data(),
                 tb->gpu_data(), beta, check->mutable_gpu_data());

  Blob<float>* bc = create_empty_blob<float>(rc, cc);

  gpu_gemm_float(TransA, TransB, ra, cc, ca, alpha, ba->gpu_data(),
                 tb->gpu_data(), beta, bc->mutable_gpu_data());

  for (int i = 0; i < bc->count(); ++i) {
    EXPECT_NEAR(bc->cpu_data()[i], check->cpu_data()[i], 1E-2);
  }

  delete check;
  delete tb;
  delete bc;
}

// gemm with tanspose of A
TEST_F(GEMMTest, gemm_float_transpose_ab) {
  Blob<float>* ta = create_empty_blob<float>(ca, ra);
  init_blob(ta);

  Blob<float>* tb = create_empty_blob<float>(cb, rb);
  init_blob(tb);

  const int rc = ra;
  const int cc = cb;
  Blob<float>* check = create_empty_blob<float>(rc, cc);

  const float alpha = 1.;
  const float beta = 0.;

  CBLAS_TRANSPOSE TransA = CblasTrans;
  CBLAS_TRANSPOSE TransB = CblasTrans;

  caffe_gpu_gemm(TransA, TransB, ra, cc, ca, alpha, ta->gpu_data(),
                 tb->gpu_data(), beta, check->mutable_gpu_data());

  Blob<float>* bc = create_empty_blob<float>(rc, cc);

  gpu_gemm_float(TransA, TransB, ra, cc, ca, alpha, ta->gpu_data(),
                 tb->gpu_data(), beta, bc->mutable_gpu_data());

  for (int i = 0; i < bc->count(); ++i) {
    EXPECT_NEAR(bc->cpu_data()[i], check->cpu_data()[i], 1E-2);
  }

  delete check;
  delete ta;
  delete tb;
  delete bc;
}

// gemm with tanspose of A
TEST_F(GEMMTest, gemm_float_alpha_beta) {
  const int rc = ra;
  const int cc = cb;
  Blob<float>* check = create_empty_blob<float>(rc, cc);

  init_blob(check);

  const float alpha = 3.;
  const float beta = 1.;

  CBLAS_TRANSPOSE TransA = CblasNoTrans;
  CBLAS_TRANSPOSE TransB = CblasNoTrans;

  caffe_gpu_gemm(TransA, TransB, ra, cc, ca, alpha, ba->gpu_data(),
                 bb->gpu_data(), beta, check->mutable_gpu_data());

  Blob<float>* bc = create_empty_blob<float>(rc, cc);
  init_blob(bc);

  gpu_gemm_float(TransA, TransB, ra, cc, ca, alpha, ba->gpu_data(),
                 bb->gpu_data(), beta, bc->mutable_gpu_data());

  for (int i = 0; i < bc->count(); ++i) {
    EXPECT_NEAR(bc->cpu_data()[i], check->cpu_data()[i], 1E-2);
  }

  delete check;
  delete bc;
}
}  // namespace caffe