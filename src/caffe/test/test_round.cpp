#include "gtest/gtest.h"
#include <cmath>
#include <cstring>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"

#include "caffe/fixpoint/math_functions.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
vector<Dtype> GenerateAllPossibleNumbers(int fl, int il) {
  Dtype eps = 1. / static_cast<Dtype>(1 << fl);
  Dtype min = -static_cast<Dtype>(1 << (il - 1));
  // 能够表示的定点数的数量
  int num = 1 << (il + fl);
  vector<Dtype> range(num);
  range[0] = min;
  for (int i = 1; i <= num; ++i) {
    range[i] = min + i * eps;
  }
  return range;
}

template <typename Dtype>
Dtype FindNearest(const vector<Dtype> &range, Dtype data) {
  if (data < range[0])
    return range[0];
  if (data > range[range.size() - 1])
    return range[range.size() - 1];

  int idx = -1;
  Dtype delta = 10000;
  for (int i = 0; i < range.size(); ++i) {
    Dtype d = fabs(data - range[i]);
    if (d < delta) {
      delta = d;
      idx = i;
    }
  }
  return range[idx];
}

TEST(RoundingTest, NearestRoundTest) {
  // Q 8.4
  // X X X X. X X X X
  int il = 4;
  int fl = 4;

  vector<float> range = GenerateAllPossibleNumbers<float>(fl, il);
  float min = -(1 << (il - 1));
  float max = (1 << (il - 1)) - 1. / (1 << fl);

  for (int i = 0; i < range.size(); ++i) {
    ASSERT_LE(range[i], max);
    ASSERT_GE(range[i], min);
  }

  float tol = 0.01 / (1 << fl);
  float a1 = 2.8;

  float ra1 = RoundToNearest(a1, 1 << fl, 1 << (il + fl - 1));
  float ea1 = FindNearest<float>(range, a1);

  EXPECT_NEAR(ra1, ea1, tol);

  float a2 = 1 << il;

  float ra2 = RoundToNearest(a2, 1 << fl, 1 << (il + fl - 1));
  float ea2 = FindNearest<float>(range, a2);

  EXPECT_NEAR(ra2, ea2, tol);

  float a3 = -(1 << il);

  float ra3 = RoundToNearest(a3, 1 << fl, 1 << (il + fl - 1));
  float ea3 = FindNearest<float>(range, a3);

  EXPECT_NEAR(ra3, ea3, tol);

  float a4 = -3.645;

  float ra4 = RoundToNearest(a4, 1 << fl, 1 << (il + fl - 1));
  float ea4 = FindNearest<float>(range, a4);

  EXPECT_NEAR(ra4, ea4, tol);
}

} // namespace caffe