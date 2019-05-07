#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/fixpoint/inner_product_float_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

TEST(InnerProductTest, ForwardTest) {
  Caffe::set_mode(Caffe::GPU);

  QCodeConfig config;
  config.set_il(5);
  config.set_fl(20);

  Blob<float>* blob_bottom = new Blob<float>(1, 3, 4, 5);
  Blob<float>* blob_top = new Blob<float>();

  FillerParameter filler_param;
  filler_param.set_value(1.);
  GaussianFiller<float> filler(filler_param);
  filler.Fill(blob_bottom);

  Blob<float>* blob_bottom_t = new Blob<float>();
  blob_bottom_t->ReshapeLike(*blob_bottom);
  caffe_copy(blob_bottom->count(), blob_bottom->cpu_data(),
             blob_bottom_t->mutable_cpu_data());
  Blob<float>* blob_top_t = new Blob<float>();

  vector<Blob<float>*> blob_bottom_vec;
  blob_bottom_vec.push_back(blob_bottom_t);

  vector<Blob<float>*> blob_top_vec;
  blob_top_vec.push_back(blob_top_t);

  LayerParameter layer_param;
  for (int i = 0; i < 3; ++i) {
    layer_param.add_qcodeconfig();
  }

  for (int i = 0; i < 3; ++i) {
    layer_param.mutable_qcodeconfig(i)->set_il(config.il());
    layer_param.mutable_qcodeconfig(i)->set_fl(config.fl());
  }

  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("constant");
  inner_product_param->mutable_bias_filler()->set_type("constant");
  inner_product_param->mutable_weight_filler()->set_value(1.);
  inner_product_param->mutable_bias_filler()->set_value(0.);

  shared_ptr<InnerProductLayer<float> > ip_t(
      new InnerProductLayer<float>(layer_param));
  ip_t->SetUp(blob_bottom_vec, blob_top_vec);
  ip_t->Forward(blob_bottom_vec, blob_top_vec);

  blob_bottom_vec[0] = blob_bottom;
  blob_top_vec[0] = blob_top;

  shared_ptr<InnerProductFloatLayer<float> > layer(
      new InnerProductFloatLayer<float>(layer_param));
  layer->SetUp(blob_bottom_vec, blob_top_vec);
  layer->Forward(blob_bottom_vec, blob_top_vec);

  EXPECT_EQ(blob_top->shape(0), blob_top_t->shape(0));
  EXPECT_EQ(blob_top->shape(1), blob_top_t->shape(1));
  EXPECT_EQ(blob_top->count(), blob_top_t->count());

  const float tol = 1E-4;
  for (int i = 0; i < blob_top->count(); ++i) {
    EXPECT_NEAR(blob_top->cpu_data()[i], blob_top_t->cpu_data()[i], tol);
  }

  delete blob_bottom;
  delete blob_top;
  delete blob_bottom_t;
  delete blob_top_t;
}
}  // namespace caffe
