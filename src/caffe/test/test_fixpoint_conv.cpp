#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/fixpoint/conv_float_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include <fstream>

namespace caffe {

void write_to_file(const Blob<float>* blob, const std::string& name, int code) {
  std::ofstream writer(name.c_str());
  writer.setf(std::ios::fixed, std::ios::floatfield);
  writer.precision(8);  
  const float* data = blob->cpu_data();
  if(code == 0) {
    // weight
    vector<int> shape = blob->shape();
    int cnt = shape[1] * shape[2] * shape[3];
    for(int i = 0; i < shape[0]; ++i) {
      for(int j = 0; j < cnt; ++j) {
        writer << *data++ << " ";
      }
      writer << "\n";
    }
  } else if(code == 1) {
    // bias
    vector<int> shape = blob->shape();
    int cnt = shape[0];
    for(int i = 0; i < cnt; ++i) {
      writer << *data++ << " ";
    }
    writer << "\n";
  } else {
    // data
    vector<int> shape = blob->shape();
    int cnt = shape[0] * shape[1] * shape[2];
    for(int i = 0; i < cnt; ++i) {
      for(int j = 0; j < shape[3]; ++j) {
        writer << *data++ << " ";
      }
      writer << "\n";
    }
  }
  writer.close();
}

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
                const vector<shared_ptr<Blob<Dtype> > >& weights,
                Blob<Dtype>* out) {
  const bool has_depth = (out->num_axes() == 5);
  if (!has_depth) {
    CHECK_EQ(4, out->num_axes());
  }
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (conv_param->has_kernel_h() || conv_param->has_kernel_w()) {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  } else {
    kernel_h = kernel_w = conv_param->kernel_size(0);
  }
  int pad_h, pad_w;
  if (conv_param->has_pad_h() || conv_param->has_pad_w()) {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  } else {
    pad_h = pad_w = conv_param->pad_size() ? conv_param->pad(0) : 0;
  }
  int stride_h, stride_w;
  if (conv_param->has_stride_h() || conv_param->has_stride_w()) {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  } else {
    stride_h = stride_w = conv_param->stride_size() ? conv_param->stride(0) : 1;
  }
  int dilation_h, dilation_w;
  dilation_h = dilation_w =
      conv_param->dilation_size() ? conv_param->dilation(0) : 1;
  int kernel_d, pad_d, stride_d, dilation_d;
  if (has_depth) {
    kernel_d = kernel_h;
    stride_d = stride_h;
    pad_d = pad_h;
    dilation_d = dilation_h;
  } else {
    kernel_d = stride_d = dilation_d = 1;
    pad_d = 0;
  }
  // Groups
  int groups = conv_param->group();
  int o_g = out->shape(1) / groups;
  int k_g = in->shape(1) / groups;
  int o_head, k_head;
  // Convolution
  vector<int> weight_offset(4 + has_depth);
  vector<int> in_offset(4 + has_depth);
  vector<int> out_offset(4 + has_depth);
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->shape(0); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
            for (int y = 0; y < out->shape(2 + has_depth); y++) {
              for (int x = 0; x < out->shape(3 + has_depth); x++) {
                for (int r = 0; r < kernel_d; r++) {
                  for (int p = 0; p < kernel_h; p++) {
                    for (int q = 0; q < kernel_w; q++) {
                      int in_z = z * stride_d - pad_d + r * dilation_d;
                      int in_y = y * stride_h - pad_h + p * dilation_h;
                      int in_x = x * stride_w - pad_w + q * dilation_w;
                      if (in_z >= 0 && in_z < (has_depth ? in->shape(2) : 1) &&
                          in_y >= 0 && in_y < in->shape(2 + has_depth) &&
                          in_x >= 0 && in_x < in->shape(3 + has_depth)) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) {
                          weight_offset[2] = r;
                        }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) {
                          in_offset[2] = in_z;
                        }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) {
                          out_offset[2] = z;
                        }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;
                        out_data[out->offset(out_offset)] +=
                            in->data_at(in_offset) *
                            weights[0]->data_at(weight_offset);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (conv_param->bias_term()) {
    const Dtype* bias_data = weights[1]->cpu_data();
    for (int n = 0; n < out->shape(0); n++) {
      for (int o = 0; o < out->shape(1); o++) {
        for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
          for (int y = 0; y < out->shape(2 + has_depth); y++) {
            for (int x = 0; x < out->shape(3 + has_depth); x++) {
              out_offset[0] = n;
              out_offset[1] = o;
              if (has_depth) {
                out_offset[2] = z;
              }
              out_offset[2 + has_depth] = y;
              out_offset[3 + has_depth] = x;
              out_data[out->offset(out_offset)] += bias_data[o];
            }
          }
        }
      }
    }
  }
}

template void caffe_conv(const Blob<float>* in,
                         ConvolutionParameter* conv_param,
                         const vector<shared_ptr<Blob<float> > >& weights,
                         Blob<float>* out);
template void caffe_conv(const Blob<double>* in,
                         ConvolutionParameter* conv_param,
                         const vector<shared_ptr<Blob<double> > >& weights,
                         Blob<double>* out);

template <typename Dtype>
Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
  Blob<Dtype>* ref_top = new Blob<Dtype>();
  ref_top->ReshapeLike(*top);
  return ref_top;
}

TEST(ConvTest, ForwardTest) {
  const int N = 10;
  Caffe::set_mode(Caffe::GPU);
  // create bottom and top blobs
  Blob<float>* blob_bottom = new Blob<float>(N, 3, 6, 4);
  Blob<float>* blob_top = new Blob<float>();
  FillerParameter filler_param;
  filler_param.set_value(1.);
  GaussianFiller<float> filler(filler_param);
  filler.Fill(blob_bottom);
  vector<Blob<float>*> blob_bottom_vec;
  blob_bottom_vec.push_back(blob_bottom);
  vector<Blob<float>*> blob_top_vec;
  blob_top_vec.push_back(blob_top);

  // create conv layer
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(1);

  QCodeConfig config;
  config.set_il(5);
  config.set_fl(20);

  for (int i = 0; i < 3; ++i) {
    layer_param.add_qcodeconfig();
  }
  for (int i = 0; i < 3; ++i) {
    layer_param.mutable_qcodeconfig(i)->set_il(config.il());
    layer_param.mutable_qcodeconfig(i)->set_fl(config.fl());
  }
  // for (int i = 0; i < 3; ++i) {
  //   int il = 1;
  //   int fl = 7;
  //   if(i == 3) {
  //     il = 2;
  //     fl = 6;
  //   }
  //   layer_param.mutable_qcodeconfig(i)->set_il(il);
  //   layer_param.mutable_qcodeconfig(i)->set_fl(fl);
  // }
  shared_ptr<ConvolutionFloatLayer<float> > layer(
      new ConvolutionFloatLayer<float>(layer_param));
  layer->SetUp(blob_bottom_vec, blob_top_vec);
  EXPECT_EQ(blob_top->num(), N);
  EXPECT_EQ(blob_top->channels(), 4);
  EXPECT_EQ(blob_top->height(), 2);
  EXPECT_EQ(blob_top->width(), 1);

  layer->Forward(blob_bottom_vec, blob_top_vec);
  // Check against reference convolution.
  const float* top_data;
  const float* ref_top_data;

  Blob<float>* ref_top = MakeReferenceTop<float>(blob_top);
  caffe_conv(blob_bottom, convolution_param, layer->blobs(), ref_top);
  top_data = blob_top->cpu_data();
  ref_top_data = ref_top->cpu_data();

  
  std::cout << "top blob: \n";

  for (int i = 0; i < blob_top->count(); ++i) {
    std::cout << top_data[i] << "\t";
    if ((i + 1) % 5 == 0) {
      std::cout << "\n";
    }
  }
  std::cout << "\nref_top blob: \n";
  for (int i = 0; i < blob_top->count(); ++i) {
    std::cout << ref_top_data[i] << "\t";
    if ((i + 1) % 5 == 0) {
      std::cout << "\n";
    }
  }
  std::cout << std::endl;

  // write_to_file(blob_bottom, "./simulator/test-data.txt", 2);
  // write_to_file(ref_top, "./simulator/test-out.txt", 2);
  // write_to_file(&layer->qweights_, "./simulator/test-w.txt", 0);
  // write_to_file(&layer->qbias_, "./simulator/test-bias.txt", 1);


  float tol = 1E-2;
  for (int i = 0; i < blob_top->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], tol);
  }

  delete ref_top;
  delete blob_bottom;
  delete blob_top;
}
}  // namespace caffe
