#include <cstdio>
#include <fstream>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/fixpoint/inner_product_float_layer.hpp"
#include "caffe/fixpoint/math_functions.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
inline void check_shape(const Blob<Dtype>& lhs, const Blob<Dtype>& rhs) {
  CHECK_EQ(lhs.shape_string(), rhs.shape_string());
}

template <>
void InnerProductFloatLayer<double>::Forward_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top) {
  NOT_IMPLEMENTED;
}

template <>
void InnerProductFloatLayer<float>::Forward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top) {
  const float* bottom_data = bottom[0]->gpu_data();
  float* top_data = top[0]->mutable_gpu_data();
  const float* weight = this->blobs_[0]->gpu_data();

  // generate random number
  caffe_gpu_rng_uniform(this->prob_weight_.count(), float(0), float(1),
                        this->prob_weight_.mutable_gpu_data());
  caffe_gpu_rng_uniform(this->prob_input_.count(), float(0), float(1),
                        this->prob_input_.mutable_gpu_data());
  caffe_gpu_rng_uniform(this->prob_output_.count(), float(0), float(1),
                        this->prob_output_.mutable_gpu_data());
  // quantize for weight
  Quantize(this->qweights_.count(), weight, this->qweights_.mutable_gpu_data(),
           this->prob_weight_.gpu_data(), this->weight_config_);

  weight = qweights_.gpu_data();
  const float* bias = NULL;
  if (this->bias_term_) {
    caffe_gpu_rng_uniform(prob_bias_.count(), float(0), float(1),
                          prob_bias_.mutable_gpu_data());
    Quantize(qbias_.count(), this->blobs_[1]->gpu_data(),
             qbias_.mutable_gpu_data(), prob_bias_.gpu_data(),
             this->bias_config_);

    bias = qbias_.gpu_data();
  }
  // quantize for input
  Quantize(bottom[0]->count(), bottom_data, bottom[0]->mutable_gpu_data(),
           prob_input_.gpu_data(), input_config_);

  if (M_ == 1) {
    // caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
    //                      weight, bottom_data, (Dtype)0., top_data);
    gpu_gemv_fp(CblasNoTrans, N_, K_, 1.f, weight, bottom_data, 0.f, top_data,
                output_max_);
    if (bias_term_)
      // caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
      //                       this->blobs_[1]->gpu_data(), top_data);
      gpu_axpy_fp(N_, bias_multiplier_.cpu_data()[0], bias, top_data);
  } else {
    // caffe_gpu_gemm<Dtype>(CblasNoTrans,
    //                       transpose_ ? CblasNoTrans : CblasTrans,
    //                       M_, N_, K_, (Dtype)1.,
    //                       bottom_data, weight, (Dtype)0., top_data);
    gpu_gemm_fp(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans, M_, N_,
                K_, 1.f, bottom_data, weight, 0.f, top_data, output_max_);
    if (bias_term_)
      // caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
      //                       (Dtype)1.,
      //                       bias_multiplier_.gpu_data(),
      //                       this->blobs_[1]->gpu_data(), (Dtype)1.,
      //                       top_data);
      gpu_gemm_fp(CblasNoTrans, CblasNoTrans, M_, N_, 1, 1.f,
                  bias_multiplier_.gpu_data(), bias, 1.f, top_data,
                  output_max_);
  }

  Quantize(top[0]->count(), top_data, top_data, prob_output_.gpu_data(),
           output_config_);
}

template <typename Dtype>
void InnerProductFloatLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  //// let's write some info to file for debugging
  // static int cnt = 0;
  // char buf[32];
  //// for quantized bias
  // sprintf(buf, "./debug/fc-bias-%06d.txt", cnt);
  // std::ofstream writer(buf);

  // writer << "bias of fc in iteration: " << cnt << "\n";

  // for(int i = 0; i < this->qbias_.count(); ++i) {
  //   writer << this->qbias_.cpu_data()[i] << "\t";
  //   if((i+1) % 10 == 0) {
  //     writer << "\n";
  //   }
  // }
  // writer.close();

  //// for un-quantized bias
  // sprintf(buf, "./debug/true-fc-bias-%06d.txt", cnt);
  // writer.open(buf);
  // writer << "true bias of fc in iteration: " << cnt << "\n";

  // for(int i = 0; i < this->qbias_.count(); ++i) {
  //   writer << this->blobs_[1]->cpu_data()[i] << "\t";
  //   if((i+1) % 10 == 0) {
  //     writer << "\n";
  //   }
  // }
  // writer.close();

  //// for prob used when quantize bias
  // sprintf(buf, "./debug/prob-fc-bias-%06d.txt", cnt);
  // writer.open(buf);
  // writer << "prob bias of fc in iteration: " << cnt << "\n";

  // for(int i = 0; i < this->qbias_.count(); ++i) {
  //   writer << this->prob_bias_.cpu_data()[i] << "\t";
  //   if((i+1) % 10 == 0) {
  //     writer << "\n";
  //   }
  // }
  // writer.close();

  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype)1.,
                            bottom_data, top_diff, (Dtype)1.,
                            this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
                            top_diff, bottom_data, (Dtype)1.,
                            this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
                          bias_multiplier_.gpu_data(), (Dtype)1.,
                          this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    const Dtype* weight = this->qweights_.gpu_data();
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype)1.,
                            top_diff, weight, (Dtype)0.,
                            bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
                            top_diff, weight, (Dtype)0.,
                            bottom[0]->mutable_gpu_diff());
    }
  }

  //// for diff of bias
  // sprintf(buf, "./debug/diff-fc-bias-%06d.txt", cnt);
  // writer.open(buf);
  // writer << "diff of bias of fc in iteration: " << cnt << "\n";

  // for(int i = 0; i < this->qbias_.count(); ++i) {
  //   writer << this->blobs_[1]->cpu_diff()[i] << "\t";
  //   if((i+1) % 10 == 0) {
  //     writer << "\n";
  //   }
  // }
  // writer.close();
  // cnt += 1;
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductFloatLayer);

}  // namespace caffe
