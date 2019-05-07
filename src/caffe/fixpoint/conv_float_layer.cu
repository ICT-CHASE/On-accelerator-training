#include <vector>
#include "caffe/fixpoint/math_functions.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/fixpoint/conv_float_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionFloatLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // generate random number
  caffe_gpu_rng_uniform(this->prob_weight_.count(), Dtype(0), Dtype(1),
                        this->prob_weight_.mutable_gpu_data());
  caffe_gpu_rng_uniform(this->prob_input_.count(), Dtype(0), Dtype(1),
                        this->prob_input_.mutable_gpu_data());
  caffe_gpu_rng_uniform(this->prob_output_.count(), Dtype(0), Dtype(1),
                        this->prob_output_.mutable_gpu_data());

  const Dtype* weight = this->blobs_[0]->gpu_data();
  // quantize for weight
  Quantize(this->qweights_.count(), weight, this->qweights_.mutable_gpu_data(),
           this->prob_weight_.gpu_data(), this->weight_config_);

  weight = this->qweights_.gpu_data();
  const Dtype* bias = NULL;
  if (this->bias_term_) {
    caffe_gpu_rng_uniform(this->prob_bias_.count(), Dtype(0), Dtype(1),
                          this->prob_bias_.mutable_gpu_data());
    Quantize(this->qbias_.count(), this->blobs_[1]->gpu_data(),
             this->qbias_.mutable_gpu_data(), this->prob_bias_.gpu_data(),
             this->bias_config_);

    bias = this->qbias_.gpu_data();
  }

  // const int output_size = conv_out_channels_ * conv_out_spatial_dim_;
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    // quantize for input
    Quantize(bottom[i]->count(), bottom_data, bottom[i]->mutable_gpu_data(),
             this->prob_input_.gpu_data(), this->input_config_);
    // Quantize(this->qinput_.count(), bottom_data, this->qinput_.mutable_gpu_data(),
    //          this->prob_input_.gpu_data(), this->input_config_);
    // bottom_data = this->qinput_.gpu_data();

    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                             top_data + n * this->top_dim_);
      if (this->bias_term_) {
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
    Quantize(top[i]->count(), top_data, top_data, this->prob_output_.gpu_data(),
             this->output_config_);
  }
}

template <typename Dtype>
void ConvolutionFloatLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* weight = this->qweights_.gpu_data();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
                                top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
                                  bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionFloatLayer);

}  // namespace caffe
