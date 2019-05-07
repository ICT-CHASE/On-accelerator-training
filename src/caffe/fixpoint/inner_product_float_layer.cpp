#include <vector>

#include "caffe/filler.hpp"
#include "caffe/fixpoint/inner_product_float_layer.hpp"
#include "caffe/fixpoint/math_functions.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

inline void CheckQCode(const QCodeConfig& config, const int qlength) {
  CHECK_EQ(config.il() + config.fl(), qlength)
      << "Q Format: " << config.il() << ", " << config.fl();
}
const int default_weight_il = 2;
const int default_weight_fl = 6;

const int default_data_il = 3;
const int default_data_fl = 5;

template <typename Dtype>
void InnerProductFloatLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // for qcode config
  const int num_q_config = this->layer_param_.qcodeconfig_size();
  if (num_q_config >= 1) {
    weight_config_ = this->layer_param_.qcodeconfig(0);
  } else {
    weight_config_.set_il(default_weight_il);
    weight_config_.set_fl(default_weight_fl);
  }
  if (num_q_config >= 2) {
    input_config_ = this->layer_param_.qcodeconfig(1);
  } else {
    input_config_.set_il(default_data_il);
    input_config_.set_fl(default_data_fl);
  }
  if (num_q_config >= 3) {
    output_config_ = this->layer_param_.qcodeconfig(2);
  } else {
    output_config_.set_il(default_data_il);
    output_config_.set_fl(default_data_fl);
  }

  // CheckQCode(weight_config_, 8);
  // CheckQCode(input_config_, 8);
  // CheckQCode(output_config_, 8);
  LOG(INFO) << "qcode config for weight: Q" << weight_config_.il() << "."
            << weight_config_.fl();
  LOG(INFO) << "qcode config for input: Q" << input_config_.il() << ","
            << input_config_.fl();
  LOG(INFO) << "qcode config for output: Q" << output_config_.il() << ","
            << output_config_.fl();

  if (this->bias_term_) {
    this->bias_config_.set_il(input_config_.il() + weight_config_.il());
    this->bias_config_.set_fl(input_config_.fl() + weight_config_.fl());

    LOG(INFO) << "qcode config for bias: Q" << this->bias_config_.il() << ","
              << this->bias_config_.fl();
  }

  qweights_.ReshapeLike(*this->blobs_[0]);
  prob_weight_.ReshapeLike(*this->blobs_[0]);
  LOG(INFO) << "Initialize qweight with shape: " << qweights_.shape_string();

  if (bias_term_) {
    qbias_.ReshapeLike(*this->blobs_[1]);
    prob_bias_.ReshapeLike(*this->blobs_[1]);
    LOG(INFO) << "Initialize qbias with shape: " << qbias_.shape_string();
  } else {
    LOG(WARNING) << "this conv doesn't have bias term";
  }
  output_max_ = 1 << (input_config_.il() + weight_config_.il() - 1);
  CHECK_GT(output_max_, 0) << "should be larger than 0";
}

template <typename Dtype>
void InnerProductFloatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
  // Reshape of quantized input
  qinput_.ReshapeLike(*bottom[0]);
  prob_input_.ReshapeLike(*bottom[0]);
  prob_output_.ReshapeLike(*top[0]);

  // LOG(INFO) << "Reshape qinput with shape: " << qinput_.shape_string();
  // LOG(INFO) << "Rehshape output with shape: " << prob_output_.shape_string();
}

template <typename Dtype>
void InnerProductFloatLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
                        M_, N_, K_, (Dtype)1., bottom_data, weight, (Dtype)0.,
                        top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                          bias_multiplier_.cpu_data(),
                          this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductFloatLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype)1.,
                            bottom_data, top_diff, (Dtype)1.,
                            this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
                            top_diff, bottom_data, (Dtype)1.,
                            this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
                          bias_multiplier_.cpu_data(), (Dtype)1.,
                          this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype)1.,
                            top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
                            bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
                            top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
                            bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductFloatLayer);
#endif

INSTANTIATE_CLASS(InnerProductFloatLayer);
REGISTER_LAYER_CLASS(InnerProductFloat);

}  // namespace caffe
