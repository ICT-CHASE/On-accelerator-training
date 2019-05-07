#ifndef CAFFE_INNER_PRODUCT_FLOAT_LAYER_HPP_
#define CAFFE_INNER_PRODUCT_FLOAT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class InnerProductFloatLayer : public Layer<Dtype> {
 public:
  explicit InnerProductFloatLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InnerProductFloat"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  // Override `ToProto` to save quantized weight
  virtual void ToProto(LayerParameter* param, bool write_diff = false) {
    param->Clear();
    param->CopyFrom(this->layer_param_);
    param->clear_blobs();

    qweights_.ToProto(param->add_blobs(), write_diff);

    if (bias_term_) {
      qbias_.ToProto(param->add_blobs(), write_diff);
    }
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights

  QCodeConfig weight_config_;
  QCodeConfig input_config_;
  QCodeConfig output_config_;

  QCodeConfig bias_config_;

  int output_max_;  // max value of output when do intermidiate computing

  // blob for quantized weight
  Blob<Dtype> qweights_;
  // blob for quantized bias
  Blob<Dtype> qbias_;
  // blob for quantized inpu data
  Blob<Dtype> qinput_;
  // blob of prob for stochastic rounding
  Blob<Dtype> prob_weight_;
  Blob<Dtype> prob_bias_;
  Blob<Dtype> prob_input_;
  Blob<Dtype> prob_output_;
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_FLOAT_LAYER_HPP_
