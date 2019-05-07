#ifndef CAFFE_IMAGE_OUTPUT_LAYER_HPP_
#define CAFFE_IMAGE_OUTPUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ImageOutputLayer : public Layer<Dtype> {
 public:
  explicit ImageOutputLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageOutput"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) {
    Forward_cpu(bottom, top);
  }
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {
    LOG(WARNING) << "Cannot do backward";
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {
    Backward_cpu(top, propagate_down, bottom);
  }
  int iteration_;

  std::string GetFilename(const std::string& base_dir, int batch_idx,
                          int iteration, bool is_image) {
    char buf[64];
    if (is_image) {
      sprintf(buf, "iter_%06d_idx_%03d_image.txt", iteration, batch_idx);
    } else {
      sprintf(buf, "iter_%06d_label.txt", iteration);
    }
    return std::string(base_dir) + std::string(buf);
  }

  void WriteImageBlobToFile(const Blob<Dtype>* blob,
                            const std::string& filename, int idx);
  void WriteLabelBlobToFile(const Blob<Dtype>* blob,
                            const std::string& filename);
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_OUTPUT_LAYER_HPP_
