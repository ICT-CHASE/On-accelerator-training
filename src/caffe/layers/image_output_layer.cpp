#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>

#include "caffe/layers/image_output_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void ImageOutputLayer<Dtype>::WriteImageBlobToFile(const Blob<Dtype>* blob,
                                                   const std::string& filename,
                                                   int idx) {
  std::ofstream writer(filename.c_str());
  const int c = blob->shape(1);
  const int h = blob->shape(2);
  const int w = blob->shape(3);

  const Dtype* ptr = blob->cpu_data() + idx * c * h * w;
  for (int i = 0; i < c; ++i) {
    for (int j = 0; j < h; ++j) {
      for (int k = 0; k < w; ++k) {
        writer << std::setprecision(8) << *ptr++;
        if (k != w - 1) {
          writer << " ";
        }
      }
      writer << "\n";
    }
  }
  writer.close();
}
template <typename Dtype>
void ImageOutputLayer<Dtype>::WriteLabelBlobToFile(
    const Blob<Dtype>* blob, const std::string& filename) {
  std::ofstream writer(filename.c_str());
  const Dtype* ptr = blob->cpu_data();
  for (int i = 0; i < blob->count(); ++i) {
    writer << *ptr++ << "\n";
  }
  writer.close();
}
template <typename Dtype>
void ImageOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  iteration_ = 0;
}

template <typename Dtype>
void ImageOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void ImageOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  LOG(INFO) << "Iteration: " << iteration_;
  const int batch_size = bottom[0]->shape(0);
  for (int i = 0; i < batch_size; ++i) {
    // get image file name
    std::string filename = GetFilename("./image_out/", i, iteration_, true);
    WriteImageBlobToFile(bottom[0], filename, i);
  }
  // get label file name
  std::string filename = GetFilename("./image_out/", 0, iteration_, false);
  WriteLabelBlobToFile(bottom[1], filename);
  ++iteration_;
}

#ifdef CPU_ONLY
//STUB_GPU(ImageOutputLayer);
#endif

INSTANTIATE_CLASS(ImageOutputLayer);
REGISTER_LAYER_CLASS(ImageOutput);
}  // namespace caffe
