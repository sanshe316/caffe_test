#include <caffe/caffe.hpp>
#include "caffe/util/upgrade_proto.hpp"
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;



class Classifier {
 public:
  Classifier(const string& model_file, const string& trained_file);
  //std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

 private:
  void SetMean(const string& mean_file);
  std::vector<float> Predict(const cv::Mat& img);
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};


Classifier::Classifier(const string& model_file, const string& trained_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  std::cout << "net_->name(): " << net_->name() << std::endl;
  std::cout << "net_->num_inputs(): " << net_->num_inputs() << std::endl;
  std::cout << "net_->num_outputs(): " << net_->num_outputs() << std::endl;

  vector<string> layer_names = net_->layer_names();
  std::cout << "net_->layer_names() (total: " << layer_names.size() << "): " << std::endl;
  for (size_t i = 0; i < layer_names.size(); i++)
      std::cout << layer_names[i] << std::endl;

  vector<string> blob_names = net_->blob_names();
  std::cout << "net_->blob_names() (total: " << blob_names.size() << "): " << std::endl;
  for (size_t i = 0; i < blob_names.size(); i++)
      std::cout << blob_names[i] << std::endl;

  //vector<string> param_names = net_->param_display_names();
  //std::cout << "net_->param_display_names() (total: " << param_names.size() << "): " << std::endl;
  //for (size_t i = 0; i < param_names.size(); i++)
  //    std::cout << param_names[i] << std::endl; // => 0/1

  Blob<float>* input_layer;
  for (size_t i = 0; i < net_->input_blobs().size(); i++) {
      input_layer = net_->input_blobs()[i];
      for (size_t j = 0; j < input_layer->num_axes(); j++) {
          std::cout << "net_->input_blobs()[" << i << "]->shape(" << j << "): " << input_layer->shape(j) << std::endl;
      }
  }

  NetParameter param;
  ReadNetParamsFromTextFileOrDie(model_file, &param);
  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    const LayerParameter& layer_param = param.layer(layer_id);
    if (layer_param.name() == "cls_score") {
        std::cout << "layer_param.name: " << layer_param.name() << std::endl;
        std::cout << "layer_param.type: " << layer_param.type() << std::endl;
        std::cout << "layer_param.bottom_size: " << layer_param.bottom_size() << std::endl;
        std::cout << "layer_param.top_size: " << layer_param.top_size() << std::endl;
        std::cout << "layer_param.bottom: " << layer_param.bottom(0) << std::endl;
        std::cout << "layer_param.top: " << layer_param.top(0) << std::endl;
        std::cout << "layer_param.phase: " << layer_param.phase() << std::endl;
        std::cout << "layer_param.param_size: " << layer_param.param_size() << std::endl;
        std::cout << "layer_param.param_size: " << layer_param.param_size() << std::endl;
        std::cout << "layer_param.param(0).has_name: " << layer_param.param(0).has_name() << std::endl;
        std::cout << "layer_param.param(1).has_name: " << layer_param.param(1).has_name() << std::endl;
        std::cout << "layer_param.param(0).name: " << layer_param.param(0).name() << std::endl;
        std::cout << "layer_param.param(1).name: " << layer_param.param(1).name() << std::endl;
        std::cout << "layer_param.blobs_size: " << layer_param.blobs_size() << std::endl;
        std::cout << "layer_param.has_inner_product_param: " << layer_param.has_inner_product_param() << std::endl;
        std::cout << "layer_param.inner_product_param().num_output: " << layer_param.inner_product_param().num_output() << std::endl;
        //std::cout << layer_param << std::endl;
    }
  }
}


int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " deploy.prototxt network.caffemodel" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  Classifier classifier(model_file, trained_file);
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
