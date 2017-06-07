
#include"extract_wrap.h"
#include <iostream>
#include <bitset>
#include <cstring>
#include <fstream>



void Extractor::SetExtractor(const string &trained_file, const string mode,const int num,string model_file){
    if(mode == "GPU")
        Caffe::set_mode(Caffe::GPU);
    else
        Caffe::set_mode(Caffe::CPU);
    caffe::NetParameter param;
    if(model_file == "none"){
        ReadProtoFromBinaryFile(trained_file, &param);
        UpgradeNetAsNeeded(trained_file, &param);
        param.mutable_state()->set_phase(caffe::TEST);
	    net_.reset(new Net<float>(param));
	    net_->CopyTrainedLayersFrom(param);
    }
    else{
        net_.reset(new Net<float>(model_file, TEST));
        net_->CopyTrainedLayersFrom(trained_file);
    }
    input_nums_ = num;
	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    input_layer->Reshape(input_nums_, num_channels_,
                       input_geometry_.height, input_geometry_.width);
    net_->Reshape();

}


std::vector<float> Extractor::Extract(const vector<cv::Mat>& imgs){
    if(imgs.size() != input_nums_){
        std::cout << "imgs.size wrong" << std::endl;
        std::cout << "imgs.size():" << imgs.size() << std::endl;
        std::cout << "input_nums" << input_nums_ << std::endl;
        vector<float> none(0);
        return none;
    }
	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);
    Preprocess(imgs, &input_channels);
	net_->Forward();
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels()*input_nums_;
	return std::vector<float>(begin, end);
}

std::vector<float> Extractor::Extract(const cv::Mat& img){
    std::vector<cv::Mat> imgs;
    imgs.push_back(img);
    return Extract(imgs);
}

void Extractor::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int j = 0; j < input_layer->num(); ++j)
      for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
      }
}

void Extractor::Preprocess(const vector<cv::Mat>& imgs,
                            std::vector<cv::Mat>* input_channels){
    for(int i = 0; i< imgs.size(); ++i){
        cv::Mat sample = imgs[i];
        cv::Mat sample_float;
        sample.convertTo(sample_float,CV_32FC3);
        cv::Mat sample_normalized;
//       	cv::subtract(sample_float,cv::Scalar(127.5,127.5,127.5),sample_normalized);
       	sample_normalized = sample_float*0.003922;
        for(int j=0;j<num_channels_;++j)
            for(int h=0;h<sample_normalized.rows;++h)
                for(int w=0;w<sample_normalized.cols;++w){
                    (* input_channels)[i*3+j].at<float>(h,w) = sample_normalized.at<cv::Vec3f>(h,w)[j];
                }
    }
}
