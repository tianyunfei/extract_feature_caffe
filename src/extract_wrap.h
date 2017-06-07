#ifndef EXTRACT_WRAP_BATCH_H
#define EXTRACT_WRAP_BATCH_H
//--------------------------------------------------
#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
//------------------------------------------------------

using namespace caffe;
using std::string;
using std::vector;

class Extractor{
	public:
        Extractor(){
	        //nothing done
	    }
        void SetExtractor(const string &trained_file, const string mode,const int num = 1,string model_file ="none");
        std::vector<float> Extract(const vector<cv::Mat>& imgs);
        std::vector<float> Extract(const cv::Mat& img);

	private:
		void WrapInputLayer(std::vector<cv::Mat>* input_channels);
        void Preprocess(const vector<cv::Mat>& imgs,
                  std::vector<cv::Mat>* input_channels);
  		
  	private:
        boost::shared_ptr<Net<float> > net_;
  		  cv::Size input_geometry_;
  		  int num_channels_;
        int input_nums_;
};

#endif
