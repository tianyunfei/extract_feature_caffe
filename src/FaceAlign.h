#ifndef _FACE_ALIGN_H__
#define _FACE_ALIGN_H__

#include "opencv2/opencv.hpp"



int FDA_Init(void **hdl, int fd = 1, int fa = 1);
int FDA_Delete(void *hdl);

// 针对已抠取的人脸小图做检测、关键点定位、矫正
// img， 输入图像
// faceRegion，输出图像矩形区域
// faceImg，输出人脸图, NULL表示不输出
// fw，fh，输出图像宽高
// eye, mouth 眼睛和嘴巴的点，NULL表示不输出
// angle 输出三个角度，pitch/yaw/roll, NULL表示不输出
// maxNum，最多检测人脸个数，目前仅支持1和非1
int FDA_DetectAndAlign(void *hdl, cv::Mat &img, std::vector<cv::Rect>& faceRegion, std::vector<cv::Mat> *faceImg, 
	int fw, int fh, std::vector<cv::Point2f> *points, std::vector<cv::Point3f> *angle, int maxNum, float *ratio = NULL);


// 针对已抠取的人脸小图做关键点定位、矫正
// img， 输入图像
// faceRegion，输入图像矩形区域
// faceImg，输出人脸图, NULL表示不输出
// fw，fh，输出图像宽高
// eye, mouth 眼睛和嘴巴的点，NULL表示不输出
// angle 输出三个角度，pitch/yaw/roll, NULL表示不输出
// maxNum，最多检测人脸个数，目前仅支持1和非1
int FDA_Align(void *hdl, cv::Mat &img, std::vector<cv::Rect> &faceRegion, std::vector<cv::Mat> *faceImg,
	int fw, int fh, std::vector<cv::Point2f> *points, std::vector<cv::Point3f> *angle, int maxNum, float *ratio = NULL);


// 人脸检测
// da, 句柄
// img, 输入图像
// tryflip, 是否尝试翻转
// faceRegion, 输出人脸区域
int FDA_FaceDetect(void *da, cv::Mat& img,
	int tryflip, std::vector<cv::Rect>& faceRegion);

// 人脸检测, 带角度
// da, 句柄
// img, 输入图像
// tryflip, 是否尝试翻转
// faceRegion, 输出人脸区域
int FDA_FaceDetectAngle(void *da, cv::Mat& img,
	int tryflip, std::vector<cv::Rect>& faceRegion, std::vector<float>& angle);

#endif