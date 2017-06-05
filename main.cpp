#include "stdio.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <gflags/gflags.h>
#include "extract_wrap.h"
#include <algorithm>
#include <locale>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include "facedetect-dll.h"
#include <windows.h>
#include <string>
#include "FaceAlign.h"
#include "i18nText.h"


#define MAX(a, b) (((a) > (b)) ? (a) : (b))

using namespace google;
using namespace cv;
using namespace std;


DEFINE_string(mode, "CPU", "CPU or GPU");
DEFINE_string(trained_file, "model/my_565000.caffemodel", "trained_file");
DEFINE_string(model_file, "model/deploy.prototxt", "model_file");
DEFINE_int32(batch, 1, "batch_size");

int ToWcharWindows(const char* src, wchar_t* dest) {

	int nLen = MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, src, -1, NULL, 0);
	if (nLen == 0)
	{
		return NULL;
	}
	//dest = new wchar_t[nLen + 1];
	MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, src, -1, dest, nLen);
	return 0;
}

int main(int argc, char *argv[])
{
	ParseCommandLineFlags(&argc, &argv, false);
	i18nText i18n;
	i18n.setFont("simhei.ttf");
	Mat frame;
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
		return -1;

	std::string winname = "camera";
	cv::namedWindow(winname, WINDOW_NORMAL);
	cv::setWindowProperty(winname, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

    #define DETECT_BUFFER_SIZE 0x20000
	int * pResults = NULL;
	unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);

	if (!pBuffer)
	{
		fprintf(stderr, "Can not alloc buffer.\n");
		return -1;
	}
	Extractor featureExtractor;
	featureExtractor.SetExtractor(FLAGS_trained_file, FLAGS_mode, FLAGS_batch, FLAGS_model_file);
	void *fa;
	FDA_Init(&fa);
	float ratio[2] = { 110.0 / 218.0f, 45.0 / 218.0f };

	while (1)
	{
		cap >> frame;
		
		if (frame.empty())
		{
			printf("image null\n");
			continue;

		}
		
		Mat  gray;
		cvtColor(frame, gray, CV_BGR2GRAY);

		
		pResults = facedetect_frontal_surveillance(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
			1.2f, 2, 48, 0);
		

		if (pResults == NULL) {
			cv::imshow(winname, frame);
			continue;
		}
			
		int num = *pResults;
		if (num > 0) {
			vector<cv::Rect> rois;
			vector<cv::Rect> faceRegion;
			int maxface = 0, idx = 0;
			for (int i = 0; i < num; i++)
			{
				short * p = ((short*)(pResults + 1)) + 142 * i;
				int x = p[0];
				int y = p[1];
				int w = p[2];
				int h = p[3];
				int neighbors = p[4];
				int angle = p[5];
				rectangle(frame, Rect(x, y, w, h), cv::Scalar(255, 0, 0), 1);
				rois.push_back(Rect(x, y, w, h));
			}
			if (rois.size() > 0)
			{
				for (int i = 0; i < rois.size(); i++)
				{
					if (rois[i].width > maxface)
					{
						maxface = rois[i].width;
						idx = i;
					}
				}
				rectangle(frame, rois[idx], cv::Scalar(0, 0, 255), 1);
				faceRegion.push_back(rois[idx]);
			}
			else
				continue;

			vector<Point2f> points;
			vector<Point3f> angle;
			vector<cv::Mat> faceImage;
			FDA_Align(fa, frame, faceRegion, &faceImage, 178, 218, &points, &angle, 1, ratio);

			//if (faceImage[0].empty())
				//continue;
			if (faceImage.size() == 0) {

				cv::imshow(winname, frame);
				continue;
			}
			
			std::vector<float> features = featureExtractor.Extract(faceImage[0]);


			//std::string pos_attribute[] = { "Bald", "Bangs", "Black_Hair", "Bushy_Eyebrows", "Eyeglasses",  "Male", "No_Beard",  "Pale_Skin",  "Sideburns", "Smiling",  "Wearing_Hat",  "Young" };
			//std::string  nega_attribute[] = { "No_Bald", "No_Bangs", "No_Black_Hair", "No_Bushy_Eyebrows", "No_Eyeglasses",  "female", "Beard",  "No_Pale_Skin",  "No_Sideburns", "No_Smiling",  "No_Wearing_Hat",  "oldness" };	
			vector<char *> pos_attribute = { "秃头","刘海","黑色头发","浓眉", "戴眼镜", "男性", "没有胡子","脸色苍白","有络腮胡", "微笑", "戴帽子", "年轻的" };
			vector<char *> nega_attribute = { "有头发的","没有刘海", "不是黑头发", "眉毛不浓的", "没戴眼镜",  "女性", "有胡子", "脸色正常的", "没有络腮胡", "没有微笑","没戴帽子", "年老的" };
			//if (features.empty())
				//continue;
			int distance = 1;
			for (int i = 0; i < 12; i++)
			{
				if (features[i] > 0)
				{
					wchar_t p[100];
					ToWcharWindows(pos_attribute[i], p);
					i18n.putText(frame, p, Point(MAX(faceRegion[0].x - 60, 0), faceRegion[0].y + distance * 15), CV_RGB(255, 0, 0));
					distance++;
				}
				else
				{
					wchar_t n[100];
					ToWcharWindows(nega_attribute[i], n);
					i18n.putText(frame, n, Point(MAX(faceRegion[0].x - 60, 0), faceRegion[0].y + distance * 15), CV_RGB(255, 0, 0));
					distance++;
				}
			}
			/*for (int i = 0; i < 12; i++)
			{
				if (features[i] > 0)
				{
					putText(frame, pos_attribute[i], Point(MAX(faceRegion[0].x - 60, 0), faceRegion[0].y + distance * 15), CV_FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 255));
					distance++;
				}
				else
				{
					putText(frame, nega_attribute[i], Point(MAX(faceRegion[0].x - 60, 0), faceRegion[0].y + distance * 15), CV_FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 255));
					distance++;
				}
			}*/
			cv::imshow(winname, frame);
			char c = cv::waitKey(5);
			if (c == 'c' || c == ' ')
				break;
		}
		else {
			cv::imshow(winname, frame);
			char c = cv::waitKey(5);
			if (c == 'c' || c == ' ')
				break;
		}
	}
	
	free(pBuffer);
	FDA_Delete(fa);

	return 0;
}



//int main(int argc, char **argv)
//{
//	ParseCommandLineFlags(&argc, &argv, false);
//
//    #define DETECT_BUFFER_SIZE 0x20000
//	int * pResults = NULL;
//	unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
//
//	if (!pBuffer)
//	{
//		fprintf(stderr, "Can not  buffer.\n");
//		return -1;
//	}
//	int num = 1;
//	char img_name[20];
//	string image_name;
//	int number =1;
//	char img_namere[20];
//	string image_namere;
//	Mat frame, gray;
//
//	for (int i =0;i<100;i++)
//	{	
//
//		sprintf(img_name,"%s%d%s","test\\testsur700wan\\",num++,".jpg");
//		image_name = img_name;
//		frame = imread(image_name);		
//		cvtColor(frame, gray, CV_BGR2GRAY);
//		pResults = facedetect_frontal_surveillance(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
//			1.2f, 1, 30, 0);
//
//		if (pResults == NULL)
//			continue;
//
//		int num = *pResults;
//		
//		vector<Rect> rois;
//		vector<Rect> faceRegion;
//		int maxface = 0, idx = 0;
//		for (int i = 0; i < num; i++)
//		{
//			short * p = ((short*)(pResults + 1)) + 142 * i;
//			int x = p[0];
//			int y = p[1];
//			int w = p[2];
//			int h = p[3];
//			int neighbors = p[4];
//			int angle = p[5];
//			rectangle(frame, Rect(x, y, w, h), cv::Scalar(255, 0, 0), 1);
//			rois.push_back(Rect(x, y, w, h));
//		}
//		if (rois.size() > 0)
//		{
//			for (int i = 0; i < rois.size(); i++)
//			{
//				if (rois[i].width > maxface)
//				{
//					maxface = rois[i].width;
//					idx = i;
//				}        
//			}
//			rectangle(frame, rois[idx], cv::Scalar(0, 0, 255), 2);
//			faceRegion.push_back(rois[idx]);
//		}
//		else
//			continue;
//		if (faceRegion.empty())
//			continue;
//		Mat align_result_img = align_img(frame, faceRegion);
//		if (align_result_img.empty())
//			continue;
//		Extractor featureExtractor;
//		featureExtractor.SetExtractor(FLAGS_trained_file, FLAGS_mode, FLAGS_batch, FLAGS_model_file);
//		std::vector<float> features = featureExtractor.Extract(align_result_img);
//		if (features.empty())
//			continue;
//		//std::string attribute[] = { "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair","Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young" };
//			
//		std::string pos_attri[] = {  "Bald", "Bangs", "Black_Hair", "Bushy_Eyebrows", "Eyeglasses",  "Male", "No_Beard",  "Pale_Skin",  "Sideburns", "Smiling",  "Wearing_Hat",  "Young" };
//		std::string nega_attri[] = { "No_Bald", "No_Bangs", "No_Black_Hair", "No_Bushy_Eyebrows", "No_Eyeglasses",  "female", "Beard",  "No_Pale_Skin",  "No_Sideburns", "No_Smiling",  "No_Wearing_Hat",  " oldness" };
//		//int count = -1;
//		//int dis[] = { -2,-1,0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
//		//int attri[] = { 4, 5, 8, 12, 15, 20, 24, 26, 30, 31, 35, 39 };
//		if (features.empty())
//			continue;
//		for (int i = 0; i < 12; i++)
//		{
//		
//			if (features[i] > 0)
//			{
//				//++count;
//				putText(frame, pos_attri[i], Point(MAX(faceRegion[0].x-60,0), faceRegion[0].y + i * 15), CV_FONT_HERSHEY_COMPLEX, 0.3, Scalar(255, 0, 255));
//			}
//			else
//			{
//				//++count;
//				putText(frame, nega_attri[i], Point(MAX(faceRegion[0].x - 60, 0), faceRegion[0].y + i * 15), CV_FONT_HERSHEY_COMPLEX, 0.3, Scalar(255, 0, 0));
//			}
//		}
//				
//	
//		
//
//		/*	int label;
//	vector<int>temp_labels;
//		if (features.size() > 0) {
//
//			for (int i = 0; i < 12; i++)
//			{
//				if (features[attri[i]] > 0) {
//					label = 1;
//					temp_labels.push_back(label);
//				}
//				else {
//					label = -1;
//					temp_labels.push_back(label);
//				}
//			}
//	
//		fout << image_name<<" ";
//		for (int j = 0; j < 12; j++) {
//			fout << temp_labels[j]<<"  ";
//		}
//		fout << endl;
//
//		}
//*/
//
//		sprintf(img_namere, "%s%d%s", "test\\testsur700wan_result\\", number++, ".jpg");
//		image_namere = img_namere;
//		imwrite(image_namere,frame);
//		char c = waitKey(1);
//		if (c == 'c' || c == ' ')
//			break;
//	}
//	free(pBuffer);
//	return 0;
//}
//
//
//
