#ifndef _FACE_ALIGN_H__
#define _FACE_ALIGN_H__

#include "opencv2/opencv.hpp"



int FDA_Init(void **hdl, int fd = 1, int fa = 1);
int FDA_Delete(void *hdl);

// ����ѿ�ȡ������Сͼ����⡢�ؼ��㶨λ������
// img�� ����ͼ��
// faceRegion�����ͼ���������
// faceImg���������ͼ, NULL��ʾ�����
// fw��fh�����ͼ����
// eye, mouth �۾�����͵ĵ㣬NULL��ʾ�����
// angle ��������Ƕȣ�pitch/yaw/roll, NULL��ʾ�����
// maxNum�����������������Ŀǰ��֧��1�ͷ�1
int FDA_DetectAndAlign(void *hdl, cv::Mat &img, std::vector<cv::Rect>& faceRegion, std::vector<cv::Mat> *faceImg, 
	int fw, int fh, std::vector<cv::Point2f> *points, std::vector<cv::Point3f> *angle, int maxNum, float *ratio = NULL);


// ����ѿ�ȡ������Сͼ���ؼ��㶨λ������
// img�� ����ͼ��
// faceRegion������ͼ���������
// faceImg���������ͼ, NULL��ʾ�����
// fw��fh�����ͼ����
// eye, mouth �۾�����͵ĵ㣬NULL��ʾ�����
// angle ��������Ƕȣ�pitch/yaw/roll, NULL��ʾ�����
// maxNum�����������������Ŀǰ��֧��1�ͷ�1
int FDA_Align(void *hdl, cv::Mat &img, std::vector<cv::Rect> &faceRegion, std::vector<cv::Mat> *faceImg,
	int fw, int fh, std::vector<cv::Point2f> *points, std::vector<cv::Point3f> *angle, int maxNum, float *ratio = NULL);


// �������
// da, ���
// img, ����ͼ��
// tryflip, �Ƿ��Է�ת
// faceRegion, �����������
int FDA_FaceDetect(void *da, cv::Mat& img,
	int tryflip, std::vector<cv::Rect>& faceRegion);

// �������, ���Ƕ�
// da, ���
// img, ����ͼ��
// tryflip, �Ƿ��Է�ת
// faceRegion, �����������
int FDA_FaceDetectAngle(void *da, cv::Mat& img,
	int tryflip, std::vector<cv::Rect>& faceRegion, std::vector<float>& angle);

#endif