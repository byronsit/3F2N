#ifndef EXRIO_OPENCV_H
#define EXRIO_OPENCV_H

#include <vector>
#include <cstring>
#include <string>
#include <opencv2/opencv.hpp>

//Ŀǰֻ��֧�ֶ�ȡfloat����exr����,���Ǵ��������㹻��


void readexr_opencv(std::string path, std::vector<cv::Mat> &image,
             std::vector<std::string> &channel_name);

#endif
