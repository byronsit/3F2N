#ifndef EXRIO_OPENCV_H
#define EXRIO_OPENCV_H

#include <vector>
#include <cstring>
#include <string>
#include <opencv2/opencv.hpp>

//目前只能支持读取float类型exr数据,但是大多数情况足够了


void readexr_opencv(std::string path, std::vector<cv::Mat> &image,
             std::vector<std::string> &channel_name);

#endif
