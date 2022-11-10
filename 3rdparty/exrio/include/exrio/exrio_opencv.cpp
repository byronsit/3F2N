
/* 
* 一个脱离opencv的版本的read EXR格式. 主要读取float类型,其他再说.
* 只要读取float类型就足够我用了
*/


#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <vector>

#include "exrio/exrio.h"
#include "exrio/exrio_opencv.h"

//#include <iostream> //debug

void readexr_opencv(std::string path, std::vector<cv::Mat> &image,
             std::vector<std::string> &channel_name) {
  int rows, cols;
  std::vector<float> image_vector;
  readexr(path, image_vector, channel_name, rows, cols);

//  std::cout << "asdf" << channel_name.size() << std::endl; //debug

  image.resize(channel_name.size());
  for (int i = 0; i < channel_name.size(); ++i) {
    image.at(i) = cv::Mat(rows, cols, CV_32F);
    memmove(image.at(i).data, &image_vector.at(cols * rows * i),
            sizeof(float) * cols * rows);
  }
}
