#ifndef EXRIO_H
#define EXRIO_H



//#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

/**
 * @brief read .exr format
 * @param[in] path
 * @param[out] image
 * @parm[out] channel_names
 * */
//extern

void readexr(std::string path, std::vector<float> &image,
             std::vector<std::string> &channel_name, int &rows, int &cols);


#endif //EXRIO_H
