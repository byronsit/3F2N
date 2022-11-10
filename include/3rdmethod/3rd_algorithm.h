//
// Created by xxyy on 22-11-10.
//

#ifndef TRAD_INCLUDE_3RDMETHOD_3RD_ALGORITHM_H_
#define TRAD_INCLUDE_3RDMETHOD_3RD_ALGORITHM_H_

#include <opencv2/opencv.hpp>

/* All of the methods you can see in
 * 'Comparison of Surface Normal Estimation
 *  Methods for Range Sensing Applications'.
 * **/
enum NEAREST_METHOD{
  PLANESVD,
  PLANEPCA,
  VECTORSVD,
  QUADSVD,
  QUADTRANSSVD,
  AREAWEIGHTED,
  ANGLEWEIGHTED
};


/**
 * @brief calculate the normal use different method
 * @param[in] range_image the input range image
 * @param[out] result the output result*/
void GetNormal(const cv::Mat &range_image, cv::Mat *result, const NEAREST_METHOD& METHOD);

#endif //TRAD_INCLUDE_3RDMETHOD_3RD_ALGORITHM_H_
