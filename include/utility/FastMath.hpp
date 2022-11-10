//
// Created by udi on 22-6-16.
//

#ifndef TRAD_INCLUDE_UTILITY_FASTMATH_HPP_
#define TRAD_INCLUDE_UTILITY_FASTMATH_HPP_

#include <cmath>
/**
 * @brief 通过位运算求绝对值，在大多数平台下，比调用fabs更快*/
inline
static float FastAbs(float &&a){
  return fabs(a);
  //浮点数 todo
  int &&tmp =  *((int*)&a) & 0x7fffffff;
  return *(float*)(&tmp);
}

/**
 * @brief 平方*/
inline
static  float Pow2(const float &a){
  return a*a;
}

#endif //TRAD_INCLUDE_UTILITY_FASTMATH_HPP_
