#include <cstdlib>
#include <iostream>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/opencv.hpp>

//???????? 1000??0.190636 ??
//????????????????????????????ü?????????????????
void csne_simple(const cv::Mat depth, cv::Matx33f K, cv::Mat *result) {
  float fx = K(0, 0);
  float fy = K(1, 1);
  float u0 = K(0, 2);  // col
  float v0 = K(1, 2);  // row

  const size_t cols = depth.cols;  // col
  const size_t rows = depth.rows;  // row

  cv::Mat z_laplace(rows, cols, CV_32FC1);
  //???????????û?????????ã???ô????????????
  if (!(result->rows == rows && result->cols == cols &&
        result->type() == CV_32FC3)) {
    *result = cv::Mat(rows, cols, CV_32FC3);  //??????
  }

  cv::Mat_<float> N(*result);

  float *idx_m1 = reinterpret_cast<float *>(depth.data);  //?????
  float *idx_p1 = reinterpret_cast<float *>(depth.data) + cols + cols;  //?????
  float *idx_o = reinterpret_cast<float *>(depth.data) + cols;  //?????
  float *end_idx;  //????????
  cv::Vec3f *idx_n = reinterpret_cast<cv::Vec3f *>(N.data) + cols;
  int ONE_MINUS_COL(cols - 1);
  int v(2), dv(1 - v0);

#define UPDATE_IDX idx_m1++, idx_p1++, idx_o++, du++, idx_n++
  for (; v != rows; ++v, ++dv) {  //???ÿ???
    end_idx = idx_o + ONE_MINUS_COL;
    int du = -u0;
    UPDATE_IDX;

    for (; idx_o != end_idx; UPDATE_IDX) {
      const float &depth_r = *(idx_o + 1);
      const float &depth_l = *(idx_o - 1);
      const float &depth_u = *idx_m1;  // ok
      const float &depth_d = *idx_p1;  // ok
      const float &c = *idx_o;

      const float &gu = (c - depth_l);
      const float &gv = (c - depth_u);
      *idx_n = {gu * fx, gv * fy, -(c + (dv)*gv + (du)*gu)};
    }
    UPDATE_IDX;
  }
#undef UPDATE_IDX
}

void csne(const cv::Mat depth, cv::Matx33f K, cv::Mat *result) {
  float fx = K(0, 0);
  float fy = K(1, 1);
  float u0 = K(0, 2);  // col
  float v0 = K(1, 2);  // row

  const size_t cols = depth.cols;  // col
  const size_t rows = depth.rows;  // row

  cv::Mat z_laplace(rows, cols, CV_32FC1); //保存拉普拉斯的数值
  cv::Mat N(rows, cols, CV_32FC3);  //???????normal????

  if (!(result->rows == rows && result->cols == cols &&
        result->type() == CV_32FC3)) {
    *result = cv::Mat(rows, cols, CV_32FC3);  //??????
  }

  float *idx_m1 = reinterpret_cast<float *>(depth.data);  //?????
  float *idx_p1 = reinterpret_cast<float *>(depth.data) + cols + cols;  //?????
  float *idx_o = reinterpret_cast<float *>(depth.data) + cols;  //?????
  float *idx_la = reinterpret_cast<float *>(z_laplace.data) + cols;  //?????

  float *end_idx;
  cv::Vec3f *idx_n =
      reinterpret_cast<cv::Vec3f *>(N.data) + cols;  //?????????normal???
  int ONE_MINUS_COL(cols - 1);

#define UPDATE_IDX idx_m1++, idx_p1++, idx_o++, du++, idx_n++ , idx_la++
  for (int v(2), dv(1 - v0); v != rows; ++v, ++dv) {           //???ÿ???
    end_idx = idx_o + ONE_MINUS_COL;
    int du = -u0;
    UPDATE_IDX;

    for (; idx_o != end_idx; UPDATE_IDX) {
      //cv::Vec3f &nn = *idx_n;

      float gu = *idx_o - *(idx_o - 1);  //?°?????
      float gv = *idx_o - *idx_m1;       //?°?????
      //?õ???normal?????õ????
      *idx_n = {gu * fx, gv * fy, -(*idx_o + (dv)*gv + (du)*gu)};
      *idx_la =
          fabs(((*(idx_o + 1) - *idx_o) + (*(idx_o - 1) - *idx_o)) +
              ((*idx_m1 - *idx_o) + (*idx_p1 - *idx_o)));
    }
    UPDATE_IDX;
  }
#undef UPDATE_IDX
#define UPDATE_IDX_N n_idx_o++, idx_res++, n_idx_m1++, n_idx_p1++, n_idx_la_o++, n_idx_la_m1++, n_idx_la_p1++

  cv::Vec3f *n_idx_m1 = reinterpret_cast<cv::Vec3f *>(N.data);
  cv::Vec3f *n_idx_p1 = reinterpret_cast<cv::Vec3f *>(N.data) + cols + cols;
  cv::Vec3f *n_idx_o = reinterpret_cast<cv::Vec3f *>(N.data) + cols;

  float *n_idx_la_m1 = reinterpret_cast<float *>(z_laplace.data) ;  //?????
  float *n_idx_la_p1 = reinterpret_cast<float *>(z_laplace.data) + cols+cols;  //?????
  float *n_idx_la_o = reinterpret_cast<float *>(z_laplace.data) + cols;  //?????

  //cv::v_float32x4 nx, ny, nz, equal, tmp, min_value;
  //float cal[10];
  float min_value;

  cv::Vec3f *n_end_idx;
  cv::Vec3f *idx_res = reinterpret_cast<cv::Vec3f *>(result->data) + cols;

  cv::v_uint32x4 a, sse_min_value;
  for (int v(2); v != rows; v++) {  //???ÿ???
    n_end_idx = n_idx_o + ONE_MINUS_COL;
    UPDATE_IDX_N;
    for (; n_idx_o != n_end_idx; UPDATE_IDX_N) {
      //static float k=0; //赋值没有花时间，比较花时间,5个数字最小值计算
      min_value = std::min(std::min(std::min(*n_idx_la_m1, *n_idx_la_p1
                                          ), *(n_idx_la_o)
                                 ),
                                 std::min(*(n_idx_la_o-1), *(n_idx_la_o+1))
      ); //比指令集代码更快


      //比较大小走if else特别花时间
      if (min_value == *(n_idx_la_o-1)) *idx_res = *(n_idx_o-1);
      else if (min_value == *(n_idx_la_o)) *idx_res = *(n_idx_o);
      else if (min_value == *(n_idx_la_o+1))*idx_res = *(n_idx_o+1);
      else if (min_value == *n_idx_la_p1) *idx_res = *n_idx_p1;
      else *idx_res = *n_idx_m1;
      continue;


      /*
      cv::v_float32x4 tmp(min_value, min_value, min_value,min_value);
      auto equal = (cv::v_float32x4(_mm_set_ps1(min_value)) == tmp);
      auto nx = cv::v_float32x4((*n_idx_m1)(0), (*n_idx_p1)(0), (*n_idx_o)(-1), (*n_idx_o)(3));
      float a = cv::v_reduce_sum(equal & nx); //1.111秒
      auto ny = cv::v_float32x4((*n_idx_m1)(0+1), (*n_idx_p1)(0+1), (*n_idx_o)(-1+1), (*n_idx_o)(3+1));
      float b = cv::v_reduce_sum(equal & ny);
      auto nz = cv::v_float32x4((*n_idx_m1)(0+2), (*n_idx_p1)(0+2), (*n_idx_o)(-1+2), (*n_idx_o)(3+2));
      float c = cv::v_reduce_sum(equal & nz);
      *idx_res={a,b,c};

      continue;
      */


      /*
      *idx_res={1,1,1};
      continue;
      float a = cv::v_reduce_sum(equal & nx);
      float b = cv::v_reduce_sum(equal & ny);
      float c = cv::v_reduce_sum(equal & nz);

      *idx_res = {cv::v_reduce_sum(equal & nx), cv::v_reduce_sum(equal & ny),
                  cv::v_reduce_sum(equal & nz)};
                  */

    }

    UPDATE_IDX_N;
  }
  return;
}
#undef UPDATE_IDX_N
