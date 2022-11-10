#include <opencv2/opencv.hpp>

void csne_easy_read(cv::Mat depth, cv::Matx33f K, cv::Mat *result) {
  float fx = K(0, 0);
  float fy = K(1, 1);
  float u0 = K(0, 2);  // col
  float v0 = K(1, 2);  // row

  const float cols = depth.cols;  // col
  const float rows = depth.rows;  // row

  cv::Mat Gu(rows, cols, CV_32FC1);
  cv::Mat Gv(rows, cols, CV_32FC1);
  cv::Mat N(rows, cols, CV_32FC3);
  cv::Mat z_laplace(rows, cols, CV_32FC1);

  if (!(result->rows == rows && result->cols == cols &&
        result->type() == CV_32FC3)) {
    *result = cv::Mat(rows, cols, CV_32FC3);  //�ع��ڴ�
  }

  //__builtin_expect(!!(1), true);
  for (int v = 1; v < rows - 1; ++v) {    //��
    for (int u = 1; u < cols - 1; ++u) {  //��
      auto &gu = Gu.at<float>(v, u);
      auto &gv = Gv.at<float>(v, u);
      auto &depth_r = depth.at<float>(v, u + 1);
      auto &depth_l = depth.at<float>(v, u - 1);
      auto &depth_u = depth.at<float>(v - 1, u);
      auto &depth_d = depth.at<float>(v + 1, u);
      auto &c = depth.at<float>(v, u);

      auto &la = z_laplace.at<float>(v, u);

      gv = (c - depth_u);  //�°汾����
      gu = (c - depth_l);  //�°汾����

      auto &n = N.at<cv::Vec3f>(v, u);
      n(0) = gu * fx;
      n(1) = gv * fy;
      n(2) = -(c + (v - v0) * gv + (u - u0) * gu);
      la = fabs(4 * c - (depth_r + depth_l + depth_u + depth_d));
    }
  }

  for (int v = 1; v < rows - 1; ++v) {
    for (int u = 1; u < cols - 1; ++u) {
      auto &la_r = z_laplace.at<float>(v, u + 1);
      auto &la_l = z_laplace.at<float>(v, u - 1);
      auto &la_u = z_laplace.at<float>(v - 1, u);
      auto &la_d = z_laplace.at<float>(v + 1, u);
      auto &la_c = z_laplace.at<float>(v, u);

      auto &n_r = N.at<cv::Vec3f>(v, u + 1);
      auto &n_l = N.at<cv::Vec3f>(v, u - 1);
      auto &n_u = N.at<cv::Vec3f>(v - 1, u);
      auto &n_d = N.at<cv::Vec3f>(v + 1, u);
      auto &n_c = N.at<cv::Vec3f>(v, u);
      if (la_r < la_l && la_r < la_u && la_r < la_u && la_r < la_d &&
          la_r < la_c) {
        result->at<cv::Vec3f>(v, u) = n_r;
        continue;
      }
      if (la_l < la_u && la_l < la_d && la_l < la_c) {
         result->at<cv::Vec3f>(v, u) = n_l;
        continue;
      }
      if (la_u < la_d && la_u < la_c) {
        result->at<cv::Vec3f>(v, u) = n_u;
        continue;
      }
      if (la_d < la_c) {
        result->at<cv::Vec3f>(v, u) = n_d;
        continue;
      }
        result->at<cv::Vec3f>(v, u)= n_c;
      continue;
    }
  }
}

void csne_simple(cv::Mat depth, cv::Matx33f K, cv::Mat *result) {
  float fx = K(0, 0);
  float fy = K(1, 1);
  float u0 = K(0, 2);  // col
  float v0 = K(1, 2);  // row

  const float cols = depth.cols;  // col
  const float rows = depth.rows;  // row

  cv::Mat Gu(rows, cols, CV_32FC1);
  cv::Mat Gv(rows, cols, CV_32FC1);
  cv::Mat_<float> N(*result);
  cv::Mat z_laplace(rows, cols, CV_32FC1);
  if (!(result->rows == rows && result->cols == cols &&
        result->type() == CV_32FC3)) {
    *result = cv::Mat(rows, cols, CV_32FC3);  //�ع��ڴ�
  }



  //__builtin_expect(!!(1), true);
  for (int v = 1; v < rows - 1; ++v) {    //��
    for (int u = 1; u < cols - 1; ++u) {  //��
      auto &gu = Gu.at<float>(v, u);
      auto &gv = Gv.at<float>(v, u);
      auto &depth_r = depth.at<float>(v, u + 1);
      auto &depth_l = depth.at<float>(v, u - 1);
      auto &depth_u = depth.at<float>(v - 1, u);
      auto &depth_d = depth.at<float>(v + 1, u);
      auto &c = depth.at<float>(v, u);

      auto &la = z_laplace.at<float>(v, u);

      gv = (c - depth_u);  //�°汾����
      gu = (c - depth_l);  //�°汾����

      auto &n = N.at<cv::Vec3f>(v, u);
      n(0) = gu * fx;
      n(1) = gv * fy;
      n(2) = -(c + (v - v0) * gv + (u - u0) * gu);
    }
  }
}


