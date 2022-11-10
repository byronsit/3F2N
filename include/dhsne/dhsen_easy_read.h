//
// Created by udi on 22-5-13.
//

#include <opencv2/opencv.hpp>

#include <opencv2/core/hal/intrin.hpp>


//需要开辟大量提前准备的内存
namespace trad{
cv::Mat ZX, ZY;
cv::Mat ZXS, ZYS; //ZX的和， ZY的和
cv::Mat ZZ; //Z*X Z*Y Z*Z
cv::Mat ZS, YS, XS;// X Y Z的9宫格之和
cv::Mat ZZS; //Z
int rows, cols;
int window_size;
cv::Matx33f K;
}


using namespace trad;

//类似opencv normal的初始化
namespace dhsne {
void initialize() {
  ZX.create(rows, cols, CV_32FC1);
  ZY.create(rows, cols, CV_32FC1);
  ZZ.create(rows, cols, CV_32FC1);

  XS.create(rows, cols, CV_32FC1);
  YS.create(rows, cols, CV_32FC1);
  ZS.create(rows, cols, CV_32FC1);
  ZZS.create(rows, cols, CV_32FC1);
}
}

//预处理得到XY XZ等快捷运算数组,尽可能减少卷积运算
void PreStep(cv::Mat &X, cv::Mat &Y, cv::Mat &Z, cv::Mat kernal){
  //cv::Mat kernal = cv::Mat::ones(3,3, CV_32FC1);
  ZX = Z.mul(X);
  ZY = Z.mul(Y);
  ZZ = Z.mul(Z);
  cv::filter2D(X, XS, X.depth(), kernal);
  cv::filter2D(Y, YS, Y.depth(), kernal);
  cv::filter2D(Z, ZS, Z.depth(), kernal);
  kernal.at<float>(1,1) = 0;
  cv::filter2D(ZZ, ZZS, Z.depth(), kernal);
  cv::filter2D(ZX, ZXS, Z.depth(), kernal);
  cv::filter2D(ZY, ZYS, Z.depth(), kernal);
  return ;
}

void dhsne_easy_read(cv::Mat &X, cv::Mat &Y, cv::Mat &Z, cv::Matx33f K, cv::Mat *result) {
  float fx = K(0, 0);
  float fy = K(1, 1);
  float u0 = K(0, 2);  // col
  float v0 = K(1, 2);  // row
  cv::Mat kernal=cv::Mat::ones(3,3,CV_32FC1);
  PreStep(X,Y,Z,kernal);

  const float cols = X.cols;  // col
  const float rows = X.rows;  // row
  const int dx[]={-1, -1, -1, 1, 1, 1,0, 0 ,0 };
  const int dy[]={-1, 1, 0, -1, 1, 0, -1, 1, 0};

  cv::Mat Gu(rows, cols, CV_32FC1);
  cv::Mat Gv(rows, cols, CV_32FC1);
  cv::Mat Nx(rows, cols, CV_32FC3);
  cv::Mat Ny(rows, cols, CV_32FC3);
  cv::Mat Nz(rows, cols, CV_32FC3);
  cv::Mat z_laplace(rows, cols, CV_32FC1);

  if (!(result->rows == rows && result->cols == cols &&
      result->type() == CV_32FC3)) {
    *result = cv::Mat(rows, cols, CV_32FC3);  //result数组的大小
  }


//__builtin_expect(!!(1), true);
  for (int v = 2; v < rows - 2; ++v) {
    for (int u = 2; u < cols - 2; ++u) {
      auto &gu = Gu.at<float>(v, u);
      auto &gv = Gv.at<float>(v, u);
      auto &zs = ZS.at<float>(v, u);
      auto &xs = XS.at<float>(v, u);
      auto &ys = YS.at<float>(v, u);
      auto &zzs = ZZS.at<float>(v, u);
      auto &x = X.at<float>(v, u);
      auto &y = Y.at<float>(v, u);
      auto &z = Z.at<float>(v, u);
      auto &zxs = ZXS.at<float>(v, u);
      auto &zys = ZYS.at<float>(v, u);

      gu = 1 / Z.at<float>(v, u - 1) - 1 / Z.at<float>(v, u + 1);
      gv = 1 / Z.at<float>(v - 1, u) - 1 / Z.at<float>(v + 1, u);
      auto &nx = Nx.at<float>(v, u) = gu * fx;
      auto &ny = Ny.at<float>(v, u) = gv * fy;
      //float A = (-10 * zs * xs + 81 * zzs + 9 * x * zs + 9 * xs * z) / 81;
      //float B = (-10 * zs * ys + 81 * zzs + 9 * y * zs + 9 * ys * z) / 81;

      float A = (-10 * zs * xs + 81 * zxs + 9 * x * zs + 9 * xs * z) / 81;
      float B = (-10 * zs * ys + 81 * zys + 9 * y * zs + 9 * ys * z) / 81;
      float C = (81 * zzs - 10 * zs * zs + 18 * zs * z) / 81;
      auto &nz = Nz.at<float>(v, u) = (-nx * A - ny * B) / C;
      /*
      if (v == 399 && u == 499) {
        std::cout << x << " " << y<< " " <<z << std::endl;
        std::cout << A << " " << B << " " << C << std::endl;
        std::cout << zzs <<" "<<ZZ.at<float>(399,499)<<std::endl;
        std::cout << nx <<" "<< ny <<" " <<nz << std::endl;
        std::cout << xs <<" "<< ys<< " " << zs << std::endl;
        std::cout << zxs <<" " << zys << std::endl;
        std::cout << -10 * zs * xs  <<" " << 81 * zxs <<" "<<9 * x * zs <<" "<<9 * xs * z << std::endl;
        printf("%.10f %.10f %.10f %.10f\n",
               -10 * zs * xs , 81 * zxs ,9 * x * zs ,9 * xs * z
               );
        std::cout << AA <<std::endl;
        exit(-1);
      }
       */
      continue;
    }
  }

}
