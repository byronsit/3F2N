#ifndef TFTN_PLUSS_H_
#define TFTN_PLUSS_H_

#include <immintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <avxintrin.h>
#include <avx2intrin.h>
#include <x86intrin.h>
#include <unistd.h>
#include <numeric>
#include <VCL/vectorclass.h>
//#include <sse_mathfun.h>

#include "utility/FastMath.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/intrin.hpp>

/** @brief 核心思路，求出曲率，利用曲率最小的点，作为3F2N的展开点
 * 曲率计算公式如下
 * Z_ulaplace = abs(conv2(Z,u_laplace,'same'))./min(cat(3,1+conv2(Z,gx,'same').^2,1+conv2(Z,gxL,'same').^2,1+conv2(Z,gxR,'same').^2),[],3).^1.5;
 * Z_vlaplace = abs(conv2(Z,v_laplace,'same'))./min(cat(3,1+conv2(Z,gy,'same').^2,1+conv2(Z,gyU,'same').^2,1+conv2(Z,gyD,'same').^2),[],3).^1.5;
 * */

//非常容易改为opencv的normal的格式，先这么用，懒得改
class TFTN_Plus  {
 public:
  TFTN_Plus(int rows, int cols, int depth, cv::InputArray K, int windows_size):
      rows_ (rows),
      cols_( cols),
      depth_(depth),//这个好像暂时没用啊
      window_size_(windows_size), K_(K.getMat()){}



#define SURREND(x) idx_m1[-1][x], idx_m1[0][x], idx_m1[1][x],idx[-1][x], idx[1][x],idx_p1[-1][x], idx_p1[0][x], idx_p1[1][x]
#define SR5(x) idx_m1[0][x], idx[-1][x] , idx[0][x], idx[1][x], idx_p1[0][x] //上左中右下 5个数值,x选0 1 2分别对应xyz
  void Work(const cv::Mat &input, //range image
                          cv::Mat &output){

    //const Vec8f kernel_x(1, 0, -1, 2, -2, 1, 0, -1);
    //const Vec8f kernel_y(1, 2, 1, 0, 0, -1, -2, -1);
    const Vec8f kernel_x(0, 0, 0, 1, -1, 0, 0, 0);
    const Vec8f kernel_y(0, 1, 0, 0, 0, 0, -1, 0);

    cv::Mat Z_l = cv::Mat(input.rows, input.cols, CV_32FC2);//可以优化，但是无所谓了,倦了，差不多了
    float fx = K_(0, 0);
    float fy = K_(1, 1);

    output.create(input.rows,input.cols, CV_32FC3);
    int COL = input.cols;
    cv::Vec3f* idx = (cv::Vec3f*)input.data + COL;
    cv::Vec2f * idx_zl = (cv::Vec2f*)Z_l.data + COL;
    cv::Vec3f* idx_m1 =(cv::Vec3f*)input.data;
    cv::Vec3f* idx_p1 = (cv::Vec3f*)input.data + COL + COL;
    cv::Vec3f* idx_o = (cv::Vec3f*)output.data + COL;
    cv::Vec3f* end_idx;

    int ONE_MINUS_COL = COL - 1;
    int v(2);
    const Vec8f ZERO(1e5);
    const Vec8f NEG_INF(-1e5);
    const float mul[9]={0,1,1/2.0,1/3.0,1/4.0,1/5.0,1/6.0,1/7.0,1/8.0};
#define UPDATE_IDX idx++, idx_m1++, idx_p1++, idx_o++, idx_zl ++
    for (; v!=input.rows; ++v){
      end_idx = idx + ONE_MINUS_COL;
      UPDATE_IDX;
      for (; idx != end_idx;UPDATE_IDX){
        Vec8f D = Vec8f(SURREND(2));
        float& nx = idx_o->operator()(0)=horizontal_add(kernel_x * fx / D);
        float& ny = idx_o->operator()(1)=horizontal_add(kernel_y * fy / D);

        //不做无穷判定
        //if ((*((int*)&nx) & 0x7FFFFFFF) < 1e-7 &&(*((int*)&ny) & 0x7FFFFFFF) < 1e-7) {
        //  idx_o->operator()(2) = -1;
        //  continue;
        //}
        Vec8f DX =  (idx->operator()(0) - Vec8f(SURREND(0))) * nx;
        Vec8f DY =  (idx->operator()(1) - Vec8f(SURREND(1))) * ny;

        Vec8f Z = D - idx->operator()(2);
        DX = (DX + DY) / Z;

        float* tmp=((float*)&DX);
        int c=0;
        for (int i = 0; i <= 7; isnan(tmp[i]) ? i++: tmp[c++] = tmp[i++]);

        std::sort(tmp, tmp+c);

        if (c)  idx_o->operator()(2) = (c&1) ? tmp[c>>1] : (tmp[c>>1] + tmp[(c>>1)-1]) * 0.5;
        else idx_o->operator()(2) = 0;

        //上左中 右下
        //0 1 2 3 4
        Vec8f Z5(SR5(2), 0,0,0);
        float up_u = FastAbs((Z5[2] +Z5[2]) - (Z5[1] + Z5[3]));
        float A_u = Pow2(0.5 * (Z5[1] - Z5[3]));
        float B_u = Pow2(Z5[1] - Z5[2]);
        float C_u = Pow2(Z5[3] - Z5[2]);
        float down_u = pow(std::min(std::min(A_u, B_u), C_u)+1, 1.5);
        float zu = up_u / down_u;

        float up_v = FastAbs((Z5[2] + Z5[2]) - (Z5[0] + Z5[4]));
        float A_v = Pow2(0.5 * (Z5[0] - Z5[4]));
        float B_v = Pow2(Z5[0] - Z5[2]);
        float C_v = Pow2(Z5[4] - Z5[2]);
        float down_v = pow(std::min(std::min(A_v, B_v), C_v) +1, 1.5);
        float zv = up_v / down_v;
        cv::Vec2f t;
        (*idx_zl)(0) = std::max(zu, zv);
        (*idx_zl)(1) = idx_o->operator()(2);
//        continue;
      }
      UPDATE_IDX;
    }

#undef UPDATE_IDX
#define UPDATE_IDX idx_o++, idx_zl ++
    //根据曲率选择合适的
    idx_o = (cv::Vec3f*)output.data + COL;
    idx_zl = (cv::Vec2f*)Z_l.data + COL;
    v=2;
    for (; v!=input.rows; ++v){
      end_idx = idx_o + ONE_MINUS_COL;
      UPDATE_IDX;
      for (; idx_o != end_idx;UPDATE_IDX) {
        //std::cout<<"@@"<<std::endl;
        Vec4f v={(*(idx_zl-COL))(0), (*(idx_zl-1))(0), (*(idx_zl+1))(0), (*(idx_zl+ COL))(0) };
        //std::cout<<v[0]<<" "<<v[1]<<" " <<v[2]<<" "<<v[3]<<std::endl;
        //疯狂check哪一个是最小的,速度可能比较慢
        if ( (*idx_zl)(0) < v[0] &&
            (*idx_zl)(0) < v[1] &&
            (*idx_zl)(0) < v[2] &&
            (*idx_zl)(0) < v[3]){
          (*idx_o)(2) = (*idx_zl)(1);
        //  std::cout<<"@@@@"<<std::endl;
        }else if (v[0] < v[1] &&
            v[0] < v[2] &&
            v[0] < v[3]){ //上面的最大
          (*idx_o)(2) = (*(idx_zl-COL))(1);
         // std::cout<<"@@@@"<<std::endl;
        }else if (v[1] < v[2] &&
            v[1] < v[3]){
          (*idx_o)(2) =  (*(idx_zl-1))(1);
        }else if (v[2] < v[3]){
          (*idx_o)(2) =  (*(idx_zl+1))(1);
        }else{
          (*idx_o)(2) =  (*(idx_zl+COL))(1);
        }
      }
    }
  }
#undef SURREND



  /** @brief 简单代码可读版本
   * 目的是便于阅读的简单版本,不涉及太多复杂的优化 */
  void SimpleWork(cv::Mat &X, cv::Mat &Y, cv::Mat &Z, cv::Mat *result) {
    X.convertTo(X, CV_32FC1);
    Y.convertTo(Y, CV_32FC1);
    Z.convertTo(Z, CV_32FC1); //转换为float类型
    float fx = K_(0, 0);
    float fy = K_(1, 1);
    std::cout<<fx<<" " <<fy << std::endl;
    std::cout<<X.rows << std::endl;

    Z_l = cv::Mat(X.rows, X.cols, CV_32FC1);
    cv::Mat D = 1.0 / Z;

    for (int v = 1; v < X.rows - 1; ++v) {
      for (int u = 1; u < X.cols - 1; ++u) {
        auto &nx = result->at<cv::Vec3f>(v, u)[0];
        auto &ny = result->at<cv::Vec3f>(v, u)[1];

        float gu = (D.at<float>(v, u - 1) - D.at<float>(v, u + 1));
        float gv = (D.at<float>(v - 1, u) - D.at<float>(v + 1, u));
        nx = gu * fx;
        ny = gv * fy;

        //计算曲率
        float up_u = fabs(Z.at<float>(v, u) + Z.at<float>(v, u) - Z.at<float>(v, u - 1) - Z.at<float>(v, u + 1));
        float A_u = 1 + pow((0.5 * (Z.at<float>(v, u - 1) - Z.at<float>(v, u + 1))), 2);
        float B_u = 1 + pow(Z.at<float>(v, u - 1) - Z.at<float>(v, u), 2);
        float C_u = 1 + pow(Z.at<float>(v, u + 1) - Z.at<float>(v, u), 2);
        float down_u = std::min(std::min(A_u, B_u), C_u);
        float zu = up_u / down_u;

        float up_v = fabs(Z.at<float>(v, u) + Z.at<float>(v, u) - Z.at<float>(v - 1, u ) - Z.at<float>(v + 1, u ));
        float A_v = 1 + pow((0.5 * (Z.at<float>(v - 1, u ) - Z.at<float>(v + 1, u ))), 2);
        float B_v = 1 + pow(Z.at<float>(v - 1, u) - Z.at<float>(v, u), 2);
        float C_v = 1 + pow(Z.at<float>(v + 1, u) - Z.at<float>(v, u), 2);
        float down_v = std::min(std::min(A_v, B_v), C_v);
        float zv = up_v / down_v;

        float z_lp = std::max(zu, zv);
        //if (v == 199 && u == 199){
        //  std::cout<< A_u<<" "<<B_u<<" "<<C_u << std::endl;
        //  std::cout<<up_u << std::endl;
        //  std::cout << zu <<" " << zv << " "<< z_lp << std::endl;
        //}
        Z_l.at<float>(v, u) = z_lp;
      }
    }
    std::cout<< result->at<cv::Vec3f>(199, 199)<<std::endl;


    for (int v = 1; v < X.rows - 1; ++v) {
      for (int u = 1; u < X.cols - 1; ++u) {
        //选最小
        float dx, dy, dz;
        auto &nx = result->at<cv::Vec3f>(v, u)[0];
        auto &ny = result->at<cv::Vec3f>(v, u)[1];
#define ZZ(x, y) Z_l.at<float>(x, y)
#define ZZ0 ZZ(v-1, u - 1)
#define ZZ1 ZZ(v-1, u)
#define ZZ2 ZZ(v-1, u + 1)
#define ZZ3 ZZ(v, u - 1)
#define ZZ4 ZZ(v, u)
#define ZZ5 ZZ(v, u + 1)
#define ZZ6 ZZ(v + 1, u - 1)
#define ZZ7 ZZ(v + 1, u)
#define ZZ8 ZZ(v + 1, u + 1)

        Vec8f  ZL={ZZ0, ZZ1, ZZ2, ZZ3, ZZ5, ZZ6, ZZ7, ZZ8};
        //__m256 ZL={ZZ0, ZZ1, ZZ2, ZZ3, ZZ5, ZZ6, ZZ7, ZZ8}; //注意没有ZZ4


        float max01 = std::max(ZZ0, ZZ1);
        float max23 = std::max(ZZ2, ZZ3);
        float max78 = std::max(ZZ7, ZZ8);
        float max56 = std::max(ZZ5,ZZ6);






        //v+1,u最小
        /*
        if (Z_l.at<float>(v + 1, u)  < Z_l.at<float>(v - 1, u) &&
            Z_l.at<float>(v + 1, u)  < Z_l.at<float>(v , u + 1) &&
            Z_l.at<float>(v + 1, u)  < Z_l.at<float>(v , u - 1)){
          dx = X.at<float>(v + 1, u) - X.at<float>(v, u);
          dy = Y.at<float>(v + 1, u) - Y.at<float>(v, u);
          dz = Z.at<float> (v + 1, u) - Z.at<float>(v, u);
        }else if (Z_l.at<float>(v -1, u) < Z_l.at<float>(v, u + 1) &&
            Z_l.at<float>(v - 1, u) < Z_l.at<float>(v, u - 1)){
          dx = X.at<float>(v - 1, u) - X.at<float>(v, u);
          dy = Y.at<float>(v - 1, u) - Y.at<float>(v, u);
          dz = Z.at<float> (v - 1, u) - Z.at<float>(v, u);
        }else if (Z_l.at<float>(v, u-1) < Z_l.at<float>(v, u + 1)){
          dx = X.at<float>(v , u - 1) - X.at<float>(v, u);
          dy = Y.at<float>(v , u - 1) - Y.at<float>(v, u);
          dz = Z.at<float> (v , u - 1) - Z.at<float>(v, u);
        }else{
          dx = X.at<float>(v , u + 1) - X.at<float>(v, u);
          dy = Y.at<float>(v , u + 1) - Y.at<float>(v, u);
          dz = Z.at<float> (v , u + 1) - Z.at<float>(v, u);
        }
         */
        //根据dx, dy, dz求解nz
        auto &nz = result->at<cv::Vec3f>(v, u)[2];
        nz = -(nx*dx+ny*dy)/dz;
      }
    }
  }

 public:
  cv::Mat Z_l;
  int rows_, cols_;
  int window_size_;
  int depth_;
  cv::Matx33f K_;


};



#endif //TFTN_PLUSS_H_
