//
// Created by udi on 22-5-13.
//

#include <opencv2/opencv.hpp>

#include <opencv2/core/hal/intrin.hpp>



//非常容易改为opencv的normal的格式，先这么用，懒得改
class DHSNE  {
 public:
  DHSNE(int rows, int cols, int depth, cv::InputArray K, int windows_size):
      rows_ (rows),
      cols_( cols),
      depth_(depth),
      window_size_(windows_size),
      K_(K.getMat()){}


  //提前分配内存,目前强制算法为CV_32F
  void initialize() {
    XS.create(rows_, cols_, CV_32FC1);
    YS.create(rows_, cols_, CV_32FC1);
    ZS.create(rows_, cols_, CV_32FC1);
    sse_x.create(rows_, cols_, CV_32FC4);
    sse_y.create(rows_, cols_, CV_32FC4);
    sse_z.create(rows_, cols_, CV_32FC4);
  }


  //预处理得到XY XZ等快捷运算数组,尽可能减少卷积运算
  void PreStep(cv::Mat &X, cv::Mat &XS, cv::Mat &sse_x){
    int COL = cols_;
    const float a19 = 1.0/9;

    float *idx = (float *) X.data + COL;
    float *idx_o = (float*)XS.data + COL;
    float *idx_m1 = (float *) X.data;
    float *idx_p1 = (float *) X.data + COL + COL;
    float *end_idx;
    cv::Vec4f *idx_sse = (cv::Vec4f*) sse_x.data + COL;

    int ONE_MINUS_COL = COL - 1;
    int v(2);
#define UPDATE_IDX idx++, idx_m1++, idx_p1++, idx_o++, idx_sse++

    for (; v!=rows_; ++v) {
      end_idx = idx + ONE_MINUS_COL;
      UPDATE_IDX;
      for (; idx != end_idx; UPDATE_IDX) {

        *idx_o =(*(idx_m1-1) + *(idx_m1) + *(idx_m1+1) +
            *(idx-1) + *(idx) + *(idx+1) +
            *(idx_p1-1) + *(idx_p1) + *(idx_p1+1)) *a19;

        *idx_sse = cv::Vec4f(*idx_m1, *(idx-1), *(idx+1), *idx_p1) ;

      //  static int cnt=0;
     //   std::cout<<"cn=" << idx<<" "<<end_idx<<" "<<idx-end_idx << std::endl;
      }
      //cv::v_float32x4
      UPDATE_IDX;
    }
#undef  UPDATE_IDX

  }


  void dhsne_fast1(cv::Mat &X, cv::Mat &Y, cv::Mat &Z, cv::Mat *result) {
    X.convertTo(X, CV_32FC1);
    Y.convertTo(Y, CV_32FC1);
    Z.convertTo(Z, CV_32FC1);
    float fx = K_(0, 0);
    float fy = K_(1, 1);
    float u0 = K_(0, 2);  // col
    float v0 = K_(1, 2);  // row

//    std::cout<<"pre"<<std::endl;

    PreStep(X,XS, sse_x);
    PreStep(Y,YS, sse_y);
    PreStep(Z,ZS, sse_z);
    Z = 1.0 / Z;

    //std::cout<<"st"<<std::endl;
    int COL = cols_;
    float *idx_x = (float *) XS.data + COL;
    float *idx_y = (float *) YS.data + COL;
    float *idx_z = (float *) ZS.data + COL;
    float *idx_d = (float *) Z.data + COL;

    cv::Vec4f *idx_sse_x = (cv::Vec4f*) sse_x.data + COL;
    cv::Vec4f *idx_sse_y = (cv::Vec4f*) sse_y.data + COL;
    cv::Vec4f *idx_sse_z = (cv::Vec4f*) sse_z.data + COL;
    cv::Vec3f *idx_o     = (cv::Vec3f*) result->data + COL;

    float *end_idx;

    int ONE_MINUS_COL = COL - 1;
    int v(2);
#define UPDATE_IDX idx_x++, idx_y++, idx_z++, idx_o++, idx_sse_x++, idx_sse_y++, idx_sse_z++, idx_d ++

    float A,B,C;
    cv::v_float32x4 AA, BB, CC, ZZ;
    float a19 = 1/9;
    for (; v!=rows_; ++v) {
      end_idx = idx_x + ONE_MINUS_COL;
      UPDATE_IDX;
      for (; idx_x != end_idx; UPDATE_IDX) {
        cv::v_float32x4 Z = cv::v_float32x4(*idx_z, *idx_z, *idx_z, *idx_z) ;
        cv::v_float32x4 X = cv::v_float32x4(*idx_x, *idx_x, *idx_x, *idx_x) ;
        cv::v_float32x4 Y = cv::v_float32x4(*idx_y, *idx_y, *idx_y, *idx_y) ;

        ZZ = (Z-*reinterpret_cast<cv::v_float32x4*>(idx_sse_z));

        AA = ZZ * (X-*reinterpret_cast<cv::v_float32x4*>(idx_sse_x));
        A=((float*)(&AA))[0] + ((float*)(&AA))[1] + ((float*)(&AA))[2] + ((float*)(&AA))[3];

        BB = ZZ * (Y-*reinterpret_cast<cv::v_float32x4*>(idx_sse_y));
        B=((float*)(&BB))[0] + ((float*)(&BB))[1] + ((float*)(&BB))[2] + ((float*)(&BB))[3];

        CC=ZZ*ZZ;
        C = ((float*)(&CC))[0] + ((float*)(&CC))[1] + ((float*)(&CC))[2] + ((float*)(&CC))[3];


        auto &nx = idx_o->operator[](0) =  (*(idx_d-1) - *(idx_d + 1)) * fx ;
        auto &ny = idx_o->operator[](1) = (*(idx_d-COL) - *(idx_d + COL)) * fy;
        auto &nz = idx_o->operator[](2) = (-nx * A - ny*B) /C;
      }
      //cv::v_float32x4
      UPDATE_IDX;
    }
#undef  UPDATE_IDX
    /*
    std::cout<<"@"<<std::endl;
    std::cout << result->size() << std::endl;
    std::cout << result->at<cv::Vec3f>(399, 499) << std::endl;

    std::cout << XS.at<float>(399, 499) << std::endl;
    std::cout << sse_x.at<cv::Vec4f>(399, 499) << std::endl;

    std::cout << ZS.at<float>(399, 499) << std::endl;
    std::cout << sse_z.at<cv::Vec4f>(399, 499) << std::endl;
    exit(-1);
     */




  }




  void dhsne_fast3(cv::Mat &X, cv::Mat &Y, cv::Mat &Z, cv::Mat *result) {
    X.convertTo(X, CV_32FC1);
    Y.convertTo(Y, CV_32FC1);
    Z.convertTo(Z, CV_32FC1);
    float fx = K_(0, 0);
    float fy = K_(1, 1);
    float u0 = K_(0, 2);  // col
    float v0 = K_(1, 2);  // row

    cv::Mat D(1.0 / Z);
    int COL = cols_;
    float *idx_x = (float *) X.data + COL;
    float *idx_y = (float *) Y.data + COL;
    float *idx_z = (float *) Z.data + COL;
    float *idx_d = (float *) D.data + COL;
    cv::Vec3f *idx_o   = (cv::Vec3f*) result->data + COL;

    float *end_idx;

    int ONE_MINUS_COL = COL - 1;
    int v(2);
#define UPDATE_IDX idx_x++, idx_y++, idx_z++, idx_d ++, idx_o++

    float A,B,C;
    cv::v_float32x4 AA, BB, CC, ZZ;

    for (; v!=rows_; ++v) {
      end_idx = idx_x + ONE_MINUS_COL;
      UPDATE_IDX;
      for (; idx_x != end_idx; UPDATE_IDX) {
        cv::v_float32x4 X = cv::v_float32x4(*idx_x, *idx_x, *idx_x, *idx_x) ;
        cv::v_float32x4 Y = cv::v_float32x4(*idx_y, *idx_y, *idx_y, *idx_y) ;
        cv::v_float32x4 Z = cv::v_float32x4(*idx_z, *idx_z, *idx_z, *idx_z) ;

        cv::v_float32x4 XD(  *(idx_x-COL), *(idx_x+COL), *(idx_x-1), *(idx_x+1) );
        cv::v_float32x4 YD(  *(idx_y-COL), *(idx_y+COL), *(idx_y-1), *(idx_y+1) );
        cv::v_float32x4 ZD(  *(idx_z-COL), *(idx_z+COL), *(idx_z-1), *(idx_z+1) );

        ZZ = (Z-ZD);
        AA = (X-XD) * ZZ;
        A=((float*)(&AA))[0] + ((float*)(&AA))[1] + ((float*)(&AA))[2] + ((float*)(&AA))[3];

        BB = (Y-YD) * ZZ;
        B=((float*)(&BB))[0] + ((float*)(&BB))[1] + ((float*)(&BB))[2] + ((float*)(&BB))[3];

        CC=ZZ*ZZ;
        C = ((float*)(&CC))[0] + ((float*)(&CC))[1] + ((float*)(&CC))[2] + ((float*)(&CC))[3];

        auto &nx = idx_o->operator[](0) =  (*(idx_d-1) - *(idx_d + 1)) * fx ;
        auto &ny = idx_o->operator[](1) = (*(idx_d-COL) - *(idx_d + COL)) * fy;
        auto &nz = idx_o->operator[](2) = (-nx * A - ny*B) /C;
      }
      //cv::v_float32x4
      UPDATE_IDX;
    }
#undef  UPDATE_IDX

    /*
    std::cout<<"@"<<std::endl;
    std::cout << result->size() << std::endl;

    std::cout << X.at<float>(399, 499) << std::endl;
    std::cout << result->at<cv::Vec3f>(399, 499) << std::endl;
    exit(-1);
     */




  }


 public:
  cv::Mat sse_x, sse_y, sse_z;
  cv::Mat ZX, ZY;
  cv::Mat ZXS, ZYS; //ZX的和， ZY的和
  cv::Mat ZZ; //Z*X Z*Y Z*Z
  cv::Mat ZS, YS, XS;// X Y Z的9宫格之和
  cv::Mat ZZS; //Z
  int rows_, cols_;
  int window_size_;
  int depth_;
  cv::Matx33f K_;


};





