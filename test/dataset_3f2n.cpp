//确定针对3F2N的paper里的数据集的读写格式,进行速度测试
#include <iostream>
#include <exrio/exrio_opencv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/rgbd.hpp>

#include <tftn/tftn.h>
#include "utility/tictoc.hpp" //计时器
#include <csne/csne_fast.h>   //主函数
#include <dhsne/dhsen_fast.h>
#include <3f2n+/3f2n+.h>


void ShowNormal(std::string win, cv::Mat result) {
  cv::Mat output(result.rows, result.cols, CV_16UC3);
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; ++j) {
      result.at<cv::Vec3f>(i, j) =
          result.at<cv::Vec3f>(i, j) / cv::norm(result.at<cv::Vec3f>(i, j));
      if (result.at<cv::Vec3f>(i, j)[2] < 0) {
        result.at<cv::Vec3f>(i, j) = -result.at<cv::Vec3f>(i, j);
      }
      output.at<cv::Vec3w>(i, j)[2] =
          (result.at<cv::Vec3f>(i, j)[0] + 1) * (65535 / 2.0);
      output.at<cv::Vec3w>(i, j)[1] =
          (result.at<cv::Vec3f>(i, j)[1] + 1) * (65535 / 2.0);
      output.at<cv::Vec3w>(i, j)[0] =
          (result.at<cv::Vec3f>(i, j)[2] + 1) * (65535 / 2.0);
    }
  }
  cv::imshow(win.c_str(), output);
  cv::waitKey(-1);
}


/**
  * @brief Read depth images (.bin files)
  * */
cv::Mat LoadDepthImage(const std::string &path, const size_t width = 640,
                       const size_t height = 480){
  const int buffer_size = sizeof(float) * height * width;
  //char *buffer = new char[buffer_size];

  cv::Mat mat(cv::Size(width, height), CV_32FC1);

  // open filestream && read buffer
  std::ifstream fs_bin_(path, std::ios::binary);
  fs_bin_.read(reinterpret_cast<char*>(mat.data), buffer_size);
  fs_bin_.close();
  return mat;
}

void TEST_CSNE_SIMPLE(const cv::Mat &depth_image, cv::Matx33d K, cv::Mat &result) {
  csne_simple(depth_image, K, &result);
}

void TEST_3F2N_PLUS(const cv::Mat &depth_image, cv::Matx33d K, cv::Mat &result){
  cv::Mat range_image;
  cv::rgbd::depthTo3d(depth_image, K, range_image);
  std::vector<cv::Mat> matpart(3);

  static bool flag = 0;
  static TFTN_Plus *sne;


  if (!flag){

    sne = new TFTN_Plus(depth_image.rows, depth_image.cols, CV_32F, K, 3);
    flag = 1;
  }
  sne->Work(range_image, result);
}


void TEST_DHSNE1(const cv::Mat &depth_image, cv::Matx33d K, cv::Mat &result){
  cv::Mat range_image;
  cv::rgbd::depthTo3d(depth_image, K, range_image);
  std::vector<cv::Mat> matpart(3);
  cv::split(range_image, matpart);

  static bool flag = 0;
  static DHSNE *dhsne;


  if (!flag){

    dhsne = new DHSNE(depth_image.rows, depth_image.cols, CV_32F, K, 3);
    dhsne->initialize();
    flag = 1;
  }
  dhsne->dhsne_fast1(matpart.at(0), matpart.at(1), matpart.at(2),  &result);
}


void TEST_DHSNE3(const cv::Mat &depth_image, cv::Matx33d K, cv::Mat &result){
  cv::Mat range_image;
  cv::rgbd::depthTo3d(depth_image, K, range_image);
  std::vector<cv::Mat> matpart(3);
  cv::split(range_image, matpart);

  static bool flag = 0;
  static DHSNE *dhsne;


  if (!flag){

    dhsne = new DHSNE(depth_image.rows, depth_image.cols, CV_32F, K, 3);
    dhsne->initialize();
    flag = 1;
  }
  dhsne->dhsne_fast3(matpart.at(0), matpart.at(1), matpart.at(2),  &result);
}

//本算法的测试
void TEST_CSNE(const cv::Mat &depth_image, cv::Matx33d K, cv::Mat &result){
  csne(depth_image, K, &result);
}



//对3F2N的算法的测试
void TEST_3F2N_MEAN(const cv::Mat &depth_image, cv::Matx33d K, cv::Mat &result){
  cv::Mat range_image;
  cv::rgbd::depthTo3d(depth_image, K, range_image);
  std::vector<cv::Mat> matpart(3);
  cv::split(range_image, matpart);

  TFTN(range_image, K,  R_MEANS_8, &result);
}

void TEST_3F2N_MEDIAN(const cv::Mat &depth_image, cv::Matx33d K, cv::Mat &result){
  cv::Mat range_image;
  cv::rgbd::depthTo3d(depth_image, K, range_image);
  std::vector<cv::Mat> matpart(3);
  cv::split(range_image, matpart);
  result.create(matpart[0].rows, matpart[0].cols, CV_32FC3);
  TFTN(range_image, K, R_MEDIAN_STABLE_8 , &result);
}


void TEST_FLAS(const cv::Mat &depth_image, cv::Matx33d K, cv::Mat &result){
  cv::Mat range_image;
  cv::rgbd::depthTo3d(depth_image, K, range_image);
  //std::vector<cv::Mat> matpart(3);
  //cv::split(range_image, matpart);

  /*
  cv::rgbd::RgbdNormals FLAS(depth_image.rows, depth_image.cols, CV_32F, K, 3,
                             cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_FALS);
  FLAS(range_image, result);
   */

  static cv::rgbd::RgbdNormals *NORMAL;
  static bool flag = 0;
  if (!flag){ //确保只有一次
     NORMAL = new cv::rgbd::RgbdNormals(depth_image.rows, depth_image.cols, CV_32F, K, 3,
        cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_FALS);
    NORMAL->initialize();
    flag =1;
  }
  (*NORMAL)(range_image, result);
  return;
}


void TEST_FLAS2(const cv::Mat &depth_image, cv::Matx33d K, cv::Mat &result){
  cv::Mat range_image;
  cv::rgbd::depthTo3d(depth_image, K, range_image);
  //std::vector<cv::Mat> matpart(3);
  //cv::split(range_image, matpart);

  static cv::rgbd::RgbdNormals *FLAS;
  static bool flag = 0;
  if (!flag){
     FLAS = new cv::rgbd::RgbdNormals(depth_image.rows, depth_image.cols, CV_32F, K, 3,
        cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_FALS);
    FLAS->initialize();
    flag =1;
  }
  (*FLAS)(range_image, result);
  return;
}


static double d0=0;
static double d1=0;
static double d3=0;
static double csne_time=0;
static double tftn_mean=0;
static double tftn_median=0;
static double flas_time=0;


void TEST_ALL(cv::Mat depth_image, cv::Matx33d camera, cv::Mat result);

int main() {
  int n; //the number of depth images.
  //std::string param = std::string(DATA_PATH) + "/android/params.txt";


  std::string param =   "/media/xxyy/34874a5f-87c8-4c8b-94ef-925ce14ae22d/home/udi/Desktop/param";

  FILE *f = fopen(param.c_str(), "r");

  cv::Matx33d camera(0, 0, 0, 0, 0, 0, 0, 0, 1);
  fscanf(f, "%lf %lf %lf %lf %d", &camera(0, 0),
         &camera(1, 1), &camera(0, 2), &camera(1, 2), &n);
  camera(0, 2)--;
  camera(1, 2)--;

  cv::Mat result;





  for (int i = 1; i <= 1800; ++ i){
    std::stringstream  ss;
    char a[1000]={0};
    sprintf(a, "/media/xxyy/34874a5f-87c8-4c8b-94ef-925ce14ae22d/home/udi/3f2n的debug数据集/room/depth/%6d.bin", i);
    cv::Mat depth_image = LoadDepthImage(a, 640, 480);
    cv::Mat_<float> s(depth_image);
    for (auto &it : s) {
      if (fabs(it) < 1e-7) { //If the value equals 0, the point is infinite
        it = 1e10;
      }
    }
    cv::Mat range_image;
    result.create(depth_image.rows, depth_image.cols, CV_32FC3);
    TEST_ALL(depth_image,  camera,  result);
   if (i%100 == 0){
     std::cout<< "3f2n+" << d0/i << std::endl;
     std::cout <<"dhsen1:"<< d1/i << std::endl;
     std::cout << "dhsen3" << d3/i << std::endl;
     std::cout << "csne" <<csne_time/i << std::endl;
     std::cout << "tftn_mean" << tftn_mean/i<<std::endl;
     std::cout << "tftn_median" << tftn_median/i<<std::endl;
     std::cout << "flas" << flas_time/i<<std::endl;
     std::cout<<"----"<<std::endl;
   }


  }

  std::cout<< "3f2n+" << d0/1800 << std::endl;
  std::cout <<"dhsen1:"<< d1/1800 << std::endl;
  std::cout << "dhsen3" << d3/1800 << std::endl;
  std::cout << "csne" <<csne_time/1800 << std::endl;
  std::cout << "tftn_mean" << tftn_mean/1800<<std::endl;
  std::cout << "tftn_median" << tftn_median/1800<<std::endl;
  std::cout << "flas" << flas_time/1800<<std::endl;





  return 0;
}


void TEST_ALL(cv::Mat depth_image, cv::Matx33d camera, cv::Mat result){
  TIC();
  TEST_3F2N_PLUS(depth_image, camera, result);
  d0+=TOC();



  TIC();
  TEST_DHSNE1( depth_image, camera, result);
  d1+=TOC();

  TIC();
  TEST_DHSNE3( depth_image, camera, result);
  d3+=TOC();

  TIC();
  TEST_CSNE(depth_image, camera, result);
  csne_time+=TOC();



  TIC();
  TEST_3F2N_MEAN( depth_image, camera, result);
  tftn_mean+=TOC();


  TIC();

  TEST_3F2N_MEDIAN(depth_image, camera, result);
  tftn_median+=TOC();



  TIC();

  TEST_FLAS(depth_image, camera, result);
  flas_time += TOC();


  TIC();
  TEST_FLAS2(depth_image, camera, result);
  TOC();


}