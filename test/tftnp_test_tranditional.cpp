//
// Created by udi on 22-6-15.
//
//只跑传统的方法的算法


#include <exrio/exrio_opencv.h> //为了读取exr格式用的函数,对应的github为https://github.com/byronsit/exrio，下载编译安装即可
#include <opencv2/rgbd.hpp>
#include "utility/tftnp_io.hpp"
#include "utility/tictoc.hpp"


void MakeVisual(cv::Mat &result, cv::Mat &output){
  output = cv::Mat(result.rows, result.cols, CV_16UC3);
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; ++j) {
      if (result.at<cv::Vec3f>(i, j)[2] < 0){
        result.at<cv::Vec3f>(i, j) = - result.at<cv::Vec3f>(i, j);
      }

      result.at<cv::Vec3f>(i, j) =
          result.at<cv::Vec3f>(i, j) / cv::norm(result.at<cv::Vec3f>(i, j));

     output.at<cv::Vec3w>(i, j)[2] =
          (result.at<cv::Vec3f>(i, j)[0] + 1) * (65535 / 2.0);
      output.at<cv::Vec3w>(i, j)[1] =
          (result.at<cv::Vec3f>(i, j)[1] + 1) * (65535 / 2.0);
      output.at<cv::Vec3w>(i, j)[0] =
          (result.at<cv::Vec3f>(i, j)[2] + 1) * (65535 / 2.0);
    }
  }
}

/**
 * @brief 可视化用的*/
void ShowNormal(std::string win, cv::Mat result) {
  cv::Mat output;
  MakeVisual(result, output);
  cv::imshow(win.c_str(), output);
  cv::waitKey(-1);
}

//计算误差
double compare(cv::Mat result, cv::Mat gt){
  //返回2个图片的compare的ae结果
  float ag=0;
  int cnt = 0;


  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; ++j) {
      auto &r = result.at<cv::Vec3f>(i,j);
      auto &g = gt.at<cv::Vec3f>(i,j);
      cnt++;

      if (isnan(r[0]) || !isfinite(r[0])){
        r[0]=0;
        r[1]=0;
        r[2]=-1;
      }
      if (r[2]>0){
        r=-r;
      }


      float tmp = (r.dot(g));
      if (tmp<-1){
        tmp = -1;
      }
      if (tmp>1){
        tmp = 1;
      }

      if (isnan(r[0]) || !isfinite(r[0]) || isnan(tmp)){
        ShowNormal("sdfa", result);
       // std::cout<<r[0]<<" "<<r[1]<<" "<<r[2]<<std::endl;
       // std::cout<<g[0]<<" "<<g[1]<<" "<<g[2]<<std::endl;
       // std::cout<<tmp<<std::endl;
     }

      ag += acos(tmp);

      if (isnan(ag)){
       // std::cout<<r[0]<<" "<<r[1]<<" "<<r[2]<<std::endl;
       // std::cout<<g[0]<<" "<<g[1]<<" "<<g[2]<<std::endl;
       // std::cout<<tmp<<std::endl;
         if (tmp<-0.9){
           std::cout<<"fu"<<std::endl;
          tmp = -0.9;
        }
        if (tmp>0.9){

          std::cout<<"zheng"<<std::endl;
          tmp = 0.9;
        }
        std::cout<<tmp<<std::endl;
        std::cout<<acos(tmp)<<std::endl;
        std::cout<<ag<<std::endl;
        exit(-1);
      }
    }
  }
  return ag/cnt;
}

TFTNP_IO io("/media/xxyy/data/dataset/3f2n+data/");

#include "3rdmethod/3rd_algorithm.h"

void TEST_PLANESVD(cv::Mat Z, cv::Matx33f K, cv::Mat &result){
  cv::Mat range_image;
  cv::rgbd::depthTo3d(Z, K, range_image);
  GetNormal(range_image, &result, PLANESVD);
}


int main(){
  std::vector<cv::Mat> image;
  std::vector<std::string> channel_names;
  cv::Mat depth_image, range_image;
  cv::Mat gtx, gty, gtz, tmp;
  cv::Mat Z, gt;
  cv::Matx33f K;

  //indoor
  FILE *file = fopen("/media/xxyy/data/dataset/result/a1/result.txt","w");
  for (int i = 0; i < 10; ++ i){
    float ae=0;
    for (int j = 0; j < 500; ++ j){
      std::cout<<j<<std::endl;
      io.GetData(Z,gt,K);
      //TODO something
      cv::Mat result;
      //ShowNormal("asdfa", gt);

      TEST_PLANESVD(Z, K, result);
      //ShowNormal("asdfa", result);

      //去这个文件夹下保存结果


      cv::Mat output;
      MakeVisual(result, output);

      ae += compare(result, gt);


      io.SaveData("/media/xxyy/data/dataset/result/a1/", output);
      std::cout<<"ae" << ae << std::endl;
      std::cout<<"ae:"<< ae/(j+1)<<std::endl;
    }
    fprintf(file, "%lf\n", ae/500.0);



  }




/*
  readexr_opencv(path, image, channel_names);
  for (int i = 0; i < channel_names.size(); ++i) {
    if (channel_names.at(i) == "Z") {
      depth_image = image.at(i);
    }

    if (channel_names.at(i) == "R") {
      gtx = image.at(i);
    }

    if (channel_names.at(i) == "G") {
      gty = image.at(i);
    }

    if (channel_names.at(i) == "B") {
      gtz = image.at(i);
    }
  }
  cv::merge(std::vector<cv::Mat>{gtx, gty, gtz}, tmp);
  //if (need_show_gt) ShowNormal("gt", tmp);

  cv::Matx33f K(281.6000, 0.0, 256.0,
      0.0, 414.5777, 212.0,
      0.0, 0.0, 1.0);
  auto KK = K;

  //考虑到计算的问题，减1是必须的
  if (1) {
    KK(0, 2)--;
    KK(1, 2)--;
  }

  if (depth_image.type() != CV_32FC1) {
    depth_image.convertTo(depth_image, depth_image.rows, depth_image.cols,
                          CV_32FC1);
  }
  TFTN_Plus sne(depth_image.rows, depth_image.cols, CV_32FC1, K, 3);
  cv::Mat result;
  std::vector<cv::Mat> matpart(3);
  result.create(matpart[0].rows, matpart[0].cols, CV_32FC3);


  TIC();
  for (int i = 0; i < 1000; ++ i){
    cv::rgbd::depthTo3d(depth_image, KK, range_image);
    sne.Work(range_image, result);
  }
  auto k = TOC();
  std::cout << "3f2n+"<<k/1000 << std::endl;


  TIC();

  for (int i = 0; i < 1000; ++ i)
  TEST_3F2N_MEDIAN(depth_image, K, result);

  k = TOC();
  std::cout << "3f2n mid"<<k/1000 << std::endl;


  TIC();

  for (int i = 0; i < 1000; ++ i)
  TEST_3F2N_MEAN(depth_image, K, result);
  k = TOC();
  std::cout << "3f2n means"<<k/1000 << std::endl;


  return 0;
  */
}
