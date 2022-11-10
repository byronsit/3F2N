//
// Created by udi on 22-6-15.
//

#ifndef TRAD_INCLUDE_UTILITY_IO_HPP_
#define TRAD_INCLUDE_UTILITY_IO_HPP_


#include <exrio/exrio_opencv.h> //为了读取exr格式用的函数,对应的github为https://github.com/byronsit/exrio，下载编译安装即可

#include <utility>
#include <sys/stat.h>

enum DATASET_NAME{
  TFTN = 0,
  TFTN_PLUS = 1
};


/**
 * @brief 给数据集提供IO接口,精确的读取1个数据
 * @todo 基本没写完，肯定都要TODO*/
class TFTNP_IO{
 public:
  std::vector<cv::Mat> image;
  std::vector<std::string> channel_names;
  cv::Mat depth_image, range_image;
  cv::Mat gtx, gty, gtz, tmp;
  const std::string m_dataset_path;


  std::string m_sub_data_set[20]={ "auditorium", "Cafe",   "cosmetics_shop",  "library",  "office",
      "basketball_court",  "class",  "entertainment_area",  "museum",   "shopping_center",
      "alley", "city",   "Santoriri",   "shanghai", "street",
  "Amusement Park",   "dust",   "school",      "shopping_street",   "street2"};


  //初始化:数据集根目录
  TFTNP_IO(std::string dataset_path):
    m_dataset_path(std::move(dataset_path)){
  }

  //当前数据集图片的idx
  int m_sub_data_set_idx;
  //当前数据集的idx
  int m_data_set_idx;

  void init(){
    m_sub_data_set_idx=0;
    m_data_set_idx = 0;
  }

  //每次调用一次getdata,就可以算一次
  //通过一些数量，卡下帧数
  void GetData(cv::Mat &Z, cv::Mat &gt, cv::Matx33f &camera_K){
    if (m_sub_data_set_idx >= 500){
      m_sub_data_set_idx = 0;
      m_data_set_idx ++ ;
    }
    m_sub_data_set_idx ++ ;

    std::string in_out_door;
    if (m_data_set_idx <= 9) in_out_door = "indoor";
    else in_out_door ="outdoor";
    char s[100];
    sprintf(s,"%04d.exr",m_sub_data_set_idx);
    std::string path=m_dataset_path+""+ in_out_door+ "/"+m_sub_data_set[m_data_set_idx] + "/standard_exr/" + s;
    readexr_opencv(path, image, channel_names);
    for (int i = 0; i < channel_names.size(); ++i) {
      if (channel_names.at(i) == "Z") {
        Z = image.at(i);
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
    //给他们合并到gt傻瓜你去
    cv::merge(std::vector<cv::Mat>{gtx, gty, gtz}, gt);
    cv::Matx33f K(281.6000, 0.0, 256.0,
                  0.0, 414.5777, 212.0,
                  0.0, 0.0, 1.0);
    auto KK = K;

    //考虑到计算的问题，减1是必须的
    if (1) {
      KK(0, 2)--;
      KK(1, 2)--;
    }
    camera_K = KK;
  }


  void SaveData(std::string path, cv::Mat image){
    std::string in_out_door;
    if (m_data_set_idx <= 9) in_out_door = "indoor";
    else in_out_door ="outdoor";

    std::string save_path = path+"/"+ in_out_door+ "/"+m_sub_data_set[m_data_set_idx] + "/";
    if (m_sub_data_set_idx == 1){
      std::cout<<"mkdir"<<std::endl;
      mkdir(save_path.c_str(), S_IRWXU);
    }

    char s[100];
    sprintf(s,"%04d.png",m_sub_data_set_idx);
    cv::imwrite(save_path+s, image);


  }
};
#endif //TRAD_INCLUDE_UTILITY_IO_HPP_
