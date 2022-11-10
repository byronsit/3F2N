#include <iostream>
#include <exrio/exrio_opencv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/rgbd.hpp>

#include "utility/tictoc.hpp" //计时器
#include <csne/csne_fast.h>   //主函数

using namespace std;



const int need_show_gt = false;

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
//��������abs
float myabs(float nx) {
  (*((int *)&nx) & 0x7FFFFFFF);
  // if ( < 1e-7 &&(*((int*)&ny) & 0x7FFFFFFF) < 1e-7) {
}

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

#define TEST_IDX 299, 299

int main() {
  // һ���򵥵Ķ�exr��ʽ��IO
  std::string path = "/home/zpmc/code/csne/CmakeExample/0001.exr";
  std::vector<cv::Mat> image;
  std::vector<std::string> channel_names;
  cv::Mat depth_image, range_image;
  cv::Mat gtx, gty, gtz, tmp;
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
  if (need_show_gt) ShowNormal("gt", tmp);

  //��������

  cv::Matx33f K(365.70001220703125, 0.0, 256.0, 0.0, 365.70001220703125, 212.0,
                0.0, 0.0, 1.0);
  if (1) {
    //���ǵ�matlab��1�±�ģ�������ͼ���һ����
    //��Ҳ��֪��matlab�ǲ���Ӧ��K��������1
    K(0, 2)--;
    K(1, 2)--;
  }

  cv::Mat result;  //����Ľ��normal
  if (depth_image.type() != CV_32FC1) {
    //ת������
    std::cout << "����ת���ˣ�" << std::endl;
    depth_image.convertTo(depth_image, depth_image.rows, depth_image.cols,
                          CV_32FC1);
  }
  TIC();
  for (int i = 0; i < 1000; ++i) {
   //  csne_simple(depth_image, K, &result);
    csne(depth_image, K, &result);
  }
  std::cout<<depth_image.cols<<" "<<depth_image.rows<<std::endl;
  TOC();
  // �Ż�ǰ��0 .4183796644 - 0.1599930525 -1.1167380810 0.000002861022949
  //�Ż���0 .4175513685 - 0.1596443057 -1.1144365072 0.000002861022949
  printf("%.10f\n", result.at<cv::Vec3f>(299, 299)(0));
  printf("%.10f\n", result.at<cv::Vec3f>(299, 299)(1));
  printf("%.10f\n", result.at<cv::Vec3f>(299, 299)(2));
  //ShowNormal("resl", result); //可视化

  return 0;

  /*
  cv::rgbd::depthTo3d(depth_image, K, range_image);

  if (range_image.type() != CV_32FC3) {
    //ת������
    std::cout << "����ת���ˣ�" << std::endl;
    range_image.convertTo(range_image, range_image.rows, range_image.cols,
                          CV_32FC3);
  }
  //�°汾�������ֵ 0.1329384297 0.2658768594 1.1048996449
  std::cout << range_image.at<cv::Vec3f>(TEST_IDX) << std::endl;

  for (int i = 0; i < 12345; ++i) {
    TIC();
    csne_simple(depth_image, K, &result);
    TOC();
  }
  return 0;
   //640*480 时间还补
   //

  */
}