#include <opencv2/cvconfig.h>

#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>

#include <exrio/exrio_opencv.h>

std::string input_file = std::string(INPUT_FILE) + "/0001.exr";


//show a normal image.
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

int main() {
  // camera's K matrix
  cv::Matx33d K(1056, 0, 1920 / 2, 0, 1056, 1080 / 2, 0, 0, 1);
  // input depth image, you need to reinstall opencv(do not use apt-get install)
  // with EXR try to use : sudo apt-get install libopenexr*, then install opencv
  cv::Mat depth_image, range_image;
  std::vector<cv::Mat> image;
  std::vector<std::string> channel_names;
  std::cout << "the exr file path:" << input_file << std::endl;
  readexr_opencv(input_file, image, channel_names);  // load a exr image.

  cv::Mat q(image.at(3));
  cv::Mat gtx(image.at((3))), gty(-image.at(2)), gtz(image.at(1));
  cv::Mat tmp;  // normal ground truth
  cv::merge(std::vector<cv::Mat>{gtx, gty, gtz}, tmp);
  ShowNormal("ground truth result", tmp);
  return 0;
}