#include <cuda.h>

#include <thrust/extrema.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cuda_runtime.h>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "device_launch_parameters.h"

#include "stdafx.h"


#define Block_x 32
#define Block_y 32

using namespace std;
using namespace cv;

__global__ void GetLaplaceNormal(float* r_nx, float* r_ny, float* r_nz,
                                 float* laplace, float* Z) {
  int v = blockDim.y * blockIdx.y + threadIdx.y;
  int u = blockDim.x * blockIdx.x + threadIdx.x;
  if ((u >= 1) && (u < umax - 1) && (v >= 1) && (v < vmax - 1)) {
    const int idx0 = v * umax + u;
    const int left_idx = idx0 - 1;
    const int right_idx = idx0 + 1;
    const int up_idx = (v - 1) * umax + u;
    const int down_idx = (v + 1) * umax + u;
    float gv = Z[down_idx] - Z[up_idx];
    float gu = Z[right_idx] - Z[left_idx];
    r_nx[idx0] = gu * fx;
    r_ny[idx0] = gv * fy;
    r_nz[idx0] = -(Z[idx0] + (v - vo) * gv + (u - uo) * gu);

    if (r_nz[idx0] > 0) {
      r_nx[idx0] = -r_nx[idx0];
      r_ny[idx0] = -r_ny[idx0];
      r_nz[idx0] = -r_nz[idx0];
    }
    

    laplace[idx0] = fabs(
        4 * Z[idx0] - (Z[left_idx] + Z[right_idx] + Z[up_idx] + Z[down_idx]));
//    if (laplace[idx0] < 0) {
//      laplace[idx0] = -laplace[idx0];
//    }
  }
}

__global__ void GetFinalNormal(float* r_nx, float* r_ny, float* r_nz,
                               float* laplace, float* nx, float* ny,
                               float* nz) {
  int v = blockDim.y * blockIdx.y + threadIdx.y;
  int u = blockDim.x * blockIdx.x + threadIdx.x;
  if ((u >= 1) && (u < umax - 1) && (v >= 1) && (v < vmax - 1)) {
    const int idx0 = v * umax + u;
    const int left_idx = idx0 - 1;
    const int right_idx = idx0 + 1;
    const int up_idx = (v - 1) * umax + u;
    const int down_idx = (v + 1) * umax + u;
    float min_value = min(min(min(laplace[left_idx], laplace[right_idx]),
                              min(laplace[up_idx], laplace[down_idx])),
                          laplace[idx0]);

    if (min_value == laplace[idx0]) {
      nx[idx0] = r_nx[idx0];
      ny[idx0] = r_ny[idx0];
      nz[idx0] = r_nz[idx0];
    }
    if (min_value == laplace[left_idx]) {
      nx[idx0] = r_nx[left_idx];
      ny[idx0] = r_ny[left_idx];
      nz[idx0] = r_nz[left_idx];
    }
    if (min_value == laplace[right_idx]) {
      nx[idx0] = r_nx[right_idx];
      ny[idx0] = r_ny[right_idx];
      nz[idx0] = r_nz[right_idx];
    }

    if (min_value ==laplace[up_idx]) {
      nx[idx0] = r_nx[up_idx];
      ny[idx0] = r_ny[up_idx];
      nz[idx0] = r_nz[up_idx];
    }

    if (min_value == laplace[down_idx]) {
      nx[idx0] = r_nx[down_idx];
      ny[idx0] = r_ny[down_idx];
      nz[idx0] = r_nz[down_idx];
    }
  }
}

void cal(float* Z, float* cpu_nx, float* cpu_ny, float* cpu_nz);

int main(int, char) {

  check_gpu_compute_capability();

  const int pixel_number = vmax * umax;
  const int float_memsize = sizeof(float) * pixel_number;

  float* x = (float*)calloc(pixel_number, sizeof(float));
  float* y = (float*)calloc(pixel_number, sizeof(float));
  float* z = (float*)calloc(pixel_number, sizeof(float));
  float* cpu_z = (float*)calloc(pixel_number, sizeof(float));

  load_data(1, cpu_z);

  double st = clock();

  for (int i = 0; i < 1000; ++ i) {
    cal(cpu_z, x, y, z);
  }
  std::cout << (clock() - st) / CLOCKS_PER_SEC << std::endl;

  vis(x, y, z);
}


void cal(float* cpu_z, float* cpu_nx, float* cpu_ny, float* cpu_nz){
  // std::cout << "copy1 " << std::endl;
  const int pixel_number = vmax * umax;
  dim3 threads = dim3(Block_x, Block_y);
  dim3 blocks = dim3(idivup(umax, threads.x), idivup(vmax, threads.y));

  //下面都是GPU计算用的变量
  float* r_nx;       //= (float*)calloc(pixel_number, sizeof(float));
  float* r_ny;       //= (float*)calloc(pixel_number, sizeof(float));
  float* r_nz;       //= (float*)calloc(pixel_number, sizeof(float));
  float* z_laplace;  //= (float*)calloc(pixel_number, sizeof(float));
  float* nx;         //= (float*)calloc(pixel_number, sizeof(float));
  float* ny;         //= (float*)calloc(pixel_number, sizeof(float));
  float* nz;         //= (float*)calloc(pixel_number, sizeof(float));

  float* Z;
  ////最后结果的CPU变量
  //float* cpu_nx = (float*)calloc(pixel_number, sizeof(float));
  //float* cpu_ny = (float*)calloc(pixel_number, sizeof(float));
  //float* cpu_nz = (float*)calloc(pixel_number, sizeof(float));
  //float* cpu_z = (float*)calloc(pixel_number, sizeof(float));

  // std::cout << "copy  2" << std::endl;

  const int float_memsize = sizeof(float) * pixel_number;

  // std::cout << "copy  3" << std::endl;

  // std::cout << "copy  4" << std::endl;

  static int flag = 0;
  if (flag == 0) {
    cudaMalloc((void**)&r_nx, float_memsize);
    cudaMalloc((void**)&r_ny, float_memsize);
    cudaMalloc((void**)&r_nz, float_memsize);
    cudaMalloc((void**)&z_laplace, float_memsize);

    cudaMalloc((void**)&nx, float_memsize);
    cudaMalloc((void**)&ny, float_memsize);
    cudaMalloc((void**)&nz, float_memsize);

    cudaMalloc((void**)&Z, float_memsize);
    flag = 1;
  }
  // std::cout << "copy " << std::endl;

  //从主机上复制内存Z到GPU上
  cudaMemcpy(Z, cpu_z, float_memsize, cudaMemcpyHostToDevice);

  // 第一步，首先求解出laplace的所有z的数值

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, NULL);

  // std::cout << "@@" << std::endl;
  GetLaplaceNormal<<<blocks, threads>>>(r_nx, r_ny, r_nz, z_laplace, Z);
  cudaDeviceSynchronize();
  GetFinalNormal<<<blocks, threads>>>(r_nx, r_ny, r_nz, z_laplace, nx, ny, nz);
  cudaDeviceSynchronize();
  // normal_estimation_bg_median<<<blocks, threads>>>(
  //    nx_dev, ny_dev, nz_dev, Volume_dev, normalization, visualization);

   cudaEventRecord(stop, NULL);
   cudaEventSynchronize(stop);
   float msecTotal = 1.0f;
   cudaEventElapsedTime(&msecTotal, start, stop);
   std::cout << "runtime: " << msecTotal << std::endl;

  //从gpu把数据弄回CPU

  cudaMemcpy(cpu_nx, nx, float_memsize, cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_ny, ny, float_memsize, cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_nz, nz, float_memsize, cudaMemcpyDeviceToHost);

  cudaMemcpy(cpu_nx, r_nx, float_memsize, cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_ny, r_ny, float_memsize, cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_nz, r_nz, float_memsize, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  return;
}

/*
int main2(int, char) {
  check_gpu_compute_capability();

  // Setting kernel and nz_filter types
  kernel_type kernel = SOBEL;
  nz_filter_type nz_filter = MEDIAN;
  normalization_type normalization = POS;
  visualization_type visualization = OPEN;

  float min_runtime = 100;
  float max_runtime = 0;

  // Setting parameters
  const int pixel_number = vmax * umax;

  // Create blocks and threads
  dim3 threads = dim3(Block_x, Block_y);
  dim3 blocks = dim3(idivup(umax, threads.x), idivup(vmax, threads.y));

  // compute memsize
  const int char_memsize = sizeof(char) * pixel_number;
  const int float_memsize = sizeof(float) * pixel_number;

  // declare eight arrays
  char* M = (char*)calloc(pixel_number, sizeof(char));
  float* D = (float*)calloc(pixel_number, sizeof(float));
  float* Z = (float*)calloc(pixel_number, sizeof(float));
  float* X = (float*)calloc(pixel_number, sizeof(float));
  float* Y = (float*)calloc(pixel_number, sizeof(float));
  float* nx = (float*)calloc(pixel_number, sizeof(float));
  float* ny = (float*)calloc(pixel_number, sizeof(float));
  float* nz = (float*)calloc(pixel_number, sizeof(float));

  cv::Mat M_mat(vmax, umax, CV_8U, M);
  cv::Mat D_mat(vmax, umax, CV_32F, D);
  cv::Mat X_mat(vmax, umax, CV_32F, X);
  cv::Mat Y_mat(vmax, umax, CV_32F, Y);
  cv::Mat Z_mat(vmax, umax, CV_32F, Z);
  cv::Mat nx_mat(vmax, umax, CV_32F, nx);
  cv::Mat ny_mat(vmax, umax, CV_32F, ny);
  cv::Mat nz_mat(vmax, umax, CV_32F, nz);

  // Bind X, Y, Z and D with texture memory;
  cudaChannelFormatDesc desc_X = cudaCreateChannelDesc<float>();
  cudaChannelFormatDesc desc_Y = cudaCreateChannelDesc<float>();
  cudaChannelFormatDesc desc_Z = cudaCreateChannelDesc<float>();
  cudaChannelFormatDesc desc_D = cudaCreateChannelDesc<float>();

  cudaArray *X_texture, *Y_texture, *Z_texture, *D_texture;

  cudaMallocArray(&X_texture, &desc_X, umax, vmax);
  cudaMallocArray(&Y_texture, &desc_Y, umax, vmax);
  cudaMallocArray(&Z_texture, &desc_Z, umax, vmax);
  cudaMallocArray(&D_texture, &desc_D, umax, vmax);

  // Create four arrays to store nx, ny, nz and volume;
  float *nx_dev, *ny_dev, *nz_dev, *Volume_dev;

  cudaMalloc((void**)&nx_dev, float_memsize);
  cudaMalloc((void**)&ny_dev, float_memsize);
  cudaMalloc((void**)&nz_dev, float_memsize);
  cudaMalloc((void**)&Volume_dev, float_memsize * 9);

  for (int frm = 1; frm <= 2500; frm++) {
            load_data(
                    torusknot,
                    1,
                    X,
                    Y,
                    Z,
                    D,
                    M);

    cudaMemcpyToArray(X_texture, 0, 0, X, float_memsize,
                      cudaMemcpyHostToDevice);
    cudaMemcpyToArray(Y_texture, 0, 0, Y, float_memsize,
                      cudaMemcpyHostToDevice);
    cudaMemcpyToArray(Z_texture, 0, 0, Z, float_memsize,
                      cudaMemcpyHostToDevice);
    cudaMemcpyToArray(D_texture, 0, 0, D, float_memsize,
                      cudaMemcpyHostToDevice);

    cudaBindTextureToArray(X_tex, X_texture, desc_X);
    cudaBindTextureToArray(Y_tex, Y_texture, desc_Y);
    cudaBindTextureToArray(Z_tex, Z_texture, desc_Z);
    cudaBindTextureToArray(D_tex, D_texture, desc_D);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);

    normal_estimation_bg_median<<<blocks, threads>>>(
        nx_dev, ny_dev, nz_dev, Volume_dev, normalization, visualization);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float msecTotal = 1.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    std::cout << "runtime: " << msecTotal << std::endl;

    if (msecTotal < min_runtime) {
      min_runtime = msecTotal;
    }
    if (msecTotal > max_runtime) {
      max_runtime = msecTotal;
    }

    cudaMemcpy(nx, nx_dev, float_memsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(ny, ny_dev, float_memsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(nz, nz_dev, float_memsize, cudaMemcpyDeviceToHost);

    cv::Mat vis_mat(vmax, umax, CV_16UC3);
    output_visualization(nx, ny, nz, vis_mat);

    std::cout << "finish" << endl;

    namedWindow("result", WINDOW_AUTOSIZE);
    imshow("result", vis_mat);
    waitKey(30);

    std::cout << frm << endl;
  }
  std::cout << std::endl
            << std::endl
            << "runtime: " << min_runtime << std::endl;
  return 0;
}
*/
