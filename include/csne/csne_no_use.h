

//����һ��sse�ĵ�idx��float����
#define IDX_AT(VAR, IDX) reinterpret_cast<float *>(&VAR)[IDX]
//����ָ��Ż��İ汾��������,���Բ�����
//�汾2,Ҫ�����,����ָ�����
//���Ǽ���ͼ��ֱ�����ż������ż��Ҳ��Ӧ���ģ�����Ŀǰ����BUG��
//���ǲ�һ�����죡 ��Ҫ���ݸ��Եĵ�������
void csne_simple_sse(cv::Mat depth, cv::Matx33f K, cv::Mat *result) {
  float fx = K(0, 0);
  float fy = K(1, 1);
  float u0 = K(0, 2);  // col
  float v0 = K(1, 2);  // row

  const size_t cols = depth.cols;  // col
  const size_t rows = depth.rows;  // row

  cv::Mat z_laplace(rows, cols, CV_32FC1);
  //�����������û����ǰ����ã���ô����Ҫ�����ڴ�
  if (!(result->rows == rows && result->cols == cols &&
        result->type() == CV_32FC3)) {
    *result = cv::Mat(rows, cols, CV_32FC3);  //�ع��ڴ�
  }
  cv::Mat_<float> N(*result);

  float *idx_m1 = reinterpret_cast<float *>(depth.data);  //��һ��
  float *idx_p1 = reinterpret_cast<float *>(depth.data) + cols + cols;  //��һ��
  float *idx_o = reinterpret_cast<float *>(depth.data) + cols;  //��ǰ��
  float *end_idx;
  cv::Vec3f *idx_n = reinterpret_cast<cv::Vec3f *>(N.data) + cols;
  int ONE_MINUS_COL(cols - 1);
  int v(2), dv(1 - v0);
  bool flag = 0;  // 2��һ�飬��Ϊ1��ʱ�򣬾��ر���һ�¡�
  cv::v_float32x4 fxfyfxfy(fx, fy, fx, fy);

//#define UPDATE_IDX idx++, idx_m1++, idx_p1++, idx_o++, ++du
#define UPDATE_IDX idx_m1++, idx_p1++, idx_o++, du++, idx_n++

  for (; v != rows; ++v, ++dv) {  //���ÿһ��
    end_idx = idx_o + ONE_MINUS_COL;
    int du = -u0;
    // int dv = v - 1 - v0;
    UPDATE_IDX;

    for (; idx_o != end_idx; UPDATE_IDX) {
      //������
      // float &depth_r_0 = *(idx_o + 1);
      float &depth_l_0 = *(idx_o - 1);
      float &depth_u_0 = *idx_m1;  // ok
      // float &depth_d_0 = *idx_p1;  // ok
      float &c_0 = *idx_o;  //?
      cv::Vec3f &idx_n_0 = *idx_n;

      UPDATE_IDX;
      // float &depth_r_1 = *(idx_o + 1);
      float &depth_l_1 = *(idx_o - 1);
      float &depth_u_1 = *idx_m1;  // ok
      // float &depth_d_1 = *idx_p1;  // ok
      float &c_1 = *idx_o;  //?
      cv::Vec3f &idx_n_1 = *idx_n;

      //�ϲ�����
      cv::v_float32x4 depth01(depth_l_0, depth_u_0, depth_l_1, depth_u_0);
      cv::v_float32x4 c0c1(c_0, c_0, c_1, c_1);
      cv::v_float32x4 gugv01 = c0c1 - depth01;  //���зֱ���gu0 gv0 gu1 gv1
      cv::v_float32x4 normal_xy_01 = gugv01 * fxfyfxfy;  // normal��(x,y)��(x,y)

      // memmove(idx_n, &normal_xy_01, sizeof(char) * 1);
      // idx_n_1[0] = 1;
      cv::v_float32x4 dvgvdugu = cv::v_float32x4(du, dv - 1, du, dv) * gugv01;

      *(idx_n - 1) = {
          cv::v_extract_n<0>(normal_xy_01), cv::v_extract_n<1>(normal_xy_01),
          -(c_0 + cv::v_extract_n<0>(dvgvdugu), cv::v_extract_n<1>(dvgvdugu))};

      /*
      *(idx_n - 1) = {IDX_AT(normal_xy_01, 0), IDX_AT(normal_xy_01, 1),
                      -(c_0 + IDX_AT(dvgvdugu, 0) + IDX_AT(dvgvdugu, 1))};

      *(idx_n) = {IDX_AT(normal_xy_01, 2), IDX_AT(normal_xy_01, 3),
                  -(c_1, IDX_AT(dvgvdugu, 2), IDX_AT(dvgvdugu, 3))};
                  */
      // float gu = (c - depth_l);  //�°汾����
      // float gv = (c - depth_u);  //�°汾����
      //*idx_n = {gu * fx, gv * fy, -(c +   (du)*gu + (dv)*gv)};
    }
    UPDATE_IDX;
  }
}