# English version
## 1. TODO



# ����˵��
## 1. EXR��ʽ����
EXR��ʽ��������ͼƬ�Ĵ洢����Ҫ���������ø���������ͼƬ��ÿһ�����ء���������Ƕ��Կ��Ժܷ���ı���һ��ͼƬ�ĺܶ����ݣ�
����RGB�Լ�depth���ر���depth������������ʽ���棬��Ҫ�������ݸ�ʽת������ʮ�ָ��ӡ����Ƶģ�����normal���������ݣ��������и�����
�ô�ͳ�Ķ����ƻ���ͼƬ���棬������ȡ�����㣬���ᶪʧ���ȡ�
## 2.��װ��˵��
������Ŀǰֻ֧��linux��Ҳ��֧��windows�����ǿ���fork�����ο�����
���ǵ�������Ŀǰ��Ҫ���ڷ��������ƣ����Զ�ȡ������ȫ����float��ʽΪ׼����float��ʽ��ʱ��֧�֡�
���ȱ�����֧��opencv,��python���á���ͳ��opencv��ʽ��openexr��ʽ֧�ֲ�����.
�������cmake-gui����ֱ���޸�CMakeLists.txt�Ĵ��룬ѡ���Ƿ�װopencv����python�汾��
## 2.1  python������
```
sudo apt-get install openexr
```
�����Ƕ�python��֧�֣���python3.6Ϊ��������python�汾ֻҪ�޸�3.6Ϊ3.7�����������ּ��ɡ�
��Ȼ�������ͨ��������ʽ���а�װpython��������û�в����
```
sudo apt-get install pybind11-dev 
sudo apt-get install python3.6-dev
sudo apt-get install libpython3.6
sudo apt-get install libpython3.6-dev
sudo apt-get install libpython3.6-stdlib
```

## 2.2 opencv������
```
sudo apt-get install libopencv-dev
```
��Ȼ����Ҳ����ʹ��������ʽ���а�װ,������û�в����



# 3.���밲װ

��������ʹ��cmake-gui��������޸ĵĲ��������¼���
![figure1](figure/figure1.bmp)

��cmake������ⲿ�֣�ǰ��������԰�false��Ϊtrue����������OPENCV����PYTHON��֧�֡�������Ĭ���ǿ�����python��opencv��֧�֡�����㲻��Ҫ����ĳ�����ܣ�����ȡ������
��������Ĭ��3.6�汾��python������Ը�����������޸�Ϊ����Ҫ��python�汾��
```
set(BUILD_WITH_OPENCV false CACHE BOOL "build opencv lib, then you can load exr to pencv")
set(BUILD_WITH_PYTHON false CACHE BOOL "build python lib, then you can use python to import exrio")
set(PYTHON_VERSION 3.6 CACHE STRING "set your python version, you can set 3.6 3.7 3.8 2.7 or others")
```

Ȼ��ͺܼ��ˣ�ִ�����´��뼴��
```
mkdir build
cd build
cmake ..
make -j1
sudo make install
```
����㶮cmake-gui�Ļ�����ô�Ҳ���˵����Ҳ֪����ô�á�

# 4 ����
# 4.1 ��ȡopencv��mat
����Ŀ¼����һ��CmakeExample���ļ��У�������readexr_opencv.cpp�ļ�����CMakeLists.txt������Ϊʹ�õĲο���
# 4.2 ����python��ȡ
����Ŀ¼����һ��PythonExample�ļ��У�������demo.py,����������ȡexr��ʽ��
̫æ�ˣ���������û��ʵ�ַ���channel name�Ĺ��ܣ����Է��ص�ͼ��ᰴ���ֵ�������
���Ҳ�æ�ˣ������ˣ���ʵ��������ܡ�����������㹻���star.

# 5. һЩBUG
�����python��������work�Ļ������Կ������·�����
sudo make install�󣬻���Ŀ¼������һ���ļ���python_module_output
�������һ��exrio_python���Ƶ��ļ���
�������������exrio_python�ĺ�׺����so������û��soΪ��׺����ô�����޸�Ϊexrio_python.so
��������ļ��������Ӧpython��site-packages/exrio/���ļ��м���
�磺/usr/lib/python3.6/site-packages/exrio/exrio_python.so
����ٳ���


# ����Ĳ��ÿ�
then you can use it.
an CMAKE example can be see in 'CmakeExample'

and you can easily to use our code to load EXR format file, and convert it to opencv cv::Mat


д�����ĵ�˵���ȡ�

��ͷ�ٸġ�

# ��������
```
sudo apt-get install openexr
sudo apt-get install pybind11-dev
```


# C++ �̳�

```
 mkdir build
 cmake ..
 sudo make install
```
�����Բο�Ŀ¼��CmakeExample�Ľṹ������һ��cmake��Ŀ�����Է����ʹ�ñ����롣


# Python �̳�

Ŀǰpythonֻʵ���˶�ȡopenexr��ʽ�Ķ�ȡ����,

����ȷ�������Ѿ���װpython. ubuntu 18.04Ĭ����python2.7��python3.6
ubuntu 20.04 Ĭ�ϵ�python�汾��python3.8
��Ȼ�������apt-get install libpython3.8 ���� apt-get install libpython3.7 ֮�������װ����Ҫ�İ汾
��Ȼ��������ʹ��sudo apt-get install python3.8-dev ���� sudo apt-get install python3.7-dev ����װ��ص�ͷ�ļ�

```
mkdir build
cmake ..
sudo make install
```

��ʱ�ļ����Ӧ�ó���һ���µ��ļ��н� python_module_output
������һ��readexr_py����readexr_py.so���ļ���
����������ɶ��������Ϊreadexr_py.so���ɡ�
����һ�������ļ������ǿ��Ա�import�ġ�
������������/usr/local/lib/python3.6/dist-packages/���棬�½�һ���ļ��оͽ�readexr_py,Ȼ����������ļ�����Ϳ����ˡ�
ע�⣡��������python3.6����python3.7��ȡ������cmake��ʱ������ĵ�cmake�汾��
�����ָ��python�İ汾�Ļ�����CMakeLists.txt���޸�PythonLibs����İ汾�ż��ɡ�

��ʱ�����������PythonExample�ļ�����ĳ���demo.py
����һ��ʾ����ζ�ȡexr��ʽ�ĳ��򡣡���Ϊ�һ�û��������python�ж������ֵ�������Ƚ���һ�£���ЪЪ��д��





