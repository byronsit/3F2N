
then you can use it.
an CMAKE example can be see in 'CmakeExample'

and you can easily to use our code to load EXR format file, and convert it to opencv cv::Mat


д�����ĵ�˵���ȡ�

��ͷ�ٸġ�

# ��������
```
sudo apt-get install openexr
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






