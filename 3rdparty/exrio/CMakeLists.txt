cmake_minimum_required(VERSION 2.8.3)
project(exrio)
#todo 用模块区分，不一定安装opencv模块或者python模块
# sudo apt-get install openexr*
#sudo apt-get install libopenexr*


string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(ColourBold  "${Esc}[1m")
set(Red         "${Esc}[31m")
set(Green       "${Esc}[32m")
set(Yellow      "${Esc}[33m")
set(Blue        "${Esc}[34m")
set(Magenta     "${Esc}[35m")
set(Cyan        "${Esc}[36m")
set(White       "${Esc}[37m")
set(BoldRed     "${Esc}[1;31m")
set(BoldGreen   "${Esc}[1;32m")
set(BoldYellow  "${Esc}[1;33m")
set(BoldBlue    "${Esc}[1;34m")
set(BoldMagenta "${Esc}[1;35m")
set(BoldCyan    "${Esc}[1;36m")
set(BoldWhite   "${Esc}[1;37m")

set(BUILD_WITH_OPENCV true CACHE BOOL "build opencv lib, then you can load exr to pencv")
set(BUILD_WITH_PYTHON false CACHE BOOL "build python lib, then you can use python to import exrio")
set(PYTHON_VERSION 3.6 CACHE STRING "set your python version, you can set 3.6 3.7 3.8 2.7 or others")

set(PYTHON_EXE "python${PYTHON_VERSION}")

#you can change this version

if (BUILD_WITH_OPENCV)
    find_package(OpenCV 4.2 REQUIRED)
endif()


if (BUILD_WITH_PYTHON)
#    find_package(Boost COMPONENTS system thread program_options)
    find_package(PythonLibs ${PYTHON_VERSION} REQUIRED EXACT) #这里可以修改python的版本号
    find_package(pybind11 REQUIRED)
endif()




set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mfma -msse2 -msse3 -msse4")

message("${Green}install path: ${CMAKE_INSTALL_PREFIX}${ColourReset}")

#add_definitions(-std=c++17)
#add_definitions(-mfma)
#add_definitions(-mno-avx512f)
#add_definitions(-mno-avx512vl)
#add_definitions(-DUSE_SSE_AUTO)
#add_definitions(-DSSE_MATHFUN_WITH_CODE)
#set(CPU_ALL_OPTIMIZATIONS "SSE;SSE2;SSE3;SSSE3;SSE4_1;SSE4_2;POPCNT;AVX;FP16;AVX2;FMA3;AVX_512F;AVX512VL;AVX512F;AVX_512VL")



# 一个不错的cmake教程 https://blog.csdn.net/u012564117/article/details/95085360
# 另一个不错的cmake教程 https://zhuanlan.zhihu.com/p/87738686
file(GLOB ALL_HEADERS include/exrio/*.h)
file(GLOB ALL_SOURCES include/*/*.cpp)
#aux_source_directory(include/tftfn/ ALL_SOURCES )

message("${BoldBlue}ALL_HEADERS: ${ColourReset}")
message(${ALL_HEADERS})
message("${BoldBlue}ALL_SOURCES: ${ColourReset}")
message(${ALL_SOURCES})

#debug用的代码
#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
#




#应该只要动态库即可
add_library(exrio         SHARED   include/exrio/exrio.cpp )

if (BUILD_WITH_OPENCV)
add_library(exrio_opencv SHARED   ${CMAKE_SOURCE_DIR}/3rdparty/exrio/include/exrio/exrio_opencv.cpp
                                  ${CMAKE_SOURCE_DIR}/3rdparty/exrio/include/exrio/exrio.cpp)
endif()

#if (BUILD_WITH_PYTHON)
#add_library(exrio_python SHARED   ${CMAKE_SOURCE_DIR}/include/exrio/exrio_python.cpp ${CMAKE_SOURCE_DIR}/include/exrio/exrio.cpp) 
#endif()

# 这里应该不用重命名了
#SET_TARGET_PROPERTIES(tftn_static PROPERTIES OUTPUT_NAME "tftn") #modify the static file name
#SET_TARGET_PROPERTIES(tftn_dynamic PROPERTIES OUTPUT_NAME "tftn") #modify the static file name
#SET_TARGET_PROPERTIES(tftn_readexr_dynamic PROPERTIES OUTPUT_NAME "tftn_readexr") #modify the static file name


include_directories(
        ${PROJECT_NAME}
        ${CMAKE_SOURCE_DIR}/include
        ${PYTHON_INCLUDE_DIR}
        ${pybind11_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

if (BUILD_WITH_OPENCV)
add_dependencies(exrio_opencv PRIVATE exrio )
endif()


#链接第三方库的时候，要加上PRIVATE
#pybind11_add_module(Pypcc ${PROJECT_SOURCE_DIR}/include/tftn/py_readexr.cpp)
if (BUILD_WITH_PYTHON)
pybind11_add_module(exrio_python ${PROJECT_SOURCE_DIR}/pybind11/exrio_python.cpp ) 
#add_library(exrio_python SHARED ${PROJECT_SOURCE_DIR}/pybind11/exrio_python.cpp) #没有so结尾,不nice
add_dependencies(exrio_python PRIVATE exrio )
SET_TARGET_PROPERTIES(exrio_python PROPERTIES OUTPUT_NAME "exrio_python.so") #modify the static file name
endif()

target_link_libraries(exrio
        -lIlmImf        #OpenEXR用的
        )

#安装python脚本

#execute_process(COMMAND ${PYTHON_EXE}  "-c \"import setuptools as _; print(_.__path__)\""  OUTPUT_VARIABLE python-dist-packages )
#execute_process(COMMAND ${PYTHON_EXE}  "-c  'import site; site.getsitepackages()[0]'"  OUTPUT_VARIABLE python-dist-packages )
#execute_process(COMMAND "${CMAKE_SOURCE_DIR}/cmake/install_python.bash")

message("your python exe is :" ${PYTHON_EXE})
execute_process(COMMAND ${PYTHON_EXE} "-c" "import sysconfig; print(sysconfig.get_paths()['platstdlib'])" OUTPUT_VARIABLE  python-dist-packages)

string(REPLACE "\n" "" python-dist-packages "${python-dist-packages}") #删除多余的回车符号
message("your platstdlib path: "  ${python-dist-packages})
message("python so install path:" [${python-dist-packages}/lib-dynload])
        


if (BUILD_WITH_PYTHON)
target_link_libraries(exrio_python
        exrio
        -lIlmImf        #OpenEXR用的
        )
endif()

if (BUILD_WITH_OPENCV)
target_link_libraries(exrio_opencv
        ${OpenCV_LIBS}
        exrio #肯定得依赖这个
        -lIlmImf        #OpenEXR用的
        )
endif()


return()


# https://blog.csdn.net/qq_38410730/article/details/102837401
set(CMAKE_INSTALL_LIBDIR  ${CMAKE_INSTALL_PREFIX}/lib)
set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/include)

message( "${BoldBlue}install library dir:"${ColourReset}\n ${CMAKE_INSTALL_LIBDIR})
message( "${BoldBlue}install include dir:"${ColourReset}\n ${CMAKE_INSTALL_INCLUDEDIR}	)

#export(EXPORT tftn NAMESPACE tftn:: FILE hahaha.cmake)
#主要用于生成一个 TFTNTargets.cmake，用来给其他目录做子目录时候用。教程https://blog.csdn.net/weixin_30379531/article/details/98589931


set(EXRIO_TARGETS exrio)
set(EXRIO_LIBS "-lIlmImf -lexrio")
#message(EXPORTD ${EXRIO_TARGETS})
if (BUILD_WITH_OPENCV)
    set(EXRIO_TARGETS  ${EXRIO_TARGETS} exrio_opencv)
    set(EXRIO_LIBS "${EXRIO_LIBS} -lexrio_opencv")
endif()

export(TARGETS ${EXRIO_TARGETS}
APPEND FILE "${CMAKE_BINARY_DIR}/EXRIOTargets.cmake")

install(
        TARGETS  ${EXRIO_TARGETS}
        DESTINATION lib
        EXPORT ${PROJECT_NAME} #不知道为啥这里只能是这个project_name
) #安装静态链接库


#python需要装在其他地方
if (BUILD_WITH_PYTHON)
install (
    TARGETS exrio_python
    LIBRARY DESTINATION  ${python-dist-packages}/lib-dynload )

install (
    TARGETS exrio_python
    LIBRARY DESTINATION   ${CMAKE_SOURCE_DIR}/python_module_output)

endif()


message(exio_export : ${EXRIO_TARGETS})

#install(FILES ${PROJECT_SOURCE_DIR}/include/tftn/tftn.h DESTINATION include)
#安装头文件
install(FILES ${ALL_HEADERS} DESTINATION include/${PROJECT_NAME})

#安装exrio.cmake exrio-release.cmake
#debug
install(EXPORT ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/exrio)



#配置动态库
configure_file(${CMAKE_SOURCE_DIR}/cmake/EXRIOConfig.cmake.in ${PROJECT_BINARY_DIR}/EXRIOConfig.cmake @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/EXRIOConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/exrio)

message("!!"${CMAKE_CURRENT_LIST_FILE})


#set(CMAKE_MODULE_PATH ${CMAKE_ROOT}/Modules)
# https://zhuanlan.zhihu.com/p/422919663 CMake编写install export

return()

#debug用的代码
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
