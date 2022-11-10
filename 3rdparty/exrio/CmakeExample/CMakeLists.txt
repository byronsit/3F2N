cmake_minimum_required(VERSION 2.8.3)
project(exrio_cmake_example)

add_definitions(-std=c++17)
find_package(OpenCV  4.5 REQUIRED)
find_package(EXRIO REQUIRED)

add_definitions(-DINPUT_FILE="${CMAKE_SOURCE_DIR}")

#dont need here, just need to reboot your system, or run <sudo ldconfig> 
link_directories(
    #TODO
    #/usr/local/lib/
)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${EXRIO_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIR}
)

message(EXRIO_INCLUDE_DIR ${EXRIO_INCLUDE_DIRS})

add_executable(demo_opencv
        ${CMAKE_SOURCE_DIR}/demo_opencv.cpp
        )
#very important!!
#otherwise in some machine,it will be built with c++11, you will find some error!!!!
#openexr can only be used in CXX14
set_property(TARGET demo_opencv PROPERTY CXX_STANDARD 17)

message("EXRIO libs:" ${EXRIO_LIBS})

target_link_libraries(demo_opencv
        ${EXRIO_LIBS}
        ${OpenCV_LIBS}
        #-lIlmImf        #OpenEXR用的
        )

        return ()
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
