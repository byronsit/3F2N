

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})
message("eigen include dir:"${EIGEN3_INCLUDE_DIRS} "\n\n---------\n")

add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/exrio  ${CMAKE_SOURCE_DIR}/build/exrio/build)

#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

#add_executable(dataset_3f2n ${CMAKE_CURRENT_SOURCE_DIR}/dataset_3f2n.cpp)

add_executable(tftnp_test_tranditional ${CMAKE_CURRENT_SOURCE_DIR}/tftnp_test_tranditional.cpp
        ${CMAKE_SOURCE_DIR}/include/3rdmethod/3rd_algorithm.cpp)


link_directories(${CMAKE_SOURCE_DIR}/3rdparty/exrio/build)
#link_libraries()

#find_package(TFTN REQUIRED PATHS /home/xxyy/github/Three-Filters-to-Normal/c_code/build)
#include_directories(${TFTN_INCLUDE_DIRS})



#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

target_link_libraries(tftnp_test_tranditional
        ${OpenCV_LIBS}
        #libexrio.so
        -lexrio_opencv
        -lIlmImf        #OpenEXR用的
        )