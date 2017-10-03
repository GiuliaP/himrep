# Copyright: (C) 2014 iCub Facility - Istituto Italiano di Tecnologia
# Authors: Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

cmake_minimum_required(VERSION 2.8.8)

set(PROJECTNAME faster-rcnn_tensorrt)
project(${PROJECTNAME})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
       -O3 -gencode arch=compute_53,code=sm_53
)

option(USE_CUDNN "Build faster-rcnn_tensorrt with cuDNN library support" ON)

if(USE_CUDNN)

###########################################################
  
  set(CUDNN_ROOT "" CACHE PATH "CUDNN root folder")

  find_path(CUDNN_INCLUDE cudnn.h
            PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDA_TOOLKIT_INCLUDE}
            DOC "Path to cuDNN include directory." )

  # dynamic libs have different suffix in mac and linux
  if(APPLE)
    set(CUDNN_LIB_NAME "libcudnn.dylib")
  else()
    set(CUDNN_LIB_NAME "libcudnn.so")
  endif()

  get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)
  find_library(CUDNN_LIBRARY NAMES ${CUDNN_LIB_NAME}
   PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE} ${__libpath_hist} ${__libpath_hist}/../lib
   DOC "Path to cuDNN library.")
  
  if(CUDNN_INCLUDE AND CUDNN_LIBRARY)
    set(HAVE_CUDNN  TRUE PARENT_SCOPE)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)

    file(READ ${CUDNN_INCLUDE}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)

    # cuDNN v3 and beyond
    string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
           CUDNN_VERSION_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
           CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
    string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
           CUDNN_VERSION_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
           CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
    string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
           CUDNN_VERSION_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
           CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")

    if(NOT CUDNN_VERSION_MAJOR)
      set(CUDNN_VERSION "???")
    else()
      set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
    endif()

    message(STATUS "Found cuDNN: ver. ${CUDNN_VERSION} found (include: ${CUDNN_INCLUDE}, library: ${CUDNN_LIBRARY})")

    string(COMPARE LESS "${CUDNN_VERSION_MAJOR}" 3 cuDNNVersionIncompatible)
    if(cuDNNVersionIncompatible)
      message(FATAL_ERROR "cuDNN version >3 is required.")
    endif()

    set(CUDNN_VERSION "${CUDNN_VERSION}" PARENT_SCOPE)
    mark_as_advanced(CUDNN_INCLUDE CUDNN_LIBRARY CUDNN_ROOT)

  endif()

###########################################################

endif()

if(HAVE_CUDNN)
    include_directories(${CUDNN_INCLUDE})
endif()

include_directories(${YARP_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR})

set(folder_source main.cpp cudaUtility.cpp fasterRCNNtensorRTExtractor.cpp)
set(folder_header fasterRCNNtensorRTExtractor.h)
source_group("Source Files" FILES ${folder_source})
source_group("Header Files" FILES ${folder_header})

cuda_add_executable(${PROJECTNAME} ${folder_source} ${folder_header})

# cudnn
if(HAVE_CUDNN)
   target_link_libraries(${PROJECTNAME} ${CUDNN_LIBRARY})
endif()

target_link_libraries(${PROJECTNAME} ${YARP_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECTNAME} ${CUDA_LIBRARIES} nvparsers nvinfer nvinfer_plugin)

install(TARGETS ${PROJECTNAME} DESTINATION bin)
yarp_install(FILES ../${PROJECTNAME}.xml DESTINATION ${ICUBCONTRIB_MODULES_INSTALL_DIR})
