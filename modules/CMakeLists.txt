# Copyright: 2017 iCub Facility
# Author: Sean Ryan Fanello, Carlo Ciliberto, Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

option(BUILD_sparseCoder "Build sparseCoder" OFF)
option(BUILD_GIECoder "Build GIECoder" OFF)
option(BUILD_faster-rcnn_tensorrt "Build faster-rcnn_tensorrt" ON)
option(BUILD_caffeCoder "Build caffeCoder" OFF)
option(BUILD_linearClassifierModule "Build linearClassifierModule" OFF)

if (BUILD_GIECoder)
	add_subdirectory(GIECoder)
endif()

if (BUILD_faster-rcnn_tensorrt)
	add_subdirectory(faster-rcnn_tensorrt)
endif()

if (BUILD_sparseCoder)
	add_subdirectory(sparseCoder)
endif()

if (BUILD_caffeCoder)
	find_package(Caffe)
	if (Caffe_FOUND)
    		message(STATUS "Caffe is found => caffeCoder will be compiled")
    		add_subdirectory(caffeCoder)
	else()
    		message(STATUS "Caffe not found => skip compiling caffeCoder")
	endif()
endif()

if (BUILD_linearClassifierModule)
	add_subdirectory(linearClassifierModule)
endif()
