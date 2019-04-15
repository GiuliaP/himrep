# Copyright: 2017 iCub Facility
# Author: Sean Ryan Fanello, Carlo Ciliberto, Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

option(BUILD_sparseCoder "Build sparseCoder" OFF)
option(BUILD_GIECoder "Build GIECoder" OFF)
option(BUILD_caffeCoder "Build caffeCoder" ON)
option(BUILD_linearClassifierModule "Build linearClassifierModule" ON)

if (BUILD_GIECoder)
	add_subdirectory(GIECoder)
endif()

if (BUILD_sparseCoder)
	add_subdirectory(sparseCoder)
endif()

if (BUILD_caffeCoder)
	add_subdirectory(caffeCoder)
endif()

if (BUILD_linearClassifierModule)
	add_subdirectory(linearClassifierModule)
endif()
