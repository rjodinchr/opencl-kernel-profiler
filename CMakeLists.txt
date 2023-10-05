cmake_minimum_required (VERSION 3.1)
project (layer)

set (SOURCES layer.c)

if (OPENCL_HEADER_PATH)
	include_directories(${OPENCL_HEADER_PATH})
endif ()

add_library (Layer MODULE ${SOURCES})
