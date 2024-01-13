# Copyright 2023 The OpenCL Kernel Profiler authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required (VERSION 3.24)
project (opencl-kernel-profiler)

set(CMAKE_CXX_STANDARD 17)

if (NOT PERFETTO_SDK_PATH)
  message(FATAL_ERROR "PERFETTO_SDK_PATH not defined")
endif ()
if (NOT OPENCL_HEADER_PATH)
  message(FATAL_ERROR "OPENCL_HEADER_PATH not defined")
endif ()

if (NOT PERFETTO_LIBRARY)
  add_library(perfetto STATIC ${PERFETTO_SDK_PATH}/perfetto.cc)
  set_target_properties(perfetto PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set(PERFETTO_LIBRARY perfetto)
endif()

add_library (opencl-kernel-profiler MODULE src/opencl-kernel-profiler.cpp)
target_include_directories(opencl-kernel-profiler PUBLIC ${PERFETTO_SDK_PATH} ${OPENCL_HEADER_PATH})
target_link_libraries(opencl-kernel-profiler ${PERFETTO_LIBRARY})

set(BACKEND "InProcess"
  CACHE STRING "Select perfetto backend")
set(BACKEND_OPTIONS
  System
  InProcess
)
set_property(CACHE BACKEND PROPERTY STRINGS ${BACKEND_OPTIONS})
if (${BACKEND} STREQUAL InProcess)
  target_compile_definitions(opencl-kernel-profiler PUBLIC BACKEND_INPROCESS)
endif()

set(TRACE_MAX_SIZE 1024 CACHE STRING "Perfetto trace maximum size")
target_compile_options(opencl-kernel-profiler PUBLIC "-DTRACE_MAX_SIZE=${TRACE_MAX_SIZE}")

set(TRACE_DEST "opencl-kernel-profiler.trace" CACHE STRING "Perfetto trace filename")
target_compile_options(opencl-kernel-profiler PUBLIC "-DTRACE_DEST=\"${TRACE_DEST}\"")

if (CLKP_BUILD_TESTS)
  add_subdirectory(tests)
endif ()
