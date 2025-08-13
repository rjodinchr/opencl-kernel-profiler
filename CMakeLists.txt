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

# cmake quirk - needed to pick up perfetto when compiled as a shared library on windows
if (WIN32)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".dll" ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

if (NOT PERFETTO_SDK_PATH)
  message(FATAL_ERROR "PERFETTO_SDK_PATH not defined")
endif ()

if (NOT PERFETTO_LIBRARY)
  add_library(perfetto STATIC ${PERFETTO_SDK_PATH}/perfetto.cc)
  set_target_properties(perfetto PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set(PERFETTO_LIBRARY perfetto)
endif()

option(SPIRV_DISASSEMBLY "Enable SPIR-V disassembly support" OFF)
if (SPIRV_DISASSEMBLY)
  find_library(SPIRV_TOOLS_LIBRARY NAMES SPIRV-Tools libSPIRV-Tools)
  find_path(SPIRV_TOOLS_INCLUDE_DIR NAMES spirv-tools/libspirv.hpp)
  if (NOT SPIRV_TOOLS_LIBRARY)
    message(FATAL_ERROR "SPIRV-Tools library not found")
  endif()
  if (NOT SPIRV_TOOLS_INCLUDE_DIR)
    message(FATAL_ERROR "SPIRV-Tools headers not found")
  endif()
endif()

add_library (opencl-kernel-profiler SHARED src/opencl-kernel-profiler.cpp)
target_include_directories(opencl-kernel-profiler PUBLIC ${PERFETTO_SDK_PATH})
if (OPENCL_HEADER_PATH)
  target_include_directories(opencl-kernel-profiler PUBLIC ${OPENCL_HEADER_PATH})
endif()

target_link_libraries(opencl-kernel-profiler ${PERFETTO_LIBRARY})

if (ANDROID)
  target_link_libraries(opencl-kernel-profiler log)
endif()

if (WIN32)
  target_link_libraries(opencl-kernel-profiler ws2_32)
endif()

if (SPIRV_DISASSEMBLY)
  target_include_directories(opencl-kernel-profiler PUBLIC ${SPIRV_TOOLS_INCLUDE_DIR})
  target_link_libraries(opencl-kernel-profiler ${SPIRV_TOOLS_LIBRARY})
  target_compile_definitions(opencl-kernel-profiler PRIVATE SPIRV_DISASSEMBLY)
endif()

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

target_compile_options(opencl-kernel-profiler PUBLIC "-DCLKP_PERFETTO_AMALGAMATED")

install(TARGETS opencl-kernel-profiler
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

if (CLKP_BUILD_TESTS)
  add_subdirectory(tests)
endif ()
