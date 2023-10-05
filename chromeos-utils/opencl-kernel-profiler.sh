#!/bin/bash

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

set -x

CLVK_LIBRARY="$(ldconfig -p | grep "libOpenCL.so$" | head -n 1 | sed 's|^.*=> \([^ ]*libOpenCL.so\)$|\1|')"
OPENCL_ICD_LOADER_LIBRARY_DIR="$(dirname $(find "/usr/local/opencl/" -name libOpenCL.so))"

OPENCL_LAYERS="/usr/local/opencl/libopencl-kernel-profiler.so" \
OCL_ICD_FILENAMES="${CLVK_LIBRARY}" \
LD_LIBRARY_PATH="${OPENCL_ICD_LOADER_LIBRARY_DIR}":$LD_LIBRARY_PATH \
"$@"
