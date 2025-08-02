#!/usr/bin/bash

# Copyright 2024 The OpenCL Kernel Profiler authors.
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

set -xe

[[ $# -eq 1 ]] || (echo "missing input 'opencl-kernel-profiler-test'" && exit -1)

SCRIPT_DIR="$(dirname $(realpath "${BASH_SOURCE[0]}"))"
CLKP_HOST_TEST="$1"
EXPECTATION_FILE="${SCRIPT_DIR}/trace-expectation.txt"
GPU_SRC_FILE="${SCRIPT_DIR}/gpu.cl"
GPU_SRC_SPVASM_FILE="${SCRIPT_DIR}/gpu.spvasm"

# Either it is in your path, or you need to define the environment variable
TRACE_PROCESSOR_SHELL=${TRACE_PROCESSOR_SHELL:-"trace_processor_shell"}

TMP_DIR="$(mktemp -d)"
KERNELS_DIR="${TMP_DIR}/kernels"
TRACE_FILE="${TMP_DIR}/trace"
OUTPUT_FILE="${TMP_DIR}/output.txt"
EXPECTATION_SORTED_FILE="${TMP_DIR}/trace-expectation.sorted"
GPU_SPV_FILE="${TMP_DIR}/gpu.spv"
GPU_SPVASM_FILE="${TMP_DIR}/gpu.spvasm"

function clean() {
    tree "${TMP_DIR}"
    rm -rf "${TMP_DIR}"
}
trap clean EXIT

mkdir -p "${KERNELS_DIR}"
spirv-as "${GPU_SRC_SPVASM_FILE}" -o "${GPU_SPV_FILE}"
spirv-dis --no-header --no-indent "${GPU_SPV_FILE}" -o "${GPU_SPVASM_FILE}"

CLKP_TRACE_DEST="${TMP_DIR}/trace" CLKP_KERNEL_DIR="${TMP_DIR}/kernels" "${CLKP_HOST_TEST}" "${GPU_SRC_FILE}" "${GPU_SPV_FILE}"

echo "SELECT name FROM slice WHERE slice.category='clkp'" \
    | "${TRACE_PROCESSOR_SHELL}" -q /dev/stdin "${TRACE_FILE}" \
    | sed 's|clkp-queue_[0-9]*|clkp_queue|' \
    | sort \
    | uniq \
          > "${OUTPUT_FILE}"
cat "${OUTPUT_FILE}"

sort "${EXPECTATION_FILE}" > "${EXPECTATION_SORTED_FILE}"
cat "${EXPECTATION_SORTED_FILE}"

diff "${OUTPUT_FILE}" "${EXPECTATION_SORTED_FILE}"

echo "SELECT EXTRACT_ARG(arg_set_id, 'debug.string') FROM slice WHERE slice.name='clCreateProgramWithSource-args'" \
    | "${TRACE_PROCESSOR_SHELL}" -q /dev/stdin "${TRACE_FILE}" \
                                 > "${OUTPUT_FILE}"
cat "${OUTPUT_FILE}"
grep -F "$(grep kernel ${GPU_SRC_FILE})" "${OUTPUT_FILE}"

diff "${GPU_SRC_FILE}" "${KERNELS_DIR}/clkp_p0.cl"

echo "SELECT EXTRACT_ARG(arg_set_id, 'debug.disassembly') FROM slice WHERE slice.name='clCreateProgramWithIL-args'" \
    | "${TRACE_PROCESSOR_SHELL}" -q /dev/stdin "${TRACE_FILE}" \
                                 > "${OUTPUT_FILE}"
cat "${OUTPUT_FILE}"
grep -F "OpName %inc \"inc\"" ${OUTPUT_FILE}

diff "${GPU_SPVASM_FILE}" "${KERNELS_DIR}/clkp_p1.spvasm"
