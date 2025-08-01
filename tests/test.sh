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

[[ $# -eq 3 ]] || (echo "missing input trace file, kernels directory, and/or test type (cl|spv)" && exit -1)

SCRIPT_DIR="$(dirname $(realpath "${BASH_SOURCE[0]}"))"
TRACE_FILE="$1"
KERNELS_DIR="$2"
TEST_TYPE="$3"
OUTPUT_FILE="${SCRIPT_DIR}/output.txt"

if [[ "$TEST_TYPE" == "cl" ]]; then
    EXPECTATION_FILE="${SCRIPT_DIR}/trace-expectation.txt"
    CREATE_PROGRAM_EVENT="clCreateProgramWithSource-args"
elif [[ "$TEST_TYPE" == "spv" ]]; then
    EXPECTATION_FILE="${SCRIPT_DIR}/trace-expectation-spv.txt"
    CREATE_PROGRAM_EVENT="clCreateProgramWithIL-args"
else
    echo "Test type must be 'cl' or 'spv'"
    exit 1
fi

EXPECTATION_SORTED_FILE="${EXPECTATION_FILE}.sorted"
GPU_SRC_FILE="${SCRIPT_DIR}/gpu.cl"

# Either it is in your path, or you need to define the environment variable
TRACE_PROCESSOR_SHELL=${TRACE_PROCESSOR_SHELL:-"trace_processor_shell"}

function clean() {
    rm -f "${OUTPUT_FILE}" "${EXPECTATION_SORTED_FILE}"
}
trap clean EXIT

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

echo "SELECT EXTRACT_ARG(arg_set_id, 'debug.string') FROM slice WHERE slice.name='${CREATE_PROGRAM_EVENT}'" \
    | "${TRACE_PROCESSOR_SHELL}" -q /dev/stdin "${TRACE_FILE}" \
                                 > "${OUTPUT_FILE}"
cat "${OUTPUT_FILE}"

if [[ "$TEST_TYPE" == "cl" ]]; then
    grep -F "$(grep kernel ${GPU_SRC_FILE})" "${OUTPUT_FILE}"
    diff "${GPU_SRC_FILE}" "${KERNELS_DIR}/clkp_p0.cl"
elif [[ "$TEST_TYPE" == "spv" ]]; then
    # For SPIR-V, we expect to see disassembled code in the debug string
    grep -i "OpCapability\|OpMemoryModel\|OpEntryPoint" "${OUTPUT_FILE}"
    # Check that a .spv file was written to disk
    [[ -f "${KERNELS_DIR}/clkp_p0.spv" ]] || (echo "Expected SPIR-V file not found" && exit 1)
fi
