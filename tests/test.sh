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

[[ $# -eq 2 ]] || (echo "missing input trace file and/or kernels directory" && exit -1)

SCRIPT_DIR="$(dirname $(realpath "${BASH_SOURCE[0]}"))"
TRACE_FILE="$1"
KERNELS_DIR="$2"
OUTPUT_FILE="${SCRIPT_DIR}/output.txt"
EXPECTATION_FILE="${SCRIPT_DIR}/trace-expectation.txt"
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

# Check for OpenCL C source events
echo "SELECT EXTRACT_ARG(arg_set_id, 'debug.string') FROM slice WHERE slice.name='clCreateProgramWithSource-args'" \
    | "${TRACE_PROCESSOR_SHELL}" -q /dev/stdin "${TRACE_FILE}" \
                                 > "${OUTPUT_FILE}"

# Check for SPIR-V binary events
echo "SELECT EXTRACT_ARG(arg_set_id, 'debug.string') FROM slice WHERE slice.name='clCreateProgramWithIL-args'" \
    | "${TRACE_PROCESSOR_SHELL}" -q /dev/stdin "${TRACE_FILE}" \
                                 >> "${OUTPUT_FILE}"

cat "${OUTPUT_FILE}"

# Check that both types of programs were created
echo "Checking for OpenCL C source program (clkp_p0):"
if echo "SELECT name FROM slice WHERE slice.name='clCreateProgramWithSource'" \
   | "${TRACE_PROCESSOR_SHELL}" -q /dev/stdin "${TRACE_FILE}" | grep -q "clCreateProgramWithSource"; then
    echo "OpenCL C source program detected"
    grep -F "$(grep kernel ${GPU_SRC_FILE})" "${OUTPUT_FILE}"
    diff "${GPU_SRC_FILE}" "${KERNELS_DIR}/clkp_p0.cl"
    echo "OpenCL C source validation passed"
else
    echo "OpenCL C source program not found"
    exit 1
fi

echo "Checking for SPIR-V binary program (clkp_p1):"
if echo "SELECT name FROM slice WHERE slice.name='clCreateProgramWithIL'" \
   | "${TRACE_PROCESSOR_SHELL}" -q /dev/stdin "${TRACE_FILE}" | grep -q "clCreateProgramWithIL"; then
    echo "SPIR-V binary program detected"
    if grep -i "OpCapability\|OpMemoryModel\|OpEntryPoint" "${OUTPUT_FILE}"; then
        echo "SPIR-V disassembly found in trace"
    else
        echo "No SPIR-V disassembly found in trace"
        exit 1
    fi
    if [[ -f "${KERNELS_DIR}/clkp_p1.il" ]]; then
        echo "IL binary file clkp_p1.il found"
    else
        echo "IL binary file clkp_p1.il not found"
        exit 1
    fi
    echo "SPIR-V binary validation passed"
else
    echo "SPIR-V binary program not found"
    exit 1
fi
