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

[[ $# -eq 1 ]] || (echo "missing input trace file" && exit -1)

SCRIPT_DIR="$(dirname $(realpath "${BASH_SOURCE[0]}"))"
TRACE_FILE="$1"
OUTPUT_FILE="${SCRIPT_DIR}/output.txt"

# Either it is in your path, or you need to define the environment variable
TRACE_PROCESSOR_SHELL=${TRACE_PROCESSOR_SHELL:-"trace_processor_shell"}

echo "SELECT EXTRACT_ARG(arg_set_id, 'debug.string') FROM slice WHERE slice.name='clCreateProgramWithSource-args'" \
    | "${TRACE_PROCESSOR_SHELL}" -q /dev/stdin "${TRACE_FILE}" \
                                 > "${OUTPUT_FILE}"
cat "${OUTPUT_FILE}"
grep -F "$(grep kernel ${SCRIPT_DIR}/gpu.cl)" "${OUTPUT_FILE}"

echo "SELECT name FROM slice WHERE slice.category='clkp'" \
    | "${TRACE_PROCESSOR_SHELL}" -q /dev/stdin "${TRACE_FILE}" \
    | sort \
    | uniq \
    | sed 's|clkp-queue_[0-9]*|clkp_queue|' \
          > "${OUTPUT_FILE}"
cat "${OUTPUT_FILE}"
diff "${OUTPUT_FILE}" "${SCRIPT_DIR}/trace-expectation.txt"

rm "${OUTPUT_FILE}"


