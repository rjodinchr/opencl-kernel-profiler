// Copyright 2024 The OpenCL Kernel Profiler authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

void vector_inc(cl_device_id device, cl_context context, cl_command_queue command_queue, size_t buffer_size,
    void *buffer, const char **source, const size_t *source_length, const size_t *global_work_size)
{
    // Check if source is SPIR-V binary (starts with SPIR-V magic number 0x07230203)
    bool is_spirv = (*source_length >= 4) && (((const uint32_t *)*source)[0] == 0x07230203);

    cl_program program;
    if (is_spirv) {
        printf("Creating program from SPIR-V binary\n");
        program = clCreateProgramWithIL(context, *source, *source_length, NULL);
    } else {
        printf("Creating program from OpenCL C source\n");
        program = clCreateProgramWithSource(context, 1, source, source_length, NULL);
    }

    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "inc", NULL);
    cl_mem cl_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buffer);

    // Write buffer to device, execute kernel and read buffer from device
    clEnqueueWriteBuffer(command_queue, cl_buffer, CL_BLOCKING, 0, buffer_size, buffer, 0, NULL, NULL);
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(command_queue, cl_buffer, CL_BLOCKING, 0, buffer_size, buffer, 0, NULL, NULL);

    clReleaseMemObject(cl_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

#define NB_ELEM 1024

int main(int argc, char **argv)
{
    printf("Starting OpenCL application\n");

    if (argc < 2) {
        fprintf(stderr,
            "At least 1 argument is expected. It should be the path(s) to the kernel source code or SPIR-V "
            "binary\n");
        return -2;
    }

    bool success = true;

    // Initialization
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, NULL, NULL);

    char platform_name[512];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    printf("CL_PLATFORM_NAME: %s\n", platform_name);

    // Process each file argument
    for (int file_idx = 1; file_idx < argc; file_idx++) {
        printf("Processing file: %s\n", argv[file_idx]);

        FILE *f_source = fopen(argv[file_idx], "rb"); // Open in binary mode to support both text and binary
        if (!f_source) {
            fprintf(stderr, "Failed to open file: %s\n", argv[file_idx]);
            return -2;
        }

        fseek(f_source, 0, SEEK_END);
        const size_t source_length = ftell(f_source);
        fseek(f_source, 0, SEEK_SET);
        char *source = (char *)malloc(source_length);
        size_t size_read = 0;
        do {
            size_read += fread(&source[size_read], 1, source_length - size_read, f_source);
        } while (size_read != source_length);
        fclose(f_source);

        uint32_t buffer[NB_ELEM];
        for (unsigned i = 0; i < NB_ELEM; i++) {
            buffer[i] = i + 42;
        }

        size_t global_work_size = NB_ELEM;
        vector_inc(device, context, command_queue, sizeof(buffer), buffer, (const char **)&source, &source_length,
            &global_work_size);

        for (unsigned i = 0; i < NB_ELEM; i++) {
            if (buffer[i] != i + 43) {
                fprintf(stderr, "[%u] Error in kernel execution: expected %u got %u\n", i, 43 + i, buffer[i]);
                success = false;
            }
        }

        free(source);
    }

    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    printf("OpenCL application completed!\n");
    return success ? 0 : -1;
}
