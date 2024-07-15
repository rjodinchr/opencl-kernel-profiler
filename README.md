# OpenCL Kernel Profiler

`opencl-kernel-profiler` is a perfetto-based OpenCL kernel profiler using the layering capability of the [OpenCL-ICD-Loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader#about-layers)

# Legal

`opencl-kernel-profiler` is licensed under the terms of the [Apache 2.0 license](LICENSE).

# Dependencies

`opencl-kernel-profiler` depends on the following:

* [OpenCL-ICD-Loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader)
* [OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers)
* [perfetto](https://github.com/google/perfetto)

`opencl-kernel-profiler` also (obviously) depends on a OpenCL implementation.

# Building

`opencl-kernel-profiler` uses CMake for its build system.

To compile it, please run:
```
cmake -B <build_dir> -S <path-to-opencl-kernel-profiler> -DOPENCL_HEADER_PATH=<path-to-opencl-header> -DPERFETTO_SDK_PATH<path-to-perfetto-sdk>
cmake --build <build_dir>
```

For real life examples, have a look at:
- ChromeOS [ebuild](https://chromium.googlesource.com/chromiumos/overlays/chromiumos-overlay/+/main/dev-libs/opencl-kernel-profiler/opencl-kernel-profiler-0.0.1.ebuild)
- Github presubmit [configuration](https://github.com/rjodinchr/opencl-kernel-profiler/blob/main/.github/workflows/presubmit.yml)

# Build options

* `OPENCL_HEADER_PATH` (REQUIRED): path to [OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers).
* `PERFETTO_SDK_PATH` (REQUIRED): path to [perfetto](https://github.com/google/perfetto) sdk (`opencl-kernel-profiler` is looking for `PERFETTO_SDK_PATH/perfetto.cc` and `PERFETTO_SDK_PATH/perfetto.h`).
* `PERFETTO_LIBRARY`: name of a perfetto library already available (avoid having to compile `perfetto.cc`).
* `BACKEND`: [perfetto](https://github.com/google/perfetto) backend to use
  * `InProcess` (default): the application will generate the traces ([perfetto documentation](https://perfetto.dev/docs/instrumentation/tracing-sdk#in-process-mode)). Build options and environment variables can be used to control the maximum size of traces and the destination file where the traces will be recorded.
  * `System`: perfetto `traced` daemon will be responsible for generating the traces ([perfetto documentation](https://perfetto.dev/docs/instrumentation/tracing-sdk#system-mode)).
* `TRACE_MAX_SIZE` (only with `InProcess` backend): Maximum size (in KB) of traces that can be recorded. Can be overriden at runtime using the following environment variable: `CLKP_TRACE_MAX_SIZE` (Default: `1024`).
* `TRACE_DEST` (only with `InProcess` backend): File where the traces will be recorded. Can be overriden at runtime using the following environment variable: `CLKP_TRACE_DEST` (Default: `opencl-kernel-profiler.trace`).

# Running with OpenCL Kernel Profiler

To run an application with the `opencl-kernel-profiler`, one need to ensure the following point

* The application will link with the [OpenCL-ICD-Loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader). If not the case, one can override `LD_LIBRARY_PATH` to point to where the `libOpenCL.so` coming from the ICD Loader is.
* The ICD Loader is build with [layers enable](https://github.com/KhronosGroup/OpenCL-ICD-Loader#about-layers) (`ENABLE_OPENCL_LAYERS=ON`).
* The ICD Loader is using the correct [OpenCL implementation](https://github.com/KhronosGroup/OpenCL-ICD-Loader#about-layers). If not the case, one can override `OCL_ICD_FILENAMES` to point to the appropriate OpenCL implementation library.

## On ChromeOS

Make sure to have emerged and deployed the `opencl-icd-loader` as well as the `opencl-kernel-profiler`.

Then run the application using `opencl-kernel-profiler.sh`. This script will take care of setting all the environment variables needed to run with the `opencl-kernel-profiler`.

# Using the trace

Once traces have been generated, on can view them using the [perfetto trace viewer](https://ui.perfetto.dev).

It is also possible to make SQL queries using the [trace_processor](https://perfetto.dev/docs/analysis/trace-processor) tool of perfetto.
[Link](https://perfetto.dev/docs/quickstart/trace-analysis) to perfetto quickststart with SQL-based analysis.

Here is simple example to extract every kernel source code from the trace:
```
echo "SELECT EXTRACT_ARG(arg_set_id, 'debug.string') FROM slice WHERE slice.name='clCreateProgramWithSource-args'" | ./trace_processor -q /dev/stdin <opencl-kernel-profiler.trace>
```

# Extracting the kernel sources without perfetto

Running an application without perfetto but with the opencl-kernel-profiler layer enabled will dump the kernel sources code inside the directory pointed by `CLKP_KERNEL_DIR`. If `CLKP_KERNEL_DIR` is not set, nothing get written on disk.

# How does it work

`opencl-kernel-profiler` intercept to following calls to generate perfetto traces:

* `clCreateCommandQueue`: it modifies `properties` to enable profiling (`CL_QUEUE_PROFILING_ENABLE`).
* `clCreateCommandQueueWithProperties`: it adds `CL_QUEUE_PROPERTIES` with `CL_QUEUE_PROFILING_ENABLE`, or just set `CL_QUEUE_PROFILING_ENABLE` if `CL_QUEUE_PROPERTIES` is already set.
* `clCreateProgramWithSource`: it creates instant traces with the program source strings and initializes internal structures.
* `clCreateKernel`: it initializes internal structures.
* `clEnqueueNDRangekernel`: it creates a callback on the kernel completion. The callback will create traces with the proper timestamp for the kernel using timestamp coming from `clGetEventProfilinginfo`.

Every intercept call also generates a trace for the function.
