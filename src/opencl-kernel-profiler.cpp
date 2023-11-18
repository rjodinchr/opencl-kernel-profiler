// Copyright 2023 The OpenCL Kernel Profiler authors.
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

#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl_layer.h>
#include <assert.h>
#include <fstream>
#include <map>
#include <mutex>
#include <perfetto.h>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*****************************************************************************/
/* PERFETTO GLOBAL VARIABLES *************************************************/
/*****************************************************************************/

#define CLKP_PERFETTO_CATEGORY "clkp"

PERFETTO_DEFINE_CATEGORIES(perfetto::Category(CLKP_PERFETTO_CATEGORY).SetDescription("OpenCL Kernel Profiler Events"));

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

#ifdef BACKEND_INPROCESS
static std::unique_ptr<perfetto::TracingSession> gTracingSession;
#endif

/*****************************************************************************/
/* ICD DISPATCH GLOBAL VARIABLES *********************************************/
/*****************************************************************************/

static struct _cl_icd_dispatch dispatch;
static const struct _cl_icd_dispatch *tdispatch;

/*****************************************************************************/
/* MACROS ********************************************************************/
/*****************************************************************************/

#define PRINT(message, ...)                                                                                            \
    do {                                                                                                               \
        fprintf(stderr, "[CLKP] %s: " message "\n", __func__, ##__VA_ARGS__);                                          \
    } while (0)
#define CHECK(test, statement, message, ...)                                                                           \
    do {                                                                                                               \
        if (!(test)) {                                                                                                 \
            PRINT(message, ##__VA_ARGS__);                                                                             \
            statement;                                                                                                 \
        }                                                                                                              \
    } while (0)
#define CHECK_CL(err, statement, message, ...) CHECK((err) == CL_SUCCESS, statement, message, ##__VA_ARGS__)
#define CHECK_ALLOC(ptr, statement) CHECK(ptr != nullptr, statement, "allocation failed")

/*****************************************************************************/
/* CREATE KERNEL & PROGRAM ***************************************************/
/*****************************************************************************/

std::mutex g_lock;

static uint32_t program_number = 0;
static std::map<cl_program, std::string> program_to_string;
static cl_program clkp_clCreateProgramWithSource(
    cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret)
{
    std::lock_guard<std::mutex> lock(g_lock);
    std::string program_str = std::string("clkp_p") + std::to_string(program_number++);
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateProgramWithSource", "program", perfetto::DynamicString(program_str),
        "count", count);
    cl_program program = tdispatch->clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
    program_to_string[program] = program_str;
    for (unsigned i = 0; i < count; i++) {
        TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, "clCreateProgramWithSource-args", "program",
            perfetto::DynamicString(program_str), "string", perfetto::DynamicString(strings[i]));
    }
    return program;
}

static uint32_t kernel_number = 0;
static std::map<cl_kernel, std::string> kernel_to_kernel_name;
static std::map<cl_kernel, cl_program> kernel_to_program;
static cl_kernel clkp_clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret)
{
    std::lock_guard<std::mutex> lock(g_lock);
    cl_kernel kernel = tdispatch->clCreateKernel(program, kernel_name, errcode_ret);
    kernel_to_program[kernel] = program;
    kernel_to_kernel_name[kernel] = std::string(kernel_name);
    return kernel;
}

/*****************************************************************************/
/* ENQUEUE NDRANGE KERNEL ****************************************************/
/*****************************************************************************/

struct callback_data {
    cl_command_queue queue;
    cl_kernel kernel;
    size_t gidX, gidY, gidZ;
};

std::set<cl_command_queue> track_named;
static void callback(cl_event event, cl_int event_command_exec_status, void *user_data)
{
    struct callback_data *data = (struct callback_data *)user_data;
    assert(data != nullptr);
    assert(event_command_exec_status == CL_COMPLETE);

    cl_ulong start, end;
    cl_int err;
    err = tdispatch->clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
    CHECK_CL(err, free(data); return, "clGetEventProfilingInfo(CL_PROFILING_COMMAND_START) failed (%i)", err);
    err = tdispatch->clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
    CHECK_CL(err, free(data); return, "clGetEventProfilingInfo(CL_PROFILING_COMMAND_END) failed (%i)", err);
    assert(end > start);

    if (track_named.count(data->queue) == 0) {
        track_named.insert(data->queue);
        TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY,
            perfetto::DynamicString("clkp-queue_" + std::to_string((uintptr_t)data->queue)),
            perfetto::Track((uintptr_t)data->queue), (uint64_t)start - 1000);
    }

    std::string name = program_to_string[kernel_to_program[data->kernel]] + "-" + kernel_to_kernel_name[data->kernel]
        + "-" + std::to_string(data->gidX) + "." + std::to_string(data->gidY) + "." + std::to_string(data->gidZ);

    TRACE_EVENT_BEGIN(CLKP_PERFETTO_CATEGORY, perfetto::DynamicString(name), perfetto::Track((uintptr_t)data->queue),
        (uint64_t)start, "program", perfetto::DynamicString(program_to_string[kernel_to_program[data->kernel]]),
        "kernel_name", perfetto::DynamicString(kernel_to_kernel_name[data->kernel]), "gidX", data->gidX, "gidY",
        data->gidY, "gidZ", data->gidZ);
    TRACE_EVENT_END(CLKP_PERFETTO_CATEGORY, perfetto::Track((uintptr_t)data->queue), (uint64_t)end);

    free(data);
}

static cl_int clkp_clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
    const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    size_t gidX = work_dim > 0 ? global_work_size[0] : 1;
    size_t gidY = work_dim > 1 ? global_work_size[1] : 1;
    size_t gidZ = work_dim > 2 ? global_work_size[2] : 1;
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clEnqueueNDRangeKernel", "program",
        perfetto::DynamicString(program_to_string[kernel_to_program[kernel]]), "kernel_name",
        perfetto::DynamicString(kernel_to_kernel_name[kernel]), "gidX", gidX, "gidY", gidY, "gidZ", gidZ);

    bool event_is_null = event == nullptr;
    if (event_is_null) {
        event = (cl_event *)malloc(sizeof(cl_event));
        CHECK_ALLOC(event, return CL_OUT_OF_HOST_MEMORY);
    }

    cl_int err = tdispatch->clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset,
        global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);

    struct callback_data *data = nullptr;
    auto clean = [&data, &event_is_null, &err, &event](bool clean_user_data = true) {
        if (clean_user_data) {
            free(data);
        }
        if (event_is_null) {
            if (err == CL_SUCCESS) {
                cl_int err_release = tdispatch->clReleaseEvent(*event);
                CHECK_CL(err_release, /* do nothing */, "clReleaseEvent failed (%i)", err);
            }
            free(event);
        }
    };
    CHECK_CL(err, clean(); return err, "clEnqueueNDRangeKernel failed (%i)", err);

    data = (struct callback_data *)malloc(sizeof(struct callback_data));
    CHECK_ALLOC(data, clean(); return err);
    data->queue = command_queue;
    data->kernel = kernel;
    data->gidX = gidX;
    data->gidY = gidY;
    data->gidZ = gidZ;

    cl_int err_cb = tdispatch->clSetEventCallback(*event, CL_COMPLETE, callback, data);
    CHECK_CL(err_cb, clean(); return err, "clSetEventCallback failed (%i)", err_cb);

    clean(false);
    return err;
}

/*****************************************************************************/
/* CREATE COMMAND QUEUE ******************************************************/
/*****************************************************************************/

static cl_command_queue clkp_clCreateCommandQueue(
    cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int *errcode_ret)
{
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateCommandQueue", "properties", properties);
    properties |= CL_QUEUE_PROFILING_ENABLE;
    auto queue = tdispatch->clCreateCommandQueue(context, device, properties, errcode_ret);

    return queue;
}

static cl_command_queue clkp_clCreateCommandQueueWithProperties(
    cl_context context, cl_device_id device, const cl_queue_properties *properties, cl_int *errcode_ret)
{
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateCommandQueueWithProperties");
    std::vector<cl_queue_properties> properties_array;
    bool cl_queue_properties_found = false;
    if (properties) {
        for (unsigned i = 0; properties[i] != 0; i += 2) {
            cl_queue_properties key = properties[i];
            cl_queue_properties val = properties[i];
            if (key == CL_QUEUE_PROPERTIES) {
                TRACE_EVENT_INSTANT(
                    CLKP_PERFETTO_CATEGORY, "clCreateCommandQueueWithProperties-properties", "properties", val);

                val |= CL_QUEUE_PROFILING_ENABLE;
                cl_queue_properties_found = true;
            }
            properties_array.push_back(key);
            properties_array.push_back(val);
        }
    }
    if (!cl_queue_properties_found) {
        TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, "clCreateCommandQueueWithProperties-properties-not-found");
        properties_array.push_back(CL_QUEUE_PROPERTIES);
        properties_array.push_back(CL_QUEUE_PROFILING_ENABLE);
    }
    properties_array.push_back(0);

    auto queue = tdispatch->clCreateCommandQueueWithProperties(context, device, properties_array.data(), errcode_ret);

    return queue;
}

/*****************************************************************************/
/* PERFETTO TRACE PARAMETERS *************************************************/
/*****************************************************************************/

#ifdef BACKEND_INPROCESS
static const char *get_trace_dest()
{
    if (auto trace_dest = getenv("CLKP_TRACE_DEST")) {
        return trace_dest;
    }
    return TRACE_DEST;
}

static const uint32_t get_trace_max_size()
{
    if (auto trace_max_size = getenv("CLKP_TRACE_MAX_SIZE")) {
        return atoi(trace_max_size);
    }
    return TRACE_MAX_SIZE;
}
#endif

/*****************************************************************************/
/* LAYER FUNCTIONS ***********************************************************/
/*****************************************************************************/

CL_API_ENTRY cl_int CL_API_CALL clGetLayerInfo(
    cl_layer_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret)
{
    switch (param_name) {
    case CL_LAYER_API_VERSION:
        if (param_value) {
            if (param_value_size < sizeof(cl_layer_api_version))
                return CL_INVALID_VALUE;
            *((cl_layer_api_version *)param_value) = CL_LAYER_API_VERSION_100;
        }
        if (param_value_size_ret)
            *param_value_size_ret = sizeof(cl_layer_api_version);
        break;
    default:
        return CL_INVALID_VALUE;
    }
    return CL_SUCCESS;
}

#ifdef BACKEND_INPROCESS
void clDeinitLayer()
{
    gTracingSession->StopBlocking();
    std::vector<char> trace_data(gTracingSession->ReadTraceBlocking());

    std::ofstream output;
    output.open(get_trace_dest(), std::ios::out | std::ios::binary);
    output.write(&trace_data[0], trace_data.size());
    output.close();
}
#endif

CL_API_ENTRY cl_int CL_API_CALL clInitLayer(cl_uint num_entries, const struct _cl_icd_dispatch *target_dispatch,
    cl_uint *num_entries_out, const struct _cl_icd_dispatch **layer_dispatch_ret)
{
    if (!target_dispatch || !num_entries_out || !layer_dispatch_ret)
        return CL_INVALID_VALUE;
    if (num_entries < sizeof(dispatch) / sizeof(dispatch.clGetPlatformIDs))
        return CL_INVALID_VALUE;

    memset(&dispatch, 0, sizeof(dispatch));
    dispatch.clCreateProgramWithSource = clkp_clCreateProgramWithSource;
    dispatch.clCreateKernel = clkp_clCreateKernel;
    dispatch.clEnqueueNDRangeKernel = clkp_clEnqueueNDRangeKernel;
    dispatch.clCreateCommandQueue = clkp_clCreateCommandQueue;
    dispatch.clCreateCommandQueueWithProperties = clkp_clCreateCommandQueueWithProperties;

    tdispatch = target_dispatch;
    *layer_dispatch_ret = &dispatch;
    *num_entries_out = sizeof(dispatch) / sizeof(dispatch.clGetPlatformIDs);

    perfetto::TracingInitArgs args;
#ifdef BACKEND_INPROCESS
    args.backends |= perfetto::kInProcessBackend;
#else
    args.backends |= perfetto::kSystemBackend;
#endif
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();

#ifdef BACKEND_INPROCESS
    perfetto::protos::gen::TrackEventConfig track_event_cfg;
    perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(get_trace_max_size());
    auto *ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");
    ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

    gTracingSession = perfetto::Tracing::NewTrace();
    gTracingSession->Setup(cfg);
    gTracingSession->StartBlocking();

    bool atexit_registered = atexit(clDeinitLayer) == 0;
    CHECK(atexit_registered, return CL_OUT_OF_RESOURCES, "Could not register clDeinitLayer using atexit()");
#endif

    return CL_SUCCESS;
}
