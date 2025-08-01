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
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <perfetto.h>
#include <queue>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#ifdef SPIRV_DISASSEMBLY
#include <spirv-tools/libspirv.hpp>
#endif

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

static std::mutex g_lock;

static void writeKernelOnDisk(
    const char *dir, std::string &program_name, cl_uint count, const char **strings, const size_t *lengths)
{
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "writeKernelOnDisk", "dir", perfetto::DynamicString(dir), "program",
        perfetto::DynamicString(program_name));
    std::filesystem::path filename(dir);
    if (!std::filesystem::exists(filename)) {
        PRINT("'%s' does not exist, could not write kernel on disk", dir);
        return;
    }
    filename /= program_name;
    filename += ".cl";
    FILE *file = fopen(filename.c_str(), "w");
    for (unsigned i = 0; i < count; i++) {
        size_t size_written = 0;
        const uint8_t *data = (const uint8_t *)strings[i];
        size_t code_size = lengths == nullptr ? strlen(strings[i]) : lengths[i];
        do {
            size_written += fwrite(&data[size_written], 1, code_size - size_written, file);
        } while (size_written != code_size);
    }
    fclose(file);
}

#ifdef SPIRV_DISASSEMBLY
static std::string disassembleSpirv(const void *il, size_t length, const std::string &program_name)
{
    const uint32_t *spirv_data = (const uint32_t *)(il);
    size_t spirv_words = length / sizeof(uint32_t);

    // Check for SPIR-V magic number
    if (spirv_words == 0 || *spirv_data != 0x07230203) {
        return "";
    }

    spvtools::SpirvTools tools(SPV_ENV_OPENCL_2_2);
    std::string disassembly;

    if (tools.Disassemble(spirv_data, spirv_words, &disassembly)) {
        return disassembly;
    } else {
        PRINT("Failed to disassemble SPIR-V for program %s", program_name.c_str());
        return "";
    }
}
#endif

static void writeILOnDisk(
    const char *dir, std::string &program_name, const void *il, size_t length, const std::string &disassembly)
{
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "writeILOnDisk", "dir", perfetto::DynamicString(dir), "program",
        perfetto::DynamicString(program_name));

    std::filesystem::path base_path(dir);
    if (!std::filesystem::exists(base_path)) {
        PRINT("'%s' does not exist, could not write SPIR-V on disk", dir);
        return;
    }

    // Write the raw IL binary
    std::filesystem::path il_filename = base_path / program_name;
    il_filename += ".il";
    FILE *file = fopen(il_filename.c_str(), "wb");
    if (file) {
        size_t size_written = 0;
        const uint8_t *data = (const uint8_t *)il;
        do {
            size_written += fwrite(&data[size_written], 1, length - size_written, file);
        } while (size_written != length);
        fclose(file);
    }

    // Write the disassembly if provided
    if (!disassembly.empty()) {
        std::filesystem::path asm_filename = base_path / program_name;
        asm_filename += ".spvasm";
        FILE *asm_file = fopen(asm_filename.c_str(), "w");
        if (asm_file) {
            size_t size_written = 0;
            const uint8_t *data = (const uint8_t *)disassembly.c_str();
            size_t code_size = disassembly.length();
            do {
                size_written += fwrite(&data[size_written], 1, code_size - size_written, asm_file);
            } while (size_written != code_size);
            fclose(asm_file);
        }
    }
}

static uint32_t program_number = 0;
static std::map<cl_program, std::string> program_to_string;
static cl_program clkp_clCreateProgramWithSource(
    cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret)
{
    std::lock_guard<std::mutex> lock(g_lock);
    std::string program_str = std::string("clkp_p") + std::to_string(program_number++);
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateProgramWithSource", "program", perfetto::DynamicString(program_str),
        "count", count);

    if (auto dir = getenv("CLKP_KERNEL_DIR")) {
        writeKernelOnDisk(dir, program_str, count, strings, lengths);
    }

    cl_program program = tdispatch->clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
    program_to_string[program] = program_str;
    for (unsigned i = 0; i < count; i++) {
        TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, "clCreateProgramWithSource-args", "program",
            perfetto::DynamicString(program_str), "string", perfetto::DynamicString(strings[i]));
    }
    return program;
}

static cl_program clkp_clCreateProgramWithIL(cl_context context, const void *il, size_t length, cl_int *errcode_ret)
{
    std::lock_guard<std::mutex> lock(g_lock);
    std::string program_str = std::string("clkp_p") + std::to_string(program_number++);

    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateProgramWithIL", "program",
        perfetto::DynamicString(program_str.c_str()), "length", length);

    std::string disassembly;
#ifdef SPIRV_DISASSEMBLY
    disassembly = disassembleSpirv(il, length, program_str);
    if (!disassembly.empty()) {
        TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, "clCreateProgramWithIL-disassembly", "program",
            perfetto::DynamicString(program_str), "disassembly", perfetto::DynamicString(disassembly));
    }
#endif

    if (auto dir = getenv("CLKP_KERNEL_DIR")) {
        writeILOnDisk(dir, program_str, il, length, disassembly);
    }

    cl_program program = tdispatch->clCreateProgramWithIL(context, il, length, errcode_ret);
    program_to_string[program] = program_str;

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
    cl_event event;
    size_t gidX, gidY, gidZ;
    int64_t time_offset;
};

struct ThreadInfo {
    std::mutex lock;
    std::condition_variable cv;
    std::queue<callback_data *> callbacks;
    bool stop;
};

static std::map<cl_command_queue, ThreadInfo *> queue_to_thread_info;
static std::map<cl_command_queue, std::thread> queue_to_thread;
static std::map<cl_command_queue, int64_t> queue_to_time_offset;

static void callback(cl_event event, cl_int event_command_exec_status, void *user_data)
{
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clkp-callback");

    struct callback_data *data = (struct callback_data *)user_data;
    assert(data != nullptr);
    assert(event_command_exec_status == CL_COMPLETE);
    cl_command_queue queue = data->queue;
    ThreadInfo *thread_info = queue_to_thread_info[queue];
    {
        std::lock_guard<std::mutex> lock(thread_info->lock);
        thread_info->callbacks.push(data);
        thread_info->cv.notify_all();
    }
}

static callback_data *get_callback(ThreadInfo *thread_info)
{
    std::unique_lock<std::mutex> lock(thread_info->lock);
    while (thread_info->callbacks.empty()) {
        if (thread_info->stop) {
            return nullptr;
        }
        TRACE_EVENT_BEGIN(CLKP_PERFETTO_CATEGORY, "clkp_wait");
        thread_info->cv.wait(lock);
        TRACE_EVENT_END(CLKP_PERFETTO_CATEGORY);
    }
    callback_data *data = thread_info->callbacks.front();
    thread_info->callbacks.pop();
    return data;
}

static void trace_callback(callback_data *data)
{
    cl_command_queue queue = data->queue;
    cl_kernel kernel = data->kernel;
    cl_event event = data->event;
    size_t gidX = data->gidX, gidY = data->gidY, gidZ = data->gidZ;
    int64_t time_offset = data->time_offset;

    cl_ulong start, end;
    cl_int err;
    err = tdispatch->clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
    CHECK_CL(err, return, "clGetEventProfilingInfo(CL_PROFILING_COMMAND_START) failed (%i)", err);
    err = tdispatch->clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
    CHECK_CL(err, return, "clGetEventProfilingInfo(CL_PROFILING_COMMAND_END) failed (%i)", err);
    if (end < start) {
        TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, perfetto::StaticString("INVALID_TIMESTAMPS"),
            perfetto::Track((uintptr_t)queue), "start", start, "end", end);
        return;
    }

    std::string kernel_name = "?";
    if (kernel_to_kernel_name.count(kernel)) {
        kernel_name = kernel_to_kernel_name[kernel];
    }

    std::string program_string = "clkp_p?";
    if (kernel_to_program.count(kernel) && program_to_string.count(kernel_to_program[kernel])) {
        program_string = program_to_string[kernel_to_program[kernel]];
    }

    std::string name = program_string + "-" + kernel_name + "-" + std::to_string(gidX) + "." + std::to_string(gidY)
        + "." + std::to_string(gidZ);

    TRACE_EVENT_BEGIN(CLKP_PERFETTO_CATEGORY, perfetto::DynamicString(name), perfetto::Track((uintptr_t)queue),
        (uint64_t)(start + time_offset), "program", perfetto::DynamicString(program_string), "kernel_name",
        perfetto::DynamicString(kernel_name), "gidX", gidX, "gidY", gidY, "gidZ", gidZ);
    TRACE_EVENT_END(CLKP_PERFETTO_CATEGORY, perfetto::Track((uintptr_t)queue), (uint64_t)(end + time_offset));

    tdispatch->clReleaseEvent(event);
}

static void queue_thread_function(ThreadInfo *thread_info)
{
    pthread_setname_np(pthread_self(), "clkp");
    while (true) {
        callback_data *data = get_callback(thread_info);
        if (data == nullptr) {
            return;
        }
        trace_callback(data);
        free(data);
    }
}

static cl_int clkp_clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
    const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    std::lock_guard<std::mutex> lock(g_lock);
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
        if (!event_is_null) {
            tdispatch->clRetainEvent(*event);
        }
    };
    CHECK_CL(err, clean(); return err, "clEnqueueNDRangeKernel failed (%i)", err);

    data = (struct callback_data *)malloc(sizeof(struct callback_data));
    CHECK_ALLOC(data, clean(); return err);
    data->queue = command_queue;
    data->kernel = kernel;
    data->event = *event;
    data->gidX = gidX;
    data->gidY = gidY;
    data->gidZ = gidZ;
    data->time_offset = queue_to_time_offset[command_queue];

    cl_int err_cb = tdispatch->clSetEventCallback(*event, CL_COMPLETE, callback, data);
    CHECK_CL(err_cb, clean(); return err, "clSetEventCallback failed (%i)", err_cb);

    clean(false);
    return err;
}

/*****************************************************************************/
/* CREATE COMMAND QUEUE ******************************************************/
/*****************************************************************************/

static cl_int clkp_clReleaseCommandQueue(cl_command_queue command_queue)
{
    std::lock_guard<std::mutex> lock(g_lock);
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clReleaseCommandQueue");
    auto ret = tdispatch->clReleaseCommandQueue(command_queue);

    ThreadInfo *thread_info = queue_to_thread_info[command_queue];
    {
        std::lock_guard<std::mutex> lock(thread_info->lock);
        thread_info->stop = true;
        thread_info->cv.notify_all();
    }
    queue_to_thread[command_queue].join();
    queue_to_thread.erase(command_queue);
    queue_to_thread_info.erase(command_queue);
    delete thread_info;

    return ret;
}

static cl_command_queue create_command_queue(
    cl_context context, cl_device_id device, const cl_queue_properties *properties, cl_int *errcode_ret)
{
    std::lock_guard<std::mutex> lock(g_lock);
    std::vector<cl_queue_properties> properties_array;
    bool cl_queue_properties_found = false;
    if (properties) {
        for (unsigned i = 0; properties[i] != 0; i += 2) {
            cl_queue_properties key = properties[i];
            cl_queue_properties val = properties[i + 1];
            if (key == CL_QUEUE_PROPERTIES) {
                TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, "clCreateCommandQueue-properties", "properties", val);

                val |= CL_QUEUE_PROFILING_ENABLE;
                cl_queue_properties_found = true;
            }
            properties_array.push_back(key);
            properties_array.push_back(val);
        }
    }
    if (!cl_queue_properties_found) {
        TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY, "clCreateCommandQueue-properties-not-found");
        properties_array.push_back(CL_QUEUE_PROPERTIES);
        properties_array.push_back(CL_QUEUE_PROFILING_ENABLE);
    }
    properties_array.push_back(0);

    auto command_queue
        = tdispatch->clCreateCommandQueueWithProperties(context, device, properties_array.data(), errcode_ret);

    TRACE_EVENT_INSTANT(CLKP_PERFETTO_CATEGORY,
        perfetto::DynamicString("clkp-queue_" + std::to_string((uintptr_t)command_queue)),
        perfetto::Track((uintptr_t)command_queue));
    ThreadInfo *thread_info = new ThreadInfo();
    thread_info->stop = false;
    queue_to_thread_info[command_queue] = thread_info;
    queue_to_thread.emplace(command_queue, [thread_info] { queue_thread_function(thread_info); });

    cl_ulong device_timestamp, host_timestamp;
    tdispatch->clGetDeviceAndHostTimer(device, &device_timestamp, &host_timestamp);
    uint64_t perfetto_timestamp = perfetto::TrackEvent::GetTraceTimeNs();
    queue_to_time_offset[command_queue] = perfetto_timestamp - device_timestamp;

    return command_queue;
}

static cl_command_queue clkp_clCreateCommandQueue(
    cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int *errcode_ret)
{
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateCommandQueue");
    cl_queue_properties props[4] = { CL_QUEUE_PROPERTIES, properties, 0, 0 };
    return create_command_queue(context, device, props, errcode_ret);
}

static cl_command_queue clkp_clCreateCommandQueueWithProperties(
    cl_context context, cl_device_id device, const cl_queue_properties *properties, cl_int *errcode_ret)
{
    TRACE_EVENT(CLKP_PERFETTO_CATEGORY, "clCreateCommandQueueWithProperties");
    return create_command_queue(context, device, properties, errcode_ret);
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

void clDeinitLayer()
{
#ifdef BACKEND_INPROCESS
    gTracingSession->StopBlocking();
    std::vector<char> trace_data(gTracingSession->ReadTraceBlocking());

    std::ofstream output;
    output.open(get_trace_dest(), std::ios::out | std::ios::binary);
    output.write(&trace_data[0], trace_data.size());
    output.close();
#else
    perfetto::TrackEvent::Flush();
#endif
}

CL_API_ENTRY cl_int CL_API_CALL clInitLayer(cl_uint num_entries, const struct _cl_icd_dispatch *target_dispatch,
    cl_uint *num_entries_out, const struct _cl_icd_dispatch **layer_dispatch_ret)
{
    std::lock_guard<std::mutex> lock(g_lock);
    if (!target_dispatch || !num_entries_out || !layer_dispatch_ret)
        return CL_INVALID_VALUE;
    if (num_entries < sizeof(dispatch) / sizeof(dispatch.clGetPlatformIDs))
        return CL_INVALID_VALUE;

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
#endif

    const uint32_t max_retry = 100;
    uint32_t retry = 0;
    while ((retry++ < max_retry) && !TRACE_EVENT_CATEGORY_ENABLED(CLKP_PERFETTO_CATEGORY)) {
        usleep(1);
    }
    if (!TRACE_EVENT_CATEGORY_ENABLED(CLKP_PERFETTO_CATEGORY)) {
        PRINT("perfetto category does not seem to be enabled");
    }

    memset(&dispatch, 0, sizeof(dispatch));
    dispatch.clCreateProgramWithSource = clkp_clCreateProgramWithSource;
    dispatch.clCreateProgramWithIL = clkp_clCreateProgramWithIL;
    dispatch.clCreateKernel = clkp_clCreateKernel;
    dispatch.clEnqueueNDRangeKernel = clkp_clEnqueueNDRangeKernel;
    dispatch.clCreateCommandQueue = clkp_clCreateCommandQueue;
    dispatch.clCreateCommandQueueWithProperties = clkp_clCreateCommandQueueWithProperties;
    dispatch.clReleaseCommandQueue = clkp_clReleaseCommandQueue;

    tdispatch = target_dispatch;
    *layer_dispatch_ret = &dispatch;
    *num_entries_out = sizeof(dispatch) / sizeof(dispatch.clGetPlatformIDs);

    bool atexit_registered = atexit(clDeinitLayer) == 0;
    CHECK(atexit_registered, return CL_OUT_OF_RESOURCES, "Could not register clDeinitLayer using atexit()");

    return CL_SUCCESS;
}
