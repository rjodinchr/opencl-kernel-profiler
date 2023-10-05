#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl_layer.h>
#include <string.h>

static struct _cl_icd_dispatch dispatch;
static const struct _cl_icd_dispatch *tdispatch;

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

static cl_program _clCreateProgramWithSource(
    cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret)
{
    cl_program program = tdispatch->clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
    return program;
}

static cl_kernel _clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret)
{
    cl_kernel kernel = tdispatch->clCreateKernel(program, kernel_name, errcode_ret);
    return kernel;
}

static cl_int _clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
    const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    return tdispatch->clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size,
        local_work_size, num_events_in_wait_list, event_wait_list, event);
}

CL_API_ENTRY cl_int CL_API_CALL clInitLayer(cl_uint num_entries, const struct _cl_icd_dispatch *target_dispatch,
    cl_uint *num_entries_out, const struct _cl_icd_dispatch **layer_dispatch_ret)
{
    if (!target_dispatch || !num_entries_out || !layer_dispatch_ret)
        return CL_INVALID_VALUE;
    /* Check that the loader does not provide us with a dispatch table
     * smaller than the one we've been compiled with. */
    if (num_entries < sizeof(dispatch) / sizeof(dispatch.clGetPlatformIDs))
        return CL_INVALID_VALUE;

    memset(&dispatch, 0, sizeof(dispatch));
    dispatch.clCreateProgramWithSource = _clCreateProgramWithSource;
    dispatch.clCreateKernel = _clCreateKernel;
    dispatch.clEnqueueNDRangeKernel = _clEnqueueNDRangeKernel;

    tdispatch = target_dispatch;
    *layer_dispatch_ret = &dispatch;
    *num_entries_out = sizeof(dispatch) / sizeof(dispatch.clGetPlatformIDs);
    return CL_SUCCESS;
}
