#include "fe_gpu_runtime.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace {

thread_local char g_last_error[512] = "OK";

int store_success() {
    std::snprintf(g_last_error, sizeof(g_last_error), "OK");
    return 0;
}

int store_custom_error(const char* message) {
    if (!message) {
        std::snprintf(g_last_error, sizeof(g_last_error), "Unknown CUDA error");
    } else {
        std::snprintf(g_last_error, sizeof(g_last_error), "%s", message);
    }
    return 1;
}

int store_cuda_error(cudaError_t err, const char* context) {
    if (err == cudaSuccess) {
        return store_success();
    }

    if (context) {
        std::snprintf(g_last_error, sizeof(g_last_error), "%s: %s", context, cudaGetErrorString(err));
    } else {
        std::snprintf(g_last_error, sizeof(g_last_error), "%s", cudaGetErrorString(err));
    }
    return static_cast<int>(err);
}

template <typename T>
__global__ void fe_accumulate_kernel(const T* __restrict__ y,
                                     const T* __restrict__ W,
                                     const int* __restrict__ fe_ids,
                                     size_t n_obs,
                                     int n_reg,
                                     size_t leading_dim,
                                     T* group_sum_y,
                                     T* group_sum_W,
                                     int* group_counts) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_obs) {
        return;
    }

    int gid = fe_ids[idx] - 1;
    if (gid < 0) {
        return;
    }

    atomicAdd(&group_sum_y[gid], y[idx]);

    size_t offset = idx;
    for (int k = 0; k < n_reg; ++k) {
        atomicAdd(&group_sum_W[gid * n_reg + k], W[offset]);
        offset += leading_dim;
    }

    atomicAdd(&group_counts[gid], 1);
}

template <typename T>
__global__ void fe_compute_means_kernel(T* group_sum_y,
                                        T* group_sum_W,
                                        const int* __restrict__ group_counts,
                                        int n_groups,
                                        int n_reg) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_groups) {
        return;
    }

    int count = group_counts[gid];
    T inv = (count > 0) ? T(1.0) / static_cast<T>(count) : T(0.0);

    group_sum_y[gid] *= inv;
    T* row = group_sum_W + static_cast<size_t>(gid) * n_reg;
    for (int k = 0; k < n_reg; ++k) {
        row[k] *= inv;
    }
}

template <typename T>
__global__ void fe_subtract_kernel(T* y,
                                   T* W,
                                   const int* __restrict__ fe_ids,
                                   size_t n_obs,
                                   int n_reg,
                                   size_t leading_dim,
                                   const T* __restrict__ group_mean_y,
                                   const T* __restrict__ group_mean_W) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_obs) {
        return;
    }

    int gid = fe_ids[idx] - 1;
    if (gid < 0) {
        return;
    }

    T mean_y = group_mean_y[gid];
    y[idx] -= mean_y;

    size_t offset = idx;
    const T* row = group_mean_W + static_cast<size_t>(gid) * n_reg;
    for (int k = 0; k < n_reg; ++k) {
        W[offset] -= row[k];
        offset += leading_dim;
    }
}

template <typename Kernel, typename... Args>
int launch_simple(size_t n, Kernel kernel, Args... args) {
    if (n == 0) {
        return store_success();
    }
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    kernel<<<blocks, threads>>>(args...);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return store_cuda_error(err, "kernel launch");
    }
    err = cudaDeviceSynchronize();
    return store_cuda_error(err, "kernel execution");
}

template <typename T>
int fe_gpu_fe_accumulate_impl(const T* y,
                              const T* W,
                              const int* fe_ids,
                              size_t n_obs,
                              int n_reg,
                              size_t leading_dim,
                              T* group_sum_y,
                              T* group_sum_W,
                              int* group_counts) {
    return launch_simple(n_obs, fe_accumulate_kernel<T>, y, W, fe_ids, n_obs, n_reg, leading_dim, group_sum_y, group_sum_W,
                         group_counts);
}

template <typename T>
int fe_gpu_fe_compute_means_impl(T* group_sum_y,
                                 T* group_sum_W,
                                 const int* group_counts,
                                 int n_groups,
                                 int n_reg) {
    return launch_simple(static_cast<size_t>(n_groups), fe_compute_means_kernel<T>, group_sum_y, group_sum_W, group_counts,
                         n_groups, n_reg);
}

template <typename T>
int fe_gpu_fe_subtract_impl(T* y,
                            T* W,
                            const int* fe_ids,
                            size_t n_obs,
                            int n_reg,
                            size_t leading_dim,
                            const T* group_mean_y,
                            const T* group_mean_W) {
    return launch_simple(n_obs, fe_subtract_kernel<T>, y, W, fe_ids, n_obs, n_reg, leading_dim, group_mean_y, group_mean_W);
}

}  // namespace

extern "C" {

int fe_gpu_runtime_is_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        return store_cuda_error(err, "cudaGetDeviceCount");
    }
    if (device_count <= 0) {
        return store_custom_error("No CUDA devices detected");
    }
    return store_success();
}

int fe_gpu_runtime_init(int device_id, fe_gpu_device_info* info) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return store_cuda_error(err, "cudaGetDeviceCount");
    }
    if (count <= 0) {
        return store_custom_error("No CUDA devices detected");
    }

    int chosen = device_id;
    if (chosen < 0 || chosen >= count) {
        chosen = 0;
    }

    err = cudaSetDevice(chosen);
    if (err != cudaSuccess) {
        return store_cuda_error(err, "cudaSetDevice");
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, chosen);
    if (err != cudaSuccess) {
        return store_cuda_error(err, "cudaGetDeviceProperties");
    }

    if (info) {
        info->device_id = chosen;
        info->total_global_mem = prop.totalGlobalMem;
        info->multiprocessor_count = prop.multiProcessorCount;
        info->major = prop.major;
        info->minor = prop.minor;
        std::strncpy(info->name, prop.name, FE_GPU_NAME_LEN - 1);
        info->name[FE_GPU_NAME_LEN - 1] = '\0';
    }

    return store_success();
}

int fe_gpu_runtime_shutdown(void) {
    cudaError_t err = cudaDeviceReset();
    return store_cuda_error(err, "cudaDeviceReset");
}

int fe_gpu_runtime_malloc(void** ptr, size_t bytes) {
    if (!ptr) {
        return store_custom_error("Null pointer passed to fe_gpu_runtime_malloc");
    }
    if (bytes == 0) {
        *ptr = nullptr;
        return store_success();
    }
    cudaError_t err = cudaMalloc(ptr, bytes);
    return store_cuda_error(err, "cudaMalloc");
}

int fe_gpu_runtime_free(void* ptr) {
    if (ptr == nullptr) {
        return store_success();
    }
    cudaError_t err = cudaFree(ptr);
    return store_cuda_error(err, "cudaFree");
}

int fe_gpu_runtime_memcpy_htod(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) {
        return store_success();
    }
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    return store_cuda_error(err, "cudaMemcpy (host->device)");
}

int fe_gpu_runtime_memcpy_dtoh(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) {
        return store_success();
    }
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    return store_cuda_error(err, "cudaMemcpy (device->host)");
}

int fe_gpu_runtime_memcpy_dtod(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) {
        return store_success();
    }
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    return store_cuda_error(err, "cudaMemcpy (device->device)");
}

int fe_gpu_runtime_memset(void* ptr, int value, size_t bytes) {
    if (!ptr || bytes == 0) {
        return store_success();
    }
    cudaError_t err = cudaMemset(ptr, value, bytes);
    return store_cuda_error(err, "cudaMemset");
}

int fe_gpu_runtime_get_last_error(char* buffer, size_t length) {
    if (!buffer || length == 0) {
        return 1;
    }
    size_t stored_len = std::strlen(g_last_error);
    size_t copy_len = (stored_len < length - 1) ? stored_len : (length - 1);
    std::memcpy(buffer, g_last_error, copy_len);
    buffer[copy_len] = '\0';
    return 0;
}

int fe_gpu_fe_accumulate(const double* y,
                         const double* W,
                         const int* fe_ids,
                         size_t n_obs,
                         int n_reg,
                         size_t leading_dim,
                         double* group_sum_y,
                         double* group_sum_W,
                         int* group_counts) {
    if (n_obs == 0) {
        return store_success();
    }
    return fe_gpu_fe_accumulate_impl(y, W, fe_ids, n_obs, n_reg, leading_dim, group_sum_y, group_sum_W, group_counts);
}

int fe_gpu_fe_compute_means(double* group_sum_y,
                            double* group_sum_W,
                            const int* group_counts,
                            int n_groups,
                            int n_reg) {
    if (n_groups == 0) {
        return store_success();
    }
    return fe_gpu_fe_compute_means_impl(group_sum_y, group_sum_W, group_counts, n_groups, n_reg);
}

int fe_gpu_fe_subtract(double* y,
                       double* W,
                       const int* fe_ids,
                       size_t n_obs,
                       int n_reg,
                       size_t leading_dim,
                       const double* group_mean_y,
                       const double* group_mean_W) {
    if (n_obs == 0) {
        return store_success();
    }
    return fe_gpu_fe_subtract_impl(y, W, fe_ids, n_obs, n_reg, leading_dim, group_mean_y, group_mean_W);
}

}  // extern "C"
