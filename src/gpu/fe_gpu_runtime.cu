#include "fe_gpu_runtime.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
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
                                     const T* __restrict__ Z,
                                     const int* __restrict__ fe_ids,
                                     size_t n_obs,
                                     int n_reg_W,
                                     int n_reg_Z,
                                     size_t leading_dim,
                                     T* group_sum_y,
                                     T* group_sum_W,
                                     T* group_sum_Z,
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
    for (int k = 0; k < n_reg_W; ++k) {
        atomicAdd(&group_sum_W[gid * n_reg_W + k], W[offset]);
        offset += leading_dim;
    }

    if (n_reg_Z > 0 && Z && group_sum_Z) {
        size_t offset_z = idx;
        for (int k = 0; k < n_reg_Z; ++k) {
            atomicAdd(&group_sum_Z[gid * n_reg_Z + k], Z[offset_z]);
            offset_z += leading_dim;
        }
    }

    atomicAdd(&group_counts[gid], 1);
}

template <typename T>
__global__ void fe_compute_means_kernel(T* group_sum_y,
                                        T* group_sum_W,
                                        T* group_sum_Z,
                                        const int* __restrict__ group_counts,
                                        int n_groups,
                                        int n_reg_W,
                                        int n_reg_Z) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_groups) {
        return;
    }

    int count = group_counts[gid];
    T inv = (count > 0) ? T(1.0) / static_cast<T>(count) : T(0.0);

    group_sum_y[gid] *= inv;
    T* row = group_sum_W + static_cast<size_t>(gid) * n_reg_W;
    for (int k = 0; k < n_reg_W; ++k) {
        row[k] *= inv;
    }

    if (n_reg_Z > 0 && group_sum_Z) {
        T* row_z = group_sum_Z + static_cast<size_t>(gid) * n_reg_Z;
        for (int k = 0; k < n_reg_Z; ++k) {
            row_z[k] *= inv;
        }
    }
}

template <typename T>
__global__ void fe_subtract_kernel(T* y,
                                   T* W,
                                   T* Z,
                                   const int* __restrict__ fe_ids,
                                   size_t n_obs,
                                   int n_reg_W,
                                   int n_reg_Z,
                                   size_t leading_dim,
                                   const T* __restrict__ group_mean_y,
                                   const T* __restrict__ group_mean_W,
                                   const T* __restrict__ group_mean_Z) {
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
    const T* row = group_mean_W + static_cast<size_t>(gid) * n_reg_W;
    for (int k = 0; k < n_reg_W; ++k) {
        W[offset] -= row[k];
        offset += leading_dim;
    }

    if (n_reg_Z > 0 && Z && group_mean_Z) {
        size_t offset_z = idx;
        const T* row_z = group_mean_Z + static_cast<size_t>(gid) * n_reg_Z;
        for (int k = 0; k < n_reg_Z; ++k) {
            Z[offset_z] -= row_z[k];
            offset_z += leading_dim;
        }
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
                              const T* Z,
                              const int* fe_ids,
                              size_t n_obs,
                              int n_reg_W,
                              int n_reg_Z,
                              size_t leading_dim,
                              T* group_sum_y,
                              T* group_sum_W,
                              T* group_sum_Z,
                              int* group_counts) {
    return launch_simple(n_obs,
                         fe_accumulate_kernel<T>,
                         y,
                         W,
                         Z,
                         fe_ids,
                         n_obs,
                         n_reg_W,
                         n_reg_Z,
                         leading_dim,
                         group_sum_y,
                         group_sum_W,
                         group_sum_Z,
                         group_counts);
}

template <typename T>
int fe_gpu_fe_compute_means_impl(T* group_sum_y,
                                 T* group_sum_W,
                                 T* group_sum_Z,
                                 const int* group_counts,
                                 int n_groups,
                                 int n_reg_W,
                                 int n_reg_Z) {
    return launch_simple(static_cast<size_t>(n_groups),
                         fe_compute_means_kernel<T>,
                         group_sum_y,
                         group_sum_W,
                         group_sum_Z,
                         group_counts,
                         n_groups,
                         n_reg_W,
                         n_reg_Z);
}

template <typename T>
int fe_gpu_fe_subtract_impl(T* y,
                            T* W,
                            T* Z,
                            const int* fe_ids,
                            size_t n_obs,
                            int n_reg_W,
                            int n_reg_Z,
                            size_t leading_dim,
                            const T* group_mean_y,
                            const T* group_mean_W,
                            const T* group_mean_Z) {
    return launch_simple(n_obs,
                         fe_subtract_kernel<T>,
                         y,
                         W,
                         Z,
                         fe_ids,
                         n_obs,
                         n_reg_W,
                         n_reg_Z,
                         leading_dim,
                         group_mean_y,
                         group_mean_W,
                         group_mean_Z);
}

__global__ void init_sequence_kernel(int* data, long long n) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    while (idx < n) {
        data[idx] = static_cast<int>(idx);
        idx += stride;
    }
}

__global__ void build_keys_kernel(const int* const* fe_ptrs,
                                  int n_dims,
                                  long long n_obs,
                                  const unsigned long long* strides,
                                  unsigned long long* keys) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    while (idx < n_obs) {
        unsigned long long key = 0ULL;
        for (int d = 0; d < n_dims; ++d) {
            int value = fe_ptrs[d][idx];
            key += (static_cast<unsigned long long>(value - 1) * strides[d]);
        }
        keys[idx] = key;
        idx += stride;
    }
}

__global__ void compute_flags_kernel(const unsigned long long* keys, int* flags, long long n) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    while (idx < n) {
        if (idx == 0) {
            flags[idx] = 1;
        } else {
            flags[idx] = (keys[idx] != keys[idx - 1]) ? 1 : 0;
        }
        idx += stride;
    }
}

__global__ void scatter_ids_kernel(const int* order,
                                   const int* cluster_sorted,
                                   int* out_ids,
                                   long long n) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    while (idx < n) {
        int pos = order[idx];
        out_ids[pos] = cluster_sorted[idx];
        idx += stride;
    }
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
                         const double* Z,
                         const int* fe_ids,
                         size_t n_obs,
                         int n_reg,
                         int n_inst,
                         size_t leading_dim,
                         double* group_sum_y,
                         double* group_sum_W,
                         double* group_sum_Z,
                         int* group_counts) {
    if (n_obs == 0) {
        return store_success();
    }
    return fe_gpu_fe_accumulate_impl(
        y, W, Z, fe_ids, n_obs, n_reg, n_inst, leading_dim, group_sum_y, group_sum_W, group_sum_Z, group_counts);
}

int fe_gpu_fe_compute_means(double* group_sum_y,
                            double* group_sum_W,
                            double* group_sum_Z,
                            const int* group_counts,
                            int n_groups,
                            int n_reg,
                            int n_inst) {
    if (n_groups == 0) {
        return store_success();
    }
    return fe_gpu_fe_compute_means_impl(group_sum_y, group_sum_W, group_sum_Z, group_counts, n_groups, n_reg, n_inst);
}

int fe_gpu_fe_subtract(double* y,
                       double* W,
                       double* Z,
                       const int* fe_ids,
                       size_t n_obs,
                       int n_reg,
                       int n_inst,
                       size_t leading_dim,
                       const double* group_mean_y,
                       const double* group_mean_W,
                       const double* group_mean_Z) {
    if (n_obs == 0) {
        return store_success();
    }
    return fe_gpu_fe_subtract_impl(
        y, W, Z, fe_ids, n_obs, n_reg, n_inst, leading_dim, group_mean_y, group_mean_W, group_mean_Z);
}

int fe_gpu_copy_columns(const double* src,
                        int ld_src,
                        const int* indices,
                        int n_indices,
                        int n_rows,
                        double* dst,
                        int ld_dst,
                        int dest_offset) {
    if (n_indices <= 0 || n_rows <= 0) {
        return store_success();
    }
    if (!src || !dst || !indices) {
        return store_custom_error("Invalid buffers for column copy");
    }
    for (int j = 0; j < n_indices; ++j) {
        int src_col = indices[j] - 1;
        if (src_col < 0) {
            return store_custom_error("Column index out of range");
        }
        const double* src_ptr = src + static_cast<size_t>(src_col) * static_cast<size_t>(ld_src);
        double* dst_ptr = dst + static_cast<size_t>(dest_offset + j) * static_cast<size_t>(ld_dst);
        cudaError_t err = cudaMemcpy(dst_ptr, src_ptr, static_cast<size_t>(n_rows) * sizeof(double), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            return store_cuda_error(err, "cudaMemcpy column copy");
        }
    }
    return store_success();
}

int fe_gpu_build_multi_cluster_ids(const void* const* fe_ptrs_host,
                                   const unsigned long long* strides_host,
                                   int n_dims,
                                   long long n_obs,
                                   int* out_ids,
                                   int* out_n_clusters) {
    if (!fe_ptrs_host || !strides_host || !out_ids || !out_n_clusters || n_dims <= 0) {
        return store_custom_error("Invalid arguments to GPU cluster builder");
    }
    if (n_obs <= 0) {
        *out_n_clusters = 0;
        return store_success();
    }

    size_t n = static_cast<size_t>(n_obs);
    const int threads = 256;
    int blocks = static_cast<int>((n_obs + threads - 1) / threads);
    blocks = std::min(blocks, 65535);

    const int** d_ptrs = nullptr;
    unsigned long long* d_strides = nullptr;
    unsigned long long* d_keys_in = nullptr;
    unsigned long long* d_keys_out = nullptr;
    int* d_order_in = nullptr;
    int* d_order_out = nullptr;
    int* d_flags = nullptr;
    int* d_cluster_sorted = nullptr;
    void* temp_sort = nullptr;
    void* temp_scan = nullptr;
    size_t temp_sort_bytes = 0;
    size_t temp_scan_bytes = 0;
    cudaError_t err;
    auto cleanup = [&]() {
        if (d_ptrs) cudaFree(d_ptrs);
        if (d_strides) cudaFree(d_strides);
        if (d_keys_in) cudaFree(d_keys_in);
        if (d_keys_out) cudaFree(d_keys_out);
        if (d_order_in) cudaFree(d_order_in);
        if (d_order_out) cudaFree(d_order_out);
        if (d_flags) cudaFree(d_flags);
        if (d_cluster_sorted) cudaFree(d_cluster_sorted);
        if (temp_sort) cudaFree(temp_sort);
        if (temp_scan) cudaFree(temp_scan);
    };

    err = cudaMalloc(&d_ptrs, sizeof(int*) * static_cast<size_t>(n_dims));
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cudaMalloc cluster ptrs");
    }
    err = cudaMemcpy(d_ptrs, fe_ptrs_host, sizeof(int*) * static_cast<size_t>(n_dims), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cudaMemcpy cluster ptrs");
    }

    err = cudaMalloc(&d_strides, sizeof(unsigned long long) * static_cast<size_t>(n_dims));
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cudaMalloc strides");
    }
    err = cudaMemcpy(d_strides, strides_host, sizeof(unsigned long long) * static_cast<size_t>(n_dims),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cudaMemcpy strides");
    }

    err = cudaMalloc(&d_keys_in, sizeof(unsigned long long) * n);
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cudaMalloc keys");
    }
    err = cudaMalloc(&d_keys_out, sizeof(unsigned long long) * n);
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cudaMalloc keys buffer");
    }
    err = cudaMalloc(&d_order_in, sizeof(int) * n);
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cudaMalloc order");
    }
    err = cudaMalloc(&d_order_out, sizeof(int) * n);
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cudaMalloc order buffer");
    }
    err = cudaMalloc(&d_flags, sizeof(int) * n);
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cudaMalloc flags");
    }
    err = cudaMalloc(&d_cluster_sorted, sizeof(int) * n);
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cudaMalloc cluster buffer");
    }

    init_sequence_kernel<<<blocks, threads>>>(d_order_in, n_obs);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "init_sequence kernel");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "init_sequence sync");
    }

    build_keys_kernel<<<blocks, threads>>>(d_ptrs, n_dims, n_obs, d_strides, d_keys_in);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "build_keys kernel");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "build_keys sync");
    }

    cub::DeviceRadixSort::SortPairs(nullptr, temp_sort_bytes, d_keys_in, d_keys_out, d_order_in, d_order_out, n);
    err = cudaMalloc(&temp_sort, temp_sort_bytes);
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cudaMalloc sort temp");
    }
    cub::DeviceRadixSort::SortPairs(temp_sort, temp_sort_bytes, d_keys_in, d_keys_out, d_order_in, d_order_out, n);

    compute_flags_kernel<<<blocks, threads>>>(d_keys_out, d_flags, n_obs);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "flags kernel");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "flags sync");
    }

    cub::DeviceScan::InclusiveSum(nullptr, temp_scan_bytes, d_flags, d_cluster_sorted, n);
    err = cudaMalloc(&temp_scan, temp_scan_bytes);
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cudaMalloc scan temp");
    }
    cub::DeviceScan::InclusiveSum(temp_scan, temp_scan_bytes, d_flags, d_cluster_sorted, n);

    err = cudaMemcpy(out_n_clusters, d_cluster_sorted + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cudaMemcpy cluster count");
    }

    scatter_ids_kernel<<<blocks, threads>>>(d_order_out, d_cluster_sorted, out_ids, n_obs);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "scatter kernel");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "scatter sync");
    }

    cleanup();
    return store_success();
}

}  // extern "C"
