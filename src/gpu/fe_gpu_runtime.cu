#include "fe_gpu_runtime.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <limits>

namespace {

thread_local char g_last_error[512] = "OK";

struct fe_order_comparator_device {
    const int* const* fe_ptrs;
    int n_dims;

    __host__ __device__ fe_order_comparator_device(const int* const* ptrs = nullptr, int dims = 0)
        : fe_ptrs(ptrs), n_dims(dims) {}

    __host__ __device__ bool operator()(const int lhs, const int rhs) const {
#ifdef __CUDA_ARCH__
        for (int d = 0; d < n_dims; ++d) {
            int lhs_val = fe_ptrs[d][lhs];
            int rhs_val = fe_ptrs[d][rhs];
            if (lhs_val < rhs_val) return true;
            if (lhs_val > rhs_val) return false;
        }
        return false;
#else
        // Host comparator only needed for compilation when using device policy.
        return lhs < rhs;
#endif
    }
};

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

__global__ void compute_flags_from_order_kernel(const int* order,
                                                const int* const* fe_ptrs,
                                                int n_dims,
                                                long long n,
                                                int* flags) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    while (idx < n) {
        if (idx == 0) {
            flags[idx] = 1;
        } else {
            int current = order[idx];
            int prev = order[idx - 1];
            int diff = 0;
            for (int d = 0; d < n_dims; ++d) {
                int val_curr = fe_ptrs[d][current];
                int val_prev = fe_ptrs[d][prev];
                if (val_curr != val_prev) {
                    diff = 1;
                    break;
                }
            }
            flags[idx] = diff;
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

__global__ void compute_keys_kernel(const int* const* fe_ptrs,
                                    const unsigned long long* strides,
                                    int n_dims,
                                    long long n,
                                    unsigned long long* keys) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    while (idx < n) {
        unsigned long long key = 0ULL;
        for (int d = 0; d < n_dims; ++d) {
            int val = fe_ptrs[d][idx];
            if (val > 0) {
                key += (static_cast<unsigned long long>(val - 1) * strides[d]);
            }
        }
        keys[idx] = key;
        idx += stride;
    }
}

__global__ void compute_keys_u32_kernel(const int* const* fe_ptrs,
                                        const unsigned int* strides,
                                        int n_dims,
                                        long long n,
                                        unsigned int* keys) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    while (idx < n) {
        unsigned int key = 0U;
        for (int d = 0; d < n_dims; ++d) {
            int val = fe_ptrs[d][idx];
            if (val > 0) {
                key += (static_cast<unsigned int>(val - 1) * strides[d]);
            }
        }
        keys[idx] = key;
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

int fe_gpu_runtime_clear_error() {
    cudaGetLastError();
    return store_success();
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
    int* d_order = nullptr;
    int* d_flags = nullptr;
    int* d_cluster_sorted = nullptr;
    void* temp_scan = nullptr;
    size_t temp_scan_bytes = 0;
    cudaError_t err;
    bool debug_cluster = (std::getenv("FE_GPU_DEBUG_CLUSTER") != nullptr);
    auto cleanup = [&]() {
        if (d_ptrs) cudaFree(d_ptrs);
        if (d_order) cudaFree(d_order);
        if (d_flags) cudaFree(d_flags);
        if (d_cluster_sorted) cudaFree(d_cluster_sorted);
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
    err = cudaMalloc(&d_order, sizeof(int) * n);
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cudaMalloc order");
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

    if (debug_cluster) {
        unsigned long long* d_strides_dbg = nullptr;
        unsigned long long* d_keys_dbg = nullptr;
        unsigned long long* d_keys_alt = nullptr;
        int* d_order_dbg = nullptr;
        int* d_order_alt = nullptr;
        unsigned int* d_strides_dbg32 = nullptr;
        unsigned int* d_keys_dbg32 = nullptr;
        unsigned int* d_keys_alt32 = nullptr;
        int* d_order_dbg32 = nullptr;
        int* d_order_alt32 = nullptr;
        void* temp_dbg = nullptr;
        void* temp_dbg32 = nullptr;
        size_t temp_dbg_bytes = 0;
        size_t temp_dbg_bytes32 = 0;
        cudaError_t dbg_err;

        dbg_err = cudaMalloc(&d_strides_dbg, sizeof(unsigned long long) * static_cast<size_t>(n_dims));
        if (dbg_err == cudaSuccess) {
            dbg_err = cudaMemcpy(d_strides_dbg, strides_host, sizeof(unsigned long long) * static_cast<size_t>(n_dims),
                cudaMemcpyHostToDevice);
        }
        if (dbg_err == cudaSuccess) {
            std::vector<unsigned int> stride32(n_dims);
            for (int i = 0; i < n_dims; ++i) {
                stride32[i] = static_cast<unsigned int>(std::min<unsigned long long>(
                    strides_host[i], std::numeric_limits<unsigned int>::max()));
            }
            dbg_err = cudaMalloc(&d_strides_dbg32, sizeof(unsigned int) * static_cast<size_t>(n_dims));
            if (dbg_err == cudaSuccess) {
                dbg_err = cudaMemcpy(d_strides_dbg32, stride32.data(),
                    sizeof(unsigned int) * static_cast<size_t>(n_dims), cudaMemcpyHostToDevice);
            }
        }
        if (dbg_err == cudaSuccess) dbg_err = cudaMalloc(&d_keys_dbg, sizeof(unsigned long long) * n);
        if (dbg_err == cudaSuccess) dbg_err = cudaMalloc(&d_keys_alt, sizeof(unsigned long long) * n);
        if (dbg_err == cudaSuccess) dbg_err = cudaMalloc(&d_order_dbg, sizeof(int) * n);
        if (dbg_err == cudaSuccess) dbg_err = cudaMalloc(&d_order_alt, sizeof(int) * n);
        if (dbg_err == cudaSuccess) dbg_err = cudaMalloc(&d_keys_dbg32, sizeof(unsigned int) * n);
        if (dbg_err == cudaSuccess) dbg_err = cudaMalloc(&d_keys_alt32, sizeof(unsigned int) * n);
        if (dbg_err == cudaSuccess) dbg_err = cudaMalloc(&d_order_dbg32, sizeof(int) * n);
        if (dbg_err == cudaSuccess) dbg_err = cudaMalloc(&d_order_alt32, sizeof(int) * n);

        if (dbg_err == cudaSuccess) {
            compute_keys_kernel<<<blocks, threads>>>(d_ptrs, d_strides_dbg, n_dims, n_obs, d_keys_dbg);
            dbg_err = cudaGetLastError();
            if (dbg_err == cudaSuccess) dbg_err = cudaDeviceSynchronize();
        }

        if (dbg_err == cudaSuccess) {
            size_t sample = (n < 16) ? static_cast<size_t>(n) : 16;
            std::vector<unsigned long long> host_keys(sample);
            cudaMemcpy(host_keys.data(), d_keys_dbg, sizeof(unsigned long long) * sample, cudaMemcpyDeviceToHost);
            unsigned long long key_min = ULLONG_MAX;
            unsigned long long key_max = 0ULL;
            for (size_t s = 0; s < sample; ++s) {
                key_min = std::min(key_min, host_keys[s]);
                key_max = std::max(key_max, host_keys[s]);
            }
            fprintf(stderr, "GPU cluster debug: key sample min=%llu max=%llu (n=%lld)\n",
                static_cast<unsigned long long>(key_min), static_cast<unsigned long long>(key_max),
                static_cast<long long>(n_obs));
        }

        if (dbg_err == cudaSuccess) {
            cub::DoubleBuffer<unsigned long long> key_buf(d_keys_dbg, d_keys_alt);
            cub::DoubleBuffer<int> val_buf(d_order_dbg, d_order_alt);
            cudaError_t tmp_req = cub::DeviceRadixSort::SortPairs(nullptr, temp_dbg_bytes, key_buf, val_buf,
                static_cast<int>(n));
            if (tmp_req != cudaSuccess) {
                fprintf(stderr, "GPU cluster debug: temp size request error: %s\n", cudaGetErrorString(tmp_req));
                dbg_err = tmp_req;
            } else {
                fprintf(stderr, "GPU cluster debug: CUB temp bytes=%zu\n", temp_dbg_bytes);
                dbg_err = cudaMalloc(&temp_dbg, temp_dbg_bytes);
                if (dbg_err == cudaSuccess) {
                    dbg_err = cub::DeviceRadixSort::SortPairs(temp_dbg, temp_dbg_bytes, key_buf, val_buf,
                        static_cast<int>(n), 0, 32);
                }
                if (dbg_err != cudaSuccess) {
                    fprintf(stderr, "GPU cluster debug: CUB radix sort error: %s\n", cudaGetErrorString(dbg_err));
                } else {
                    dbg_err = cudaDeviceSynchronize();
                    if (dbg_err != cudaSuccess) {
                        fprintf(stderr, "GPU cluster debug: CUB sort sync error: %s\n", cudaGetErrorString(dbg_err));
                    } else {
                        size_t sample = (n < 8) ? static_cast<size_t>(n) : 8;
                        std::vector<unsigned long long> host_keys_sorted(sample);
                        std::vector<int> host_order_sorted(sample);
                        cudaMemcpy(host_keys_sorted.data(), key_buf.Current(), sizeof(unsigned long long) * sample,
                            cudaMemcpyDeviceToHost);
                        cudaMemcpy(host_order_sorted.data(), val_buf.Current(), sizeof(int) * sample,
                            cudaMemcpyDeviceToHost);
                        fprintf(stderr, "GPU cluster debug: sorted key/order sample:");
                        for (size_t s = 0; s < sample; ++s) {
                            fprintf(stderr, " (%llu,%d)", static_cast<unsigned long long>(host_keys_sorted[s]),
                                host_order_sorted[s]);
                        }
                        fprintf(stderr, "\n");
                    }
                }
            }
        }

        if (dbg_err == cudaSuccess) {
            compute_keys_u32_kernel<<<blocks, threads>>>(d_ptrs, d_strides_dbg32, n_dims, n_obs, d_keys_dbg32);
            dbg_err = cudaGetLastError();
            if (dbg_err == cudaSuccess) dbg_err = cudaDeviceSynchronize();
        }
        if (dbg_err == cudaSuccess) {
            cub::DoubleBuffer<unsigned int> key_buf32(d_keys_dbg32, d_keys_alt32);
            cub::DoubleBuffer<int> val_buf32(d_order_dbg32, d_order_alt32);
            cudaError_t tmp_req32 = cub::DeviceRadixSort::SortPairs(nullptr, temp_dbg_bytes32, key_buf32, val_buf32,
                static_cast<int>(n), 0, 32);
            if (tmp_req32 != cudaSuccess) {
                fprintf(stderr, "GPU cluster debug: CUB32 temp size error: %s\n", cudaGetErrorString(tmp_req32));
                dbg_err = tmp_req32;
            } else {
                fprintf(stderr, "GPU cluster debug: CUB32 temp bytes=%zu\n", temp_dbg_bytes32);
                dbg_err = cudaMalloc(&temp_dbg32, temp_dbg_bytes32);
                if (dbg_err == cudaSuccess) {
                    dbg_err = cub::DeviceRadixSort::SortPairs(temp_dbg32, temp_dbg_bytes32, key_buf32, val_buf32,
                        static_cast<int>(n), 0, 32);
                }
                if (dbg_err != cudaSuccess) {
                    fprintf(stderr, "GPU cluster debug: CUB32 radix sort error: %s\n", cudaGetErrorString(dbg_err));
                } else {
                    dbg_err = cudaDeviceSynchronize();
                    if (dbg_err != cudaSuccess) {
                        fprintf(stderr, "GPU cluster debug: CUB32 sync error: %s\n", cudaGetErrorString(dbg_err));
                    } else {
                        size_t sample = (n < 8) ? static_cast<size_t>(n) : 8;
                        std::vector<unsigned int> host_keys_sorted(sample);
                        std::vector<int> host_order_sorted(sample);
                        cudaMemcpy(host_keys_sorted.data(), key_buf32.Current(), sizeof(unsigned int) * sample,
                            cudaMemcpyDeviceToHost);
                        cudaMemcpy(host_order_sorted.data(), val_buf32.Current(), sizeof(int) * sample,
                            cudaMemcpyDeviceToHost);
                        fprintf(stderr, "GPU cluster debug: CUB32 sorted sample:");
                        for (size_t s = 0; s < sample; ++s) {
                            fprintf(stderr, " (%u,%d)", host_keys_sorted[s], host_order_sorted[s]);
                        }
                        fprintf(stderr, "\n");
                    }
                }
            }
        }

        if (dbg_err == cudaSuccess) {
            thrust::device_ptr<unsigned long long> key_begin(d_keys_dbg);
            thrust::device_ptr<int> order_begin_dbg(d_order_dbg);
            thrust::sequence(order_begin_dbg, order_begin_dbg + n);
            thrust::sort(thrust::cuda::par.on(0), key_begin, key_begin + n, thrust::less<unsigned long long>());
            thrust::sort_by_key(thrust::cuda::par.on(0), key_begin, key_begin + n, order_begin_dbg);
            cudaError_t th_err = cudaDeviceSynchronize();
            if (th_err != cudaSuccess) {
                fprintf(stderr, "GPU cluster debug: thrust sort error: %s\n", cudaGetErrorString(th_err));
            } else {
                size_t sample = (n < 8) ? static_cast<size_t>(n) : 8;
                std::vector<unsigned long long> host_keys_sorted(sample);
                std::vector<int> host_order_sorted(sample);
                cudaMemcpy(host_keys_sorted.data(), d_keys_dbg, sizeof(unsigned long long) * sample,
                    cudaMemcpyDeviceToHost);
                cudaMemcpy(host_order_sorted.data(), d_order_dbg, sizeof(int) * sample,
                    cudaMemcpyDeviceToHost);
                fprintf(stderr, "GPU cluster debug: thrust sorted sample:");
                for (size_t s = 0; s < sample; ++s) {
                    fprintf(stderr, " (%llu,%d)", static_cast<unsigned long long>(host_keys_sorted[s]),
                        host_order_sorted[s]);
                }
                fprintf(stderr, "\n");
            }
        }

        if (d_strides_dbg) cudaFree(d_strides_dbg);
        if (d_strides_dbg32) cudaFree(d_strides_dbg32);
        if (d_keys_dbg) cudaFree(d_keys_dbg);
        if (d_keys_alt) cudaFree(d_keys_alt);
        if (d_keys_dbg32) cudaFree(d_keys_dbg32);
        if (d_keys_alt32) cudaFree(d_keys_alt32);
        if (d_order_dbg) cudaFree(d_order_dbg);
        if (d_order_alt) cudaFree(d_order_alt);
        if (d_order_dbg32) cudaFree(d_order_dbg32);
        if (d_order_alt32) cudaFree(d_order_alt32);
        if (temp_dbg) cudaFree(temp_dbg);
        if (temp_dbg32) cudaFree(temp_dbg32);
    }

    try {
        thrust::device_ptr<int> order_begin(d_order);
        thrust::sequence(order_begin, order_begin + n);
        auto exec = thrust::cuda::par.on(0);
        thrust::sort(exec, order_begin, order_begin + n, fe_order_comparator_device(d_ptrs, n_dims));
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cleanup();
            return store_cuda_error(err, "thrust::sort");
        }
    } catch (const thrust::system_error& ex) {
        cleanup();
        return store_custom_error(ex.what());
    } catch (...) {
        cleanup();
        return store_custom_error("Unknown error in sort");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "sort sync");
    }

    compute_flags_from_order_kernel<<<blocks, threads>>>(d_order, d_ptrs, n_dims, n_obs, d_flags);
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

    int num_items_scan = static_cast<int>(n);
    thrust::device_ptr<int> flags_ptr(d_flags);
    int cluster_count = thrust::reduce(thrust::device, flags_ptr, flags_ptr + n_obs);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "thrust::reduce");
    }
    if (cluster_count == 0) {
        int sample = (n_obs < 16) ? static_cast<int>(n_obs) : 16;
        std::vector<int> host_flags(sample);
        std::vector<int> host_order(sample);
        cudaMemcpy(host_flags.data(), d_flags, sizeof(int) * sample, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_order.data(), d_order, sizeof(int) * sample, cudaMemcpyDeviceToHost);
        fprintf(stderr, "GPU cluster builder debug (first %d):\\nflags: ", sample);
        for (int i = 0; i < sample; ++i) fprintf(stderr, "%d ", host_flags[i]);
        fprintf(stderr, "\\norder: ");
        for (int i = 0; i < sample; ++i) fprintf(stderr, "%d ", host_order[i]);
        fprintf(stderr, "\\n");
        cleanup();
        return store_custom_error("GPU cluster builder produced zero clusters");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "thrust reduce sync");
    }

    err = cub::DeviceScan::InclusiveSum(nullptr, temp_scan_bytes, d_flags, d_cluster_sorted, num_items_scan);
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cub::InclusiveSum temp");
    }
    err = cudaMalloc(&temp_scan, temp_scan_bytes);
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cudaMalloc scan temp");
    }
    err = cub::DeviceScan::InclusiveSum(temp_scan, temp_scan_bytes, d_flags, d_cluster_sorted, num_items_scan);
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cub::InclusiveSum run");
    }

    *out_n_clusters = cluster_count;
    if (*out_n_clusters == 0 && n > 0) {
        cleanup();
        return store_custom_error("GPU cluster builder produced zero clusters");
    }

    scatter_ids_kernel<<<blocks, threads>>>(d_order, d_cluster_sorted, out_ids, n_obs);
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
