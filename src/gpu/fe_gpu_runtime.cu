#include "fe_gpu_runtime.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/execution_policy.h>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <limits>
#include <cmath>

namespace {

thread_local char g_last_error[512] = "OK";

struct zip_key_order_less {
    __host__ __device__ bool operator()(const thrust::tuple<unsigned long long, int>& lhs,
                                        const thrust::tuple<unsigned long long, int>& rhs) const {
        unsigned long long lk = thrust::get<0>(lhs);
        unsigned long long rk = thrust::get<0>(rhs);
        if (lk < rk) return true;
        if (lk > rk) return false;
        return thrust::get<1>(lhs) < thrust::get<1>(rhs);
    }
};

struct abs_functor {
    __host__ __device__ double operator()(double x) const {
        return fabs(x);
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
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; idx < n_obs; idx += stride) {
        int gid = fe_ids[idx] - 1;
        if (gid < 0) {
            continue;
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
}

template <typename T, int kMaxGroups>
__global__ void fe_accumulate_small_kernel(const T* __restrict__ y,
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
                                           int* group_counts,
                                           int n_groups) {
    extern __shared__ unsigned char shmem_raw[];
    T* s_sum_y = reinterpret_cast<T*>(shmem_raw);
    T* s_sum_W = s_sum_y + kMaxGroups;
    T* s_sum_Z = s_sum_W + static_cast<size_t>(kMaxGroups) * n_reg_W;
    int* s_counts = reinterpret_cast<int*>(s_sum_Z + static_cast<size_t>(kMaxGroups) * n_reg_Z);

    for (int gid = threadIdx.x; gid < n_groups; gid += blockDim.x) {
        s_sum_y[gid] = 0;
        s_counts[gid] = 0;
    }
    for (int idx = threadIdx.x; idx < n_groups * n_reg_W; idx += blockDim.x) {
        s_sum_W[idx] = 0;
    }
    for (int idx = threadIdx.x; idx < n_groups * n_reg_Z; idx += blockDim.x) {
        s_sum_Z[idx] = 0;
    }
    __syncthreads();

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; idx < n_obs; idx += stride) {
        int gid = fe_ids[idx] - 1;
        if (gid < 0 || gid >= n_groups) continue;

        atomicAdd(&s_sum_y[gid], y[idx]);

        size_t offset = idx;
        for (int k = 0; k < n_reg_W; ++k) {
            atomicAdd(&s_sum_W[gid * n_reg_W + k], W[offset]);
            offset += leading_dim;
        }

        if (n_reg_Z > 0 && Z && group_sum_Z) {
            size_t offset_z = idx;
            for (int k = 0; k < n_reg_Z; ++k) {
                atomicAdd(&s_sum_Z[gid * n_reg_Z + k], Z[offset_z]);
                offset_z += leading_dim;
            }
        }

        atomicAdd(&s_counts[gid], 1);
    }
    __syncthreads();

    for (int gid = threadIdx.x; gid < n_groups; gid += blockDim.x) {
        atomicAdd(&group_sum_y[gid], s_sum_y[gid]);
        atomicAdd(&group_counts[gid], s_counts[gid]);
    }
    for (int idx = threadIdx.x; idx < n_groups * n_reg_W; idx += blockDim.x) {
        atomicAdd(&group_sum_W[idx], s_sum_W[idx]);
    }
    for (int idx = threadIdx.x; idx < n_groups * n_reg_Z; idx += blockDim.x) {
        atomicAdd(&group_sum_Z[idx], s_sum_Z[idx]);
    }
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
                                   const T* __restrict__ group_mean_Z,
                                   T relaxation) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; idx < n_obs; idx += stride) {
        int gid = fe_ids[idx] - 1;
        if (gid < 0) {
            continue;
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
}

template <typename Kernel, typename... Args>
int launch_simple(size_t n, Kernel kernel, Args... args) {
    if (n == 0) {
        return store_success();
    }
    const int threads = 256;
    const int blocks = static_cast<int>(std::min((n + threads - 1) / threads, static_cast<size_t>(65535)));
    kernel<<<blocks, threads>>>(args...);
    cudaError_t err = cudaPeekAtLastError();
    return store_cuda_error(err, "kernel launch");
}

template <typename T>
int fe_gpu_fe_accumulate_impl(const T* y,
                              const T* W,
                              const T* Z,
                              const int* fe_ids,
                              size_t n_obs,
                              int n_groups,
                              int n_reg_W,
                              int n_reg_Z,
                              size_t leading_dim,
                              T* group_sum_y,
                              T* group_sum_W,
                              T* group_sum_Z,
                              int* group_counts) {
    const int kMaxSmall = 512;
    if (n_groups > 0 && n_groups <= kMaxSmall && n_reg_W <= 8 && n_reg_Z <= 4) {
        int threads = 256;
        int blocks = static_cast<int>(std::min((n_obs + threads - 1) / threads, static_cast<size_t>(65535)));
        size_t shmem_bytes = sizeof(T) * (static_cast<size_t>(kMaxSmall) * (1 + n_reg_W + n_reg_Z)) +
                             sizeof(int) * static_cast<size_t>(kMaxSmall);
        if (shmem_bytes <= 48 * 1024) {
            fe_accumulate_small_kernel<T, kMaxSmall><<<blocks, threads, shmem_bytes>>>(
                y, W, Z, fe_ids, n_obs, n_reg_W, n_reg_Z, leading_dim, group_sum_y, group_sum_W, group_sum_Z,
                group_counts, n_groups);
            cudaError_t err = cudaPeekAtLastError();
            return store_cuda_error(err, "accumulate small kernel");
        }
    }

    // Fallback to straightforward atomic accumulation path.
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
                            const T* group_mean_Z,
                            T relaxation) {
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
                         group_mean_Z,
                         relaxation);
}

template <typename T>
__global__ void fe_mix_means_kernel(T* mean, T* prev, size_t n, T relaxation) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; idx < n; idx += stride) {
        T m = mean[idx];
        T p = prev[idx];
        T mixed = p + relaxation * (m - p);
        prev[idx] = mixed;
        mean[idx] = mixed;
    }
}

template <typename T>
int fe_gpu_mix_means_impl(T* mean, T* prev, size_t n, T relaxation) {
    if (n == 0) {
        return store_success();
    }
    return launch_simple(n, fe_mix_means_kernel<T>, mean, prev, n, relaxation);
}

template <typename T>
__global__ void axpy_kernel(int n, T alpha, const T* __restrict__ x, T* __restrict__ y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        y[i] += alpha * x[i];
    }
}

struct dot_functor {
    __host__ __device__ double operator()(const thrust::tuple<double, double>& t) const {
        return thrust::get<0>(t) * thrust::get<1>(t);
    }
};

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

__global__ void compute_flags_from_keys_kernel(const unsigned long long* keys,
                                               long long n,
                                               int* flags) {
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
                         int n_groups,
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
        y, W, Z, fe_ids, n_obs, n_groups, n_reg, n_inst, leading_dim, group_sum_y, group_sum_W, group_sum_Z, group_counts);
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
                       const double* group_mean_Z,
                       double relaxation) {
    if (n_obs == 0) {
        return store_success();
    }
    return fe_gpu_fe_subtract_impl(
        y, W, Z, fe_ids, n_obs, n_reg, n_inst, leading_dim, group_mean_y, group_mean_W, group_mean_Z, relaxation);
}

int fe_gpu_mix_means(double* mean,
                     double* prev,
                     size_t n,
                     double relaxation) {
    return fe_gpu_mix_means_impl(mean, prev, n, relaxation);
}

int fe_gpu_axpy(int n, double alpha, const double* x, double* y) {
    if (n <= 0 || alpha == 0.0 || !x || !y) {
        return store_success();
    }
    const int threads = 256;
    int blocks = std::min((n + threads - 1) / threads, 65535);
    axpy_kernel<<<blocks, threads>>>(n, alpha, x, y);
    cudaError_t err = cudaPeekAtLastError();
    return store_cuda_error(err, "axpy kernel");
}

int fe_gpu_absmax(const double* data, long long n, double* out) {
    if (!data || !out || n <= 0) {
        if (out) *out = 0.0;
        return store_custom_error("Invalid arguments to absmax");
    }
    try {
        thrust::device_ptr<const double> begin(data);
        thrust::device_ptr<const double> end = begin + n;
        auto exec = thrust::cuda::par.on(0);
        double result = thrust::transform_reduce(exec,
                                                 begin,
                                                 end,
                                                 abs_functor(),
                                                 0.0,
                                                 thrust::maximum<double>());
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return store_cuda_error(err, "absmax reduce");
        }
        *out = result;
    } catch (const thrust::system_error& ex) {
        return store_custom_error(ex.what());
    } catch (...) {
        return store_custom_error("Unknown error in absmax");
    }
    return store_success();
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
    size_t blocks_size = (n + threads - 1) / threads;
    if (blocks_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return store_custom_error("GPU cluster builder input too large for grid configuration");
    }
    int blocks = static_cast<int>(blocks_size);

    const int** d_ptrs = nullptr;
    unsigned long long* d_strides = nullptr;
    unsigned long long* d_keys = nullptr;
    int* d_order = nullptr;
    int* d_flags = nullptr;
    int* d_cluster_sorted = nullptr;
    void* temp_scan = nullptr;
    size_t temp_scan_bytes = 0;
    cudaError_t err;
    auto cleanup = [&]() {
        if (d_ptrs) cudaFree(d_ptrs);
        if (d_strides) cudaFree(d_strides);
        if (d_keys) cudaFree(d_keys);
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
    err = cudaMalloc(&d_keys, sizeof(unsigned long long) * n);
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "cudaMalloc keys");
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

    compute_keys_kernel<<<blocks, threads>>>(d_ptrs, d_strides, n_dims, n_obs, d_keys);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "keys kernel");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cleanup();
        return store_cuda_error(err, "keys sync");
    }

    try {
        thrust::device_ptr<unsigned long long> key_begin(d_keys);
        thrust::device_ptr<int> order_begin(d_order);
        thrust::sequence(order_begin, order_begin + n);
        auto exec = thrust::cuda::par.on(0);
        auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(key_begin, order_begin));
        auto zipped_end = thrust::make_zip_iterator(thrust::make_tuple(key_begin + n, order_begin + n));
        thrust::stable_sort(exec, zipped_begin, zipped_end, zip_key_order_less());
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cleanup();
            return store_cuda_error(err, "thrust::sort_by_key");
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

    compute_flags_from_keys_kernel<<<blocks, threads>>>(d_keys, n_obs, d_flags);
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
