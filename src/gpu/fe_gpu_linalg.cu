#include "fe_gpu_runtime.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace {

thread_local cublasHandle_t g_handle = nullptr;

int ensure_handle() {
    if (g_handle != nullptr) {
        return 0;
    }
    cublasStatus_t status = cublasCreate(&g_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        g_handle = nullptr;
        return 1;
    }
    return 0;
}

void destroy_handle() {
    if (g_handle) {
        cublasDestroy(g_handle);
        g_handle = nullptr;
    }
}

template <typename T>
__global__ void cluster_scores_kernel(const T* residual,
                                      const T* W,
                                      const int* cluster_ids,
                                      size_t n_rows,
                                      int n_cols,
                                      int ldW,
                                      int n_clusters,
                                      T* scores) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_rows) {
        return;
    }
    int cluster = cluster_ids[idx] - 1;
    if (cluster < 0 || cluster >= n_clusters) {
        return;
    }
    T r = residual[idx];
    for (int col = 0; col < n_cols; ++col) {
        T xij = W[idx + static_cast<size_t>(col) * ldW];
        atomicAdd(&scores[static_cast<size_t>(cluster) + static_cast<size_t>(col) * n_clusters], r * xij);
    }
}

template <typename T>
int launch_cluster_scores_impl(const T* residual,
                               const T* W,
                               const int* cluster_ids,
                               int n_rows,
                               int n_cols,
                               int ldW,
                               int n_clusters,
                               T* scores) {
    if (n_cols == 0 || n_rows <= 0 || n_clusters <= 0) {
        return 0;
    }
    int threads = 256;
    int blocks = (n_rows + threads - 1) / threads;
    cluster_scores_kernel<<<blocks, threads>>>(residual, W, cluster_ids, static_cast<size_t>(n_rows), n_cols, ldW, n_clusters, scores);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return err;
    }
    return 0;
}

}  // namespace

extern "C" {

int fe_gpu_linalg_init(void) {
    return ensure_handle();
}

int fe_gpu_linalg_shutdown(void) {
    destroy_handle();
    return 0;
}

int fe_gpu_syrk(int n_rows, int n_cols, double alpha, const double* W, int ldW, double beta, double* Q) {
    if (ensure_handle() != 0) {
        return 1;
    }
    if (n_cols == 0) {
        return 0;
    }
    cublasStatus_t status = cublasDsyrk(g_handle,
                                       CUBLAS_FILL_MODE_UPPER,
                                       CUBLAS_OP_T,
                                       n_cols,
                                       n_rows,
                                       &alpha,
                                       W,
                                       ldW,
                                       &beta,
                                       Q,
                                       n_cols);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    return 0;
}

int fe_gpu_gemv(int n_rows, int n_cols, double alpha, const double* W, int ldW, const double* y, double beta, double* b) {
    if (ensure_handle() != 0) {
        return 1;
    }
    if (n_cols == 0) {
        return 0;
    }
    cublasStatus_t status = cublasDgemv(g_handle,
                                       CUBLAS_OP_T,
                                       n_rows,
                                       n_cols,
                                       &alpha,
                                       W,
                                       ldW,
                                       y,
                                       1,
                                       &beta,
                                       b,
                                       1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    return 0;
}

int fe_gpu_residual(int n_rows,
                    int n_cols,
                    const double* W,
                    int ldW,
                    const double* beta,
                    const double* y,
                    double* residual) {
    if (ensure_handle() != 0) {
        return 1;
    }
    if (n_cols == 0) {
        return 0;
    }
    cublasStatus_t status = cublasDcopy(g_handle, n_rows, y, 1, residual, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    const double minus_one = -1.0;
    const double one = 1.0;
    status = cublasDgemv(g_handle,
                         CUBLAS_OP_N,
                         n_rows,
                         n_cols,
                         &minus_one,
                         W,
                         ldW,
                         beta,
                         1,
                         &one,
                         residual,
                         1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    return 0;
}

int fe_gpu_dot(int n_rows, const double* x, const double* y, double* result) {
    if (ensure_handle() != 0) {
        return 1;
    }
    cublasStatus_t status = cublasDdot(g_handle, n_rows, x, 1, y, 1, result);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    return 0;
}

int fe_gpu_cluster_scores(const double* residual,
                          const double* W,
                          const int* cluster_ids,
                          int n_rows,
                          int n_cols,
                          int ldW,
                          int n_clusters,
                          double* scores) {
    return launch_cluster_scores_impl(residual, W, cluster_ids, n_rows, n_cols, ldW, n_clusters, scores);
}

int fe_gpu_cluster_meat(int n_clusters,
                        int n_cols,
                        const double* scores,
                        int ldScores,
                        double* meat) {
    return fe_gpu_syrk(n_clusters, n_cols, 1.0, scores, ldScores, 0.0, meat);
}

}  // extern "C"
