#include "fe_gpu_runtime.h"

#include <string.h>

static char g_last_error[256] = "CUDA backend not available in this build.";

static int set_stub_error(void) {
    const char* message = "CUDA backend not available in this build.";
    strncpy(g_last_error, message, sizeof(g_last_error) - 1);
    g_last_error[sizeof(g_last_error) - 1] = '\0';
    return 1;
}

int fe_gpu_runtime_is_available(void) {
    return 0;
}

int fe_gpu_runtime_init(int device_id, fe_gpu_device_info* info) {
    (void)device_id;
    if (info) {
        memset(info, 0, sizeof(*info));
    }
    return set_stub_error();
}

int fe_gpu_runtime_shutdown(void) {
    return set_stub_error();
}

int fe_gpu_runtime_malloc(void** ptr, size_t bytes) {
    (void)bytes;
    if (ptr) {
        *ptr = NULL;
    }
    return set_stub_error();
}

int fe_gpu_runtime_free(void* ptr) {
    (void)ptr;
    return set_stub_error();
}

int fe_gpu_runtime_memcpy_htod(void* dst, const void* src, size_t bytes) {
    (void)dst;
    (void)src;
    (void)bytes;
    return set_stub_error();
}

int fe_gpu_runtime_memcpy_dtoh(void* dst, const void* src, size_t bytes) {
    (void)dst;
    (void)src;
    (void)bytes;
    return set_stub_error();
}

int fe_gpu_runtime_memcpy_dtod(void* dst, const void* src, size_t bytes) {
    (void)dst;
    (void)src;
    (void)bytes;
    return set_stub_error();
}

int fe_gpu_runtime_memset(void* ptr, int value, size_t bytes) {
    (void)ptr;
    (void)value;
    (void)bytes;
    return set_stub_error();
}

int fe_gpu_runtime_get_last_error(char* buffer, size_t length) {
    if (!buffer || length == 0) {
        return 1;
    }

    size_t msg_len = 0;
    while (msg_len < sizeof(g_last_error) && g_last_error[msg_len] != '\0') {
        msg_len++;
    }

    size_t copy_len = (msg_len < length - 1) ? msg_len : (length - 1);
    memcpy(buffer, g_last_error, copy_len);
    buffer[copy_len] = '\0';
    return 0;
}

int fe_gpu_residual(int n_rows,
                    int n_cols,
                    const double* W,
                    int ldW,
                    const double* beta,
                    const double* y,
                    double* residual) {
    (void)n_rows; (void)n_cols; (void)W; (void)ldW; (void)beta; (void)y; (void)residual;
    return set_stub_error();
}

int fe_gpu_dot(int n_rows,
               const double* x,
               const double* y,
               double* result) {
    (void)n_rows; (void)x; (void)y; (void)result;
    return set_stub_error();
}

int fe_gpu_cluster_scores(const double* residual,
                          const double* W,
                          const int* cluster_ids,
                          int n_rows,
                          int n_cols,
                          int ldW,
                          int n_clusters,
                          double* scores) {
    (void)residual; (void)W; (void)cluster_ids; (void)n_rows; (void)n_cols;
    (void)ldW; (void)n_clusters; (void)scores;
    return set_stub_error();
}

int fe_gpu_cluster_meat(int n_clusters,
                        int n_cols,
                        const double* scores,
                        int ldScores,
                        double* meat) {
    (void)n_clusters; (void)n_cols; (void)scores; (void)ldScores; (void)meat;
    return set_stub_error();
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
    (void)y;
    (void)W;
    (void)Z;
    (void)fe_ids;
    (void)n_obs;
    (void)n_reg;
    (void)n_inst;
    (void)leading_dim;
    (void)group_sum_y;
    (void)group_sum_W;
    (void)group_sum_Z;
    (void)group_counts;
    return set_stub_error();
}

int fe_gpu_fe_compute_means(double* group_sum_y,
                            double* group_sum_W,
                            double* group_sum_Z,
                            const int* group_counts,
                            int n_groups,
                            int n_reg,
                            int n_inst) {
    (void)group_sum_y;
    (void)group_sum_W;
    (void)group_sum_Z;
    (void)group_counts;
    (void)n_groups;
    (void)n_reg;
    (void)n_inst;
    return set_stub_error();
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
    (void)y;
    (void)W;
    (void)Z;
    (void)fe_ids;
    (void)n_obs;
    (void)n_reg;
    (void)n_inst;
    (void)leading_dim;
    (void)group_mean_y;
    (void)group_mean_W;
    (void)group_mean_Z;
    return set_stub_error();
}

int fe_gpu_copy_columns(const double* src,
                        int ld_src,
                        const int* indices,
                        int n_indices,
                        int n_rows,
                        double* dst,
                        int ld_dst,
                        int dest_offset) {
    (void)src;
    (void)ld_src;
    (void)indices;
    (void)n_indices;
    (void)n_rows;
    (void)dst;
    (void)ld_dst;
    (void)dest_offset;
    return set_stub_error();
}

int fe_gpu_build_multi_cluster_ids(const void* const* fe_ptrs,
                                   const unsigned long long* strides,
                                   int n_dims,
                                   long long n_obs,
                                   int* out_ids,
                                   int* out_n_clusters) {
    (void)fe_ptrs;
    (void)strides;
    (void)n_dims;
    (void)n_obs;
    (void)out_ids;
    (void)out_n_clusters;
    return set_stub_error();
}

int fe_gpu_linalg_init(void) {
    return set_stub_error();
}

int fe_gpu_linalg_shutdown(void) {
    return set_stub_error();
}

int fe_gpu_syrk(int n_rows, int n_cols, double alpha, const double* W, int ldW, double beta, double* Q) {
    (void)n_rows;
    (void)n_cols;
    (void)alpha;
    (void)W;
    (void)ldW;
    (void)beta;
    (void)Q;
    return set_stub_error();
}

int fe_gpu_gemv(int n_rows, int n_cols, double alpha, const double* W, int ldW, const double* y, double beta, double* b) {
    (void)n_rows;
    (void)n_cols;
    (void)alpha;
    (void)W;
    (void)ldW;
    (void)y;
    (void)beta;
    (void)b;
    return set_stub_error();
}

int fe_gpu_gemm(char transA,
                char transB,
                int m,
                int n,
                int k,
                double alpha,
                const double* A,
                int ldA,
                const double* B,
                int ldB,
                double beta,
                double* C,
                int ldC) {
    (void)transA;
    (void)transB;
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)A;
    (void)ldA;
    (void)B;
    (void)ldB;
    (void)beta;
    (void)C;
    (void)ldC;
    return set_stub_error();
}
