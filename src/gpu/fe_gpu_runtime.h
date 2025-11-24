#ifndef FE_GPU_RUNTIME_H
#define FE_GPU_RUNTIME_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef FE_GPU_NAME_LEN
#define FE_GPU_NAME_LEN 256
#endif

typedef struct {
    int device_id;
    size_t total_global_mem;
    int multiprocessor_count;
    int major;
    int minor;
    char name[FE_GPU_NAME_LEN];
} fe_gpu_device_info;

int fe_gpu_runtime_is_available(void);
int fe_gpu_runtime_init(int device_id, fe_gpu_device_info* info);
int fe_gpu_runtime_shutdown(void);
int fe_gpu_runtime_malloc(void** ptr, size_t bytes);
int fe_gpu_runtime_free(void* ptr);
int fe_gpu_runtime_memcpy_htod(void* dst, const void* src, size_t bytes);
int fe_gpu_runtime_memcpy_dtoh(void* dst, const void* src, size_t bytes);
int fe_gpu_runtime_memcpy_dtod(void* dst, const void* src, size_t bytes);
int fe_gpu_runtime_memset(void* ptr, int value, size_t bytes);
int fe_gpu_runtime_get_last_error(char* buffer, size_t length);
int fe_gpu_runtime_clear_error(void);

int fe_gpu_linalg_init(void);
int fe_gpu_linalg_shutdown(void);
int fe_gpu_syrk(int n_rows, int n_cols, double alpha, const double* W, int ldW, double beta, double* Q);
int fe_gpu_gemv(int n_rows, int n_cols, double alpha, const double* W, int ldW, const double* y, double beta, double* b);
int fe_gpu_residual(int n_rows,
                    int n_cols,
                    const double* W,
                    int ldW,
                    const double* beta,
                    const double* y,
                    double* residual);
int fe_gpu_dot(int n_rows,
               const double* x,
               const double* y,
               double* result);
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
                int ldC);
int fe_gpu_cluster_scores(const double* residual,
                          const double* W,
                          const int* cluster_ids,
                          int n_rows,
                          int n_cols,
                          int ldW,
                          int n_clusters,
                          double* scores);
int fe_gpu_cluster_meat(int n_clusters,
                        int n_cols,
                        const double* scores,
                        int ldScores,
                        double* meat);

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
                         int* group_counts);
int fe_gpu_fe_compute_means(double* group_sum_y,
                            double* group_sum_W,
                            double* group_sum_Z,
                            const int* group_counts,
                            int n_groups,
                            int n_reg,
                            int n_inst);
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
                       double relaxation);
int fe_gpu_mix_means(double* mean,
                     double* prev,
                     size_t n,
                     double relaxation);

int fe_gpu_absmax(const double* data,
                  long long n,
                  double* out);
int fe_gpu_copy_columns(const double* src,
                        int ld_src,
                        const int* indices,
                        int n_indices,
                        int n_rows,
                        double* dst,
                        int ld_dst,
                        int dest_offset);
int fe_gpu_build_multi_cluster_ids(const void* const* fe_ptrs,
                                   const unsigned long long* strides,
                                   int n_dims,
                                   long long n_obs,
                                   int* out_ids,
                                   int* out_n_clusters);

#ifdef __cplusplus
}
#endif

#endif /* FE_GPU_RUNTIME_H */
