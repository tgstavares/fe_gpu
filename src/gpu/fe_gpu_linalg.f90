module fe_gpu_linalg
    use iso_c_binding, only: c_int, c_ptr, c_double, c_char
    use iso_fortran_env, only: int64, real64
    use fe_gpu_runtime, only: fe_device_buffer, fe_gpu_check
    implicit none
    private

    public :: fe_gpu_linalg_initialize
    public :: fe_gpu_linalg_finalize
    public :: fe_gpu_compute_cross_products
    public :: fe_gpu_compute_residual
    public :: fe_gpu_dot
    public :: fe_gpu_cluster_scores
    public :: fe_gpu_cluster_meat
    public :: fe_gpu_cross_product
    public :: fe_gpu_matmul

    interface
        function c_fe_gpu_linalg_init() bind(C, name="fe_gpu_linalg_init") result(status)
            import :: c_int
            integer(c_int) :: status
        end function c_fe_gpu_linalg_init

        function c_fe_gpu_linalg_shutdown() bind(C, name="fe_gpu_linalg_shutdown") result(status)
            import :: c_int
            integer(c_int) :: status
        end function c_fe_gpu_linalg_shutdown

        function c_fe_gpu_syrk(n_rows, n_cols, alpha, W, ldW, beta, Q) bind(C, name="fe_gpu_syrk") result(status)
            import :: c_int, c_ptr, c_double
            integer(c_int), value :: n_rows, n_cols
            real(c_double), value :: alpha, beta
            type(c_ptr), value :: W
            integer(c_int), value :: ldW
            type(c_ptr), value :: Q
            integer(c_int) :: status
        end function c_fe_gpu_syrk

        function c_fe_gpu_gemv(n_rows, n_cols, alpha, W, ldW, y, beta, b) bind(C, name="fe_gpu_gemv") result(status)
            import :: c_int, c_ptr, c_double
            integer(c_int), value :: n_rows, n_cols
            real(c_double), value :: alpha, beta
            type(c_ptr), value :: W
            integer(c_int), value :: ldW
            type(c_ptr), value :: y, b
            integer(c_int) :: status
        end function c_fe_gpu_gemv

        function c_fe_gpu_residual(n_rows, n_cols, W, ldW, beta, y, residual) bind(C, name="fe_gpu_residual") result(status)
            import :: c_int, c_ptr
            integer(c_int), value :: n_rows, n_cols
            type(c_ptr), value :: W, beta, y, residual
            integer(c_int), value :: ldW
            integer(c_int) :: status
        end function c_fe_gpu_residual

        function c_fe_gpu_dot(n_rows, x, y, result) bind(C, name="fe_gpu_dot") result(status)
            import :: c_int, c_ptr, c_double
            integer(c_int), value :: n_rows
            type(c_ptr), value :: x, y
            real(c_double), intent(out) :: result
            integer(c_int) :: status
        end function c_fe_gpu_dot

        function c_fe_gpu_cluster_scores(residual, W, cluster_ids, n_rows, n_cols, ldW, n_clusters, scores) &
                bind(C, name="fe_gpu_cluster_scores") result(status)
            import :: c_int, c_ptr
            type(c_ptr), value :: residual, W, cluster_ids, scores
            integer(c_int), value :: n_rows, n_cols, ldW, n_clusters
            integer(c_int) :: status
        end function c_fe_gpu_cluster_scores

        function c_fe_gpu_cluster_meat(n_clusters, n_cols, scores, ldScores, meat) &
                bind(C, name="fe_gpu_cluster_meat") result(status)
            import :: c_int, c_ptr
            integer(c_int), value :: n_clusters, n_cols, ldScores
            type(c_ptr), value :: scores, meat
            integer(c_int) :: status
        end function c_fe_gpu_cluster_meat

        function c_fe_gpu_gemm(trans_a, trans_b, m, n, k, alpha, mat_a, ldA, mat_b, ldB, beta, mat_c, ldC) &
                bind(C, name="fe_gpu_gemm") result(status)
            import :: c_int, c_ptr, c_double, c_char
            character(c_char), value :: trans_a, trans_b
            integer(c_int), value :: m, n, k, ldA, ldB, ldC
            real(c_double), value :: alpha, beta
            type(c_ptr), value :: mat_a, mat_b, mat_c
            integer(c_int) :: status
        end function c_fe_gpu_gemm
    end interface

contains

    subroutine fe_gpu_linalg_initialize()
        integer(c_int) :: status
        status = c_fe_gpu_linalg_init()
        call fe_gpu_check(status, 'initializing cuBLAS handle')
    end subroutine fe_gpu_linalg_initialize

    subroutine fe_gpu_linalg_finalize()
        integer(c_int) :: status
        status = c_fe_gpu_linalg_shutdown()
        call fe_gpu_check(status, 'destroying cuBLAS handle')
    end subroutine fe_gpu_linalg_finalize

    subroutine fe_gpu_compute_cross_products(dataset_y, dataset_W, buffer_Q, buffer_b, n_obs, n_reg)
        type(fe_device_buffer), intent(in) :: dataset_y
        type(fe_device_buffer), intent(in) :: dataset_W
        type(fe_device_buffer), intent(in) :: buffer_Q
        type(fe_device_buffer), intent(in) :: buffer_b
        integer(int64), intent(in) :: n_obs
        integer, intent(in) :: n_reg
        integer(c_int) :: status
        real(c_double) :: one, zero

        if (n_reg == 0 .or. n_obs <= 0_int64) return

        one = real(1.0_real64, kind=c_double)
        zero = real(0.0_real64, kind=c_double)

        status = c_fe_gpu_syrk(int(n_obs, c_int), int(n_reg, c_int), one, dataset_W%ptr, int(n_obs, c_int), zero, buffer_Q%ptr)
        call fe_gpu_check(status, 'computing Q = W^T W via cuBLAS')

        status = c_fe_gpu_gemv(int(n_obs, c_int), int(n_reg, c_int), one, dataset_W%ptr, int(n_obs, c_int), &
            dataset_y%ptr, zero, buffer_b%ptr)
        call fe_gpu_check(status, 'computing b = W^T y via cuBLAS')
    end subroutine fe_gpu_compute_cross_products

    subroutine fe_gpu_compute_residual(dataset_y, dataset_W, beta, residual, n_obs, n_reg)
        type(fe_device_buffer), intent(in) :: dataset_y
        type(fe_device_buffer), intent(in) :: dataset_W
        type(fe_device_buffer), intent(in) :: beta
        type(fe_device_buffer), intent(in) :: residual
        integer(int64), intent(in) :: n_obs
        integer, intent(in) :: n_reg
        integer(c_int) :: status

        if (n_reg == 0 .or. n_obs <= 0_int64) return
        status = c_fe_gpu_residual(int(n_obs, c_int), int(n_reg, c_int), dataset_W%ptr, int(n_obs, c_int), &
            beta%ptr, dataset_y%ptr, residual%ptr)
        call fe_gpu_check(status, 'computing residual vector')
    end subroutine fe_gpu_compute_residual

    subroutine fe_gpu_dot(vec_x, vec_y, n_elements, result)
        type(fe_device_buffer), intent(in) :: vec_x
        type(fe_device_buffer), intent(in) :: vec_y
        integer(int64), intent(in) :: n_elements
        real(real64), intent(out) :: result
        integer(c_int) :: status

        status = c_fe_gpu_dot(int(n_elements, c_int), vec_x%ptr, vec_y%ptr, result)
        call fe_gpu_check(status, 'computing dot product')
    end subroutine fe_gpu_dot

    subroutine fe_gpu_cluster_scores(residual, dataset_W, cluster_ids, n_obs, n_reg, n_clusters, scores)
        type(fe_device_buffer), intent(in) :: residual
        type(fe_device_buffer), intent(in) :: dataset_W
        type(fe_device_buffer), intent(in) :: cluster_ids
        integer(int64), intent(in) :: n_obs
        integer, intent(in) :: n_reg
        integer, intent(in) :: n_clusters
        type(fe_device_buffer), intent(in) :: scores
        integer(c_int) :: status

        if (n_reg == 0 .or. n_obs <= 0_int64 .or. n_clusters <= 0) return
        status = c_fe_gpu_cluster_scores(residual%ptr, dataset_W%ptr, cluster_ids%ptr, &
            int(n_obs, c_int), int(n_reg, c_int), int(n_obs, c_int), int(n_clusters, c_int), scores%ptr)
        call fe_gpu_check(status, 'accumulating cluster scores')
    end subroutine fe_gpu_cluster_scores

    subroutine fe_gpu_cluster_meat(scores, n_clusters, n_reg, meat)
        type(fe_device_buffer), intent(in) :: scores
        integer, intent(in) :: n_clusters
        integer, intent(in) :: n_reg
        type(fe_device_buffer), intent(in) :: meat
        integer(c_int) :: status

        if (n_reg == 0 .or. n_clusters <= 0) return
        status = c_fe_gpu_cluster_meat(int(n_clusters, c_int), int(n_reg, c_int), scores%ptr, int(n_clusters, c_int), meat%ptr)
        call fe_gpu_check(status, 'forming clustered meat matrix')
    end subroutine fe_gpu_cluster_meat

    subroutine fe_gpu_cross_product(left, right, output, n_rows, n_left, n_right)
        type(fe_device_buffer), intent(in) :: left
        type(fe_device_buffer), intent(in) :: right
        type(fe_device_buffer), intent(in) :: output
        integer(int64), intent(in) :: n_rows
        integer, intent(in) :: n_left, n_right
        integer(c_int) :: status
        real(c_double) :: one, zero
        character(c_char) :: trans_t, trans_n

        if (n_left == 0 .or. n_right == 0 .or. n_rows <= 0_int64) return

        one = real(1.0_real64, kind=c_double)
        zero = real(0.0_real64, kind=c_double)
        trans_t = 'T'
        trans_n = 'N'

        status = c_fe_gpu_gemm(trans_t, trans_n, int(n_left, c_int), int(n_right, c_int), int(n_rows, c_int), one, &
            left%ptr, int(n_rows, c_int), right%ptr, int(n_rows, c_int), zero, output%ptr, int(n_left, c_int))
        call fe_gpu_check(status, 'computing cross-product A^T B')
    end subroutine fe_gpu_cross_product

    subroutine fe_gpu_matmul(left, right, result, n_rows, n_inner, n_cols)
        type(fe_device_buffer), intent(in) :: left
        type(fe_device_buffer), intent(in) :: right
        type(fe_device_buffer), intent(in) :: result
        integer(int64), intent(in) :: n_rows
        integer, intent(in) :: n_inner, n_cols
        integer(c_int) :: status
        real(c_double) :: one, zero
        character(c_char) :: trans_n

        if (n_rows <= 0_int64 .or. n_inner <= 0 .or. n_cols <= 0) return

        one = real(1.0_real64, kind=c_double)
        zero = real(0.0_real64, kind=c_double)
        trans_n = 'N'

        status = c_fe_gpu_gemm(trans_n, trans_n, int(n_rows, c_int), int(n_cols, c_int), int(n_inner, c_int), one, &
            left%ptr, int(n_rows, c_int), right%ptr, int(n_inner, c_int), zero, result%ptr, int(n_rows, c_int))
        call fe_gpu_check(status, 'computing matrix product A * B')
    end subroutine fe_gpu_matmul

end module fe_gpu_linalg
