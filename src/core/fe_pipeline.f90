module fe_pipeline
    use iso_c_binding, only: c_loc, c_null_ptr, c_associated
    use iso_fortran_env, only: int32, int64, real64
    use fe_config, only: fe_runtime_config
    use fe_types, only: fe_dataset_header, fe_host_arrays
    use fe_gpu_data, only: fe_gpu_dataset, fe_gpu_dataset_upload, fe_gpu_dataset_destroy
    use fe_gpu_demean, only: fe_gpu_within_transform
    use fe_gpu_linalg, only: fe_gpu_linalg_initialize, fe_gpu_linalg_finalize, fe_gpu_compute_cross_products, &
        fe_gpu_compute_residual, fe_gpu_dot, fe_gpu_cluster_scores, fe_gpu_cluster_meat
    use fe_gpu_runtime, only: fe_device_buffer, fe_device_alloc, fe_device_free, fe_memcpy_dtoh, fe_memcpy_htod, fe_device_memset
    use fe_solver, only: chol_solve_and_invert
    use fe_logging, only: log_warn, log_info
    use fe_cluster_utils, only: build_cluster_ids
    implicit none
    intrinsic :: system_clock
    integer(int64), parameter :: REAL64_BYTES = storage_size(0.0_real64) / 8
    integer(int64), parameter :: INT32_BYTES = storage_size(1_int32) / 8
    integer, parameter :: MAX_CLUSTER_DIMS = 3
    private

    type, public :: fe_estimation_result
        logical :: converged = .false.
        integer :: fe_iterations = 0
        integer :: solver_info = 0
        integer :: rank = 0
        integer(int32), allocatable :: cluster_fe_dims(:)
        real(real64) :: time_demean = 0.0_real64
        real(real64) :: time_regression = 0.0_real64
        real(real64) :: time_se = 0.0_real64
        real(real64) :: rss = 0.0_real64
        real(real64) :: tss_total = 0.0_real64
        real(real64) :: tss_within = 0.0_real64
        real(real64) :: r2 = 0.0_real64
        real(real64) :: r2_adj = 0.0_real64
        real(real64) :: r2_within = 0.0_real64
        real(real64) :: f_stat = 0.0_real64
        integer :: dof_model = 0
        integer :: dof_fes = 0
        integer(int64) :: dof_resid = 0_int64
        real(real64), allocatable :: beta(:)
        real(real64), allocatable :: se(:)
    end type fe_estimation_result

    public :: fe_gpu_estimate

contains

    subroutine fe_gpu_estimate(cfg, header, host, group_sizes, result)
        type(fe_runtime_config), intent(in) :: cfg
        type(fe_dataset_header), intent(in) :: header
        type(fe_host_arrays), intent(in) :: host
        integer, intent(in) :: group_sizes(:)
        type(fe_estimation_result), intent(out) :: result

        type(fe_gpu_dataset) :: gpu_data
        type(fe_device_buffer) :: d_Q, d_b
        logical :: converged
        integer :: iterations
        integer :: k, info
        integer(int64) :: bytes
        real(real64), allocatable, target :: host_Q(:, :), host_b(:)
        integer, allocatable :: keep_idx(:)
        real(real64), allocatable :: Q_inv_kept(:, :)
        integer :: it0, it1, itrate

        call system_clock(count_rate = itrate)
        call system_clock(count=it0)

        result%converged = .false.
        result%fe_iterations = 0
        result%solver_info = 0
        result%rank = 0
        result%time_demean = 0.0_real64
        result%time_regression = 0.0_real64
        result%time_se = 0.0_real64
        result%rss = 0.0_real64
        result%tss_total = 0.0_real64
        result%tss_within = 0.0_real64
        result%r2 = 0.0_real64
        result%r2_adj = 0.0_real64
        result%r2_within = 0.0_real64
        result%f_stat = 0.0_real64
        result%dof_model = 0
        result%dof_fes = 0
        result%dof_resid = 0_int64
        if (allocated(result%beta)) deallocate(result%beta)
        if (allocated(result%se)) deallocate(result%se)
        if (allocated(result%cluster_fe_dims)) deallocate(result%cluster_fe_dims)
        call initialize_cluster_dims(cfg%cluster_fe_dims, header%n_fe, result%cluster_fe_dims)

        if (size(group_sizes) /= header%n_fe) then
            call log_warn('Group size metadata does not match header; skipping estimation.')
            allocate(result%beta(0))
            allocate(result%se(0))
            return
        end if

        d_Q%ptr = c_null_ptr
        d_Q%size_bytes = 0_int64
        d_b%ptr = c_null_ptr
        d_b%size_bytes = 0_int64

        result%tss_total = compute_tss(host%y, .true.)
        result%dof_fes = estimate_fe_dof(group_sizes)

        call fe_gpu_dataset_upload(host, group_sizes, gpu_data)
        call fe_gpu_linalg_initialize()

        
        call fe_gpu_within_transform(gpu_data, cfg%fe_tolerance, cfg%fe_max_iterations, converged, iterations)

        call system_clock(count=it1)
        result%time_demean = real(it1 - it0) / real(itrate)
        result%converged = converged
        result%fe_iterations = iterations

        call system_clock(count=it0)
        k = header%n_regressors
        allocate(result%beta(max(0, k)))
        allocate(result%se(max(0, k)))
        result%beta = 0.0_real64
        result%se = 0.0_real64

        if (k <= 0) then
            call cleanup()
            return
        end if

        bytes = int(k, int64) * int(k, int64) * REAL64_BYTES
        call fe_device_alloc(d_Q, bytes)

        bytes = int(k, int64) * REAL64_BYTES
        call fe_device_alloc(d_b, bytes)

        call fe_gpu_dot(gpu_data%d_y, gpu_data%d_y, gpu_data%n_obs, result%tss_within)

        call fe_gpu_compute_cross_products(gpu_data%d_y, gpu_data%d_W, d_Q, d_b, gpu_data%n_obs, k)

        allocate(host_Q(k, k))
        allocate(host_b(k))

        call fe_memcpy_dtoh(c_loc(host_Q(1, 1)), d_Q)
        call fe_memcpy_dtoh(c_loc(host_b(1)), d_b)
        call symmetrize_upper(host_Q)

        call solve_with_column_filter(host_Q, host_b, result, keep_idx, Q_inv_kept, info)
        result%solver_info = info

        call system_clock(count=it1)
        result%time_regression = real(it1 - it0) / real(itrate)

        call system_clock(count=it0)
        if (info == 0) then
            call compute_standard_errors(keep_idx, Q_inv_kept, cfg%verbose)
            call finalize_regression_stats(result, gpu_data%n_obs, header%n_fe > 0)
        else
            deallocate(keep_idx, Q_inv_kept)
            call cleanup()
            return
        end if

        deallocate(keep_idx, Q_inv_kept)
        call cleanup()

        call system_clock(count=it1)
        result%time_se = real(it1 - it0) / real(itrate)

    contains

        subroutine solve_with_column_filter(Q_full, b_full, est, idx_out, Q_inv, info_out)
            real(real64), intent(inout) :: Q_full(:, :)
            real(real64), intent(in) :: b_full(:)
            type(fe_estimation_result), intent(inout) :: est
            integer, allocatable, intent(out) :: idx_out(:)
            real(real64), allocatable, intent(out) :: Q_inv(:, :)
            integer, intent(out) :: info_out
            logical, allocatable :: keep(:)
            real(real64), allocatable :: Q_sel(:, :), b_sel(:), beta_sel(:)
            real(real64) :: diag_max, diag_val, tol
            integer :: kept, i

            diag_max = 0.0_real64
            do i = 1, size(b_full)
                diag_val = abs(Q_full(i, i))
                if (diag_val > diag_max) diag_max = diag_val
            end do
            diag_max = max(diag_max, 1.0_real64)
            tol = 1.0e-10_real64 * diag_max

            allocate(keep(size(b_full)))
            keep = .true.
            do i = 1, size(b_full)
                if (abs(Q_full(i, i)) <= tol) keep(i) = .false.
            end do

            kept = count(keep)
            est%dof_model = kept
            if (kept == 0) then
                call log_warn('All regressors dropped after FE transformation (no within variation).')
                allocate(est%beta(0))
                allocate(est%se(0))
                info_out = -1
                deallocate(keep)
                return
            end if

            if (kept < size(b_full)) then
                call log_warn('Some regressors were dropped due to zero variance after FE transformation.')
            end if

            allocate(idx_out(kept))
            idx_out = pack([(i, i = 1, size(b_full))], keep)

            allocate(Q_sel(kept, kept))
            allocate(b_sel(kept))
            allocate(beta_sel(kept))
            allocate(Q_inv(kept, kept))

            Q_sel = Q_full(idx_out, idx_out)
            b_sel = b_full(idx_out)

            call chol_solve_and_invert(Q_sel, b_sel, beta_sel, Q_inv, info_out)
            est%beta = 0.0_real64
            if (info_out == 0) then
                est%beta(idx_out) = beta_sel
                est%rank = kept
            else
                deallocate(idx_out, Q_sel, b_sel, beta_sel, Q_inv, keep)
                return
            end if

            deallocate(Q_sel, b_sel, beta_sel, keep)
        end subroutine solve_with_column_filter

        subroutine compute_standard_errors(idx, Q_inv_kept, verbose)
            integer, intent(in) :: idx(:)
            real(real64), intent(in) :: Q_inv_kept(:, :)
            logical, intent(in) :: verbose
            type(fe_device_buffer) :: d_beta, d_residual, d_scores, d_meat, d_cluster_temp
            real(real64) :: rss, sigma2, t_start, t_end, weight
            integer :: df, kept
            integer(int64) :: bytes_obs, bytes_reg, bytes_clusters
            integer :: n_clusters
            real(real64), allocatable, target :: meat_full(:, :)
            real(real64), allocatable :: meat_kept(:, :), cov_mat(:, :), cov_accum(:, :)
            real(real64), allocatable :: diag_vals(:)
            integer :: i, n_cluster_dims, mask, subset_size, subset_pos, dim_index, status_build
            integer, allocatable :: subset_dims(:)
            real(real64), allocatable, target :: beta_copy(:)
            integer(int32), allocatable, target :: combo_ids(:)
            logical :: has_clusters, cluster_success
            character(len=256) :: warn_msg
            type(fe_device_buffer) :: ids_buffer
            real(real64) :: subset_start, subset_mid, subset_end
            character(len=64) :: subset_label

            kept = size(idx)
            if (allocated(result%se)) result%se = 0.0_real64

            call cpu_time(t_start)

            bytes_reg = int(size(result%beta), int64) * REAL64_BYTES
            call fe_device_alloc(d_beta, bytes_reg)
            if (size(result%beta) > 0) then
                allocate(beta_copy(size(result%beta)))
                beta_copy = result%beta
                call fe_memcpy_htod(d_beta, c_loc(beta_copy(1)), bytes_reg)
                deallocate(beta_copy)
            end if

            bytes_obs = gpu_data%n_obs * REAL64_BYTES
            call fe_device_alloc(d_residual, bytes_obs)
            call fe_gpu_compute_residual(gpu_data%d_y, gpu_data%d_W, d_beta, d_residual, gpu_data%n_obs, &
                size(result%beta))
            call fe_gpu_dot(d_residual, d_residual, gpu_data%n_obs, rss)
            df = max(1_int64, gpu_data%n_obs - kept)
            result%dof_resid = df
            result%rss = rss
            sigma2 = rss / real(df, real64)

            diag_vals = diag_vector(Q_inv_kept)

            has_clusters = allocated(result%cluster_fe_dims) .and. size(result%cluster_fe_dims) > 0
            cluster_success = .true.
            if (.not. has_clusters) then
                do i = 1, kept
                    result%se(idx(i)) = sqrt(max(0.0_real64, sigma2 * diag_vals(i)))
                end do
            else
                n_cluster_dims = size(result%cluster_fe_dims)
                allocate(cov_accum(kept, kept))
                cov_accum = 0.0_real64
                d_cluster_temp%ptr = c_null_ptr
                d_cluster_temp%size_bytes = 0_int64

                do mask = 1, ishft(1, n_cluster_dims) - 1
                    call cpu_time(subset_start)
                    subset_size = popcnt(mask)
                    allocate(subset_dims(subset_size))
                    subset_pos = 0
                    do i = 1, n_cluster_dims
                        if (iand(mask, ishft(1, i - 1)) /= 0) then
                            subset_pos = subset_pos + 1
                            subset_dims(subset_pos) = result%cluster_fe_dims(i)
                        end if
                    end do
                    weight = merge(1.0_real64, -1.0_real64, mod(subset_size, 2) == 1)
                    subset_size = size(subset_dims)
                    subset_label = format_subset_dim_list(subset_dims)
                    if (subset_size == 1) then
                        dim_index = subset_dims(1)
                        if (dim_index < 1 .or. dim_index > gpu_data%n_fe) then
                            write(warn_msg, '("Cluster FE dimension ",I0," is out of range; skipping cluster SEs.")') dim_index
                            call log_warn(trim(warn_msg))
                            cluster_success = .false.
                            deallocate(subset_dims)
                            exit
                        end if
                        n_clusters = group_sizes(dim_index)
                        if (n_clusters <= 0) then
                            write(warn_msg, '("Cluster FE dimension ",I0," has no groups; skipping cluster SEs.")') dim_index
                            call log_warn(trim(warn_msg))
                            cluster_success = .false.
                            deallocate(subset_dims)
                            exit
                        end if
                        ids_buffer = gpu_data%fe_dims(dim_index)%fe_ids
                    else
                        if (.not. allocated(combo_ids)) allocate(combo_ids(int(gpu_data%n_obs)))
                        call build_cluster_ids(host%fe_ids, group_sizes, subset_dims, combo_ids, n_clusters, status_build)
                        if (status_build /= 0) then
                            call log_warn('Unable to build cluster identifiers for requested subset; skipping cluster SEs.')
                            cluster_success = .false.
                            deallocate(subset_dims)
                            exit
                        end if
                        if (n_clusters <= 0) then
                            call log_warn('Cluster subset produced zero groups; skipping cluster SEs.')
                            cluster_success = .false.
                            deallocate(subset_dims)
                            exit
                        end if
                        bytes_clusters = gpu_data%n_obs * INT32_BYTES
                        call fe_device_alloc(d_cluster_temp, bytes_clusters)
                        call fe_memcpy_htod(d_cluster_temp, c_loc(combo_ids(1)), bytes_clusters)
                        ids_buffer = d_cluster_temp
                    end if
                    call cpu_time(subset_mid)

                    bytes_clusters = int(n_clusters, int64) * int(size(result%beta), int64) * REAL64_BYTES
                    call fe_device_alloc(d_scores, bytes_clusters)
                    call fe_device_memset(d_scores, 0)
                    call fe_gpu_cluster_scores(d_residual, gpu_data%d_W, ids_buffer, gpu_data%n_obs, size(result%beta), &
                        n_clusters, d_scores)

                    bytes_reg = int(size(result%beta), int64) * int(size(result%beta), int64) * REAL64_BYTES
                    call fe_device_alloc(d_meat, bytes_reg)
                    call fe_gpu_cluster_meat(d_scores, n_clusters, size(result%beta), d_meat)

                    allocate(meat_full(size(result%beta), size(result%beta)))
                    call fe_memcpy_dtoh(c_loc(meat_full(1, 1)), d_meat)
                    call symmetrize_upper(meat_full)

                    allocate(meat_kept(kept, kept))
                    allocate(cov_mat(kept, kept))
                    meat_kept = meat_full(idx, idx)
                    cov_mat = matmul(Q_inv_kept, matmul(meat_kept, Q_inv_kept))
                    cov_accum = cov_accum + weight * cov_mat
                    deallocate(meat_full, meat_kept, cov_mat)

                    call fe_device_free(d_scores)
                    call fe_device_free(d_meat)
                    if (subset_size > 1 .and. c_associated(d_cluster_temp%ptr)) then
                        call fe_device_free(d_cluster_temp)
                        d_cluster_temp%ptr = c_null_ptr
                        d_cluster_temp%size_bytes = 0_int64
                    end if
                    deallocate(subset_dims)
                    call cpu_time(subset_end)
                    if (verbose) then
                        write(warn_msg, '("SE subset ",A," -> clusters=",I0,", build=",F8.3," s, accumulate=",F8.3," s")') &
                            trim(subset_label), n_clusters, subset_mid - subset_start, subset_end - subset_mid
                        call log_info(trim(warn_msg))
                    end if
                    if (.not. cluster_success) exit
                end do

                if (cluster_success) then
                    do i = 1, kept
                        result%se(idx(i)) = sqrt(max(0.0_real64, cov_accum(i, i)))
                    end do
                else
                    call log_warn('Cluster-robust SEs failed; falling back to homoskedastic estimates.')
                    if (allocated(result%cluster_fe_dims)) then
                        deallocate(result%cluster_fe_dims)
                    end if
                    allocate(result%cluster_fe_dims(0))
                    do i = 1, kept
                        result%se(idx(i)) = sqrt(max(0.0_real64, sigma2 * diag_vals(i)))
                    end do
                end if
                if (allocated(cov_accum)) deallocate(cov_accum)
                if (c_associated(d_cluster_temp%ptr)) call fe_device_free(d_cluster_temp)
                if (allocated(combo_ids)) deallocate(combo_ids)
            end if

            call fe_device_free(d_beta)
            call fe_device_free(d_residual)
            if (allocated(diag_vals)) deallocate(diag_vals)

            call cpu_time(t_end)
            result%time_se = t_end - t_start
        end subroutine compute_standard_errors

        subroutine initialize_cluster_dims(requested, n_fe, output)
            integer(int32), intent(in) :: requested(:)
            integer, intent(in) :: n_fe
            integer, allocatable, intent(out) :: output(:)
            integer :: buffer(MAX_CLUSTER_DIMS)
            integer :: count, i
            logical :: truncated
            character(len=256) :: msg

            count = 0
            truncated = .false.
            buffer = 0

            if (size(requested) == 0) then
                allocate(output(0))
                return
            end if

            do i = 1, size(requested)
                if (requested(i) < 1 .or. requested(i) > n_fe) then
                    write(msg, '("Ignoring cluster FE dimension ",I0," (out of range).")') requested(i)
                    call log_warn(trim(msg))
                    cycle
                end if
                if (count > 0) then
                    if (any(buffer(1:count) == requested(i))) cycle
                end if
                if (count == MAX_CLUSTER_DIMS) then
                    truncated = .true.
                    exit
                end if
                count = count + 1
                buffer(count) = requested(i)
            end do

            if (truncated) then
                write(msg, '("Limiting clustered SEs to the first ",I0," dimensions.")') MAX_CLUSTER_DIMS
                call log_warn(trim(msg))
            end if

            if (count > 0) then
                allocate(output(count))
                output = buffer(1:count)
            else
                allocate(output(0))
            end if
        end subroutine initialize_cluster_dims

        function diag_vector(mat) result(diag)
            real(real64), intent(in) :: mat(:, :)
            real(real64) :: diag(size(mat, 1))
            integer :: i
            do i = 1, size(diag)
                diag(i) = mat(i, i)
            end do
        end function diag_vector

        subroutine cleanup()
            if (c_associated(d_Q%ptr)) call fe_device_free(d_Q)
            if (c_associated(d_b%ptr)) call fe_device_free(d_b)
            call fe_gpu_dataset_destroy(gpu_data)
            call fe_gpu_linalg_finalize()
            if (allocated(host_Q)) deallocate(host_Q)
            if (allocated(host_b)) deallocate(host_b)
        end subroutine cleanup

        subroutine symmetrize_upper(mat)
            real(real64), intent(inout) :: mat(:, :)
            integer :: ii, jj, n
            n = size(mat, 1)
            do jj = 1, n
                do ii = jj + 1, n
                    mat(ii, jj) = mat(jj, ii)
                end do
            end do
        end subroutine symmetrize_upper

    end subroutine fe_gpu_estimate

    function format_subset_dim_list(dims) result(text)
        integer, intent(in) :: dims(:)
        character(len=64) :: text
        integer :: j, pos

        text = '['
        pos = len_trim(text)
        do j = 1, size(dims)
            write(text(pos + 1:), '(I0)') dims(j)
            pos = len_trim(text)
            if (j < size(dims)) then
                text(pos + 1:pos + 1) = ','
                pos = pos + 1
            end if
        end do
        text = trim(text) // ']'
    end function format_subset_dim_list

    function compute_tss(vec, has_intercept) result(total)
        real(real64), intent(in) :: vec(:)
        logical, intent(in) :: has_intercept
        real(real64) :: total
        real(real64) :: mean_val
        integer :: n

        n = size(vec)
        if (n <= 0) then
            total = 0.0_real64
            return
        end if

        if (has_intercept) then
            mean_val = sum(vec) / real(n, real64)
            total = sum((vec - mean_val) ** 2)
        else
            total = sum(vec ** 2)
        end if
    end function compute_tss

    function estimate_fe_dof(group_sizes) result(total)
        integer, intent(in) :: group_sizes(:)
        integer :: total
        integer :: d

        total = 0
        do d = 1, size(group_sizes)
            total = total + max(0, group_sizes(d) - 1)
        end do
    end function estimate_fe_dof

    subroutine finalize_regression_stats(est, n_obs, has_fes)
        type(fe_estimation_result), intent(inout) :: est
        integer(int64), intent(in) :: n_obs
        logical, intent(in) :: has_fes
        real(real64) :: n, df_model_r, df_resid_r, ssr, dev, dev0
        real(real64), parameter :: EPS = 1.0e-12_real64
        logical, parameter :: has_intercept = .true.
        real(real64) :: fe_flag, denom_adj, k_total

        n = real(max(1_int64, n_obs), real64)
        df_model_r = real(max(0, est%dof_model), real64)
        df_resid_r = real(max(1_int64, est%dof_resid), real64)
        dev = est%rss
        dev0 = max(EPS, est%tss_total)

        if (dev0 > EPS) then
            est%r2 = 1.0_real64 - dev / dev0
            est%r2 = max(0.0_real64, min(1.0_real64, est%r2))
        else
            est%r2 = 0.0_real64
        end if

        if (est%tss_within > EPS) then
            est%r2_within = 1.0_real64 - dev / est%tss_within
            est%r2_within = max(0.0_real64, min(1.0_real64, est%r2_within))
        else
            est%r2_within = est%r2
        end if

        fe_flag = merge(1.0_real64, 0.0_real64, has_intercept .or. has_fes)
        k_total = real(est%dof_model + est%dof_fes + merge(1, 0, has_intercept), real64)
        denom_adj = max(n - k_total, 1.0_real64)
        if (n > 1.0_real64 .and. dev0 > EPS) then
            est%r2_adj = 1.0_real64 - ((dev * (n - fe_flag)) / (dev0 * denom_adj))
            est%r2_adj = max(0.0_real64, min(1.0_real64, est%r2_adj))
        else
            est%r2_adj = est%r2
        end if

        ssr = max(0.0_real64, dev0 - dev)
        if (df_model_r > 0.0_real64 .and. df_resid_r > 0.0_real64 .and. dev > EPS) then
            est%f_stat = (ssr / df_model_r) / (dev / df_resid_r)
        else
            est%f_stat = 0.0_real64
        end if
    end subroutine finalize_regression_stats

end module fe_pipeline
