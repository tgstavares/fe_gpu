module fe_pipeline
    use iso_c_binding, only: c_ptr, c_loc, c_null_ptr, c_associated
    use iso_fortran_env, only: int32, int64, real64
    use fe_config, only: fe_runtime_config
    use fe_types, only: fe_dataset_header, fe_host_arrays
    use fe_gpu_data, only: fe_gpu_dataset, fe_gpu_dataset_upload, fe_gpu_dataset_destroy
    use fe_gpu_demean, only: fe_gpu_within_transform
    use fe_gpu_linalg, only: fe_gpu_linalg_initialize, fe_gpu_linalg_finalize, fe_gpu_compute_cross_products, &
        fe_gpu_compute_residual, fe_gpu_dot, fe_gpu_cluster_scores, fe_gpu_cluster_meat, fe_gpu_cross_product, fe_gpu_matmul
    use fe_gpu_runtime, only: fe_device_buffer, fe_device_alloc, fe_device_free, fe_memcpy_dtoh, fe_memcpy_htod, &
        fe_device_memset, fe_gpu_copy_columns, fe_gpu_build_multi_cluster_ids, fe_gpu_last_error, fe_gpu_clear_error
    use fe_solver, only: chol_solve_and_invert
    use fe_logging, only: log_warn, log_info
    use fe_cluster_utils, only: build_cluster_ids
    implicit none
    intrinsic :: system_clock
    integer(int64), parameter :: REAL64_BYTES = storage_size(0.0_real64) / 8
    integer(int64), parameter :: INT32_BYTES = storage_size(1_int32) / 8
    integer, parameter :: MAX_CLUSTER_DIMS = 3
    logical, save :: allow_gpu_cluster_builder = .true.
    logical, parameter :: DEBUG_CLUSTERS = .false.
    private

    interface
        subroutine dsyev(jobz, uplo, n, a, lda, w, work, lwork, info) bind(C, name="dsyev_")
            import :: int32, real64
            character(len=1), intent(in) :: jobz, uplo
            integer(int32), intent(in) :: n, lda, lwork
            real(real64), intent(inout) :: a(lda, *)
            real(real64), intent(out) :: w(*)
            real(real64), intent(inout) :: work(*)
            integer(int32), intent(out) :: info
        end subroutine dsyev
    end interface

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
        logical :: is_iv = .false.
        integer :: n_instruments = 0
        real(real64), allocatable :: beta(:)
        real(real64), allocatable :: se(:)
        character(len=:), allocatable :: coef_names(:)
        character(len=:), allocatable :: depvar_name
        character(len=:), allocatable :: cluster_labels(:)
        character(len=:), allocatable :: iv_labels(:)
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
        integer :: k, info, i
        integer(int64) :: bytes
        real(real64), allocatable, target :: host_Q(:, :), host_b(:)
        real(real64), allocatable :: variance_diag(:)
        real(real64), allocatable, target :: host_WW(:, :), host_Wy(:)
        integer, allocatable :: keep_idx(:)
        real(real64), allocatable :: Q_inv_kept(:, :)
        integer :: it0, it1, itrate
        logical, allocatable :: is_endog(:)
        integer, allocatable :: idx_endog(:), idx_exog(:)
        integer :: n_endog, n_exog, n_total_instr, n_selected_instr, n_available_instr
        type(fe_device_buffer) :: d_Z_aug
        character(len=256) :: warn_buf
        integer(int32), allocatable :: idx_instruments(:), tmp_instr(:)
        logical :: use_iv
        integer :: dof_fes_nested

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
        result%is_iv = .false.
        result%n_instruments = 0
        if (allocated(result%beta)) deallocate(result%beta)
        if (allocated(result%se)) deallocate(result%se)
        if (allocated(result%cluster_fe_dims)) deallocate(result%cluster_fe_dims)
        if (allocated(result%coef_names)) deallocate(result%coef_names)
        if (allocated(result%depvar_name)) deallocate(result%depvar_name)
        if (allocated(result%cluster_labels)) deallocate(result%cluster_labels)
        if (allocated(result%iv_labels)) deallocate(result%iv_labels)
        if (allocated(header%depvar_name)) then
            if (len_trim(header%depvar_name) > 0) then
                allocate(character(len=len_trim(header%depvar_name)) :: result%depvar_name)
                result%depvar_name = trim(header%depvar_name)
            end if
        else if (allocated(cfg%depvar_name)) then
            if (len_trim(cfg%depvar_name) > 0) then
                allocate(character(len=len_trim(cfg%depvar_name)) :: result%depvar_name)
                result%depvar_name = trim(cfg%depvar_name)
            end if
        end if
        call initialize_cluster_dims(cfg%cluster_fe_dims, header%n_fe, result%cluster_fe_dims)
        call initialize_cluster_labels(header)

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
        d_Z_aug%ptr = c_null_ptr
        d_Z_aug%size_bytes = 0_int64

        result%tss_total = compute_tss(host%y, .true.)
        result%dof_fes = estimate_fe_dof(group_sizes)
        dof_fes_nested = estimate_nested_fe_dof(result%cluster_fe_dims, group_sizes)

        call fe_gpu_dataset_upload(host, group_sizes, gpu_data)
        result%is_iv = .false.
        result%n_instruments = 0
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
        call initialize_coefficient_names(header, k)

        if (k <= 0) then
            call cleanup()
            return
        end if

        bytes = int(k, int64) * int(k, int64) * REAL64_BYTES
        call fe_device_alloc(d_Q, bytes)

        bytes = int(k, int64) * REAL64_BYTES
        call fe_device_alloc(d_b, bytes)

        call fe_gpu_dot(gpu_data%d_y, gpu_data%d_y, gpu_data%n_obs, result%tss_within)

        if (k > 0) then
            allocate(host_WW(k, k))
            allocate(host_Wy(k))
            call fe_gpu_compute_cross_products(gpu_data%d_y, gpu_data%d_W, d_Q, d_b, gpu_data%n_obs, k)
            call fe_memcpy_dtoh(c_loc(host_WW(1, 1)), d_Q)
            call fe_memcpy_dtoh(c_loc(host_Wy(1)), d_b)
            call symmetrize_upper(host_WW)
            allocate(variance_diag(k))
            do i = 1, k
                variance_diag(i) = host_WW(i, i)
            end do
        else
            allocate(host_WW(0, 0))
            allocate(host_Wy(0))
            allocate(variance_diag(0))
        end if

        allocate(host_Q(k, k))
        allocate(host_b(k))

        if (k > 0) then
            allocate(is_endog(k))
            is_endog = .false.
            if (allocated(cfg%iv_regressors)) then
                do i = 1, size(cfg%iv_regressors)
                    if (cfg%iv_regressors(i) < 1 .or. cfg%iv_regressors(i) > k) then
                        write(warn_buf, '("Ignoring IV regressor index ",I0," (out of range).")') cfg%iv_regressors(i)
                        call log_warn(trim(warn_buf))
                    else
                        is_endog(cfg%iv_regressors(i)) = .true.
                    end if
                end do
            end if
            n_endog = count(is_endog)
            n_exog = k - n_endog
            allocate(idx_endog(n_endog))
            allocate(idx_exog(n_exog))
            if (n_endog > 0) idx_endog = pack([(i, i = 1, k)], is_endog)
            if (n_exog > 0) idx_exog = pack([(i, i = 1, k)], .not. is_endog)
            call initialize_iv_labels(header, cfg%iv_regressors)
        else
            allocate(is_endog(0))
            allocate(idx_endog(0))
            allocate(idx_exog(0))
            n_endog = 0
            n_exog = 0
            call initialize_iv_labels(header, cfg%iv_regressors)
        end if

        n_available_instr = header%n_instruments
        n_selected_instr = 0
        if (n_available_instr <= 0) then
            if (allocated(cfg%iv_instrument_cols) .and. size(cfg%iv_instrument_cols) > 0) then
                call log_warn('Dataset contains no instrument columns; ignoring --iv-z-cols.')
            end if
            allocate(idx_instruments(0))
        else
            if (allocated(cfg%iv_instrument_cols) .and. size(cfg%iv_instrument_cols) > 0) then
                allocate(idx_instruments(size(cfg%iv_instrument_cols)))
                do i = 1, size(cfg%iv_instrument_cols)
                    if (cfg%iv_instrument_cols(i) < 1 .or. cfg%iv_instrument_cols(i) > n_available_instr) then
                        write(warn_buf, '("Ignoring instrument column index ",I0," (out of range).")') &
                            cfg%iv_instrument_cols(i)
                        call log_warn(trim(warn_buf))
                        cycle
                    end if
                    n_selected_instr = n_selected_instr + 1
                    idx_instruments(n_selected_instr) = int(cfg%iv_instrument_cols(i), int32)
                end do
                if (n_selected_instr == 0) then
                    call log_warn('No valid instrument indices supplied; using all instrument columns.')
                    deallocate(idx_instruments)
                    allocate(idx_instruments(n_available_instr))
                    do i = 1, n_available_instr
                        idx_instruments(i) = int(i, int32)
                    end do
                    n_selected_instr = n_available_instr
                else
                    if (n_selected_instr < size(cfg%iv_instrument_cols)) then
                        call log_warn('Some instrument column indices were ignored because they are out of range.')
                    end if
                    if (n_selected_instr < size(idx_instruments)) then
                        allocate(tmp_instr(n_selected_instr))
                        tmp_instr = idx_instruments(1:n_selected_instr)
                        call move_alloc(tmp_instr, idx_instruments)
                    end if
                end if
            else
                allocate(idx_instruments(n_available_instr))
                do i = 1, n_available_instr
                    idx_instruments(i) = int(i, int32)
                end do
                n_selected_instr = n_available_instr
            end if
        end if

        n_total_instr = n_exog + n_selected_instr
        use_iv = (n_endog > 0)
        if (use_iv .and. n_selected_instr < n_endog) then
            write(warn_buf, '("Insufficient instruments (",I0," instrument columns for ",I0,' // &
                '" endogenous regressors); reverting to OLS.")') n_selected_instr, n_endog
            call log_warn(trim(warn_buf))
            use_iv = .false.
        end if
        result%is_iv = use_iv
        if (use_iv) then
            result%n_instruments = n_total_instr
        else
            result%n_instruments = 0
        end if

        if (use_iv) then
            call build_iv_normal_equations(host_Q, host_b, info)
            if (info /= 0) then
                result%solver_info = info
                call cleanup()
                return
            end if
        else
            call build_ols_normal_equations(host_Q, host_b)
        end if

        call solve_with_column_filter(host_Q, host_b, variance_diag, result, keep_idx, Q_inv_kept, info)
        result%solver_info = info

        call system_clock(count=it1)
        result%time_regression = real(it1 - it0) / real(itrate)

        call system_clock(count=it0)
        if (info == 0) then
            allow_gpu_cluster_builder = cfg%fast_mode .and. cfg%use_gpu .and. &
                .not. (cfg%use_formula_design .and. cfg%formula_has_categorical)
            call compute_standard_errors(keep_idx, Q_inv_kept, host, group_sizes, cfg%verbose)
            call finalize_regression_stats(result, gpu_data%n_obs, header%n_fe > 0)
        else
            if (allocated(keep_idx)) deallocate(keep_idx)
            if (allocated(Q_inv_kept)) deallocate(Q_inv_kept)
            call cleanup()
            return
        end if

        if (allocated(keep_idx)) deallocate(keep_idx)
        if (allocated(Q_inv_kept)) deallocate(Q_inv_kept)
        call cleanup()

        call system_clock(count=it1)
        result%time_se = real(it1 - it0) / real(itrate)

    contains

        subroutine initialize_coefficient_names(header_local, n_reg)
            type(fe_dataset_header), intent(in) :: header_local
            integer, intent(in) :: n_reg
            integer :: name_len, i
            character(len=64) :: label_buf

            if (allocated(result%coef_names)) deallocate(result%coef_names)
            if (n_reg <= 0) then
                allocate(character(len=1) :: result%coef_names(0))
                return
            end if

            name_len = 0
            if (allocated(header_local%regressor_names)) then
                if (size(header_local%regressor_names) == n_reg) then
                    do i = 1, n_reg
                        name_len = max(name_len, len_trim(header_local%regressor_names(i)))
                    end do
                end if
            end if
            do i = 1, n_reg
                write(label_buf, '(A,I0)') 'beta_', i
                name_len = max(name_len, len_trim(label_buf))
            end do
            if (name_len <= 0) name_len = 12

            allocate(character(len=name_len) :: result%coef_names(n_reg))
            result%coef_names = ''
            if (allocated(header_local%regressor_names) .and. size(header_local%regressor_names) == n_reg) then
                do i = 1, n_reg
                    call assign_padded(result%coef_names(i), trim(header_local%regressor_names(i)))
                end do
            else
                do i = 1, n_reg
                    write(label_buf, '(A,I0)') 'beta_', i
                    call assign_padded(result%coef_names(i), trim(label_buf))
                end do
            end if
        end subroutine initialize_coefficient_names

        subroutine initialize_cluster_labels(header_local)
            type(fe_dataset_header), intent(in) :: header_local
            integer :: count, i, label_len
            character(len=64) :: label_buf

            if (allocated(result%cluster_labels)) deallocate(result%cluster_labels)
            count = size(result%cluster_fe_dims)
            if (count <= 0) then
                allocate(character(len=1) :: result%cluster_labels(0))
                return
            end if

            label_len = 0
            do i = 1, count
                label_buf = cluster_dimension_label(header_local, result%cluster_fe_dims(i))
                label_len = max(label_len, len_trim(label_buf))
            end do
            if (label_len <= 0) label_len = 8

            allocate(character(len=label_len) :: result%cluster_labels(count))
            result%cluster_labels = ''
            do i = 1, count
                label_buf = cluster_dimension_label(header_local, result%cluster_fe_dims(i))
                call assign_padded(result%cluster_labels(i), trim(label_buf))
            end do
        end subroutine initialize_cluster_labels

        subroutine initialize_iv_labels(header_local, iv_indices)
            type(fe_dataset_header), intent(in) :: header_local
            integer(int32), intent(in) :: iv_indices(:)
            integer :: count, i, name_len, idx
            character(len=64) :: label_buf

            if (allocated(result%iv_labels)) deallocate(result%iv_labels)
            count = size(iv_indices)
            if (count <= 0) then
                allocate(character(len=1) :: result%iv_labels(0))
                return
            end if

            name_len = 0
            do i = 1, count
                idx = int(iv_indices(i))
                label_buf = regressor_label_from_header(header_local, idx)
                name_len = max(name_len, len_trim(label_buf))
            end do
            if (name_len <= 0) name_len = 8
            allocate(character(len=name_len) :: result%iv_labels(count))
            result%iv_labels = ''
            do i = 1, count
                idx = int(iv_indices(i))
                label_buf = regressor_label_from_header(header_local, idx)
                call assign_padded(result%iv_labels(i), trim(label_buf))
            end do
        end subroutine initialize_iv_labels

        function regressor_label_from_header(header_local, idx) result(text)
            type(fe_dataset_header), intent(in) :: header_local
            integer, intent(in) :: idx
            character(len=64) :: text

            text = ''
            if (idx >= 1) then
                if (allocated(header_local%regressor_names)) then
                    if (idx <= size(header_local%regressor_names)) then
                        if (len_trim(header_local%regressor_names(idx)) > 0) then
                            text = trim(header_local%regressor_names(idx))
                            return
                        end if
                    end if
                end if
            end if
            write(text, '(A,I0)') 'Reg ', idx
        end function regressor_label_from_header

        function cluster_dimension_label(header_local, dim_index) result(text)
            type(fe_dataset_header), intent(in) :: header_local
            integer, intent(in) :: dim_index
            character(len=64) :: text

            text = ''
            if (allocated(header_local%fe_names)) then
                if (dim_index >= 1 .and. dim_index <= size(header_local%fe_names)) then
                    if (len_trim(header_local%fe_names(dim_index)) > 0) then
                        text = trim(header_local%fe_names(dim_index))
                        return
                    end if
                end if
            end if
            write(text, '(A,I0)') 'FE dim ', dim_index
        end function cluster_dimension_label

        subroutine assign_padded(dest, source)
            character(len=*), intent(inout) :: dest
            character(len=*), intent(in) :: source
            integer :: copy_len

            dest = ''
            copy_len = min(len_trim(source), len(dest))
            if (copy_len > 0) dest(1:copy_len) = source(1:copy_len)
        end subroutine assign_padded

        subroutine build_ols_normal_equations(Q_out, b_out)
            real(real64), intent(out), target :: Q_out(:, :)
            real(real64), intent(out), target :: b_out(:)
            if (k <= 0) return
            Q_out = host_WW
            b_out = host_Wy
        end subroutine build_ols_normal_equations

        subroutine build_iv_normal_equations(Q_out, b_out, info_out)
            real(real64), intent(out) :: Q_out(:, :)
            real(real64), intent(out) :: b_out(:)
            integer, intent(out) :: info_out
            type(fe_device_buffer) :: d_Q_iv, d_b_iv, d_cross, d_temp
            real(real64), allocatable, target :: Qzz(:, :), Qinv(:, :), Qzx(:, :)
            real(real64), allocatable, target :: Qzy(:), iv_rhs(:), proj_matrix(:, :)
            integer(int64) :: bytes
            integer(int32), allocatable :: col_idx(:)
            integer :: n_instr_aug

            info_out = 0
            n_instr_aug = n_total_instr
            if (n_instr_aug <= 0) then
                info_out = -1
                return
            end if

            bytes = gpu_data%n_obs * int(n_instr_aug, int64) * REAL64_BYTES
            call fe_device_alloc(d_Z_aug, bytes)
            if (n_exog > 0) then
                allocate(col_idx(n_exog))
                col_idx = int(idx_exog, int32)
                call fe_gpu_copy_columns(gpu_data%d_W, gpu_data%n_obs, col_idx, d_Z_aug, 0)
                deallocate(col_idx)
            end if
            if (n_selected_instr > 0) then
                allocate(col_idx(n_selected_instr))
                col_idx = idx_instruments
                call fe_gpu_copy_columns(gpu_data%d_Z, gpu_data%n_obs, col_idx, d_Z_aug, n_exog)
                deallocate(col_idx)
            end if

            bytes = int(n_instr_aug, int64) * int(n_instr_aug, int64) * REAL64_BYTES
            call fe_device_alloc(d_Q_iv, bytes)

            bytes = int(n_instr_aug, int64) * REAL64_BYTES
            call fe_device_alloc(d_b_iv, bytes)

            bytes = int(n_instr_aug, int64) * int(k, int64) * REAL64_BYTES
            call fe_device_alloc(d_cross, bytes)

            call fe_gpu_compute_cross_products(gpu_data%d_y, d_Z_aug, d_Q_iv, d_b_iv, gpu_data%n_obs, n_instr_aug)
            call fe_gpu_cross_product(d_Z_aug, gpu_data%d_W, d_cross, gpu_data%n_obs, n_instr_aug, k)

            allocate(Qzz(n_instr_aug, n_instr_aug))
            allocate(Qzy(n_instr_aug))
            allocate(Qzx(n_instr_aug, k))
            allocate(Qinv(n_instr_aug, n_instr_aug))
            allocate(iv_rhs(n_instr_aug))
            allocate(proj_matrix(n_instr_aug, k))

            call fe_memcpy_dtoh(c_loc(Qzz(1, 1)), d_Q_iv)
            call fe_memcpy_dtoh(c_loc(Qzy(1)), d_b_iv)
            call fe_memcpy_dtoh(c_loc(Qzx(1, 1)), d_cross)
            call symmetrize_upper(Qzz)

            call chol_solve_and_invert(Qzz, Qzy, iv_rhs, Qinv, info_out)
            if (info_out /= 0) goto 100

            proj_matrix = matmul(Qinv, Qzx)
            Q_out = matmul(transpose(Qzx), proj_matrix)
            Q_out = 0.5_real64 * (Q_out + transpose(Q_out))
            b_out = matmul(transpose(Qzx), iv_rhs)

            bytes = int(n_instr_aug, int64) * int(k, int64) * REAL64_BYTES
            call fe_device_alloc(d_temp, bytes)
            call fe_memcpy_htod(d_temp, c_loc(proj_matrix(1, 1)), bytes)

            bytes = gpu_data%n_obs * int(k, int64) * REAL64_BYTES
            call fe_device_alloc(gpu_data%d_proj_W, bytes)
            call fe_gpu_matmul(d_Z_aug, d_temp, gpu_data%d_proj_W, gpu_data%n_obs, n_instr_aug, k)

100         continue
            call fe_device_free(d_Q_iv)
            call fe_device_free(d_b_iv)
            call fe_device_free(d_cross)
            call fe_device_free(d_temp)
            if (c_associated(d_Z_aug%ptr)) call fe_device_free(d_Z_aug)
            if (allocated(Qzz)) deallocate(Qzz)
            if (allocated(Qzy)) deallocate(Qzy)
            if (allocated(Qzx)) deallocate(Qzx)
            if (allocated(Qinv)) deallocate(Qinv)
            if (allocated(iv_rhs)) deallocate(iv_rhs)
            if (allocated(proj_matrix)) deallocate(proj_matrix)
        end subroutine build_iv_normal_equations

        subroutine solve_with_column_filter(Q_full, b_full, variance_diag, est, idx_out, Q_inv, info_out)
            real(real64), intent(inout) :: Q_full(:, :)
            real(real64), intent(in) :: b_full(:)
            real(real64), intent(in) :: variance_diag(:)
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
                if (size(variance_diag) >= i) then
                    diag_val = abs(variance_diag(i))
                else
                    diag_val = abs(Q_full(i, i))
                end if
                if (diag_val > diag_max) diag_max = diag_val
            end do
            diag_max = max(diag_max, 1.0_real64)
            tol = 1.0e-10_real64 * diag_max

            allocate(keep(size(b_full)))
            keep = .true.
            do i = 1, size(b_full)
                if (size(variance_diag) >= i) then
                    diag_val = abs(variance_diag(i))
                else
                    diag_val = abs(Q_full(i, i))
                end if
                if (diag_val <= tol) keep(i) = .false.
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

        subroutine compute_standard_errors(idx, Q_inv_kept, host_local, group_sizes_local, verbose)
            integer, intent(in) :: idx(:)
            real(real64), intent(in) :: Q_inv_kept(:, :)
            type(fe_host_arrays), intent(in) :: host_local
            integer, intent(in) :: group_sizes_local(:)
            logical, intent(in) :: verbose
            type(fe_device_buffer) :: d_beta, d_residual, d_scores, d_meat, d_cluster_temp
            type(fe_device_buffer) :: d_scores_pool, d_meat_pool
            real(real64) :: rss, sigma2, t_start, t_end, weight
            integer :: kept, intercept_count
            integer(int64) :: df
            integer(int64) :: bytes_obs, bytes_reg, bytes_cluster_ids
            integer :: n_clusters, min_cluster_size, min_single_cluster
            real(real64), allocatable, target :: meat_full(:, :)
            real(real64), allocatable :: meat_kept(:, :), cov_mat(:, :), cov_accum(:, :)
            real(real64), allocatable :: cov_cpu(:, :)
            real(real64), allocatable :: diag_vals(:)
            integer :: i, n_cluster_dims, mask, subset_size, subset_pos, dim_index, status_build
            integer :: n_clusters_cpu, status_ids
            integer, allocatable :: subset_dims(:)
            real(real64), allocatable, target :: beta_copy(:)
            integer(int32), allocatable, target :: combo_ids(:)
            logical :: has_clusters, cluster_success
            character(len=256) :: warn_msg
            type(fe_device_buffer) :: ids_buffer
            real(real64) :: subset_start, subset_mid, subset_end, subset_weight, scalar, denom_adj
            character(len=64) :: subset_label
            type(fe_device_buffer) :: reg_matrix
            logical :: use_projected, used_gpu_ids, disable_gpu_cluster_builder
            integer :: param_count_total, param_count_effective, dof_fes_effective, nested_adj
            integer :: cpu_cluster_fallbacks
            logical :: psd_fix_applied, psd_check_success
            real(real64) :: min_eig
            logical :: enable_debug
            real(real64), allocatable, target :: host_residual(:), host_reg(:, :)
            integer(int32), allocatable, target :: host_combo_ids(:)
            integer(int32), allocatable, target :: gpu_ids_host(:)


            kept = size(idx)
            if (allocated(result%se)) result%se = 0.0_real64
            disable_gpu_cluster_builder = .not. allow_gpu_cluster_builder
            cpu_cluster_fallbacks = 0

            call cpu_time(t_start)
            d_scores_pool%ptr = c_null_ptr
            d_scores_pool%size_bytes = 0_int64
            d_meat_pool%ptr = c_null_ptr
            d_meat_pool%size_bytes = 0_int64

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
            intercept_count = 1
            param_count_total = max(0, result%dof_model) + max(0, result%dof_fes) + intercept_count
            df = max(1_int64, gpu_data%n_obs - int(param_count_total, int64))
            result%dof_resid = df
            result%rss = rss
            sigma2 = rss / real(df, real64)

            diag_vals = diag_vector(Q_inv_kept)

            has_clusters = allocated(result%cluster_fe_dims) .and. size(result%cluster_fe_dims) > 0
            use_projected = result%is_iv .and. c_associated(gpu_data%d_proj_W%ptr)
            if (use_projected) then
                reg_matrix = gpu_data%d_proj_W
            else
                reg_matrix = gpu_data%d_W
            end if
            enable_debug = DEBUG_CLUSTERS .and. verbose .and. size(result%beta) <= 6
            if (enable_debug) then
                allocate(host_residual(int(gpu_data%n_obs)))
                call fe_memcpy_dtoh(c_loc(host_residual(1)), d_residual)
                allocate(host_reg(int(gpu_data%n_obs), size(result%beta)))
                call fe_memcpy_dtoh(c_loc(host_reg(1, 1)), reg_matrix)
            end if
            cluster_success = .true.
            if (.not. has_clusters) then
                do i = 1, kept
                    result%se(idx(i)) = sqrt(max(0.0_real64, sigma2 * diag_vals(i)))
                end do
            else
                n_cluster_dims = size(result%cluster_fe_dims)
                min_cluster_size = huge(0)
                min_single_cluster = huge(0)
                allocate(cov_accum(kept, kept))
                cov_accum = 0.0_real64
                bytes_cluster_ids = gpu_data%n_obs * INT32_BYTES
                call fe_device_alloc(d_cluster_temp, bytes_cluster_ids)
                allocate(meat_full(size(result%beta), size(result%beta)))
                allocate(meat_kept(kept, kept))
                allocate(cov_mat(kept, kept))

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
                        min_cluster_size = min(min_cluster_size, n_clusters)
                        min_single_cluster = min(min_single_cluster, n_clusters)
                        ids_buffer = gpu_data%fe_dims(dim_index)%fe_ids
                    else
                        used_gpu_ids = .false.
                        n_clusters = 0
                        if (.not. disable_gpu_cluster_builder) then
                            call build_gpu_cluster_ids_helper(gpu_data, subset_dims, group_sizes, d_cluster_temp, &
                                n_clusters, status_build)
                            if (status_build == 0 .and. n_clusters > 0) then
                                min_cluster_size = min(min_cluster_size, n_clusters)
                                ids_buffer = d_cluster_temp
                                used_gpu_ids = .true.
                            else
                                disable_gpu_cluster_builder = .true.
                                if (verbose) then
                                    write(warn_msg, '("GPU cluster builder failed subset ",A," status=",I0," clusters=",I0)') &
                                        trim(subset_label), status_build, n_clusters
                                    call log_warn(trim(warn_msg))
                                    call log_warn(trim(fe_gpu_last_error()))
                                end if
                                call fe_gpu_clear_error()
                            end if
                        end if

                        if (.not. used_gpu_ids) then
                            cpu_cluster_fallbacks = cpu_cluster_fallbacks + 1
                            if (verbose) then
                                write(warn_msg, '("GPU builder fallback to CPU for subset ",A)') trim(subset_label)
                                call log_info(trim(warn_msg))
                            end if
                            if (.not. c_associated(d_cluster_temp%ptr)) then
                                call fe_device_alloc(d_cluster_temp, bytes_cluster_ids)
                            end if
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
                            min_cluster_size = min(min_cluster_size, n_clusters)
                            call fe_memcpy_htod(d_cluster_temp, c_loc(combo_ids(1)), bytes_cluster_ids)
                            ids_buffer = d_cluster_temp
                        end if
                    end if
                    call cpu_time(subset_mid)

                    bytes_reg = int(n_clusters, int64) * int(size(result%beta), int64) * REAL64_BYTES
                    if (d_scores_pool%size_bytes < bytes_reg) then
                        if (c_associated(d_scores_pool%ptr)) call fe_device_free(d_scores_pool)
                        call fe_device_alloc(d_scores_pool, bytes_reg)
                    end if
                    d_scores = d_scores_pool
                    call fe_device_memset(d_scores, 0)
                    call fe_gpu_cluster_scores(d_residual, reg_matrix, ids_buffer, gpu_data%n_obs, size(result%beta), &
                        n_clusters, d_scores)

                    bytes_reg = int(size(result%beta), int64) * int(size(result%beta), int64) * REAL64_BYTES
                    if (d_meat_pool%size_bytes < bytes_reg) then
                        if (c_associated(d_meat_pool%ptr)) call fe_device_free(d_meat_pool)
                        call fe_device_alloc(d_meat_pool, bytes_reg)
                    end if
                    d_meat = d_meat_pool
                    call fe_gpu_cluster_meat(d_scores, n_clusters, size(result%beta), d_meat)

                    call fe_memcpy_dtoh(c_loc(meat_full(1, 1)), d_meat)
                    call symmetrize_upper(meat_full)

                    meat_kept = meat_full(idx, idx)
                    cov_mat = matmul(Q_inv_kept, matmul(meat_kept, Q_inv_kept))
                    subset_weight = weight
                    if (enable_debug .and. kept <= 6) then
                        call log_matrix('meat['//trim(subset_label)//']', meat_kept)
                        call log_matrix('cov['//trim(subset_label)//'] before weight', cov_mat)
                        write(warn_msg, '("weight=",F6.2)') subset_weight
                        call log_info(trim(warn_msg))
                        if (enable_debug) then
                            if (.not. allocated(cov_cpu)) allocate(cov_cpu(kept, kept))
                            call compute_cpu_covariance(subset_dims, host_local%fe_ids, group_sizes_local, host_residual, &
                                host_reg, idx, Q_inv_kept, cov_cpu, status_build)
                            if (status_build == 0) then
                                call log_matrix('cov_cpu['//trim(subset_label)//']', cov_cpu)
                            end if
                            deallocate(cov_cpu)
                        end if
                    end if
                    cov_accum = cov_accum + subset_weight * cov_mat
                    ! keep device buffer for reuse; freed after clustering loop
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
                    ! Small-sample adjustment (Cameron-Gelbach-Miller) should only subtract model rank;
                    ! fixed effects are already accounted for by within-transformation.
                    dof_fes_effective = 0
                    nested_adj = 0
                    if (has_clusters .and. dof_fes_nested > 0) nested_adj = 1
                    param_count_effective = max(0, result%dof_model) + max(0, dof_fes_effective) + nested_adj
                    denom_adj = real(max(1_int64, gpu_data%n_obs - int(param_count_effective, int64)), real64)
                    scalar = real(max(1_int64, gpu_data%n_obs - 1_int64), real64) / denom_adj
                    if (min_single_cluster > 1) then
                        scalar = scalar * real(min_single_cluster, real64) / real(min_single_cluster - 1, real64)
                    end if
                    cov_accum = cov_accum * scalar
                    call symmetrize_upper(cov_accum)
                    call compute_min_eigenvalue(cov_accum, min_eig, psd_check_success)
                    if (verbose .and. psd_check_success) then
                        write(warn_msg, '("Min eigenvalue of clustered VCV before PSD fix: ",ES12.4)') min_eig
                        call log_info(trim(warn_msg))
                    end if
                    call enforce_psd_covariance(cov_accum, psd_fix_applied, psd_check_success)
                    if (.not. psd_check_success) then
                        call log_warn('Unable to ensure PSD covariance; eigen decomposition failed.')
                    else if (psd_fix_applied) then
                        call log_warn('VCV matrix was non-positive semi-definite; CGM PSD fix applied.')
                    end if
                    if (verbose .and. kept <= 6) then
                        do i = 1, kept
                            write(warn_msg, '(10(1X,ES12.4))') cov_accum(i, 1:kept)
                            call log_info(trim(warn_msg))
                        end do
                    end if
                    do i = 1, kept
                        result%se(idx(i)) = sqrt(max(0.0_real64, cov_accum(i, i)))
                    end do
                    if (min_single_cluster < huge(0)) then
                        result%dof_resid = max(1_int64, int(min_single_cluster, int64) - 1_int64)
                    end if
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
                if (allocated(meat_full)) deallocate(meat_full)
                if (allocated(meat_kept)) deallocate(meat_kept)
                if (allocated(cov_mat)) deallocate(cov_mat)
                if (c_associated(d_cluster_temp%ptr)) call fe_device_free(d_cluster_temp)
                if (allocated(combo_ids)) deallocate(combo_ids)
            end if

            if (cpu_cluster_fallbacks > 0 .and. verbose) then
                write(warn_msg, '("Cluster ID GPU builder unavailable for ",I0," subset(s); CPU fallback used.")') &
                    cpu_cluster_fallbacks
                call log_info(trim(warn_msg))
            end if

            call fe_device_free(d_beta)
            call fe_device_free(d_residual)
            if (allocated(diag_vals)) deallocate(diag_vals)
            if (allocated(host_residual)) deallocate(host_residual)
            if (allocated(host_reg)) deallocate(host_reg)
            if (allocated(host_combo_ids)) deallocate(host_combo_ids)
            if (allocated(gpu_ids_host)) deallocate(gpu_ids_host)
            if (c_associated(d_scores_pool%ptr)) call fe_device_free(d_scores_pool)
            if (c_associated(d_meat_pool%ptr)) call fe_device_free(d_meat_pool)

            call cpu_time(t_end)
            result%time_se = t_end - t_start
        end subroutine compute_standard_errors

        subroutine compute_cpu_covariance(subset_dims, fe_ids, group_sizes, residual_host, reg_host, idx_keep, &
            Q_inv_kept, cov_out, status_out)
            integer, intent(in) :: subset_dims(:)
            integer(int32), intent(in) :: fe_ids(:, :)
            integer, intent(in) :: group_sizes(:)
            real(real64), intent(in) :: residual_host(:)
            real(real64), intent(in) :: reg_host(:, :)
            integer, intent(in) :: idx_keep(:)
            real(real64), intent(in) :: Q_inv_kept(:, :)
            real(real64), intent(out) :: cov_out(:, :)
            integer, intent(out) :: status_out
            integer :: n_clusters_local, status_ids
            integer(int32), allocatable :: host_combo_ids(:)
            real(real64), allocatable :: scores(:, :)
            real(real64), allocatable :: meat_cpu(:, :)
            integer :: obs, j, n_reg
            integer(int64) :: n_obs
            character(len=64) :: subset_label

            status_out = 0
            n_obs = size(residual_host, kind=int64)
            n_reg = size(reg_host, 2)

            allocate(host_combo_ids(int(n_obs)))
            call build_cluster_ids(fe_ids, group_sizes, subset_dims, host_combo_ids, n_clusters_local, status_ids)
            if (status_ids /= 0 .or. n_clusters_local <= 0) then
                status_out = 1
                deallocate(host_combo_ids)
                return
            end if
            subset_label = format_subset_dim_list(subset_dims)
            write(subset_label, '(A,I0)') trim(subset_label)//' clusters=', n_clusters_local
            call log_info(trim(subset_label))

            allocate(scores(n_clusters_local, n_reg))
            scores = 0.0_real64
            do obs = 1, int(n_obs)
                j = host_combo_ids(obs)
                if (j < 1 .or. j > n_clusters_local) cycle
                scores(j, :) = scores(j, :) + residual_host(obs) * reg_host(obs, :)
            end do

            allocate(meat_cpu(n_reg, n_reg))
            meat_cpu = matmul(transpose(scores), scores)
            cov_out = matmul(Q_inv_kept, matmul(meat_cpu(idx_keep, idx_keep), Q_inv_kept))

            deallocate(scores, meat_cpu, host_combo_ids)
        end subroutine compute_cpu_covariance

        subroutine build_gpu_cluster_ids_helper(dataset, subset_dims, group_sizes_local, ids_buf, n_clusters, status)
            type(fe_gpu_dataset), intent(in) :: dataset
            integer, intent(in) :: subset_dims(:)
            integer, intent(in) :: group_sizes_local(:)
            type(fe_device_buffer), intent(inout) :: ids_buf
            integer, intent(out) :: n_clusters
            integer, intent(out) :: status
            integer :: subset_size, i, dim_index
            integer(int64), allocatable, target :: stride_vals(:)
            type(c_ptr), allocatable, target :: ptrs(:)
            integer(int64) :: stride_accum, dim_size
            integer(int64), parameter :: KEY_LIMIT = 2_int64 ** 60

            subset_size = size(subset_dims)
            if (subset_size <= 1) then
                status = -1
                n_clusters = 0
                return
            end if

            allocate(stride_vals(subset_size))
            stride_accum = 1_int64
            do i = 1, subset_size
                dim_index = subset_dims(i)
                if (dim_index < 1 .or. dim_index > dataset%n_fe) then
                    status = 1
                    n_clusters = 0
                    deallocate(stride_vals)
                    return
                end if
                dim_size = int(group_sizes_local(dim_index), int64)
                if (dim_size <= 0_int64) then
                    status = 2
                    n_clusters = 0
                    deallocate(stride_vals)
                    return
                end if
                stride_vals(i) = stride_accum
                if (dim_size > 0_int64 .and. stride_accum > KEY_LIMIT / dim_size) then
                    status = 3
                    n_clusters = 0
                    deallocate(stride_vals)
                    return
                end if
                stride_accum = stride_accum * dim_size
            end do

            allocate(ptrs(subset_size))
            do i = 1, subset_size
                ptrs(i) = dataset%fe_dims(subset_dims(i))%fe_ids%ptr
            end do

            call fe_gpu_build_multi_cluster_ids(ptrs, stride_vals, subset_size, dataset%n_obs, ids_buf, n_clusters, status)
            if (status /= 0) n_clusters = 0

            deallocate(ptrs)
            deallocate(stride_vals)
        end subroutine build_gpu_cluster_ids_helper

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
            if (c_associated(d_Z_aug%ptr)) call fe_device_free(d_Z_aug)
            call fe_gpu_dataset_destroy(gpu_data)
            call fe_gpu_linalg_finalize()
            if (allocated(host_Q)) deallocate(host_Q)
            if (allocated(host_b)) deallocate(host_b)
            if (allocated(host_WW)) deallocate(host_WW)
            if (allocated(host_Wy)) deallocate(host_Wy)
            if (allocated(variance_diag)) deallocate(variance_diag)
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

        subroutine compute_min_eigenvalue(mat, min_eig, success)
            real(real64), intent(in) :: mat(:, :)
            real(real64), intent(out) :: min_eig
            logical, intent(out) :: success
            real(real64), allocatable :: work(:)
            real(real64), allocatable :: eigvals(:)
            real(real64), allocatable :: scratch(:, :)
            integer :: n, lwork
            integer(int32) :: n_i, lda_i, lwork_i, info_i

            success = .false.
            min_eig = 0.0_real64
            n = size(mat, 1)
            if (n <= 0) return

            allocate(scratch(n, n))
            scratch = mat
            allocate(eigvals(n))
            lwork = max(1, 3 * n - 1)
            allocate(work(lwork))

            n_i = int(n, int32)
            lda_i = int(max(1, n), int32)
            lwork_i = int(lwork, int32)
            call dsyev('N', 'U', n_i, scratch, lda_i, eigvals, work, lwork_i, info_i)
            deallocate(work)
            if (info_i /= 0_int32) then
                deallocate(eigvals, scratch)
                return
            end if
            min_eig = minval(eigvals)
            success = .true.
            deallocate(eigvals, scratch)
        end subroutine compute_min_eigenvalue

        subroutine log_matrix(label, mat)
            character(len=*), intent(in) :: label
            real(real64), intent(in) :: mat(:, :)
            integer :: r, c
            character(len=256) :: buf
            call log_info(label)
            do r = 1, size(mat, 1)
                write(buf, '(99(1X,ES12.4))') (mat(r, c), c = 1, size(mat, 2))
                call log_info(trim(buf))
            end do
        end subroutine log_matrix

        subroutine enforce_psd_covariance(mat, fix_applied, success)
            real(real64), intent(inout) :: mat(:, :)
            logical, intent(out) :: fix_applied
            logical, intent(out) :: success
            real(real64), allocatable :: eigvec(:, :)
            real(real64), allocatable :: eigvals(:)
            real(real64), allocatable :: work(:)
            real(real64), allocatable :: scratch(:, :)
            integer :: n, lwork, idx
            integer(int32) :: n_i, lda_i, lwork_i, info_i
            real(real64) :: lambda

            fix_applied = .false.
            success = .true.
            n = size(mat, 1)
            if (n <= 0) return

            allocate(eigvec(n, n))
            eigvec = mat
            allocate(eigvals(n))
            lwork = max(1, 3 * n - 1)
            allocate(work(lwork))

            n_i = int(n, int32)
            lda_i = int(max(1, n), int32)
            lwork_i = int(lwork, int32)
            call dsyev('V', 'U', n_i, eigvec, lda_i, eigvals, work, lwork_i, info_i)
            deallocate(work)
            if (info_i /= 0_int32) then
                success = .false.
                deallocate(eigvec, eigvals)
                return
            end if

            allocate(scratch(n, n))
            scratch = 0.0_real64
            do idx = 1, n
                lambda = eigvals(idx)
                if (lambda < 0.0_real64) then
                    lambda = 0.0_real64
                    fix_applied = .true.
                end if
                if (lambda > 0.0_real64) then
                    scratch(:, idx) = sqrt(lambda) * eigvec(:, idx)
                else
                    scratch(:, idx) = 0.0_real64
                end if
            end do
            mat = matmul(scratch, transpose(scratch))
            deallocate(eigvec, eigvals, scratch)
        end subroutine enforce_psd_covariance

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

    function estimate_nested_fe_dof(cluster_dims, group_sizes) result(total)
        integer(int32), intent(in) :: cluster_dims(:)
        integer, intent(in) :: group_sizes(:)
        integer :: total
        integer :: i, dim_index

        total = 0
        if (size(cluster_dims) == 0) return
        do i = 1, size(cluster_dims)
            dim_index = cluster_dims(i)
            if (dim_index < 1 .or. dim_index > size(group_sizes)) cycle
            total = total + max(0, group_sizes(dim_index) - 1)
        end do
    end function estimate_nested_fe_dof

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
