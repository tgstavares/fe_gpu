program fe_gpu_main
    use iso_fortran_env, only: int32, real64
    use fe_config, only: fe_runtime_config, init_default_config, fe_formula_term, fe_formula_interaction
    use fe_cli, only: parse_cli_arguments, apply_config_bindings
    use fe_logging, only: log_info, log_warn
    use fe_types, only: fe_dataset_header, fe_host_arrays
    use fe_data_io, only: load_dataset_from_file, release_host_arrays
    use fe_gpu_runtime, only: fe_gpu_context, fe_gpu_initialize, fe_gpu_finalize, fe_gpu_backend_available
    use fe_grouping, only: compute_fe_group_sizes
    use fe_pipeline, only: fe_gpu_estimate, fe_estimation_result
    implicit none
    intrinsic :: system_clock, erfc
    integer, parameter :: MAX_CATEGORIES = 40
    real(real64), parameter :: CATEGORY_TOL = 1.0e-6_real64
    integer, parameter :: NAME_BUFFER_LEN = 128

    type design_column
        real(real64), allocatable :: values(:)
        character(len=:), allocatable :: name
        logical :: is_endog = .false.
    end type design_column

    type variable_info_entry
        character(len=:), allocatable :: name
        integer :: column_index = 0
        logical :: needs_levels = .false.
        integer, allocatable :: levels(:)
        logical :: valid = .false.
    end type variable_info_entry
    type(fe_runtime_config) :: cfg
    type(fe_dataset_header) :: header
    type(fe_host_arrays) :: host
    type(fe_gpu_context) :: gpu_ctx
    type(fe_estimation_result) :: est
    logical :: data_exists
    logical :: use_gpu
    integer, allocatable :: group_sizes(:)
    real(real64) :: total_time, load_time, grouping_time
    real(real64) :: t10
    logical :: have_estimate
    integer :: it0, it1, itrate, itotal0, itotal1

    print*,""

    call system_clock(count_rate = itrate)
    call system_clock(count = it0)

    load_time = 0.0_real64
    grouping_time = 0.0_real64
    have_estimate = .false.
    call system_clock(count=itotal0)

    call init_default_config(cfg)
    call parse_cli_arguments(cfg)

    use_gpu = cfg%use_gpu
    if (cfg%use_gpu .and. .not. fe_gpu_backend_available()) then
        call log_warn('GPU backend unavailable; falling back to CPU execution (not yet implemented).')
        use_gpu = .false.
    end if


    if (use_gpu) then
        call fe_gpu_initialize(gpu_ctx)
    end if
    
    inquire(file=cfg%data_path, exist=data_exists)
    if (data_exists) then
        
        call system_clock(count=it0)
        call load_dataset_from_file(cfg%data_path, header, host)
        call apply_config_bindings(cfg, header)
        call apply_fe_filter(cfg, header, host)
        if (cfg%use_formula_design) then
            call build_formula_design_matrix(cfg, header, host)
        else
            call apply_regressor_filter(cfg, header, host)
        end if
        call system_clock(count=it1)
        load_time = real(it1 - it0) / real(itrate)
        !call log_info('Dataset summary -> ' // header%summary())
        
        if (use_gpu) then


            call system_clock(count=it0)
            call compute_fe_group_sizes(host%fe_ids, group_sizes)
            if (allocated(header%fe_names) .and. size(header%fe_names) == size(group_sizes)) then
                call log_fe_dimensions(group_sizes, header%fe_names)
            else
                call log_fe_dimensions(group_sizes)
            end if
            call system_clock(it1)
            grouping_time = real(it1 - it0)/real(itrate)
            

            call fe_gpu_estimate(cfg, header, host, group_sizes, est)
            have_estimate = .true.
            deallocate(group_sizes)
        else
            call log_warn('CPU-only execution path is not available yet.')
        end if

        call release_host_arrays(host)
    else
        call log_warn('Dataset file not found: ' // trim(cfg%data_path) // '; skipping load.')
    end if

    
    if (use_gpu .and. gpu_ctx%initialized) then
        call fe_gpu_finalize(gpu_ctx)
    end if

    call system_clock(count=itotal1)
    total_time = real(itotal1 - itotal0) / real(itrate)

    if (have_estimate) then
        !print*,""
        call report_results(est, load_time, grouping_time, total_time)
        !print*,""
    end if

    !call log_info(adjustl(format_runtime_message(total_time)))

contains

    subroutine report_results(est, load_time, grouping_time, total_time)
        type(fe_estimation_result), intent(inout) :: est
        real(real64), intent(in) :: load_time, grouping_time, total_time
        character(len=256) :: msg
        character(len=128) :: cluster_list
        character(len=128) :: iv_list
        character(len=64) :: coef_label
        character(len=64) :: cluster_label
        character(len=15) :: coef_label_trim
        integer :: i, j
        real(real64) :: demeant, regt, set
        real(real64) :: t_stat, p_value, lower_ci, upper_ci
        real(real64), parameter :: INV_SQRT2 = 0.7071067811865475_real64
        real(real64) :: dof_resid, t_crit

        write(msg, '("FE iterations=",I0,", converged=",L1)') est%fe_iterations, est%converged
        call log_info(trim(msg))

        demeant = est%time_demean
        regt = est%time_regression
        set = est%time_se
        write(msg, '("Timing (s): total=",F8.3,", load=",F8.3,", grouping=",F8.3,", demeaning=",F8.3,", regression=",F8.3, &
            ", se=",F8.3)') total_time, load_time, grouping_time, demeant, regt, set
        call log_info(trim(msg))

        if (est%solver_info /= 0) then
            write(msg, '("Solver failed with info=",I0)') est%solver_info
            call log_warn(trim(msg))
            return
        end if

        if (.not. allocated(est%beta)) then
            call log_warn('No coefficients were produced.')
            return
        end if

        if (.not. allocated(est%se)) then
            allocate(est%se(size(est%beta)))
            est%se = 0.0_real64
        end if

        if (est%is_iv) then
            write(msg, '("Estimator: 2SLS (IV) with ",I0," instruments")') max(0, est%n_instruments)
        else
            msg = 'Estimator: OLS'
        end if
        call log_info(trim(msg))
        if (allocated(est%depvar_name)) then
            if (len_trim(est%depvar_name) > 0) then
                call log_info('Dependent variable: ' // trim(est%depvar_name))
            end if
        end if
        if (allocated(est%iv_labels)) then
            if (size(est%iv_labels) > 0) then
                iv_list = '['
                do i = 1, size(est%iv_labels)
                    if (len_trim(est%iv_labels(i)) == 0) cycle
                    if (len_trim(iv_list) > 1) then
                        iv_list(len_trim(iv_list)+1:len_trim(iv_list)+1) = ','
                    end if
                    write(iv_list(len_trim(iv_list)+1:), '(A)') trim(est%iv_labels(i))
                end do
                iv_list = trim(iv_list) // ']'
                call log_info('Endogenous regressors (--iv-cols): ' // trim(iv_list))
            end if
        end if

        if (allocated(est%cluster_fe_dims) .and. size(est%cluster_fe_dims) > 0) then
            cluster_list = '['
            do j = 1, size(est%cluster_fe_dims)
                if (allocated(est%cluster_labels) .and. size(est%cluster_labels) >= j) then
                    cluster_label = trim(est%cluster_labels(j))
                    if (len_trim(cluster_label) == 0) then
                        write(cluster_label, '(I0)') est%cluster_fe_dims(j)
                    end if
                else
                    write(cluster_label, '(I0)') est%cluster_fe_dims(j)
                end if
                write(cluster_list(len_trim(cluster_list)+1:), '(A)') trim(cluster_label)
                if (j < size(est%cluster_fe_dims)) then
                    cluster_list(len_trim(cluster_list)+1:len_trim(cluster_list)+1) = ','
                end if
            end do
            cluster_list = trim(cluster_list) // ']'
            write(msg, '("Standard errors clustered on FE dimensions ",A)') trim(cluster_list)
        else
            msg = 'Standard errors: homoskedastic'
        end if
        call log_info(trim(msg))

        print*, ""
        call log_info('                   Coefficient          Estimate            StdErr            t-stat          Pr(>|t|)' // &
            '         Lower 95%         Upper 95%')
        call log_info('  ----------------------------   ---------------   ---------------   ---------------   ---------------' // &
            '   ---------------   ---------------')
        do i = 1, size(est%beta)
            if (est%se(i) > 0.0_real64) then
                t_stat = est%beta(i) / est%se(i)
            else
                t_stat = 0.0_real64
            end if
            dof_resid = real(max(1, int(est%dof_resid)), real64)
            p_value = two_sided_p_from_t(t_stat, dof_resid)
            t_crit = t_quantile_975(dof_resid)
            lower_ci = est%beta(i) - t_crit * est%se(i)
            upper_ci = est%beta(i) + t_crit * est%se(i)
            if (allocated(est%coef_names) .and. size(est%coef_names) >= i) then
                coef_label = trim(est%coef_names(i))
                if (len_trim(coef_label) == 0) then
                    write(coef_label, '(A,I0)') 'beta_', i
                end if
            else
                write(coef_label, '(A,I0)') 'beta_', i
            end if
            coef_label_trim = trim(coef_label)

            write(msg, '(A30,2X,ES16.6,2X,ES16.6,2X,ES16.6,2X,ES16.6,2X,ES16.6,2X,ES16.6)') &
                adjustr(coef_label_trim), est%beta(i), est%se(i), t_stat, p_value, lower_ci, upper_ci
            call log_info(trim(msg))
        end do
        call log_info('  ----------------------------   ---------------   ---------------   ---------------   ---------------' // &
            '   ---------------   ---------------')
        print*, ""

        write(msg, '("  dof (model): ",I0,", dof (residuals): ",I0)') est%dof_model, est%dof_resid
        call log_info(trim(msg))
        write(msg, '("  R^2=",F8.5,", R^2 adj=",F8.5,", R^2 within=",F8.5)') est%r2, est%r2_adj, est%r2_within
        call log_info(trim(msg))
        write(msg, '("  F-statistic=",ES12.5," (df1=",I0,", df2=",I0,")")') est%f_stat, est%dof_model, est%dof_resid
        call log_info(trim(msg))
        print*, ""
    end subroutine report_results

    pure real(real64) function t_cdf(value, nu) result(cdf)
        real(real64), intent(in) :: value
        real(real64), intent(in) :: nu
        real(real64) :: x, betareg
        if (nu <= 0.0_real64) then
            cdf = 0.5_real64
            return
        end if
        if (value == 0.0_real64) then
            cdf = 0.5_real64
            return
        end if
        x = nu / (nu + value * value)
        betareg = regularized_incomplete_beta(0.5_real64 * nu, 0.5_real64, x)
        if (value > 0.0_real64) then
            cdf = 1.0_real64 - 0.5_real64 * betareg
        else
            cdf = 0.5_real64 * betareg
        end if
    end function t_cdf

    pure real(real64) function two_sided_p_from_t(t_stat, nu) result(p)
        real(real64), intent(in) :: t_stat, nu
        real(real64) :: cdf_val
        cdf_val = t_cdf(t_stat, nu)
        p = 2.0_real64 * min(cdf_val, 1.0_real64 - cdf_val)
    end function two_sided_p_from_t

    pure real(real64) function t_quantile_975(nu) result(tcrit)
        real(real64), intent(in) :: nu
        real(real64) :: lo, hi, mid, target, cdf_mid
        integer :: iter
        target = 0.975_real64
        lo = 0.0_real64
        hi = 10.0_real64
        do while (t_cdf(hi, nu) < target)
            hi = hi * 2.0_real64
            if (hi > 1.0e4_real64) exit
        end do
        do iter = 1, 60
            mid = 0.5_real64 * (lo + hi)
            cdf_mid = t_cdf(mid, nu)
            if (cdf_mid < target) then
                lo = mid
            else
                hi = mid
            end if
        end do
        tcrit = 0.5_real64 * (lo + hi)
    end function t_quantile_975

    pure real(real64) function regularized_incomplete_beta(a, b, x) result(res)
        real(real64), intent(in) :: a, b, x
        real(real64) :: bt, qab, qap, qam, c, d, h, ap, bp, app, bpp, am, bm, az, bz, aold, eps, fpmin
        integer :: m, m2

        eps = 1.0e-12_real64
        fpmin = 1.0e-30_real64

        if (x <= 0.0_real64) then
            res = 0.0_real64
            return
        else if (x >= 1.0_real64) then
            res = 1.0_real64
            return
        end if

        bt = exp(log_gamma(a + b) - log_gamma(a) - log_gamma(b) + a * log(x) + b * log(1.0_real64 - x))
        if (x < (a + 1.0_real64) / (a + b + 2.0_real64)) then
            res = bt * betacf(a, b, x) / a
        else
            res = 1.0_real64 - bt * betacf(b, a, 1.0_real64 - x) / b
        end if
    end function regularized_incomplete_beta

    pure real(real64) function betacf(a, b, x) result(cf)
        real(real64), intent(in) :: a, b, x
        integer, parameter :: max_it = 200
        real(real64), parameter :: eps = 1.0e-12_real64
        real(real64), parameter :: fpmin = 1.0e-30_real64
        integer :: m, m2
        real(real64) :: aa, c, d, del, h, qab, qap, qam

        qab = a + b
        qap = a + 1.0_real64
        qam = a - 1.0_real64
        c = 1.0_real64
        d = 1.0_real64 - qab * x / qap
        if (abs(d) < fpmin) d = fpmin
        d = 1.0_real64 / d
        h = d
        do m = 1, max_it
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0_real64 + aa * d
            if (abs(d) < fpmin) d = fpmin
            c = 1.0_real64 + aa / c
            if (abs(c) < fpmin) c = fpmin
            d = 1.0_real64 / d
            h = h * d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0_real64 + aa * d
            if (abs(d) < fpmin) d = fpmin
            c = 1.0_real64 + aa / c
            if (abs(c) < fpmin) c = fpmin
            d = 1.0_real64 / d
            del = d * c
            h = h * del
            if (abs(del - 1.0_real64) < eps) exit
        end do
        cf = h
    end function betacf

    subroutine log_fe_dimensions(groups, fe_names)
        integer, intent(in) :: groups(:)
        character(len=*), intent(in), optional :: fe_names(:)
        character(len=128) :: msg
        character(len=64) :: label
        integer :: i
        logical :: have_names

        have_names = present(fe_names)
        do i = 1, size(groups)
            label = ''
            if (have_names) then
                if (size(fe_names) >= i) then
                    label = trim(fe_names(i))
                end if
            end if
            if (len_trim(label) > 0) then
                write(msg, '("FE dimension ",I0," (",A,"): ",I0," groups")') i, trim(label), groups(i)
            else
                write(msg, '("FE dimension ",I0,": ",I0," groups")') i, groups(i)
            end if
            call log_info(trim(msg))
        end do
    end subroutine log_fe_dimensions

    function format_runtime_message(total_time) result(msg)
        real(real64), intent(in) :: total_time
        character(len=128) :: msg
        write(msg, '("fe_gpu runtime completed in ",F8.3," s.")') total_time
        msg = trim(msg)
    end function format_runtime_message

    subroutine apply_regressor_filter(cfg, header, host)
        type(fe_runtime_config), intent(inout) :: cfg
        type(fe_dataset_header), intent(inout) :: header
        type(fe_host_arrays), intent(inout) :: host
        integer(int32), allocatable :: selection(:)
        integer :: n_sel, j, n_obs, orig_k, idx, i
        real(real64), allocatable :: newW(:, :)
        character(len=:), allocatable :: new_names(:)
        integer :: max_len

        if (.not. allocated(cfg%regressor_selection)) return
        if (size(cfg%regressor_selection) == 0) return

        selection = cfg%regressor_selection
        n_sel = size(selection)
        orig_k = size(host%W, 2)
        n_obs = size(host%W, 1)
        if (n_sel <= 0 .or. orig_k <= 0) return

        allocate(newW(n_obs, n_sel))
        do j = 1, n_sel
            idx = selection(j)
            if (idx < 1 .or. idx > orig_k) then
                call log_warn('Regressor selection index out of range; keeping original matrix.')
                deallocate(newW)
                return
            end if
            newW(:, j) = host%W(:, idx)
        end do
        call move_alloc(newW, host%W)
        header%n_regressors = n_sel

        if (allocated(header%regressor_names)) then
            max_len = 0
            do j = 1, n_sel
                if (selection(j) >= 1 .and. selection(j) <= size(header%regressor_names)) then
                    max_len = max(max_len, len_trim(header%regressor_names(selection(j))))
                end if
            end do
            if (max_len <= 0) max_len = len(header%regressor_names(1))
            allocate(character(len=max_len) :: new_names(n_sel))
            new_names = ''
            do j = 1, n_sel
                if (selection(j) >= 1 .and. selection(j) <= size(header%regressor_names)) then
                    new_names(j)(1:len_trim(header%regressor_names(selection(j)))) = &
                        trim(header%regressor_names(selection(j)))
                end if
            end do
            call move_alloc(new_names, header%regressor_names)
        end if

        if (allocated(cfg%regressor_selection)) then
            cfg%regressor_selection = [(int(i, int32), i = 1, n_sel)]
        end if
    end subroutine apply_regressor_filter

    subroutine apply_fe_filter(cfg, header, host)
        type(fe_runtime_config), intent(inout) :: cfg
        type(fe_dataset_header), intent(inout) :: header
        type(fe_host_arrays), intent(inout) :: host
        integer(int32), allocatable :: selection(:)
        integer :: n_keep, j, n_obs, orig_fe, idx, k
        integer, allocatable :: remap(:)
        integer :: max_len
        integer(int32), allocatable :: tmp_dims(:)
        integer :: count
        integer(int32), allocatable :: new_ids(:, :)

        if (.not. allocated(cfg%fe_selection)) return
        if (size(cfg%fe_selection) == 0) return

        selection = cfg%fe_selection
        n_keep = size(selection)
        orig_fe = size(host%fe_ids, 1)
        if (n_keep <= 0 .or. orig_fe <= 0) return

        n_obs = size(host%fe_ids, 2)
        allocate(remap(orig_fe))
        remap = -1
        do j = 1, n_keep
            idx = selection(j)
            if (idx < 1 .or. idx > orig_fe) then
                call log_warn('FE selection index out of range; keeping original FE dimensions.')
                deallocate(remap)
                return
            end if
            remap(idx) = j
        end do

        allocate(new_ids(n_keep, size(host%fe_ids, 2)))
        do j = 1, n_keep
            idx = selection(j)
            new_ids(j, :) = host%fe_ids(idx, :)
        end do
        deallocate(host%fe_ids)
        allocate(host%fe_ids(n_keep, size(new_ids, 2)))
        host%fe_ids = new_ids
        deallocate(new_ids)
        header%n_fe = n_keep

        if (allocated(header%fe_names)) then
            max_len = 0
            do j = 1, n_keep
                if (selection(j) >= 1 .and. selection(j) <= size(header%fe_names)) then
                    max_len = max(max_len, len_trim(header%fe_names(selection(j))))
                end if
            end do
            if (max_len <= 0) max_len = len(header%fe_names(1))
            call resize_name_list(header%fe_names, selection, max_len)
        end if

        if (allocated(cfg%cluster_fe_dims)) then
            allocate(tmp_dims(size(cfg%cluster_fe_dims)))
            count = 0
            do j = 1, size(cfg%cluster_fe_dims)
                idx = cfg%cluster_fe_dims(j)
                if (idx < 1 .or. idx > orig_fe) cycle
                k = remap(idx)
                if (k > 0) then
                    count = count + 1
                    tmp_dims(count) = k
                end if
            end do
            if (count > 0) then
                if (size(cfg%cluster_fe_dims) /= count) then
                    deallocate(cfg%cluster_fe_dims)
                    allocate(cfg%cluster_fe_dims(count))
                end if
                cfg%cluster_fe_dims(1:count) = tmp_dims(1:count)
            else
                deallocate(cfg%cluster_fe_dims)
                allocate(cfg%cluster_fe_dims(0))
            end if
            deallocate(tmp_dims)
        end if

        if (allocated(cfg%fe_selection)) then
            deallocate(cfg%fe_selection)
            allocate(cfg%fe_selection(n_keep))
            cfg%fe_selection = [(int(k, int32), k = 1, n_keep)]
        end if

        deallocate(remap)
    end subroutine apply_fe_filter

    subroutine resize_name_list(name_list, selection, max_len)
        character(len=:), allocatable, intent(inout) :: name_list(:)
        integer(int32), intent(in) :: selection(:)
        integer, intent(in) :: max_len
        character(len=:), allocatable :: new_list(:)
        integer :: j, idx

        allocate(character(len=max_len) :: new_list(size(selection)))
        new_list = ''
        do j = 1, size(selection)
            idx = selection(j)
            if (idx >= 1 .and. idx <= size(name_list)) then
                new_list(j)(1:len_trim(name_list(idx))) = trim(name_list(idx))
            end if
        end do
        call move_alloc(new_list, name_list)
    end subroutine resize_name_list

    subroutine build_formula_design_matrix(cfg, header, host)
        type(fe_runtime_config), intent(inout) :: cfg
        type(fe_dataset_header), intent(inout) :: header
        type(fe_host_arrays), intent(inout) :: host
        type(variable_info_entry), allocatable :: var_infos(:)
        real(real64), allocatable :: design(:, :)
        character(len=:), allocatable :: names(:)
        logical, allocatable :: endog_flags(:)
        integer :: total_cols, col_ptr, max_len, count_iv
        integer :: i
        integer(int32), allocatable :: iv_idx(:)
        integer :: n_obs

        if (.not. allocated(header%regressor_names)) then
            call log_warn('Dataset lacks regressor names; cannot expand formula.')
            return
        end if
        call collect_variable_info(cfg, header, host, var_infos)
        if (.not. allocated(var_infos)) then
            call log_warn('No variables available for formula expansion.')
            return
        end if
        n_obs = size(host%W, 1)
        total_cols = 0

        do i = 1, size(cfg%formula_terms)
            total_cols = total_cols + count_columns_for_term(cfg%formula_terms(i), var_infos)
        end do
        do i = 1, size(cfg%formula_interactions)
            total_cols = total_cols + count_columns_for_interaction(cfg%formula_interactions(i), var_infos)
        end do

        if (total_cols <= 0) then
            call log_warn('Formula produced no regressors; skipping transformation.')
            return
        end if

        allocate(design(n_obs, total_cols))
        allocate(character(len=NAME_BUFFER_LEN) :: names(total_cols))
        allocate(endog_flags(total_cols))
        design = 0.0_real64
        names = ''
        endog_flags = .false.
        col_ptr = 0
        max_len = 0

        do i = 1, size(cfg%formula_terms)
            call append_main_term_columns(cfg, header, host, cfg%formula_terms(i), var_infos, &
     &          design, names, endog_flags, col_ptr, max_len)
        end do
        do i = 1, size(cfg%formula_interactions)
            call append_interaction_columns(cfg, header, host, cfg%formula_interactions(i), var_infos, &
     &          design, names, endog_flags, col_ptr, max_len)
        end do

        if (col_ptr < total_cols) then
            call shrink_matrix(design, names, endog_flags, col_ptr)
            total_cols = col_ptr
        end if

        call move_alloc(design, host%W)
        header%n_regressors = total_cols
        if (allocated(header%regressor_names)) deallocate(header%regressor_names)
        max_len = max(1, max_len)
        allocate(character(len=max_len) :: header%regressor_names(total_cols))
        do i = 1, total_cols
            header%regressor_names(i) = ''
            if (len_trim(names(i)) > 0) then
                header%regressor_names(i)(1:len_trim(names(i))) = trim(names(i))
            end if
        end do
        if (allocated(cfg%regressor_selection)) deallocate(cfg%regressor_selection)
        allocate(cfg%regressor_selection(total_cols))
        cfg%regressor_selection = [(int(i, int32), i = 1, total_cols)]

        count_iv = count(endog_flags)
        if (count_iv > 0) then
            allocate(iv_idx(count_iv))
            iv_idx = 0
            count_iv = 0
            do i = 1, total_cols
                if (endog_flags(i)) then
                    count_iv = count_iv + 1
                    iv_idx(count_iv) = int(i, int32)
                end if
            end do
        else
            allocate(iv_idx(0))
        end if
        call finalize_iv_indices(cfg, iv_idx)
        if (allocated(iv_idx)) deallocate(iv_idx)
        deallocate(names)
        deallocate(endog_flags)
    end subroutine build_formula_design_matrix

    subroutine shrink_matrix(mat, names, flags, new_cols)
        real(real64), allocatable, intent(inout) :: mat(:, :)
        character(len=:), allocatable, intent(inout) :: names(:)
        logical, allocatable, intent(inout) :: flags(:)
        integer, intent(in) :: new_cols
        real(real64), allocatable :: tmp(:, :)
        character(len=:), allocatable :: new_names(:)
        logical, allocatable :: new_flags(:)
        integer :: name_len
        if (new_cols <= 0) then
            deallocate(mat)
            allocate(mat(0, 0))
            deallocate(names)
            allocate(character(len=NAME_BUFFER_LEN) :: names(0))
            deallocate(flags)
            allocate(flags(0))
            return
        end if
        if (size(mat, 2) == new_cols) return
        allocate(tmp(size(mat, 1), new_cols))
        tmp = mat(:, 1:new_cols)
        call move_alloc(tmp, mat)
        name_len = len(names(1))
        allocate(character(len=name_len) :: new_names(new_cols))
        new_names = names(1:new_cols)
        call move_alloc(new_names, names)
            allocate(new_flags(new_cols))
        new_flags = flags(1:new_cols)
        call move_alloc(new_flags, flags)
    end subroutine shrink_matrix

    subroutine finalize_iv_indices(cfg, indices)
        type(fe_runtime_config), intent(inout) :: cfg
        integer(int32), intent(in) :: indices(:)
        integer :: count_iv
        count_iv = count(indices /= 0)
        if (count_iv <= 0) then
            if (allocated(cfg%iv_regressors)) deallocate(cfg%iv_regressors)
            allocate(cfg%iv_regressors(0))
            return
        end if
        if (allocated(cfg%iv_regressors)) deallocate(cfg%iv_regressors)
        allocate(cfg%iv_regressors(count_iv))
        cfg%iv_regressors = indices(1:count_iv)
    end subroutine finalize_iv_indices

    integer function count_columns_for_term(term, infos) result(total)
        type(fe_formula_term), intent(in) :: term
        type(variable_info_entry), intent(in) :: infos(:)
        integer :: idx
        total = 0
        idx = find_variable_info(infos, term%name)
        if (idx <= 0) return
        if (.not. infos(idx)%valid) return
        if (term%is_categorical) then
            if (.not. allocated(infos(idx)%levels)) return
            if (size(infos(idx)%levels) > 1) total = size(infos(idx)%levels) - 1
        else
            total = 1
        end if
    end function count_columns_for_term

    integer function count_columns_for_interaction(interaction, infos) result(total)
        type(fe_formula_interaction), intent(in) :: interaction
        type(variable_info_entry), intent(in) :: infos(:)
        integer :: i, idx, factor_cols
        total = 0
        if (.not. allocated(interaction%factors)) return
        if (size(interaction%factors) <= 1) return
        total = 1
        do i = 1, size(interaction%factors)
            idx = find_variable_info(infos, interaction%factors(i)%name)
            if (idx <= 0) then
                total = 0
                return
            end if
            if (.not. infos(idx)%valid) then
                total = 0
                return
            end if
            if (interaction%factors(i)%is_categorical) then
                if (.not. allocated(infos(idx)%levels)) then
                    total = 0
                    return
                end if
                factor_cols = size(infos(idx)%levels) - 1
                if (factor_cols <= 0) then
                    total = 0
                    return
                end if
                total = total * factor_cols
            end if
        end do
    end function count_columns_for_interaction

    subroutine collect_variable_info(cfg, header, host, infos)
        type(fe_runtime_config), intent(in) :: cfg
        type(fe_dataset_header), intent(in) :: header
        type(fe_host_arrays), intent(in) :: host
        type(variable_info_entry), allocatable, intent(out) :: infos(:)
        character(len=:), allocatable :: names(:)
        logical, allocatable :: need_cat(:)
        integer :: count_names, i, j
        character(len=:), allocatable :: lower_name
        integer :: idx

        call gather_factor_names(cfg, names, need_cat)
        count_names = size(names)
        if (count_names <= 0) then
            allocate(infos(0))
            return
        end if
        allocate(infos(count_names))
        do i = 1, count_names
            infos(i)%name = names(i)
            infos(i)%needs_levels = need_cat(i)
            infos(i)%column_index = find_regressor_index(header, names(i))
            if (infos(i)%column_index <= 0) then
                call log_warn('Variable "' // trim(names(i)) // '" not found in dataset; dropping.')
                cycle
            end if
            if (infos(i)%needs_levels) then
                call collect_levels_for_column(host%W(:, infos(i)%column_index), infos(i)%name, infos(i)%levels)
                if (size(infos(i)%levels) <= 1) then
                    call log_warn('Categorical variable "' // trim(names(i)) // '" has <=1 level; skipping dummy expansion.')
                    infos(i)%needs_levels = .false.
                end if
            end if
            infos(i)%valid = .true.
        end do
    end subroutine collect_variable_info

    subroutine gather_factor_names(cfg, names, need_cat)
        type(fe_runtime_config), intent(in) :: cfg
        character(len=:), allocatable, intent(out) :: names(:)
        logical, allocatable, intent(out) :: need_cat(:)
        character(len=:), allocatable :: temp(:)
        logical, allocatable :: temp_cat(:)
        integer :: i, j
        type(fe_formula_term) :: term

        do i = 1, size(cfg%formula_terms)
            call append_factor_name(temp, temp_cat, cfg%formula_terms(i)%name, cfg%formula_terms(i)%is_categorical)
        end do
        do i = 1, size(cfg%formula_interactions)
            if (.not. allocated(cfg%formula_interactions(i)%factors)) cycle
            do j = 1, size(cfg%formula_interactions(i)%factors)
                term = cfg%formula_interactions(i)%factors(j)
                call append_factor_name(temp, temp_cat, term%name, term%is_categorical)
            end do
        end do
        if (.not. allocated(temp)) then
            allocate(character(len=1) :: temp(0))
        end if
        if (.not. allocated(temp_cat)) then
            allocate(temp_cat(0))
        end if
        call move_alloc(temp, names)
        call move_alloc(temp_cat, need_cat)
    end subroutine gather_factor_names

    subroutine append_factor_name(list, cat_flags, name, is_cat)
        character(len=:), allocatable, intent(inout) :: list(:)
        logical, allocatable, intent(inout) :: cat_flags(:)
        character(len=*), intent(in) :: name
        logical, intent(in) :: is_cat
        integer :: i
        character(len=:), allocatable :: lower

        if (len_trim(name) == 0) return
        if (.not. allocated(list)) then
            call append_string_item(list, trim(name))
            call append_logical(cat_flags, is_cat)
            return
        end if
        lower = to_lower(trim(name))
        do i = 1, size(list)
            if (to_lower(trim(list(i))) == lower) then
                cat_flags(i) = cat_flags(i) .or. is_cat
                return
            end if
        end do
        call append_string_item(list, trim(name))
        call append_logical(cat_flags, is_cat)
    end subroutine append_factor_name

    subroutine append_string_item(list, value)
        character(len=:), allocatable, intent(inout) :: list(:)
        character(len=*), intent(in) :: value
        character(len=:), allocatable :: tmp(:)
        integer :: n
        if (len_trim(value) == 0) return
        if (.not. allocated(list)) then
            allocate(character(len=len_trim(value)) :: list(1))
            list(1) = trim(value)
            return
        end if
        n = size(list)
        allocate(character(len=max(len(list(1)), len_trim(value))) :: tmp(n + 1))
        if (n > 0) tmp(1:n) = list
        tmp(n + 1) = trim(value)
        call move_alloc(tmp, list)
    end subroutine append_string_item

    subroutine append_logical(array, value)
        logical, allocatable, intent(inout) :: array(:)
        logical, intent(in) :: value
        logical, allocatable :: tmp(:)
        integer :: n
        if (.not. allocated(array)) then
            allocate(array(1))
            array(1) = value
            return
        end if
        n = size(array)
        allocate(tmp(n + 1))
        tmp(1:n) = array
        tmp(n + 1) = value
        call move_alloc(tmp, array)
    end subroutine append_logical

    integer function find_variable_info(infos, name) result(idx)
        type(variable_info_entry), intent(in) :: infos(:)
        character(len=*), intent(in) :: name
        integer :: i
        character(len=:), allocatable :: lower
        lower = to_lower(trim(name))
        idx = 0
        do i = 1, size(infos)
            if (to_lower(trim(infos(i)%name)) == lower) then
                idx = i
                return
            end if
        end do
    end function find_variable_info

    integer function find_regressor_index(header, name) result(idx)
        type(fe_dataset_header), intent(in) :: header
        character(len=*), intent(in) :: name
        integer :: i
        if (.not. allocated(header%regressor_names)) then
            idx = 0
            return
        end if
        idx = 0
        do i = 1, size(header%regressor_names)
            if (to_lower(trim(header%regressor_names(i))) == to_lower(trim(name))) then
                idx = i
                return
            end if
        end do
    end function find_regressor_index

    logical function is_endogenous_variable(cfg, name) result(flag)
        type(fe_runtime_config), intent(in) :: cfg
        character(len=*), intent(in) :: name
        integer :: i
        flag = .false.
        if (.not. allocated(cfg%iv_regressor_names)) return
        do i = 1, size(cfg%iv_regressor_names)
            if (to_lower(trim(cfg%iv_regressor_names(i))) == to_lower(trim(name))) then
                flag = .true.
                return
            end if
        end do
    end function is_endogenous_variable

    subroutine append_main_term_columns(cfg, header, host, term, infos, design, names, endog_flags, col_ptr, max_len)
        type(fe_runtime_config), intent(in) :: cfg
        type(fe_dataset_header), intent(in) :: header
        type(fe_host_arrays), intent(in) :: host
        type(fe_formula_term), intent(in) :: term
        type(variable_info_entry), intent(in) :: infos(:)
        real(real64), intent(inout) :: design(:, :)
        character(len=:), allocatable, intent(inout) :: names(:)
        logical, intent(inout) :: endog_flags(:)
        integer, intent(inout) :: col_ptr
        integer, intent(inout) :: max_len
        integer :: idx, i, level_count
        integer :: info_index
        integer :: level
        integer :: row
        logical :: is_endog

        info_index = find_variable_info(infos, term%name)
        if (info_index <= 0) return
        if (.not. infos(info_index)%valid) return
        idx = infos(info_index)%column_index
        is_endog = is_endogenous_variable(cfg, term%name)

        if (.not. term%is_categorical) then
            col_ptr = col_ptr + 1
            if (col_ptr > size(design, 2)) return
            design(:, col_ptr) = host%W(:, idx)
            names(col_ptr) = trim(term%name)
            endog_flags(col_ptr) = is_endog
            max_len = max(max_len, len_trim(names(col_ptr)))
        else
            if (.not. allocated(infos(info_index)%levels)) return
            level_count = size(infos(info_index)%levels)
            if (level_count <= 1) return
            do i = 2, level_count
                col_ptr = col_ptr + 1
                if (col_ptr > size(design, 2)) return
                design(:, col_ptr) = 0.0_real64
                do row = 1, size(host%W, 1)
                    if (int(nint(host%W(row, idx))) == infos(info_index)%levels(i)) then
                        design(row, col_ptr) = 1.0_real64
                    end if
                end do
                names(col_ptr) = trim(term%name) // ':' // int_to_string(infos(info_index)%levels(i))
                endog_flags(col_ptr) = is_endog
                max_len = max(max_len, len_trim(names(col_ptr)))
            end do
        end if
    end subroutine append_main_term_columns

    subroutine append_interaction_columns(cfg, header, host, interaction, infos, design, names, endog_flags, col_ptr, max_len)
        type(fe_runtime_config), intent(in) :: cfg
        type(fe_dataset_header), intent(in) :: header
        type(fe_host_arrays), intent(in) :: host
        type(fe_formula_interaction), intent(in) :: interaction
        type(variable_info_entry), intent(in) :: infos(:)
        real(real64), intent(inout) :: design(:, :)
        character(len=:), allocatable, intent(inout) :: names(:)
        logical, intent(inout) :: endog_flags(:)
        integer, intent(inout) :: col_ptr
        integer, intent(inout) :: max_len
        integer :: n_obs
        real(real64), allocatable :: current(:, :), next(:, :), new_mat(:, :)
        character(len=:), allocatable :: current_names(:), next_names(:), new_names(:)
        logical, allocatable :: current_endog(:), next_endog(:), new_endog(:)
        integer :: i, info_index, a, b, idx
        logical :: is_endog

        if (.not. allocated(interaction%factors)) return
        if (size(interaction%factors) <= 1) return
        n_obs = size(host%W, 1)

        call build_factor_block(cfg, host, interaction%factors(1), infos, current, current_names, current_endog)
        if (.not. allocated(current)) return

        do i = 2, size(interaction%factors)
            call build_factor_block(cfg, host, interaction%factors(i), infos, next, next_names, next_endog)
            if (.not. allocated(next)) then
                deallocate(current, current_names, current_endog)
                return
            end if
        allocate(new_mat(n_obs, size(current_names) * size(next_names)))
        allocate(character(len=NAME_BUFFER_LEN) :: new_names(size(current_names) * size(next_names)))
            allocate(new_endog(size(current_names) * size(next_names)))
            idx = 0
            do a = 1, size(current_names)
                do b = 1, size(next_names)
                    idx = idx + 1
                    new_mat(:, idx) = current(:, a) * next(:, b)
                    new_names(idx) = trim(current_names(a)) // '#' // trim(next_names(b))
                    new_endog(idx) = current_endog(a) .or. next_endog(b)
                end do
            end do
            deallocate(current, current_names, current_endog)
            call move_alloc(new_mat, current)
            call move_alloc(new_names, current_names)
            call move_alloc(new_endog, current_endog)
            deallocate(next, next_names, next_endog)
        end do

        do i = 1, size(current_names)
            col_ptr = col_ptr + 1
            if (col_ptr > size(design, 2)) exit
            design(:, col_ptr) = current(:, i)
            names(col_ptr) = trim(current_names(i))
            endog_flags(col_ptr) = current_endog(i)
            max_len = max(max_len, len_trim(names(col_ptr)))
        end do
        deallocate(current, current_names, current_endog)
    end subroutine append_interaction_columns

    subroutine build_factor_block(cfg, host, term, infos, data, labels, endog_flags)
        type(fe_runtime_config), intent(in) :: cfg
        type(fe_host_arrays), intent(in) :: host
        type(fe_formula_term), intent(in) :: term
        type(variable_info_entry), intent(in) :: infos(:)
        real(real64), allocatable, intent(out) :: data(:, :)
        character(len=:), allocatable, intent(out) :: labels(:)
        logical, allocatable, intent(out) :: endog_flags(:)
        integer :: info_index, idx, n_obs, level_count, i, row
        logical :: is_endog

        if (allocated(data)) deallocate(data)
        if (allocated(labels)) deallocate(labels)
        if (allocated(endog_flags)) deallocate(endog_flags)
        info_index = find_variable_info(infos, term%name)
        if (info_index <= 0) return
        if (.not. infos(info_index)%valid) return
        idx = infos(info_index)%column_index
        is_endog = is_endogenous_variable(cfg, term%name)
        n_obs = size(host%W, 1)
        if (.not. term%is_categorical) then
            allocate(data(n_obs, 1))
            data(:, 1) = host%W(:, idx)
            allocate(character(len=NAME_BUFFER_LEN) :: labels(1))
            labels(1) = trim(term%name)
            allocate(endog_flags(1))
            endog_flags(1) = is_endog
        else
            if (.not. allocated(infos(info_index)%levels)) return
            level_count = size(infos(info_index)%levels)
            if (level_count <= 1) return
            allocate(data(n_obs, level_count - 1))
            allocate(character(len=NAME_BUFFER_LEN) :: labels(level_count - 1))
            allocate(endog_flags(level_count - 1))
            data = 0.0_real64
            do i = 2, level_count
                do row = 1, n_obs
                    if (int(nint(host%W(row, idx))) == infos(info_index)%levels(i)) then
                        data(row, i - 1) = 1.0_real64
                    end if
                end do
                labels(i - 1) = trim(term%name) // ':' // int_to_string(infos(info_index)%levels(i))
                endog_flags(i - 1) = is_endog
            end do
        end if
    end subroutine build_factor_block

    subroutine collect_levels_for_column(column, var_name, levels)
        real(real64), intent(in) :: column(:)
        character(len=*), intent(in) :: var_name
        integer, allocatable, intent(out) :: levels(:)
        integer :: temp(MAX_CATEGORIES)
        integer :: count, i, val, pos
        logical :: seen, reached_limit

        temp = 0
        count = 0
        reached_limit = .false.
        do i = 1, size(column)
            val = int(nint(column(i)))
            if (abs(column(i) - real(val, kind=real64)) > CATEGORY_TOL) cycle
            seen = .false.
            pos = 1
            do while (pos <= count)
                if (temp(pos) == val) then
                    seen = .true.
                    exit
                end if
                pos = pos + 1
            end do
            if (.not. seen) then
                count = count + 1
                temp(count) = val
                if (count == MAX_CATEGORIES) then
                    reached_limit = .true.
                    exit
                end if
            end if
        end do
        if (count <= 0) then
            allocate(levels(0))
            return
        end if
        call sort_int_array(temp, count)
        allocate(levels(count))
        levels = temp(1:count)
        if (reached_limit) then
            call log_warn('Categorical variable "' // trim(var_name) // '" truncated to first ' // &
                trim(int_to_string(MAX_CATEGORIES)) // ' levels.')
        end if
    end subroutine collect_levels_for_column

    subroutine sort_int_array(values, count)
        integer, intent(inout) :: values(MAX_CATEGORIES)
        integer, intent(in) :: count
        integer :: i, j, key
        do i = 2, count
            key = values(i)
            j = i - 1
            do while (j >= 1 .and. values(j) > key)
                values(j + 1) = values(j)
                j = j - 1
            end do
            values(j + 1) = key
        end do
    end subroutine sort_int_array

    function int_to_string(value) result(text)
        integer, intent(in) :: value
        character(len=32) :: text
        write(text, '(I0)') value
    end function int_to_string

    function to_lower(text) result(out)
        character(len=*), intent(in) :: text
        character(len=len(text)) :: out
        integer :: i
        do i = 1, len(text)
            select case (text(i:i))
            case ('A':'Z')
                out(i:i) = achar(iachar(text(i:i)) + 32)
            case default
                out(i:i) = text(i:i)
            end select
        end do
    end function to_lower

end program fe_gpu_main
