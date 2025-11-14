program fe_gpu_main
    use iso_fortran_env, only: real64
    use fe_config, only: fe_runtime_config, init_default_config
    use fe_cli, only: parse_cli_arguments
    use fe_logging, only: log_info, log_warn
    use fe_types, only: fe_dataset_header, fe_host_arrays
    use fe_data_io, only: load_dataset_from_file, release_host_arrays
    use fe_gpu_runtime, only: fe_gpu_context, fe_gpu_initialize, fe_gpu_finalize, fe_gpu_backend_available
    use fe_grouping, only: compute_fe_group_sizes
    use fe_pipeline, only: fe_gpu_estimate, fe_estimation_result
    implicit none
    intrinsic :: system_clock, erfc
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
        call system_clock(count=it1)
        load_time = real(it1 - it0) / real(itrate)
        !call log_info('Dataset summary -> ' // header%summary())
        
        if (use_gpu) then


            call system_clock(count=it0)
            call compute_fe_group_sizes(host%fe_ids, group_sizes)
            call log_fe_dimensions(group_sizes)
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
        integer :: i, j
        real(real64) :: demeant, regt, set
        real(real64) :: t_stat, p_value, lower_ci, upper_ci
        real(real64), parameter :: Z_CRIT = 1.959963984540054_real64
        real(real64), parameter :: INV_SQRT2 = 0.7071067811865475_real64

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

        if (allocated(est%cluster_fe_dims) .and. size(est%cluster_fe_dims) > 0) then
            cluster_list = '['
            do j = 1, size(est%cluster_fe_dims)
                write(cluster_list(len_trim(cluster_list)+1:), '(I0)') est%cluster_fe_dims(j)
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
        call log_info('    Coefficient          Estimate            StdErr            t-stat          Pr(>|t|)' // &
            '         Lower 95%        Upper 95%')
        call log_info('  -------------   ---------------   ---------------   ---------------   ---------------' // &
            '   ---------------   ---------------')
        do i = 1, size(est%beta)
            if (est%se(i) > 0.0_real64) then
                t_stat = est%beta(i) / est%se(i)
            else
                t_stat = 0.0_real64
            end if
            p_value = erfc(abs(t_stat) * INV_SQRT2)
            lower_ci = est%beta(i) - Z_CRIT * est%se(i)
            upper_ci = est%beta(i) + Z_CRIT * est%se(i)
            write(msg, '(A12,I3,3X,ES15.6,3X,ES15.6,3X,ES15.6,3X,ES15.6,3X,ES15.6,3X,ES15.6)') &
                'beta:', i, est%beta(i), est%se(i), t_stat, p_value, lower_ci, upper_ci
            call log_info(trim(msg))
        end do
        call log_info('  -------------   ---------------   ---------------   ---------------   ---------------' // &
            '   ---------------   ---------------')
        print*, ""

        write(msg, '("  dof (model): ",I0,", dof (residuals): ",I0)') est%dof_model, est%dof_resid
        call log_info(trim(msg))
        write(msg, '("  R^2=",F8.5,", R^2 adj=",F8.5,", R^2 within=",F8.5)') est%r2, est%r2_adj, est%r2_within
        call log_info(trim(msg))
        write(msg, '("  F-statistic=",ES12.5)') est%f_stat
        call log_info(trim(msg))
        print*, ""
    end subroutine report_results

    subroutine log_fe_dimensions(groups)
        integer, intent(in) :: groups(:)
        character(len=128) :: msg
        integer :: i
        do i = 1, size(groups)
            write(msg, '("FE dimension ",I0,": ",I0," groups")') i, groups(i)
            call log_info(trim(msg))
        end do
    end subroutine log_fe_dimensions

    function format_runtime_message(total_time) result(msg)
        real(real64), intent(in) :: total_time
        character(len=128) :: msg
        write(msg, '("fe_gpu runtime completed in ",F8.3," s.")') total_time
        msg = trim(msg)
    end function format_runtime_message

end program fe_gpu_main
