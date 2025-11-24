module fe_gpu_demean
    use iso_c_binding, only: c_ptr, c_size_t, c_int, c_long_long, c_double, c_loc, c_null_ptr, c_associated
    use iso_fortran_env, only: int64, real64
    use fe_gpu_runtime, only: fe_gpu_check, fe_device_memset, fe_memcpy_dtoh, fe_device_buffer, fe_device_alloc, &
        fe_device_free
    use fe_gpu_linalg, only: fe_gpu_dot
    use fe_gpu_data, only: fe_gpu_dataset, fe_gpu_fe_dimension
    use fe_logging, only: log_info
    implicit none
    private

    public :: fe_gpu_within_transform

    interface
        function c_fe_gpu_fe_accumulate(y, W, Z, fe_ids, n_obs, n_groups, n_reg, n_inst, leading_dim, group_sum_y, group_sum_W, &
                group_sum_Z, group_counts) bind(C, name="fe_gpu_fe_accumulate") result(status)
            import :: c_ptr, c_size_t, c_int
            type(c_ptr), value :: y, W, Z, fe_ids, group_sum_y, group_sum_W, group_sum_Z, group_counts
            integer(c_size_t), value :: n_obs, leading_dim
            integer(c_int), value :: n_groups, n_reg, n_inst
            integer(c_int) :: status
        end function c_fe_gpu_fe_accumulate

        function c_fe_gpu_fe_compute_means(group_sum_y, group_sum_W, group_sum_Z, group_counts, n_groups, n_reg, n_inst) &
            bind(C, name="fe_gpu_fe_compute_means") result(status)
            import :: c_ptr, c_int
            type(c_ptr), value :: group_sum_y, group_sum_W, group_sum_Z, group_counts
            integer(c_int), value :: n_groups, n_reg, n_inst
            integer(c_int) :: status
        end function c_fe_gpu_fe_compute_means

        function c_fe_gpu_fe_subtract(y, W, Z, fe_ids, n_obs, n_reg, n_inst, leading_dim, group_mean_y, group_mean_W, &
                group_mean_Z, relaxation) &
            bind(C, name="fe_gpu_fe_subtract") result(status)
            import :: c_ptr, c_size_t, c_int, c_double
            type(c_ptr), value :: y, W, Z, fe_ids, group_mean_y, group_mean_W, group_mean_Z
            integer(c_size_t), value :: n_obs, leading_dim
            integer(c_int), value :: n_reg, n_inst
            real(c_double), value :: relaxation
            integer(c_int) :: status
        end function c_fe_gpu_fe_subtract

        function c_fe_gpu_mix_means(mean_ptr, prev_ptr, n, relaxation) bind(C, name="fe_gpu_mix_means") result(status)
            import :: c_ptr, c_size_t, c_int, c_double
            type(c_ptr), value :: mean_ptr
            type(c_ptr), value :: prev_ptr
            integer(c_size_t), value :: n
            real(c_double), value :: relaxation
            integer(c_int) :: status
        end function c_fe_gpu_mix_means

        function c_fe_gpu_absmax(data, n, out) bind(C, name="fe_gpu_absmax") result(status)
            import :: c_ptr, c_long_long, c_int, c_double
            type(c_ptr), value :: data
            integer(c_long_long), value :: n
            real(c_double), intent(out) :: out
            integer(c_int) :: status
        end function c_fe_gpu_absmax
    end interface

contains

    subroutine fe_gpu_within_transform(dataset, tolerance, max_iterations, use_cg, converged, iterations)
        type(fe_gpu_dataset), intent(inout) :: dataset
        real(real64), intent(in) :: tolerance
        integer, intent(in) :: max_iterations
        logical, intent(in) :: use_cg
        logical, intent(out) :: converged
        integer, intent(out) :: iterations
        integer :: iter
        integer :: d, s, sweep
        integer :: sweeps_per_iter
        real(real64) :: max_update
        real(real64) :: change, change_dim
        integer(c_int) :: status
        integer :: n_reg
        integer :: n_inst
        integer(int64) :: ldW
        integer(c_size_t) :: n_obs_c
        integer(c_size_t) :: ldw_c
        logical :: hit_tolerance
        real(real64), allocatable :: relaxation(:)
        real(real64), allocatable :: prev_change(:)
        type(fe_device_buffer), allocatable :: prev_mean_y(:)
        type(fe_device_buffer), allocatable :: prev_mean_W(:)
        type(fe_device_buffer), allocatable :: prev_mean_Z(:)
        integer(int64) :: bytes_buf
        integer, allocatable :: sweep_order(:)

        converged = .true.
        iterations = 0
        n_reg = dataset%n_regressors
        n_inst = dataset%n_instruments
        ldW = dataset%n_obs
        hit_tolerance = .false.
        allocate(relaxation(dataset%n_fe))
        allocate(prev_change(dataset%n_fe))
        allocate(prev_mean_y(dataset%n_fe))
        allocate(prev_mean_W(dataset%n_fe))
        allocate(prev_mean_Z(dataset%n_fe))
        sweeps_per_iter = 2
        relaxation = 1.5_real64
        prev_change = huge(0.0_real64)
        allocate(sweep_order(dataset%n_fe))
        sweep_order = [(d, d=1,dataset%n_fe)]
        do d = 1, dataset%n_fe - 1
            do s = d + 1, dataset%n_fe
                if (dataset%fe_dims(sweep_order(s))%n_groups < dataset%fe_dims(sweep_order(d))%n_groups) then
                    bytes_buf = sweep_order(d)
                    sweep_order(d) = sweep_order(s)
                    sweep_order(s) = int(bytes_buf, kind=kind(sweep_order(s)))
                end if
            end do
        end do
        do d = 1, dataset%n_fe
            prev_mean_y(d)%ptr = c_null_ptr
            prev_mean_y(d)%size_bytes = 0_int64
            prev_mean_W(d)%ptr = c_null_ptr
            prev_mean_W(d)%size_bytes = 0_int64
            prev_mean_Z(d)%ptr = c_null_ptr
            prev_mean_Z(d)%size_bytes = 0_int64
            bytes_buf = dataset%fe_dims(d)%group_mean_y%size_bytes
            if (bytes_buf > 0_int64) then
                call fe_device_alloc(prev_mean_y(d), bytes_buf)
                call fe_device_memset(prev_mean_y(d), 0)
            end if
            bytes_buf = dataset%fe_dims(d)%group_mean_W%size_bytes
            if (bytes_buf > 0_int64) then
                call fe_device_alloc(prev_mean_W(d), bytes_buf)
                call fe_device_memset(prev_mean_W(d), 0)
            end if
            bytes_buf = dataset%fe_dims(d)%group_mean_Z%size_bytes
            if (bytes_buf > 0_int64) then
                call fe_device_alloc(prev_mean_Z(d), bytes_buf)
                call fe_device_memset(prev_mean_Z(d), 0)
            end if
        end do

        if (dataset%n_fe == 0 .or. dataset%n_obs == 0_int64) return

        n_obs_c = int(dataset%n_obs, kind=c_size_t)
        ldw_c = int(ldW, kind=c_size_t)

        do iter = 1, max_iterations
            max_update = 0.0_real64

            do sweep = 1, sweeps_per_iter
                do s = 1, dataset%n_fe
                    d = sweep_order(s)
                call fe_device_memset(dataset%fe_dims(d)%group_mean_y, 0)
                call fe_device_memset(dataset%fe_dims(d)%group_mean_W, 0)
                if (n_inst > 0) then
                    call fe_device_memset(dataset%fe_dims(d)%group_mean_Z, 0)
                end if
                call fe_device_memset(dataset%fe_dims(d)%group_counts, 0)

                status = c_fe_gpu_fe_accumulate( &
                    dataset%d_y%ptr, dataset%d_W%ptr, dataset%d_Z%ptr, dataset%fe_dims(d)%fe_ids%ptr, n_obs_c, &
                    int(dataset%fe_dims(d)%n_groups, kind=c_int), int(n_reg, kind=c_int), int(n_inst, kind=c_int), ldw_c, &
                    dataset%fe_dims(d)%group_mean_y%ptr, dataset%fe_dims(d)%group_mean_W%ptr, &
                    dataset%fe_dims(d)%group_mean_Z%ptr, dataset%fe_dims(d)%group_counts%ptr)
                call fe_gpu_check(status, 'accumulating FE statistics')

                status = c_fe_gpu_fe_compute_means( &
                    dataset%fe_dims(d)%group_mean_y%ptr, dataset%fe_dims(d)%group_mean_W%ptr, &
                    dataset%fe_dims(d)%group_mean_Z%ptr, dataset%fe_dims(d)%group_counts%ptr, &
                    int(dataset%fe_dims(d)%n_groups, kind=c_int), int(n_reg, kind=c_int), int(n_inst, kind=c_int))
                call fe_gpu_check(status, 'computing FE group means')

                ! Apply over-relaxation to the group means before subtraction.
                if (c_associated(prev_mean_y(d)%ptr)) then
                    status = c_fe_gpu_mix_means(dataset%fe_dims(d)%group_mean_y%ptr, prev_mean_y(d)%ptr, &
                        int(dataset%fe_dims(d)%n_groups, kind=c_size_t), real(relaxation(d), c_double))
                    call fe_gpu_check(status, 'relaxing FE means (y)')
                end if
                if (n_reg > 0 .and. c_associated(prev_mean_W(d)%ptr)) then
                    status = c_fe_gpu_mix_means(dataset%fe_dims(d)%group_mean_W%ptr, prev_mean_W(d)%ptr, &
                        int(dataset%fe_dims(d)%n_groups, kind=c_size_t) * int(n_reg, kind=c_size_t), &
                        real(relaxation(d), c_double))
                    call fe_gpu_check(status, 'relaxing FE means (W)')
                end if
                if (n_inst > 0 .and. c_associated(prev_mean_Z(d)%ptr)) then
                    status = c_fe_gpu_mix_means(dataset%fe_dims(d)%group_mean_Z%ptr, prev_mean_Z(d)%ptr, &
                        int(dataset%fe_dims(d)%n_groups, kind=c_size_t) * int(n_inst, kind=c_size_t), &
                        real(relaxation(d), c_double))
                    call fe_gpu_check(status, 'relaxing FE means (Z)')
                end if

                status = c_fe_gpu_fe_subtract( &
                    dataset%d_y%ptr, dataset%d_W%ptr, dataset%d_Z%ptr, dataset%fe_dims(d)%fe_ids%ptr, n_obs_c, &
                    int(n_reg, kind=c_int), int(n_inst, kind=c_int), ldw_c, dataset%fe_dims(d)%group_mean_y%ptr, &
                    dataset%fe_dims(d)%group_mean_W%ptr, dataset%fe_dims(d)%group_mean_Z%ptr, &
                    real(relaxation(d), c_double))
                call fe_gpu_check(status, 'subtracting FE means')

                change = abs_copy_to_host(dataset%fe_dims(d)%group_mean_y, int(dataset%fe_dims(d)%n_groups, int64))
                change_dim = change
                max_update = max(max_update, change_dim)

                if (n_reg > 0) then
                    change = abs_copy_to_host(dataset%fe_dims(d)%group_mean_W, &
                        int(dataset%fe_dims(d)%n_groups, int64) * int(n_reg, int64))
                    change_dim = max(change_dim, change)
                    max_update = max(max_update, change)
                end if
                if (n_inst > 0) then
                    change = abs_copy_to_host(dataset%fe_dims(d)%group_mean_Z, &
                        int(dataset%fe_dims(d)%n_groups, int64) * int(n_inst, int64))
                    change_dim = max(change_dim, change)
                    max_update = max(max_update, change)
                end if

                if (change_dim > prev_change(d) * 1.05_real64) then
                    relaxation(d) = max(0.4_real64, relaxation(d) * 0.5_real64)
                else if (change_dim < prev_change(d) * 0.8_real64) then
                    relaxation(d) = min(2.0_real64, relaxation(d) * 1.10_real64)
                end if
                prev_change(d) = change_dim
            end do

                do s = dataset%n_fe, 1, -1
                    d = sweep_order(s)
                call fe_device_memset(dataset%fe_dims(d)%group_mean_y, 0)
                call fe_device_memset(dataset%fe_dims(d)%group_mean_W, 0)
                if (n_inst > 0) then
                    call fe_device_memset(dataset%fe_dims(d)%group_mean_Z, 0)
                end if
                call fe_device_memset(dataset%fe_dims(d)%group_counts, 0)

                status = c_fe_gpu_fe_accumulate( &
                    dataset%d_y%ptr, dataset%d_W%ptr, dataset%d_Z%ptr, dataset%fe_dims(d)%fe_ids%ptr, n_obs_c, &
                    int(dataset%fe_dims(d)%n_groups, kind=c_int), int(n_reg, kind=c_int), int(n_inst, kind=c_int), ldw_c, &
                    dataset%fe_dims(d)%group_mean_y%ptr, dataset%fe_dims(d)%group_mean_W%ptr, &
                    dataset%fe_dims(d)%group_mean_Z%ptr, dataset%fe_dims(d)%group_counts%ptr)
                call fe_gpu_check(status, 'accumulating FE statistics (reverse)')

                status = c_fe_gpu_fe_compute_means( &
                    dataset%fe_dims(d)%group_mean_y%ptr, dataset%fe_dims(d)%group_mean_W%ptr, &
                    dataset%fe_dims(d)%group_mean_Z%ptr, dataset%fe_dims(d)%group_counts%ptr, &
                    int(dataset%fe_dims(d)%n_groups, kind=c_int), int(n_reg, kind=c_int), int(n_inst, kind=c_int))
                call fe_gpu_check(status, 'computing FE group means (reverse)')

                if (c_associated(prev_mean_y(d)%ptr)) then
                    status = c_fe_gpu_mix_means(dataset%fe_dims(d)%group_mean_y%ptr, prev_mean_y(d)%ptr, &
                        int(dataset%fe_dims(d)%n_groups, kind=c_size_t), real(relaxation(d), c_double))
                    call fe_gpu_check(status, 'relaxing FE means (y, reverse)')
                end if
                if (n_reg > 0 .and. c_associated(prev_mean_W(d)%ptr)) then
                    status = c_fe_gpu_mix_means(dataset%fe_dims(d)%group_mean_W%ptr, prev_mean_W(d)%ptr, &
                        int(dataset%fe_dims(d)%n_groups, kind=c_size_t) * int(n_reg, kind=c_size_t), &
                        real(relaxation(d), c_double))
                    call fe_gpu_check(status, 'relaxing FE means (W, reverse)')
                end if
                if (n_inst > 0 .and. c_associated(prev_mean_Z(d)%ptr)) then
                    status = c_fe_gpu_mix_means(dataset%fe_dims(d)%group_mean_Z%ptr, prev_mean_Z(d)%ptr, &
                        int(dataset%fe_dims(d)%n_groups, kind=c_size_t) * int(n_inst, kind=c_size_t), &
                        real(relaxation(d), c_double))
                    call fe_gpu_check(status, 'relaxing FE means (Z, reverse)')
                end if

                status = c_fe_gpu_fe_subtract( &
                    dataset%d_y%ptr, dataset%d_W%ptr, dataset%d_Z%ptr, dataset%fe_dims(d)%fe_ids%ptr, n_obs_c, &
                    int(n_reg, kind=c_int), int(n_inst, kind=c_int), ldw_c, dataset%fe_dims(d)%group_mean_y%ptr, &
                    dataset%fe_dims(d)%group_mean_W%ptr, dataset%fe_dims(d)%group_mean_Z%ptr, &
                    real(relaxation(d), c_double))
                call fe_gpu_check(status, 'subtracting FE means (reverse)')

                change = abs_copy_to_host(dataset%fe_dims(d)%group_mean_y, int(dataset%fe_dims(d)%n_groups, int64))
                change_dim = change
                max_update = max(max_update, change_dim)

                if (n_reg > 0) then
                    change = abs_copy_to_host(dataset%fe_dims(d)%group_mean_W, &
                        int(dataset%fe_dims(d)%n_groups, int64) * int(n_reg, int64))
                    change_dim = max(change_dim, change)
                    max_update = max(max_update, change)
                end if
                if (n_inst > 0) then
                    change = abs_copy_to_host(dataset%fe_dims(d)%group_mean_Z, &
                        int(dataset%fe_dims(d)%n_groups, int64) * int(n_inst, int64))
                    change_dim = max(change_dim, change)
                    max_update = max(max_update, change)
                end if

                    if (change_dim > prev_change(d) * 1.05_real64) then
                        relaxation(d) = max(0.4_real64, relaxation(d) * 0.5_real64)
                    else if (change_dim < prev_change(d) * 0.8_real64) then
                        relaxation(d) = min(2.0_real64, relaxation(d) * 1.10_real64)
                    end if
                    prev_change(d) = change_dim
                end do
            end do  ! sweeps

            iterations = iter
            if (max_update < tolerance) then
                hit_tolerance = .true.
                exit
            end if
        end do

        converged = hit_tolerance

        do d = 1, dataset%n_fe
            if (c_associated(prev_mean_y(d)%ptr)) call fe_device_free(prev_mean_y(d))
            if (c_associated(prev_mean_W(d)%ptr)) call fe_device_free(prev_mean_W(d))
            if (c_associated(prev_mean_Z(d)%ptr)) call fe_device_free(prev_mean_Z(d))
        end do

        call log_info('GPU FE solver iterations=' // trim(int_to_string(iterations)) // ', max_update=' // &
            trim(real_to_string(max_update)))
    end subroutine fe_gpu_within_transform

        real(real64) function abs_copy_to_host(buffer, length) result(max_abs)
            type(fe_device_buffer), intent(in) :: buffer
            integer(int64), intent(in) :: length
            integer(c_int) :: status

            max_abs = 0.0_real64
            if (length <= 0) return

            status = c_fe_gpu_absmax(buffer%ptr, int(length, c_long_long), max_abs)
            call fe_gpu_check(status, 'computing absmax on device')
        end function abs_copy_to_host

    function int_to_string(value) result(buffer)
        integer, intent(in) :: value
        character(len=32) :: buffer
        write(buffer, '(I0)') value
    end function int_to_string

    function real_to_string(value) result(buffer)
        real(real64), intent(in) :: value
        character(len=32) :: buffer
        write(buffer, '(ES10.3)') value
    end function real_to_string

end module fe_gpu_demean
