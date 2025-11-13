module fe_gpu_demean
    use iso_c_binding, only: c_ptr, c_size_t, c_int, c_loc
    use iso_fortran_env, only: int64, real32, real64
    use fe_gpu_runtime, only: fe_gpu_check, fe_device_memset, fe_memcpy_dtoh, fe_device_buffer
    use fe_gpu_data, only: fe_gpu_dataset, fe_gpu_fe_dimension
    use fe_logging, only: log_info
    implicit none
    private

    public :: fe_gpu_within_transform

    interface
        function c_fe_gpu_fe_accumulate(y, W, fe_ids, n_obs, n_reg, leading_dim, group_sum_y, group_sum_W, group_counts) &
            bind(C, name="fe_gpu_fe_accumulate") result(status)
            import :: c_ptr, c_size_t, c_int
            type(c_ptr), value :: y, W, fe_ids, group_sum_y, group_sum_W, group_counts
            integer(c_size_t), value :: n_obs, leading_dim
            integer(c_int), value :: n_reg
            integer(c_int) :: status
        end function c_fe_gpu_fe_accumulate
        function c_fe_gpu_fe_accumulate_f32(y, W, fe_ids, n_obs, n_reg, leading_dim, group_sum_y, group_sum_W, group_counts) &
            bind(C, name="fe_gpu_fe_accumulate_f32") result(status)
            import :: c_ptr, c_size_t, c_int
            type(c_ptr), value :: y, W, fe_ids, group_sum_y, group_sum_W, group_counts
            integer(c_size_t), value :: n_obs, leading_dim
            integer(c_int), value :: n_reg
            integer(c_int) :: status
        end function c_fe_gpu_fe_accumulate_f32

        function c_fe_gpu_fe_compute_means(group_sum_y, group_sum_W, group_counts, n_groups, n_reg) &
            bind(C, name="fe_gpu_fe_compute_means") result(status)
            import :: c_ptr, c_int
            type(c_ptr), value :: group_sum_y, group_sum_W, group_counts
            integer(c_int), value :: n_groups, n_reg
            integer(c_int) :: status
        end function c_fe_gpu_fe_compute_means
        function c_fe_gpu_fe_compute_means_f32(group_sum_y, group_sum_W, group_counts, n_groups, n_reg) &
            bind(C, name="fe_gpu_fe_compute_means_f32") result(status)
            import :: c_ptr, c_int
            type(c_ptr), value :: group_sum_y, group_sum_W, group_counts
            integer(c_int), value :: n_groups, n_reg
            integer(c_int) :: status
        end function c_fe_gpu_fe_compute_means_f32

        function c_fe_gpu_fe_subtract(y, W, fe_ids, n_obs, n_reg, leading_dim, group_mean_y, group_mean_W) &
            bind(C, name="fe_gpu_fe_subtract") result(status)
            import :: c_ptr, c_size_t, c_int
            type(c_ptr), value :: y, W, fe_ids, group_mean_y, group_mean_W
            integer(c_size_t), value :: n_obs, leading_dim
            integer(c_int), value :: n_reg
            integer(c_int) :: status
        end function c_fe_gpu_fe_subtract
        function c_fe_gpu_fe_subtract_f32(y, W, fe_ids, n_obs, n_reg, leading_dim, group_mean_y, group_mean_W) &
            bind(C, name="fe_gpu_fe_subtract_f32") result(status)
            import :: c_ptr, c_size_t, c_int
            type(c_ptr), value :: y, W, fe_ids, group_mean_y, group_mean_W
            integer(c_size_t), value :: n_obs, leading_dim
            integer(c_int), value :: n_reg
            integer(c_int) :: status
        end function c_fe_gpu_fe_subtract_f32
    end interface

contains

    subroutine fe_gpu_within_transform(dataset, tolerance, max_iterations, converged, iterations)
        type(fe_gpu_dataset), intent(inout) :: dataset
        real(real64), intent(in) :: tolerance
        integer, intent(in) :: max_iterations
        logical, intent(out) :: converged
        integer, intent(out) :: iterations
        integer :: iter
        integer :: d
        real(real64) :: max_update
        real(real64) :: change
        integer(c_int) :: status
        integer :: n_reg
        integer(int64) :: ldW
        integer(c_size_t) :: n_obs_c
        integer(c_size_t) :: ldw_c
        logical :: hit_tolerance
        logical :: use_fp32

        converged = .true.
        iterations = 0
        n_reg = dataset%n_regressors
        ldW = dataset%n_obs
        hit_tolerance = .false.
        use_fp32 = dataset%use_fp32

        if (dataset%n_fe == 0 .or. dataset%n_obs == 0_int64) return

        n_obs_c = int(dataset%n_obs, kind=c_size_t)
        ldw_c = int(ldW, kind=c_size_t)

        do iter = 1, max_iterations
            max_update = 0.0_real64

            do d = 1, dataset%n_fe
                call fe_device_memset(dataset%fe_dims(d)%group_mean_y, 0)
                call fe_device_memset(dataset%fe_dims(d)%group_mean_W, 0)
                call fe_device_memset(dataset%fe_dims(d)%group_counts, 0)

                if (use_fp32) then
                    status = c_fe_gpu_fe_accumulate_f32( &
                        dataset%d_y%ptr, dataset%d_W%ptr, dataset%fe_dims(d)%fe_ids%ptr, n_obs_c, int(n_reg, kind=c_int), ldw_c, &
                        dataset%fe_dims(d)%group_mean_y%ptr, dataset%fe_dims(d)%group_mean_W%ptr, &
                        dataset%fe_dims(d)%group_counts%ptr)
                else
                    status = c_fe_gpu_fe_accumulate( &
                        dataset%d_y%ptr, dataset%d_W%ptr, dataset%fe_dims(d)%fe_ids%ptr, n_obs_c, int(n_reg, kind=c_int), ldw_c, &
                        dataset%fe_dims(d)%group_mean_y%ptr, dataset%fe_dims(d)%group_mean_W%ptr, &
                        dataset%fe_dims(d)%group_counts%ptr)
                end if
                call fe_gpu_check(status, 'accumulating FE statistics')

                if (use_fp32) then
                    status = c_fe_gpu_fe_compute_means_f32( &
                        dataset%fe_dims(d)%group_mean_y%ptr, dataset%fe_dims(d)%group_mean_W%ptr, &
                        dataset%fe_dims(d)%group_counts%ptr, int(dataset%fe_dims(d)%n_groups, kind=c_int), int(n_reg, kind=c_int))
                else
                    status = c_fe_gpu_fe_compute_means( &
                        dataset%fe_dims(d)%group_mean_y%ptr, dataset%fe_dims(d)%group_mean_W%ptr, &
                        dataset%fe_dims(d)%group_counts%ptr, int(dataset%fe_dims(d)%n_groups, kind=c_int), int(n_reg, kind=c_int))
                end if
                call fe_gpu_check(status, 'computing FE group means')

                if (use_fp32) then
                    status = c_fe_gpu_fe_subtract_f32( &
                        dataset%d_y%ptr, dataset%d_W%ptr, dataset%fe_dims(d)%fe_ids%ptr, n_obs_c, int(n_reg, kind=c_int), ldw_c, &
                        dataset%fe_dims(d)%group_mean_y%ptr, dataset%fe_dims(d)%group_mean_W%ptr)
                else
                    status = c_fe_gpu_fe_subtract( &
                        dataset%d_y%ptr, dataset%d_W%ptr, dataset%fe_dims(d)%fe_ids%ptr, n_obs_c, int(n_reg, kind=c_int), ldw_c, &
                        dataset%fe_dims(d)%group_mean_y%ptr, dataset%fe_dims(d)%group_mean_W%ptr)
                end if
                call fe_gpu_check(status, 'subtracting FE means')

                change = abs_copy_to_host(dataset%fe_dims(d)%group_mean_y, &
                    int(dataset%fe_dims(d)%n_groups, int64), use_fp32)
                max_update = max(max_update, change)

                if (n_reg > 0) then
                    change = abs_copy_to_host(dataset%fe_dims(d)%group_mean_W, &
                        int(dataset%fe_dims(d)%n_groups, int64) * int(n_reg, int64), use_fp32)
                    max_update = max(max_update, change)
                end if
            end do

            iterations = iter
            if (max_update < tolerance) then
                hit_tolerance = .true.
                exit
            end if
        end do

        converged = hit_tolerance

        call log_info('GPU FE solver iterations=' // trim(int_to_string(iterations)) // ', max_update=' // &
            trim(real_to_string(max_update)))
    end subroutine fe_gpu_within_transform

    real(real64) function abs_copy_to_host(buffer, length, use_fp32) result(max_abs)
        type(fe_device_buffer), intent(in) :: buffer
        integer(int64), intent(in) :: length
        logical, intent(in) :: use_fp32
        integer(int64) :: bytes
        real(real32), allocatable, target :: host_copy32(:)
        real(real64), allocatable, target :: host_copy(:)

        max_abs = 0.0_real64
        if (length <= 0) return

        if (use_fp32) then
            allocate(host_copy32(length))
            bytes = length * int(storage_size(0.0_real32) / 8, int64)
            call fe_memcpy_dtoh(c_loc(host_copy32(1)), buffer, bytes)
            max_abs = maxval(abs(real(host_copy32, kind=real64)))
            deallocate(host_copy32)
        else
            allocate(host_copy(length))
            bytes = length * int(storage_size(0.0_real64) / 8, int64)
            call fe_memcpy_dtoh(c_loc(host_copy(1)), buffer, bytes)
            max_abs = maxval(abs(host_copy))
            deallocate(host_copy)
        end if
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
