program test_gpu_demean
    use iso_fortran_env, only: int32, real64
    use fe_types, only: fe_host_arrays
    use fe_grouping, only: compute_fe_group_sizes
    use fe_gpu_runtime, only: fe_gpu_context, fe_gpu_initialize, fe_gpu_finalize, fe_gpu_backend_available
    use fe_logging, only: set_log_threshold, LOG_LEVEL_ERROR
    use fe_gpu_data, only: fe_gpu_dataset, fe_gpu_dataset_upload, fe_gpu_dataset_destroy, fe_gpu_dataset_download
    use fe_gpu_demean, only: fe_gpu_within_transform
    implicit none

    type(fe_host_arrays) :: host
    type(fe_gpu_dataset) :: dataset
    type(fe_gpu_context) :: ctx
    integer, allocatable :: group_sizes(:)
    logical :: converged
    integer :: iterations

    call set_log_threshold(LOG_LEVEL_ERROR)

    if (.not. fe_gpu_backend_available()) then
        print *, 'GPU backend unavailable; skipping demeaning test.'
        stop 0
    end if

    call build_host_dataset(host)
    call compute_fe_group_sizes(host%fe_ids, group_sizes)

    call fe_gpu_initialize(ctx)
    call fe_gpu_dataset_upload(host, group_sizes, dataset, .false.)
    call fe_gpu_within_transform(dataset, 1.0e-8_real64, 200, converged, iterations)
    call fe_gpu_dataset_download(dataset, host)

    call assert_true(converged, 'FE solver did not converge')
    call verify_demeaned(host, group_sizes, 1.0e-6_real64)

    call fe_gpu_dataset_destroy(dataset)
    call fe_gpu_finalize(ctx)

    print *, 'GPU demeaning test passed.'

contains

    subroutine build_host_dataset(host)
        type(fe_host_arrays), intent(out) :: host
        integer :: n_obs
        integer :: n_reg

        n_obs = 6
        n_reg = 2

        allocate(host%y(n_obs))
        allocate(host%W(n_obs, n_reg))
        allocate(host%fe_ids(2, n_obs))

        host%y = (/ 1.0_real64, 2.0_real64, 3.0_real64, -2.0_real64, -1.0_real64, 0.5_real64 /)
        host%W(:, 1) = (/ 1.0_real64, 1.5_real64, -0.5_real64, -1.0_real64, 2.0_real64, -0.5_real64 /)
        host%W(:, 2) = (/ 0.0_real64, -2.0_real64, 1.0_real64, 0.5_real64, 1.5_real64, -1.0_real64 /)

        host%fe_ids(1, :) = (/ 1, 1, 2, 2, 3, 3 /)
        host%fe_ids(2, :) = (/ 1, 2, 1, 2, 1, 2 /)
    end subroutine build_host_dataset

    subroutine verify_demeaned(host, group_sizes, tol)
        type(fe_host_arrays), intent(in) :: host
        integer, intent(in) :: group_sizes(:)
        real(real64), intent(in) :: tol
        integer :: d
        integer :: g
        integer :: n_obs
        integer :: n_reg
        real(real64) :: mean_val

        n_obs = size(host%y)
        n_reg = size(host%W, 2)

        do d = 1, size(group_sizes)
            do g = 1, group_sizes(d)
                mean_val = group_mean(host%y, host%fe_ids(d, :), g)
                call assert_true(abs(mean_val) < tol, 'Non-zero mean detected in y for FE dim ' // trim(int_to_string(d)))

                call check_regressors(host%W, host%fe_ids(d, :), g, tol, d)
            end do
        end do
    end subroutine verify_demeaned

    subroutine check_regressors(W, fe_idx, group, tol, dim_id)
        real(real64), intent(in) :: W(:, :)
        integer(int32), intent(in) :: fe_idx(:)
        integer, intent(in) :: group
        real(real64), intent(in) :: tol
        integer, intent(in) :: dim_id
        integer :: n_reg
        integer :: k
        real(real64) :: m

        n_reg = size(W, 2)
        do k = 1, n_reg
            m = group_mean(W(:, k), fe_idx, group)
            call assert_true(abs(m) < tol, 'Non-zero mean detected in W col ' // trim(int_to_string(k)) // &
                ' for FE dim ' // trim(int_to_string(dim_id)))
        end do
    end subroutine check_regressors

    real(real64) function group_mean(values, fe_idx, group) result(mean_val)
        real(real64), intent(in) :: values(:)
        integer(int32), intent(in) :: fe_idx(:)
        integer, intent(in) :: group
        integer :: i
        real(real64) :: sum_val
        integer :: count

        sum_val = 0.0_real64
        count = 0
        do i = 1, size(values)
            if (fe_idx(i) == group) then
                sum_val = sum_val + values(i)
                count = count + 1
            end if
        end do
        if (count == 0) then
            mean_val = 0.0_real64
        else
            mean_val = sum_val / real(count, kind=real64)
        end if
    end function group_mean

    subroutine assert_true(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        if (.not. condition) then
            write(*, '(A)') 'Assertion failed: ' // trim(message)
            stop 1
        end if
    end subroutine assert_true

    function int_to_string(value) result(buffer)
        integer, intent(in) :: value
        character(len=16) :: buffer
        write(buffer, '(I0)') value
    end function int_to_string

end program test_gpu_demean
