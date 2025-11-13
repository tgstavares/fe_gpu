program test_gpu_regression
    use iso_fortran_env, only: real64, int32
    use fe_config, only: fe_runtime_config, init_default_config
    use fe_types, only: fe_dataset_header, fe_host_arrays
    use fe_grouping, only: compute_fe_group_sizes
    use fe_gpu_runtime, only: fe_gpu_context, fe_gpu_initialize, fe_gpu_finalize, fe_gpu_backend_available
    use fe_pipeline, only: fe_gpu_estimate, fe_estimation_result
    use fe_logging, only: set_log_threshold, LOG_LEVEL_ERROR
    implicit none

    type(fe_runtime_config) :: cfg
    type(fe_dataset_header) :: header
    type(fe_host_arrays) :: host
    type(fe_gpu_context) :: ctx
    type(fe_estimation_result) :: est
    integer, allocatable :: group_sizes(:)
    real(real64), dimension(2) :: beta_true = (/ 1.5_real64, -0.75_real64 /)

    call set_log_threshold(LOG_LEVEL_ERROR)

    if (.not. fe_gpu_backend_available()) then
        print *, 'GPU backend unavailable; skipping regression test.'
        stop 0
    end if

    call build_dataset(host, header, beta_true)
    call compute_fe_group_sizes(host%fe_ids, group_sizes)

    call init_default_config(cfg)
    cfg%fe_tolerance = 1.0e-10_real64
    cfg%fe_max_iterations = 500

    call fe_gpu_initialize(ctx)
    call fe_gpu_estimate(cfg, header, host, group_sizes, est)
    call fe_gpu_finalize(ctx)

    call assert_true(est%converged, 'FE solver did not converge')
    call assert_true(est%solver_info == 0, 'Cholesky solver failed')
    call assert_true(size(est%beta) == size(beta_true), 'Unexpected beta length')
    call assert_real_vector(est%beta, beta_true, 1.0e-8_real64, 'Estimated beta mismatch')

    call release_dataset(host)
    deallocate(group_sizes)

    print *, 'GPU regression test passed.'

contains

    subroutine build_dataset(host, header, beta_true)
        type(fe_host_arrays), intent(out) :: host
        type(fe_dataset_header), intent(out) :: header
        real(real64), intent(in) :: beta_true(:)
        integer :: n_obs, n_reg
        real(real64), dimension(3) :: fe1 = (/ 0.5_real64, -0.25_real64, 0.8_real64 /)
        real(real64), dimension(2) :: fe2 = (/ -0.4_real64, 0.2_real64 /)
        real(real64), allocatable :: base(:)

        n_obs = 6
        n_reg = size(beta_true)

        header%n_obs = n_obs
        header%n_regressors = n_reg
        header%n_fe = 2
        header%has_cluster = .false.
        header%has_weights = .false.
        header%precision_flag = 0

        allocate(host%y(n_obs))
        allocate(host%W(n_obs, n_reg))
        allocate(host%fe_ids(2, n_obs))

        host%W(:, 1) = (/ 1.0_real64, 1.5_real64, -0.5_real64, -1.0_real64, 0.25_real64, 2.0_real64 /)
        host%W(:, 2) = (/ -2.0_real64, 0.5_real64, 1.0_real64, -1.5_real64, 0.75_real64, 1.25_real64 /)

        host%fe_ids(1, :) = (/ 1, 1, 2, 2, 3, 3 /)
        host%fe_ids(2, :) = (/ 1, 2, 1, 2, 1, 2 /)

        allocate(base(n_obs))
        base = matmul(host%W, beta_true)
        host%y = base + fe1(host%fe_ids(1, :)) + fe2(host%fe_ids(2, :))
        deallocate(base)
    end subroutine build_dataset

    subroutine release_dataset(host)
        type(fe_host_arrays), intent(inout) :: host
        if (allocated(host%y)) deallocate(host%y)
        if (allocated(host%W)) deallocate(host%W)
        if (allocated(host%fe_ids)) deallocate(host%fe_ids)
        if (allocated(host%cluster)) deallocate(host%cluster)
        if (allocated(host%weights)) deallocate(host%weights)
    end subroutine release_dataset

    subroutine assert_real_vector(actual, expected, tol, message)
        real(real64), intent(in) :: actual(:)
        real(real64), intent(in) :: expected(:)
        real(real64), intent(in) :: tol
        character(len=*), intent(in) :: message
        call assert_true(size(actual) == size(expected), trim(message) // ' (length mismatch)')
        if (maxval(abs(actual - expected)) > tol) then
            call fail(message)
        end if
    end subroutine assert_real_vector

    subroutine assert_true(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        if (.not. condition) call fail(message)
    end subroutine assert_true

    subroutine fail(message)
        character(len=*), intent(in) :: message
        write(*, '(A)') 'Assertion failed: ' // trim(message)
        stop 1
    end subroutine fail

end program test_gpu_regression
