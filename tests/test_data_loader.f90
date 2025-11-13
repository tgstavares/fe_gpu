program test_data_loader
    use iso_fortran_env, only: int32, int64, real32, real64
    use fe_types, only: fe_dataset_header, fe_host_arrays, PRECISION_FLOAT32, PRECISION_FLOAT64
    use fe_data_io, only: load_dataset_from_file, release_host_arrays
    use fe_logging, only: set_log_threshold, LOG_LEVEL_ERROR
    implicit none

    call set_log_threshold(LOG_LEVEL_ERROR)

    call run_case_fp64_no_flag()
    call run_case_fp32_with_flag()

    print *, 'All data loader tests passed.'

contains

    subroutine run_case_fp64_no_flag()
        character(len=*), parameter :: path = 'case_fp64_noflag.bin'
        type(fe_dataset_header) :: header
        type(fe_host_arrays) :: data
        real(real64), dimension(3) :: expected_y = (/ 1.0_real64, 2.0_real64, -1.0_real64 /)
        real(real64), dimension(3, 2) :: expected_W
        integer :: i

        expected_W(:, 1) = (/ 1.0_real64, 0.0_real64, 1.0_real64 /)
        expected_W(:, 2) = (/ 0.5_real64, -0.5_real64, 2.0_real64 /)

        call write_case_fp64_no_flag(path)
        call load_dataset_from_file(path, header, data)

        call assert_equal_int64(header%n_obs, 3_int64, 'case1: N mismatch')
        call assert_equal_int32(header%n_regressors, 2, 'case1: K mismatch')
        call assert_equal_int32(header%n_fe, 1, 'case1: FE count mismatch')
        call assert_true(header%has_cluster, 'case1: cluster flag mismatch')
        call assert_true(.not. header%has_weights, 'case1: weights flag mismatch')
        call assert_equal_int32(header%precision_flag, PRECISION_FLOAT64, 'case1: precision flag mismatch')

        call assert_real_vector(data%y, expected_y, 1.0e-12_real64, 'case1: y mismatch')
        call assert_real_matrix(data%W, expected_W, 1.0e-12_real64, 'case1: W mismatch')
        call assert_int_vector(data%fe_ids(1, :), (/ 1, 1, 2 /), 'case1: FE ids mismatch')
        call assert_int_vector(data%cluster, (/ 10, 10, 99 /), 'case1: cluster mismatch')

        call release_host_arrays(data)
    end subroutine run_case_fp64_no_flag

    subroutine run_case_fp32_with_flag()
        character(len=*), parameter :: path = 'case_fp32_flag.bin'
        type(fe_dataset_header) :: header
        type(fe_host_arrays) :: data
        real(real64), dimension(2) :: expected_y = (/ 0.25_real64, 0.75_real64 /)
        real(real64), dimension(2, 1) :: expected_W
        real(real64), dimension(2) :: expected_weights = (/ 1.5_real64, 2.5_real64 /)

        expected_W(:, 1) = (/ -1.5_real64, 2.5_real64 /)

        call write_case_fp32_with_flag(path)
        call load_dataset_from_file(path, header, data)

        call assert_equal_int64(header%n_obs, 2_int64, 'case2: N mismatch')
        call assert_equal_int32(header%n_regressors, 1, 'case2: K mismatch')
        call assert_equal_int32(header%n_fe, 2, 'case2: FE count mismatch')
        call assert_true(.not. header%has_cluster, 'case2: cluster flag mismatch')
        call assert_true(header%has_weights, 'case2: weights flag mismatch')
        call assert_equal_int32(header%precision_flag, PRECISION_FLOAT32, 'case2: precision flag mismatch')

        call assert_real_vector(data%y, expected_y, 1.0e-6_real64, 'case2: y mismatch')
        call assert_real_matrix(data%W, expected_W, 1.0e-6_real64, 'case2: W mismatch')
        call assert_int_vector(data%fe_ids(1, :), (/ 3, 4 /), 'case2: FE1 mismatch')
        call assert_int_vector(data%fe_ids(2, :), (/ 1, 1 /), 'case2: FE2 mismatch')
        call assert_real_vector(data%weights, expected_weights, 1.0e-6_real64, 'case2: weights mismatch')

        call release_host_arrays(data)
    end subroutine run_case_fp32_with_flag

    subroutine write_case_fp64_no_flag(path)
        character(len=*), intent(in) :: path
        integer :: unit
        integer :: ios
        real(real64), dimension(3) :: y = (/ 1.0_real64, 2.0_real64, -1.0_real64 /)
        real(real64), dimension(3, 2) :: W
        integer(int32), dimension(3) :: fe = (/ 1_int32, 1_int32, 2_int32 /)
        integer(int32), dimension(3) :: cluster = (/ 10_int32, 10_int32, 99_int32 /)

        W(:, 1) = (/ 1.0_real64, 0.0_real64, 1.0_real64 /)
        W(:, 2) = (/ 0.5_real64, -0.5_real64, 2.0_real64 /)

        open(newunit=unit, file=path, status='replace', access='stream', form='unformatted', action='write', iostat=ios)
        call assert_true(ios == 0, 'Failed to open file for case1')

        write(unit) 3_int64
        write(unit) 2_int32
        write(unit) 1_int32
        write(unit) 1_int32
        write(unit) 0_int32

        write(unit) y
        write(unit) W
        write(unit) fe
        write(unit) cluster

        close(unit)
    end subroutine write_case_fp64_no_flag

    subroutine write_case_fp32_with_flag(path)
        character(len=*), intent(in) :: path
        integer :: unit, ios
        real(real32), dimension(2) :: y = (/ real(0.25_real64, real32), real(0.75_real64, real32) /)
        real(real32), dimension(2, 1) :: W
        real(real32), dimension(2) :: weights = (/ real(1.5_real64, real32), real(2.5_real64, real32) /)
        integer(int32), dimension(2) :: fe1 = (/ 3_int32, 4_int32 /)
        integer(int32), dimension(2) :: fe2 = (/ 1_int32, 1_int32 /)

        W(:, 1) = (/ real(-1.5_real64, real32), real(2.5_real64, real32) /)

        open(newunit=unit, file=path, status='replace', access='stream', form='unformatted', action='write', iostat=ios)
        call assert_true(ios == 0, 'Failed to open file for case2')

        write(unit) 2_int64
        write(unit) 1_int32
        write(unit) 2_int32
        write(unit) 0_int32
        write(unit) 1_int32
        write(unit) PRECISION_FLOAT32

        write(unit) y
        write(unit) W
        write(unit) fe1
        write(unit) fe2
        write(unit) weights

        close(unit)
    end subroutine write_case_fp32_with_flag

    subroutine assert_real_vector(actual, expected, tol, message)
        real(real64), intent(in) :: actual(:)
        real(real64), intent(in) :: expected(:)
        real(real64), intent(in) :: tol
        character(len=*), intent(in) :: message
        call assert_true(size(actual) == size(expected), trim(message) // ' (size mismatch)')
        if (maxval(abs(actual - expected)) > tol) then
            call fail_test(message)
        end if
    end subroutine assert_real_vector

    subroutine assert_real_matrix(actual, expected, tol, message)
        real(real64), intent(in) :: actual(:, :)
        real(real64), intent(in) :: expected(:, :)
        real(real64), intent(in) :: tol
        character(len=*), intent(in) :: message
        call assert_true(all(shape(actual) == shape(expected)), trim(message) // ' (shape mismatch)')
        if (maxval(abs(actual - expected)) > tol) then
            call fail_test(message)
        end if
    end subroutine assert_real_matrix

    subroutine assert_int_vector(actual, expected, message)
        integer(int32), intent(in) :: actual(:)
        integer(int32), intent(in) :: expected(:)
        character(len=*), intent(in) :: message
        call assert_true(all(actual == expected), message)
    end subroutine assert_int_vector

    subroutine assert_equal_int64(actual, expected, message)
        integer(int64), intent(in) :: actual, expected
        character(len=*), intent(in) :: message
        if (actual /= expected) call fail_test(message)
    end subroutine assert_equal_int64

    subroutine assert_equal_int32(actual, expected, message)
        integer(int32), intent(in) :: actual, expected
        character(len=*), intent(in) :: message
        if (actual /= expected) call fail_test(message)
    end subroutine assert_equal_int32

    subroutine assert_true(condition, message)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        if (.not. condition) call fail_test(message)
    end subroutine assert_true

    subroutine fail_test(message)
        character(len=*), intent(in) :: message
        write(*, '(A)') 'Test failed: ' // trim(message)
        error stop 1
    end subroutine fail_test

end program test_data_loader
