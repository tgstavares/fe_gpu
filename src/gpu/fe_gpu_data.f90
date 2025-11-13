module fe_gpu_data
    use iso_c_binding, only: c_loc
    use iso_fortran_env, only: int32, int64, real32, real64
    use fe_types, only: fe_host_arrays
    use fe_gpu_runtime, only: fe_device_buffer, fe_device_alloc, fe_device_free, fe_memcpy_htod, fe_memcpy_dtoh, &
        fe_device_memset, fe_gpu_check
    use fe_logging, only: log_error
    implicit none
    private

    integer(int64), parameter :: REAL64_BYTES = storage_size(0.0_real64) / 8
    integer(int64), parameter :: REAL32_BYTES = storage_size(0.0_real32) / 8
    integer(int64), parameter :: INT32_BYTES = storage_size(1_int32) / 8

    type, public :: fe_gpu_fe_dimension
        type(fe_device_buffer) :: fe_ids
        type(fe_device_buffer) :: group_mean_y
        type(fe_device_buffer) :: group_mean_W
        type(fe_device_buffer) :: group_counts
        integer :: n_groups = 0
    end type fe_gpu_fe_dimension

    type, public :: fe_gpu_dataset
        integer(int64) :: n_obs = 0_int64
        integer :: n_regressors = 0
        integer :: n_fe = 0
        logical :: use_fp32 = .false.
        integer(int64) :: scalar_bytes = REAL64_BYTES
        type(fe_device_buffer) :: d_y
        type(fe_device_buffer) :: d_W
        type(fe_gpu_fe_dimension), allocatable :: fe_dims(:)
    end type fe_gpu_dataset

    public :: fe_gpu_dataset_upload
    public :: fe_gpu_dataset_destroy
    public :: fe_gpu_dataset_download

contains

    subroutine fe_gpu_dataset_upload(host, group_sizes, dataset, use_fp32)
        type(fe_host_arrays), intent(in) :: host
        integer, intent(in) :: group_sizes(:)
        type(fe_gpu_dataset), intent(inout) :: dataset
        logical, intent(in) :: use_fp32
        integer(int64) :: bytes
        integer :: d
        integer :: n_reg
        integer(int64) :: n_obs64

        n_obs64 = int(size(host%y), int64)
        dataset%n_obs = n_obs64
        dataset%n_regressors = size(host%W, 2)
        dataset%n_fe = size(host%fe_ids, 1)
        dataset%use_fp32 = use_fp32
        dataset%scalar_bytes = merge(REAL32_BYTES, REAL64_BYTES, use_fp32)

        if (dataset%n_fe /= size(group_sizes)) then
            call fail('Group size vector does not match number of FE dimensions')
        end if

        bytes = n_obs64 * dataset%scalar_bytes
        call fe_device_alloc(dataset%d_y, bytes)
        if (n_obs64 > 0_int64) then
            call copy_real_vector_to_device(host%y, dataset%d_y, use_fp32)
        end if

        n_reg = dataset%n_regressors
        bytes = n_obs64 * int(n_reg, int64) * dataset%scalar_bytes
        call fe_device_alloc(dataset%d_W, bytes)
        if (n_obs64 > 0_int64 .and. n_reg > 0) then
            call copy_real_matrix_to_device(host%W, dataset%d_W, use_fp32)
        end if

        if (allocated(dataset%fe_dims)) call fe_gpu_dataset_destroy(dataset)
        allocate(dataset%fe_dims(dataset%n_fe))

        do d = 1, dataset%n_fe
            bytes = n_obs64 * INT32_BYTES
            call fe_device_alloc(dataset%fe_dims(d)%fe_ids, bytes)
            if (n_obs64 > 0_int64) then
                call copy_int_vector_to_device(host%fe_ids(d, :), dataset%fe_dims(d)%fe_ids)
            end if

            dataset%fe_dims(d)%n_groups = group_sizes(d)

            bytes = int(group_sizes(d), int64) * dataset%scalar_bytes
            call fe_device_alloc(dataset%fe_dims(d)%group_mean_y, bytes)
            call fe_device_memset(dataset%fe_dims(d)%group_mean_y, 0)

            bytes = int(group_sizes(d), int64) * int(n_reg, int64) * dataset%scalar_bytes
            call fe_device_alloc(dataset%fe_dims(d)%group_mean_W, bytes)
            call fe_device_memset(dataset%fe_dims(d)%group_mean_W, 0)

            bytes = int(group_sizes(d), int64) * INT32_BYTES
            call fe_device_alloc(dataset%fe_dims(d)%group_counts, bytes)
            call fe_device_memset(dataset%fe_dims(d)%group_counts, 0)
        end do
    end subroutine fe_gpu_dataset_upload

    subroutine fe_gpu_dataset_destroy(dataset)
        type(fe_gpu_dataset), intent(inout) :: dataset
        integer :: d

        call fe_device_free(dataset%d_y)
        call fe_device_free(dataset%d_W)

        if (allocated(dataset%fe_dims)) then
            do d = 1, size(dataset%fe_dims)
                call fe_device_free(dataset%fe_dims(d)%fe_ids)
                call fe_device_free(dataset%fe_dims(d)%group_mean_y)
                call fe_device_free(dataset%fe_dims(d)%group_mean_W)
                call fe_device_free(dataset%fe_dims(d)%group_counts)
            end do
            deallocate(dataset%fe_dims)
        end if

        dataset%n_obs = 0_int64
        dataset%n_regressors = 0
        dataset%n_fe = 0
        dataset%use_fp32 = .false.
        dataset%scalar_bytes = REAL64_BYTES
    end subroutine fe_gpu_dataset_destroy

    subroutine fe_gpu_dataset_download(dataset, host)
        type(fe_gpu_dataset), intent(in) :: dataset
        type(fe_host_arrays), intent(inout) :: host
        if (dataset%n_obs /= int(size(host%y), int64)) then
            call fail('Host array dimensions do not match dataset during download')
        end if

        if (dataset%n_obs > 0_int64) then
            call copy_device_to_real_vector(dataset%d_y, host%y, dataset%use_fp32)
        end if

        if (dataset%n_regressors > 0 .and. dataset%n_obs > 0_int64) then
            call copy_device_to_real_matrix(dataset%d_W, host%W, dataset%use_fp32)
        end if
    end subroutine fe_gpu_dataset_download

    subroutine copy_real_vector_to_device(src, dest, use_fp32)
        real(real64), intent(in) :: src(:)
        type(fe_device_buffer), intent(in) :: dest
        logical, intent(in) :: use_fp32
        integer(int64) :: bytes
        real(real32), allocatable, target :: tmp32(:)
        real(real64), allocatable, target :: tmp64(:)

        if (use_fp32) then
            allocate(tmp32(size(src)))
            tmp32 = real(src, kind=real32)
            bytes = int(size(tmp32), int64) * REAL32_BYTES
            call fe_memcpy_htod(dest, c_loc(tmp32(1)), bytes)
            deallocate(tmp32)
        else
            allocate(tmp64(size(src)))
            tmp64 = src
            bytes = int(size(tmp64), int64) * REAL64_BYTES
            call fe_memcpy_htod(dest, c_loc(tmp64(1)), bytes)
            deallocate(tmp64)
        end if
    end subroutine copy_real_vector_to_device

    subroutine copy_real_matrix_to_device(src, dest, use_fp32)
        real(real64), intent(in) :: src(:, :)
        type(fe_device_buffer), intent(in) :: dest
        logical, intent(in) :: use_fp32
        integer(int64) :: bytes
        real(real32), allocatable, target :: tmp32(:, :)
        real(real64), allocatable, target :: tmp64(:, :)

        if (use_fp32) then
            allocate(tmp32(size(src, 1), size(src, 2)))
            tmp32 = real(src, kind=real32)
            bytes = int(size(tmp32), int64) * REAL32_BYTES
            call fe_memcpy_htod(dest, c_loc(tmp32(1, 1)), bytes)
            deallocate(tmp32)
        else
            allocate(tmp64(size(src, 1), size(src, 2)))
            tmp64 = src
            bytes = int(size(tmp64), int64) * REAL64_BYTES
            call fe_memcpy_htod(dest, c_loc(tmp64(1, 1)), bytes)
            deallocate(tmp64)
        end if
    end subroutine copy_real_matrix_to_device

    subroutine copy_int_vector_to_device(src, dest)
        integer(int32), intent(in) :: src(:)
        type(fe_device_buffer), intent(in) :: dest
        integer(int32), allocatable, target :: tmp(:)

        allocate(tmp(size(src)))
        tmp = src
        call fe_memcpy_htod(dest, c_loc(tmp(1)))
        deallocate(tmp)
    end subroutine copy_int_vector_to_device

    subroutine copy_device_to_real_vector(src, dest, use_fp32)
        type(fe_device_buffer), intent(in) :: src
        real(real64), intent(inout) :: dest(:)
        logical, intent(in) :: use_fp32
        integer(int64) :: bytes
        real(real32), allocatable, target :: tmp32(:)
        real(real64), allocatable, target :: tmp64(:)

        if (use_fp32) then
            allocate(tmp32(size(dest)))
            bytes = int(size(tmp32), int64) * REAL32_BYTES
            call fe_memcpy_dtoh(c_loc(tmp32(1)), src, bytes)
            dest = real(tmp32, kind=real64)
            deallocate(tmp32)
        else
            allocate(tmp64(size(dest)))
            bytes = int(size(tmp64), int64) * REAL64_BYTES
            call fe_memcpy_dtoh(c_loc(tmp64(1)), src, bytes)
            dest = tmp64
            deallocate(tmp64)
        end if
    end subroutine copy_device_to_real_vector

    subroutine copy_device_to_real_matrix(src, dest, use_fp32)
        type(fe_device_buffer), intent(in) :: src
        real(real64), intent(inout) :: dest(:, :)
        logical, intent(in) :: use_fp32
        integer(int64) :: bytes
        real(real32), allocatable, target :: tmp32(:, :)
        real(real64), allocatable, target :: tmp64(:, :)

        if (use_fp32) then
            allocate(tmp32(size(dest, 1), size(dest, 2)))
            bytes = int(size(tmp32), int64) * REAL32_BYTES
            call fe_memcpy_dtoh(c_loc(tmp32(1, 1)), src, bytes)
            dest = real(tmp32, kind=real64)
            deallocate(tmp32)
        else
            allocate(tmp64(size(dest, 1), size(dest, 2)))
            bytes = int(size(tmp64), int64) * REAL64_BYTES
            call fe_memcpy_dtoh(c_loc(tmp64(1, 1)), src, bytes)
            dest = tmp64
            deallocate(tmp64)
        end if
    end subroutine copy_device_to_real_matrix

    subroutine fail(message)
        character(len=*), intent(in) :: message
        call log_error(trim(message))
        error stop 1
    end subroutine fail

end module fe_gpu_data
