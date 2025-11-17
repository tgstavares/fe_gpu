module fe_gpu_data
    use iso_c_binding, only: c_loc, c_ptr, c_null_ptr
    use iso_fortran_env, only: int32, int64, real64
    use fe_types, only: fe_host_arrays
    use fe_gpu_runtime, only: fe_device_buffer, fe_device_alloc, fe_device_free, fe_memcpy_htod, fe_memcpy_dtoh, &
        fe_device_memset, fe_gpu_check
    use fe_logging, only: log_error
    implicit none
    private

    integer(int64), parameter :: REAL64_BYTES = storage_size(0.0_real64) / 8
    integer(int64), parameter :: INT32_BYTES = storage_size(1_int32) / 8

    type, public :: fe_gpu_fe_dimension
        type(fe_device_buffer) :: fe_ids
        type(fe_device_buffer) :: group_mean_y
        type(fe_device_buffer) :: group_mean_W
        type(fe_device_buffer) :: group_mean_Z
        type(fe_device_buffer) :: group_counts
        integer :: n_groups = 0
    end type fe_gpu_fe_dimension

    type, public :: fe_gpu_dataset
        integer(int64) :: n_obs = 0_int64
        integer :: n_regressors = 0
        integer :: n_instruments = 0
        integer :: n_fe = 0
        type(fe_device_buffer) :: d_y
        type(fe_device_buffer) :: d_W
        type(fe_device_buffer) :: d_Z
        type(fe_device_buffer) :: d_proj_W
        type(fe_gpu_fe_dimension), allocatable :: fe_dims(:)
    end type fe_gpu_dataset

    public :: fe_gpu_dataset_upload
    public :: fe_gpu_dataset_destroy
    public :: fe_gpu_dataset_download

contains

    subroutine fe_gpu_dataset_upload(host, group_sizes, dataset)
        type(fe_host_arrays), intent(in) :: host
        integer, intent(in) :: group_sizes(:)
        type(fe_gpu_dataset), intent(inout) :: dataset
        integer(int64) :: bytes
        integer :: d
        integer :: n_reg
        integer(int64) :: n_obs64

        n_obs64 = int(size(host%y), int64)
        dataset%n_obs = n_obs64
        dataset%n_regressors = size(host%W, 2)
        dataset%n_instruments = size(host%Z, 2)
        dataset%n_fe = size(host%fe_ids, 1)

        if (dataset%n_fe /= size(group_sizes)) then
            call fail('Group size vector does not match number of FE dimensions')
        end if

        bytes = n_obs64 * REAL64_BYTES
        call fe_device_alloc(dataset%d_y, bytes)
        if (n_obs64 > 0_int64) then
            call copy_real_vector_to_device(host%y, dataset%d_y)
        end if

        n_reg = dataset%n_regressors
        bytes = n_obs64 * int(n_reg, int64) * REAL64_BYTES
        call fe_device_alloc(dataset%d_W, bytes)
        if (n_obs64 > 0_int64 .and. n_reg > 0) then
            call copy_real_matrix_to_device(host%W, dataset%d_W)
        end if

        if (dataset%n_instruments > 0) then
            bytes = n_obs64 * int(dataset%n_instruments, int64) * REAL64_BYTES
            call fe_device_alloc(dataset%d_Z, bytes)
            if (n_obs64 > 0_int64) then
                call copy_real_matrix_to_device(host%Z, dataset%d_Z)
            end if
        else
            dataset%d_Z%ptr = c_null_ptr
            dataset%d_Z%size_bytes = 0_int64
        end if

        dataset%d_proj_W%ptr = c_null_ptr
        dataset%d_proj_W%size_bytes = 0_int64

        if (allocated(dataset%fe_dims)) call fe_gpu_dataset_destroy(dataset)
        allocate(dataset%fe_dims(dataset%n_fe))

        do d = 1, dataset%n_fe
            bytes = n_obs64 * INT32_BYTES
            call fe_device_alloc(dataset%fe_dims(d)%fe_ids, bytes)
            if (n_obs64 > 0_int64) then
                call copy_int_vector_to_device(host%fe_ids(d, :), dataset%fe_dims(d)%fe_ids)
            end if

            dataset%fe_dims(d)%n_groups = group_sizes(d)

            bytes = int(group_sizes(d), int64) * REAL64_BYTES
            call fe_device_alloc(dataset%fe_dims(d)%group_mean_y, bytes)
            call fe_device_memset(dataset%fe_dims(d)%group_mean_y, 0)

            bytes = int(group_sizes(d), int64) * int(n_reg, int64) * REAL64_BYTES
            call fe_device_alloc(dataset%fe_dims(d)%group_mean_W, bytes)
            call fe_device_memset(dataset%fe_dims(d)%group_mean_W, 0)

            if (dataset%n_instruments > 0) then
                bytes = int(group_sizes(d), int64) * int(dataset%n_instruments, int64) * REAL64_BYTES
                call fe_device_alloc(dataset%fe_dims(d)%group_mean_Z, bytes)
                call fe_device_memset(dataset%fe_dims(d)%group_mean_Z, 0)
            else
                dataset%fe_dims(d)%group_mean_Z%ptr = c_null_ptr
                dataset%fe_dims(d)%group_mean_Z%size_bytes = 0_int64
            end if

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
        call fe_device_free(dataset%d_Z)
        call fe_device_free(dataset%d_proj_W)

        if (allocated(dataset%fe_dims)) then
            do d = 1, size(dataset%fe_dims)
                call fe_device_free(dataset%fe_dims(d)%fe_ids)
                call fe_device_free(dataset%fe_dims(d)%group_mean_y)
                call fe_device_free(dataset%fe_dims(d)%group_mean_W)
                call fe_device_free(dataset%fe_dims(d)%group_mean_Z)
                call fe_device_free(dataset%fe_dims(d)%group_counts)
            end do
            deallocate(dataset%fe_dims)
        end if

        dataset%n_obs = 0_int64
        dataset%n_regressors = 0
        dataset%n_instruments = 0
        dataset%n_fe = 0
    end subroutine fe_gpu_dataset_destroy

    subroutine fe_gpu_dataset_download(dataset, host)
        type(fe_gpu_dataset), intent(in) :: dataset
        type(fe_host_arrays), intent(inout) :: host
        if (dataset%n_obs /= int(size(host%y), int64)) then
            call fail('Host array dimensions do not match dataset during download')
        end if

        if (dataset%n_obs > 0_int64) then
            call copy_device_to_real_vector(dataset%d_y, host%y)
        end if

        if (dataset%n_regressors > 0 .and. dataset%n_obs > 0_int64) then
            call copy_device_to_real_matrix(dataset%d_W, host%W)
        end if
    end subroutine fe_gpu_dataset_download

    subroutine copy_real_vector_to_device(src, dest)
        real(real64), intent(in), target :: src(:)
        type(fe_device_buffer), intent(in) :: dest
        integer(int64) :: bytes
        type(c_ptr) :: ptr

        if (size(src) <= 0) return
        bytes = int(size(src), int64) * REAL64_BYTES
        ptr = vector_c_ptr(src)
        call fe_memcpy_htod(dest, ptr, bytes)
    end subroutine copy_real_vector_to_device

    subroutine copy_real_matrix_to_device(src, dest)
        real(real64), intent(in), target :: src(:, :)
        type(fe_device_buffer), intent(in) :: dest
        integer(int64) :: bytes
        type(c_ptr) :: ptr

        if (size(src, 1) == 0 .or. size(src, 2) == 0) return
        bytes = int(size(src, 1), int64) * int(size(src, 2), int64) * REAL64_BYTES
        ptr = matrix_c_ptr(src)
        call fe_memcpy_htod(dest, ptr, bytes)
    end subroutine copy_real_matrix_to_device

    subroutine copy_device_to_real_vector(src, dest)
        type(fe_device_buffer), intent(in) :: src
        real(real64), intent(inout), target :: dest(:)
        integer(int64) :: bytes
        type(c_ptr) :: ptr

        if (size(dest) <= 0) return
        bytes = int(size(dest), int64) * REAL64_BYTES
        ptr = vector_c_ptr(dest)
        call fe_memcpy_dtoh(ptr, src, bytes)
    end subroutine copy_device_to_real_vector

    subroutine copy_device_to_real_matrix(src, dest)
        type(fe_device_buffer), intent(in) :: src
        real(real64), intent(inout), target :: dest(:, :)
        integer(int64) :: bytes
        type(c_ptr) :: ptr

        if (size(dest, 1) == 0 .or. size(dest, 2) == 0) return
        bytes = int(size(dest, 1), int64) * int(size(dest, 2), int64) * REAL64_BYTES
        ptr = matrix_c_ptr(dest)
        call fe_memcpy_dtoh(ptr, src, bytes)
    end subroutine copy_device_to_real_matrix

    subroutine copy_int_vector_to_device(src, dest)
        integer(int32), intent(in) :: src(:)
        type(fe_device_buffer), intent(in) :: dest
        integer(int32), allocatable, target :: tmp(:)

        allocate(tmp(size(src)))
        tmp = src
        call fe_memcpy_htod(dest, c_loc(tmp(1)))
        deallocate(tmp)
    end subroutine copy_int_vector_to_device

    function vector_c_ptr(vec) result(ptr)
        real(real64), intent(in), target :: vec(:)
        type(c_ptr) :: ptr
        if (size(vec) == 0) then
            ptr = c_null_ptr
        else
            ptr = c_loc(vec(1))
        end if
    end function vector_c_ptr

    function matrix_c_ptr(mat) result(ptr)
        real(real64), intent(in), target :: mat(:, :)
        type(c_ptr) :: ptr
        if (size(mat, 1) == 0 .or. size(mat, 2) == 0) then
            ptr = c_null_ptr
        else
            ptr = c_loc(mat(1, 1))
        end if
    end function matrix_c_ptr

    subroutine fail(message)
        character(len=*), intent(in) :: message
        call log_error(trim(message))
        error stop 1
    end subroutine fail

end module fe_gpu_data
