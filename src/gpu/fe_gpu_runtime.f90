module fe_gpu_runtime
    use iso_c_binding, only: c_ptr, c_null_ptr, c_int, c_size_t, c_char, c_null_char, c_associated, c_long_long, c_loc
    use iso_fortran_env, only: int64, int32
    use fe_logging, only: log_info, log_error
    implicit none
    private

    integer, parameter :: GPU_NAME_LEN = 256
    integer, parameter :: GPU_ERROR_LEN = 512

    type, bind(C) :: c_fe_gpu_device_info
        integer(c_int) :: device_id
        integer(c_int) :: multiprocessor_count
        integer(c_int) :: major
        integer(c_int) :: minor
        integer(c_int) :: reserved
        integer(c_size_t) :: total_global_mem
        character(kind=c_char) :: name(GPU_NAME_LEN)
    end type c_fe_gpu_device_info

    type, public :: fe_gpu_device_info
        integer :: device_id = -1
        integer :: multiprocessors = 0
        integer :: major = 0
        integer :: minor = 0
        integer(int64) :: total_memory = 0_int64
        character(len=GPU_NAME_LEN) :: name = ''
    end type fe_gpu_device_info

    type, public :: fe_gpu_context
        logical :: initialized = .false.
        integer :: device_id = -1
        type(fe_gpu_device_info) :: info
    end type fe_gpu_context

    type, public :: fe_device_buffer
        type(c_ptr) :: ptr = c_null_ptr
        integer(int64) :: size_bytes = 0_int64
    end type fe_device_buffer

    public :: fe_gpu_initialize
    public :: fe_gpu_finalize
    public :: fe_device_alloc
    public :: fe_device_free
    public :: fe_memcpy_htod
    public :: fe_memcpy_dtoh
    public :: fe_memcpy_dtod
    public :: fe_device_memset
    public :: fe_gpu_backend_available
    public :: fe_gpu_check
    public :: fe_gpu_copy_columns
    public :: fe_gpu_build_multi_cluster_ids

    interface
        function c_fe_gpu_runtime_is_available() bind(C, name="fe_gpu_runtime_is_available") result(flag)
            import :: c_int
            integer(c_int) :: flag
        end function c_fe_gpu_runtime_is_available

        function c_fe_gpu_runtime_init(device_id, info) bind(C, name="fe_gpu_runtime_init") result(status)
            import :: c_int, c_fe_gpu_device_info
            integer(c_int), value :: device_id
            type(c_fe_gpu_device_info), intent(out) :: info
            integer(c_int) :: status
        end function c_fe_gpu_runtime_init

        function c_fe_gpu_runtime_shutdown() bind(C, name="fe_gpu_runtime_shutdown") result(status)
            import :: c_int
            integer(c_int) :: status
        end function c_fe_gpu_runtime_shutdown

        function c_fe_gpu_runtime_malloc(dev_ptr, bytes) bind(C, name="fe_gpu_runtime_malloc") result(status)
            import :: c_ptr, c_size_t, c_int
            type(c_ptr), intent(out) :: dev_ptr
            integer(c_size_t), value :: bytes
            integer(c_int) :: status
        end function c_fe_gpu_runtime_malloc

        function c_fe_gpu_runtime_free(dev_ptr) bind(C, name="fe_gpu_runtime_free") result(status)
            import :: c_ptr, c_int
            type(c_ptr), value :: dev_ptr
            integer(c_int) :: status
        end function c_fe_gpu_runtime_free

        function c_fe_gpu_runtime_memcpy_htod(dst, src, bytes) bind(C, name="fe_gpu_runtime_memcpy_htod") result(status)
            import :: c_ptr, c_size_t, c_int
            type(c_ptr), value :: dst
            type(c_ptr), value :: src
            integer(c_size_t), value :: bytes
            integer(c_int) :: status
        end function c_fe_gpu_runtime_memcpy_htod

        function c_fe_gpu_runtime_memcpy_dtoh(dst, src, bytes) bind(C, name="fe_gpu_runtime_memcpy_dtoh") result(status)
            import :: c_ptr, c_size_t, c_int
            type(c_ptr), value :: dst
            type(c_ptr), value :: src
            integer(c_size_t), value :: bytes
            integer(c_int) :: status
        end function c_fe_gpu_runtime_memcpy_dtoh

        function c_fe_gpu_runtime_memcpy_dtod(dst, src, bytes) bind(C, name="fe_gpu_runtime_memcpy_dtod") result(status)
            import :: c_ptr, c_size_t, c_int
            type(c_ptr), value :: dst
            type(c_ptr), value :: src
            integer(c_size_t), value :: bytes
            integer(c_int) :: status
        end function c_fe_gpu_runtime_memcpy_dtod

        function c_fe_gpu_runtime_memset(ptr, value, bytes) bind(C, name="fe_gpu_runtime_memset") result(status)
            import :: c_ptr, c_size_t, c_int
            type(c_ptr), value :: ptr
            integer(c_int), value :: value
            integer(c_size_t), value :: bytes
            integer(c_int) :: status
        end function c_fe_gpu_runtime_memset

        function c_fe_gpu_runtime_get_last_error(buffer, length) bind(C, name="fe_gpu_runtime_get_last_error") result(status)
            import :: c_char, c_size_t, c_int
            character(kind=c_char), intent(out) :: buffer(*)
            integer(c_size_t), value :: length
            integer(c_int) :: status
        end function c_fe_gpu_runtime_get_last_error

        function c_fe_gpu_copy_columns(src, ld_src, indices, n_indices, n_rows, dst, ld_dst, dest_offset) &
                bind(C, name="fe_gpu_copy_columns") result(status)
            import :: c_ptr, c_int
            type(c_ptr), value :: src, dst
            integer(c_int), value :: ld_src, n_indices, n_rows, ld_dst, dest_offset
            integer(c_int), intent(in) :: indices(*)
            integer(c_int) :: status
        end function c_fe_gpu_copy_columns

        function c_fe_gpu_build_multi_cluster_ids(fe_ptrs, strides, n_dims, n_obs, out_ids, out_clusters) &
                bind(C, name="fe_gpu_build_multi_cluster_ids") result(status)
            import :: c_ptr, c_int, c_long_long
            type(c_ptr), value :: fe_ptrs
            type(c_ptr), value :: strides
            integer(c_int), value :: n_dims
            integer(c_long_long), value :: n_obs
            type(c_ptr), value :: out_ids
            type(c_ptr), value :: out_clusters
            integer(c_int) :: status
        end function c_fe_gpu_build_multi_cluster_ids
    end interface

contains

    logical function fe_gpu_backend_available()
        integer(c_int) :: flag
        flag = c_fe_gpu_runtime_is_available()
        fe_gpu_backend_available = (flag == 0_c_int)
    end function fe_gpu_backend_available

    subroutine fe_gpu_initialize(ctx, device_id)
        type(fe_gpu_context), intent(inout) :: ctx
        integer, intent(in), optional :: device_id
        integer :: requested
        type(c_fe_gpu_device_info) :: info_c
        integer(c_int) :: status

        if (ctx%initialized) return

        if (.not. fe_gpu_backend_available()) then
            call fe_gpu_fail('GPU backend unavailable in this environment.')
        end if

        requested = 0
        if (present(device_id)) requested = device_id

        status = c_fe_gpu_runtime_init(int(requested, c_int), info_c)
        call fe_gpu_check(status, 'initializing CUDA device')

        ctx%initialized = .true.
        ctx%device_id = info_c%device_id
        ctx%info = map_device_info(info_c)
        call log_info('Initialized GPU device: ' // trim(ctx%info%name))
    end subroutine fe_gpu_initialize

    subroutine fe_gpu_finalize(ctx)
        type(fe_gpu_context), intent(inout) :: ctx
        integer(c_int) :: status

        if (.not. ctx%initialized) return

        status = c_fe_gpu_runtime_shutdown()
        call fe_gpu_check(status, 'shutting down CUDA device')
        ctx%initialized = .false.
        ctx%device_id = -1
    end subroutine fe_gpu_finalize

    subroutine fe_device_alloc(buffer, nbytes)
        type(fe_device_buffer), intent(inout) :: buffer
        integer(int64), intent(in) :: nbytes
        integer(c_int) :: status

        if (nbytes < 0_int64) then
            call fe_gpu_fail('Attempted to allocate a negative byte count on device')
        end if

        call fe_device_free(buffer)
        buffer%size_bytes = nbytes

        if (nbytes == 0_int64) then
            buffer%ptr = c_null_ptr
            return
        end if

        status = c_fe_gpu_runtime_malloc(buffer%ptr, to_c_size(nbytes))
        call fe_gpu_check(status, 'allocating device memory')
    end subroutine fe_device_alloc

    subroutine fe_device_free(buffer)
        type(fe_device_buffer), intent(inout) :: buffer
        integer(c_int) :: status

        if (c_associated(buffer%ptr)) then
            status = c_fe_gpu_runtime_free(buffer%ptr)
            call fe_gpu_check(status, 'freeing device memory')
        end if

        buffer%ptr = c_null_ptr
        buffer%size_bytes = 0_int64
    end subroutine fe_device_free

    subroutine fe_gpu_copy_columns(src, n_rows, indices, dest, dest_offset)
        type(fe_device_buffer), intent(in) :: src
        integer(int64), intent(in) :: n_rows
        integer(int32), intent(in) :: indices(:)
        type(fe_device_buffer), intent(in) :: dest
        integer, intent(in) :: dest_offset
        integer(c_int) :: status

        if (.not. c_associated(src%ptr) .or. .not. c_associated(dest%ptr)) return
        if (size(indices) == 0) return
        status = c_fe_gpu_copy_columns(src%ptr, int(n_rows, c_int), indices, int(size(indices), c_int), &
            int(n_rows, c_int), dest%ptr, int(n_rows, c_int), int(dest_offset, c_int))
        call fe_gpu_check(status, 'copying GPU columns')
    end subroutine fe_gpu_copy_columns

    subroutine fe_gpu_build_multi_cluster_ids(ptrs, strides, n_dims, n_obs, ids_buffer, n_clusters, status)
        type(c_ptr), intent(in), target :: ptrs(:)
        integer(int64), intent(in), target :: strides(:)
        integer, intent(in) :: n_dims
        integer(int64), intent(in) :: n_obs
        type(fe_device_buffer), intent(in) :: ids_buffer
        integer, intent(out) :: n_clusters
        integer, intent(out) :: status
        integer(int32), target :: n_clusters32

        if (n_dims <= 0) then
            status = -1
            n_clusters = 0
            return
        end if

        n_clusters32 = 0_int32
        status = c_fe_gpu_build_multi_cluster_ids(c_loc(ptrs(1)), c_loc(strides(1)), int(n_dims, c_int), &
            int(n_obs, c_long_long), ids_buffer%ptr, c_loc(n_clusters32))
        if (status == 0) then
            n_clusters = n_clusters32
        else
            n_clusters = 0
        end if
    end subroutine fe_gpu_build_multi_cluster_ids

    subroutine fe_memcpy_htod(dest, src_ptr, nbytes)
        type(fe_device_buffer), intent(in) :: dest
        type(c_ptr), value :: src_ptr
        integer(int64), intent(in), optional :: nbytes
        integer(int64) :: bytes
        integer(c_int) :: status

        call ensure_buffer_valid(dest, 'host-to-device copy destination')
        bytes = buffer_copy_size(dest, nbytes, 'host-to-device copy')
        if (bytes == 0_int64) return

        status = c_fe_gpu_runtime_memcpy_htod(dest%ptr, src_ptr, to_c_size(bytes))
        call fe_gpu_check(status, 'copying host data to device')
    end subroutine fe_memcpy_htod

    subroutine fe_memcpy_dtoh(dst_ptr, src, nbytes)
        type(c_ptr), value :: dst_ptr
        type(fe_device_buffer), intent(in) :: src
        integer(int64), intent(in), optional :: nbytes
        integer(int64) :: bytes
        integer(c_int) :: status

        call ensure_buffer_valid(src, 'device-to-host copy source')
        bytes = buffer_copy_size(src, nbytes, 'device-to-host copy')
        if (bytes == 0_int64) return

        status = c_fe_gpu_runtime_memcpy_dtoh(dst_ptr, src%ptr, to_c_size(bytes))
        call fe_gpu_check(status, 'copying device data to host')
    end subroutine fe_memcpy_dtoh

    subroutine fe_memcpy_dtod(dest, src, nbytes)
        type(fe_device_buffer), intent(in) :: dest
        type(fe_device_buffer), intent(in) :: src
        integer(int64), intent(in), optional :: nbytes
        integer(int64) :: bytes
        integer(c_int) :: status

        call ensure_buffer_valid(dest, 'device-to-device copy destination')
        call ensure_buffer_valid(src, 'device-to-device copy source')
        bytes = buffer_copy_size(dest, nbytes, 'device-to-device copy')

        if (bytes > src%size_bytes) then
            call fe_gpu_fail('Requested device-to-device copy exceeds source buffer size')
        end if

        if (bytes == 0_int64) return

        status = c_fe_gpu_runtime_memcpy_dtod(dest%ptr, src%ptr, to_c_size(bytes))
        call fe_gpu_check(status, 'copying device data between buffers')
    end subroutine fe_memcpy_dtod

    subroutine fe_device_memset(buffer, value)
        type(fe_device_buffer), intent(in) :: buffer
        integer, intent(in) :: value
        integer(c_int) :: status
        integer :: byte_value

        if (.not. c_associated(buffer%ptr)) return
        if (buffer%size_bytes <= 0_int64) return

        byte_value = iand(value, 255)
        status = c_fe_gpu_runtime_memset(buffer%ptr, int(byte_value, c_int), to_c_size(buffer%size_bytes))
        call fe_gpu_check(status, 'setting device memory')
    end subroutine fe_device_memset

    subroutine ensure_buffer_valid(buffer, label)
        type(fe_device_buffer), intent(in) :: buffer
        character(len=*), intent(in) :: label

        if (.not. c_associated(buffer%ptr)) then
            call fe_gpu_fail(trim(label) // ': device pointer is not allocated')
        end if
    end subroutine ensure_buffer_valid

    integer(int64) function buffer_copy_size(buffer, requested, context) result(bytes)
        type(fe_device_buffer), intent(in) :: buffer
        integer(int64), intent(in), optional :: requested
        character(len=*), intent(in) :: context

        if (present(requested)) then
            bytes = requested
        else
            bytes = buffer%size_bytes
        end if

        if (bytes < 0_int64) then
            call fe_gpu_fail(trim(context) // ': negative byte count requested')
        end if
        if (bytes > buffer%size_bytes) then
            call fe_gpu_fail(trim(context) // ': requested bytes exceed buffer size')
        end if
    end function buffer_copy_size

    integer(c_size_t) function to_c_size(nbytes) result(value)
        integer(int64), intent(in) :: nbytes

        if (nbytes < 0_int64) then
            call fe_gpu_fail('Negative byte count encountered for CUDA runtime')
        end if
        value = int(nbytes, kind=c_size_t)
    end function to_c_size

    function map_device_info(info_c) result(info)
        type(c_fe_gpu_device_info), intent(in) :: info_c
        type(fe_gpu_device_info) :: info
        character(len=:), allocatable :: name_buf
        integer :: copy_len

        info%device_id = info_c%device_id
        info%multiprocessors = info_c%multiprocessor_count
        info%major = info_c%major
        info%minor = info_c%minor
        info%total_memory = int(info_c%total_global_mem, int64)

        name_buf = c_string_to_fortran(info_c%name)
        info%name = ''
        if (len(name_buf) > 0) then
            copy_len = min(len(name_buf), len(info%name))
            info%name(1:copy_len) = name_buf(1:copy_len)
        end if
    end function map_device_info

    function c_string_to_fortran(c_array) result(str)
        character(len=:), allocatable :: str
        character(kind=c_char), intent(in) :: c_array(:)
        integer :: i, n

        n = 0
        do i = 1, size(c_array)
            if (c_array(i) == c_null_char) exit
            n = n + 1
        end do

        if (n == 0) then
            str = ''
            return
        end if

        allocate(character(len=n) :: str)
        do i = 1, n
            str(i:i) = transfer(c_array(i), ' ')
        end do
    end function c_string_to_fortran

    function fe_gpu_last_error() result(message)
        character(len=:), allocatable :: message
        character(kind=c_char) :: buffer(GPU_ERROR_LEN)
        integer(c_int) :: status

        buffer = c_null_char
        status = c_fe_gpu_runtime_get_last_error(buffer, int(GPU_ERROR_LEN, c_size_t))
        if (status /= 0_c_int) then
            message = 'Unknown GPU backend error'
            return
        end if

        message = c_string_to_fortran(buffer)
        if (len(message) == 0) message = 'Unknown GPU backend error'
    end function fe_gpu_last_error

    subroutine fe_gpu_check(status, context)
        integer(c_int), intent(in) :: status
        character(len=*), intent(in) :: context
        character(len=:), allocatable :: err_str

        if (status == 0_c_int) return

        err_str = fe_gpu_last_error()
        call fe_gpu_fail('GPU failure while ' // trim(context) // ': ' // trim(err_str))
    end subroutine fe_gpu_check

    subroutine fe_gpu_fail(message)
        character(len=*), intent(in) :: message
        call log_error(trim(message))
        error stop 1
    end subroutine fe_gpu_fail

end module fe_gpu_runtime
