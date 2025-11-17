module fe_data_io
    use iso_fortran_env, only: int32, int64, real32, real64
    use fe_types, only: fe_dataset_header, fe_host_arrays, PRECISION_FLOAT32, PRECISION_FLOAT64
    use fe_logging, only: log_info, log_error
    implicit none
    private

    integer(int64), parameter :: HEADER_BASE_BYTES_NEW = 8_int64 + 5_int64 * 4_int64
    integer(int64), parameter :: HEADER_BASE_BYTES_OLD = 8_int64 + 4_int64 * 4_int64
    integer(int64), parameter :: INT32_BYTES = 4_int64
    integer(int64), parameter :: REAL32_BYTES = 4_int64
    integer(int64), parameter :: REAL64_BYTES = 8_int64

    public :: load_dataset_from_file
    public :: release_host_arrays

contains

    subroutine load_dataset_from_file(path, header, host)
        character(len=*), intent(in) :: path
        type(fe_dataset_header), intent(out) :: header
        type(fe_host_arrays), intent(inout) :: host
        integer :: unit, ios
        logical :: exists
        integer(int64) :: file_size

        inquire(file=path, exist=exists, size=file_size)
        if (.not. exists) then
            call fail('Dataset file not found: ' // trim(path))
        end if

        if (file_size < HEADER_BASE_BYTES_OLD) then
            call fail('File too small to contain required header: ' // trim(path))
        end if

        open(newunit=unit, file=path, access='stream', form='unformatted', action='read', status='old', iostat=ios)
        if (ios /= 0) then
            call fail('Unable to open dataset file: ' // trim(path))
        end if

        call read_header(unit, file_size, header)
        call release_host_arrays(host)
        call allocate_host_arrays(header, host)

        call read_numeric_vector(unit, header%n_obs, header%precision_flag, host%y)
        call read_numeric_matrix(unit, header%n_obs, header%n_regressors, header%precision_flag, host%W)
        if (header%n_instruments > 0) then
            call read_numeric_matrix(unit, header%n_obs, header%n_instruments, header%precision_flag, host%Z)
        end if
        call read_fe_ids(unit, header, host)
        if (header%has_cluster) then
            call read_int_vector(unit, header%n_obs, host%cluster)
        end if
        if (header%has_weights) then
            call read_numeric_vector(unit, header%n_obs, header%precision_flag, host%weights)
        end if

        close(unit)

        call log_info('Loaded dataset header: ' // header%summary())
    end subroutine load_dataset_from_file

    subroutine release_host_arrays(host)
        type(fe_host_arrays), intent(inout) :: host
        if (allocated(host%y)) deallocate(host%y)
        if (allocated(host%W)) deallocate(host%W)
        if (allocated(host%Z)) deallocate(host%Z)
        if (allocated(host%fe_ids)) deallocate(host%fe_ids)
        if (allocated(host%cluster)) deallocate(host%cluster)
        if (allocated(host%weights)) deallocate(host%weights)
    end subroutine release_host_arrays

    subroutine read_header(unit, file_size, header)
        integer, intent(in) :: unit
        integer(int64), intent(in) :: file_size
        type(fe_dataset_header), intent(inout) :: header
        type(fe_dataset_header) :: header_new, header_old
        integer(int32) :: tmp_vals(4)
        integer :: ios
        integer(int64) :: total_new_64, total_new_32, total_old_64, total_old_32
        integer(int64) :: base_bytes, expected_total
        logical :: match_new, match_old
        integer(int32) :: tmp_i32

        ! First pass: read assuming new-format header to determine layout.
        read(unit, iostat=ios) header%n_obs
        call ensure_io_success(ios, 'reading N')
        call require_positive_int64(header%n_obs, 'N')

        read(unit, iostat=ios) header%n_regressors
        call ensure_io_success(ios, 'reading K')
        call require_nonnegative_int32(header%n_regressors, 'K')

        read(unit, iostat=ios) tmp_vals
        call ensure_io_success(ios, 'reading header metadata')

        header_new = header
        header_new%n_instruments = tmp_vals(1)
        header_new%n_fe = tmp_vals(2)
        header_new%has_cluster = (tmp_vals(3) /= 0)
        header_new%has_weights = (tmp_vals(4) /= 0)

        header_old = header
        header_old%n_instruments = 0
        header_old%n_fe = tmp_vals(1)
        header_old%has_cluster = (tmp_vals(2) /= 0)
        header_old%has_weights = (tmp_vals(3) /= 0)

        total_new_64 = HEADER_BASE_BYTES_NEW + compute_data_bytes(header_new, REAL64_BYTES)
        total_new_32 = HEADER_BASE_BYTES_NEW + compute_data_bytes(header_new, REAL32_BYTES)
        total_old_64 = HEADER_BASE_BYTES_OLD + compute_data_bytes(header_old, REAL64_BYTES)
        total_old_32 = HEADER_BASE_BYTES_OLD + compute_data_bytes(header_old, REAL32_BYTES)

        match_new = (file_size == total_new_64) .or. (file_size == total_new_32) .or. &
            (file_size == total_new_64 + INT32_BYTES) .or. (file_size == total_new_32 + INT32_BYTES)
        match_old = (file_size == total_old_64) .or. (file_size == total_old_32) .or. &
            (file_size == total_old_64 + INT32_BYTES) .or. (file_size == total_old_32 + INT32_BYTES)

        if (.not. match_new .and. .not. match_old) then
            match_new = .true.
        end if

        rewind(unit)

        if (match_new) then
            header = header_new
            base_bytes = HEADER_BASE_BYTES_NEW
            read(unit, iostat=ios) header%n_obs
            call ensure_io_success(ios, 'reading N')
            call require_positive_int64(header%n_obs, 'N')
            read(unit, iostat=ios) header%n_regressors
            call ensure_io_success(ios, 'reading K')
            call require_nonnegative_int32(header%n_regressors, 'K')
            read(unit, iostat=ios) header%n_instruments
            call ensure_io_success(ios, 'reading L (instruments)')
            call require_nonnegative_int32(header%n_instruments, 'L')
            read(unit, iostat=ios) header%n_fe
            call ensure_io_success(ios, 'reading n_fe')
            call require_nonnegative_int32(header%n_fe, 'n_fe')
            read(unit, iostat=ios) tmp_i32
            header%has_cluster = (tmp_i32 /= 0)
            read(unit, iostat=ios) tmp_i32
            header%has_weights = (tmp_i32 /= 0)
        else if (match_old) then
            header = header_old
            base_bytes = HEADER_BASE_BYTES_OLD
            read(unit, iostat=ios) header%n_obs
            call ensure_io_success(ios, 'reading N')
            call require_positive_int64(header%n_obs, 'N')
            read(unit, iostat=ios) header%n_regressors
            call ensure_io_success(ios, 'reading K')
            call require_nonnegative_int32(header%n_regressors, 'K')
            header%n_instruments = 0
            read(unit, iostat=ios) header%n_fe
            call ensure_io_success(ios, 'reading n_fe')
            call require_nonnegative_int32(header%n_fe, 'n_fe')
            read(unit, iostat=ios) tmp_i32
            header%has_cluster = (tmp_i32 /= 0)
            read(unit, iostat=ios) tmp_i32
            header%has_weights = (tmp_i32 /= 0)
        else
            call fail('Unable to determine dataset header format')
        end if

        total_new_64 = base_bytes + compute_data_bytes(header, REAL64_BYTES)
        total_new_32 = base_bytes + compute_data_bytes(header, REAL32_BYTES)

        if (file_size == total_new_64) then
            header%precision_flag = PRECISION_FLOAT64
            return
        else if (file_size == total_new_32) then
            header%precision_flag = PRECISION_FLOAT32
            return
        end if

        read(unit, iostat=ios) tmp_i32
        call ensure_io_success(ios, 'reading precision flag')
        header%precision_flag = tmp_i32

        select case (header%precision_flag)
        case (PRECISION_FLOAT64)
            expected_total = base_bytes + INT32_BYTES + compute_data_bytes(header, REAL64_BYTES)
        case (PRECISION_FLOAT32)
            expected_total = base_bytes + INT32_BYTES + compute_data_bytes(header, REAL32_BYTES)
        case default
            call fail('Unsupported precision flag value: ' // trim(int_to_string(tmp_i32)))
        end select

        if (file_size /= expected_total) then
            call fail('File size does not match header metadata for declared precision')
        end if
    end subroutine read_header

    subroutine allocate_host_arrays(header, host)
        type(fe_dataset_header), intent(in) :: header
        type(fe_host_arrays), intent(inout) :: host
        integer :: n_obs_i, n_reg_i, n_fe_i

        n_obs_i = safe_int(header%n_obs, 'N')
        n_reg_i = header%n_regressors
        n_fe_i = header%n_fe

        if (n_obs_i <= 0) call fail('Dataset must contain at least one observation')
        if (n_reg_i < 0) call fail('Number of regressors cannot be negative')

        allocate(host%y(n_obs_i))
        if (n_reg_i > 0) then
            allocate(host%W(n_obs_i, n_reg_i))
        else
            allocate(host%W(n_obs_i, 0))
        end if

        if (header%n_instruments > 0) then
            allocate(host%Z(n_obs_i, header%n_instruments))
        else
            allocate(host%Z(n_obs_i, 0))
        end if

        if (n_fe_i > 0) then
            allocate(host%fe_ids(n_fe_i, n_obs_i))
        else
            allocate(host%fe_ids(0, 0))
        end if

        if (header%has_cluster) then
            allocate(host%cluster(n_obs_i))
        end if
        if (header%has_weights) then
            allocate(host%weights(n_obs_i))
        end if
    end subroutine allocate_host_arrays

    subroutine read_numeric_vector(unit, n_obs, precision_flag, dest)
        integer, intent(in) :: unit
        integer(int64), intent(in) :: n_obs
        integer(int32), intent(in) :: precision_flag
        real(real64), intent(out) :: dest(:)
        integer :: n_i
        real(real32), allocatable :: tmp32(:)
        integer :: ios

        n_i = safe_int(n_obs, 'vector length')
        if (size(dest) /= n_i) then
            call fail('Destination vector size mismatch during numeric read')
        end if

        select case (precision_flag)
        case (PRECISION_FLOAT32)
            allocate(tmp32(n_i))
            read(unit, iostat=ios) tmp32
            call ensure_io_success(ios, 'reading single-precision vector')
            dest = real(tmp32, kind=real64)
            deallocate(tmp32)
        case default
            read(unit, iostat=ios) dest
            call ensure_io_success(ios, 'reading double-precision vector')
        end select
    end subroutine read_numeric_vector

    subroutine read_numeric_matrix(unit, n_obs, n_reg, precision_flag, dest)
        integer, intent(in) :: unit
        integer(int64), intent(in) :: n_obs
        integer(int32), intent(in) :: n_reg
        integer(int32), intent(in) :: precision_flag
        real(real64), intent(out) :: dest(:, :)
        integer :: n_obs_i, n_reg_i
        real(real32), allocatable :: tmp32(:, :)
        integer :: ios

        n_obs_i = safe_int(n_obs, 'matrix rows')
        n_reg_i = max(0, n_reg)

        if (n_reg_i == 0) return
        if (size(dest, 1) /= n_obs_i .or. size(dest, 2) /= n_reg_i) then
            call fail('Destination matrix shape mismatch during numeric read')
        end if

        select case (precision_flag)
        case (PRECISION_FLOAT32)
            allocate(tmp32(n_obs_i, n_reg_i))
            read(unit, iostat=ios) tmp32
            call ensure_io_success(ios, 'reading single-precision matrix')
            dest(:, :) = real(tmp32, kind=real64)
            deallocate(tmp32)
        case default
            read(unit, iostat=ios) dest(:, :)
            call ensure_io_success(ios, 'reading double-precision matrix')
        end select
    end subroutine read_numeric_matrix

    subroutine read_fe_ids(unit, header, host)
        integer, intent(in) :: unit
        type(fe_dataset_header), intent(in) :: header
        type(fe_host_arrays), intent(inout) :: host
        integer :: d, n_obs_i, n_fe_i, ios
        integer(int32), allocatable :: buffer(:)

        n_fe_i = header%n_fe
        if (n_fe_i <= 0) return

        n_obs_i = safe_int(header%n_obs, 'N for FE ids')
        allocate(buffer(n_obs_i))
        if (size(host%fe_ids, 1) /= n_fe_i .or. size(host%fe_ids, 2) /= n_obs_i) then
            call fail('Destination FE id array shape mismatch')
        end if

        do d = 1, n_fe_i
            read(unit, iostat=ios) buffer
            call ensure_io_success(ios, 'reading FE ids')
            host%fe_ids(d, :) = buffer
        end do

        deallocate(buffer)
    end subroutine read_fe_ids

    subroutine read_int_vector(unit, n_obs, dest)
        integer, intent(in) :: unit
        integer(int64), intent(in) :: n_obs
        integer(int32), intent(out) :: dest(:)
        integer :: ios
        integer :: n_i

        n_i = safe_int(n_obs, 'integer vector length')
        if (size(dest) /= n_i) then
            call fail('Destination integer vector size mismatch during read')
        end if

        read(unit, iostat=ios) dest
        call ensure_io_success(ios, 'reading integer vector')
    end subroutine read_int_vector

    integer(int64) function compute_data_bytes(header, value_bytes) result(total)
        type(fe_dataset_header), intent(in) :: header
        integer(int64), intent(in) :: value_bytes
        integer(int64) :: n_obs64, n_reg64, n_fe64, n_instr64

        n_obs64 = header%n_obs
        n_reg64 = int(header%n_regressors, int64)
        n_fe64 = int(header%n_fe, int64)
        n_instr64 = int(header%n_instruments, int64)

        total = 0_int64
        total = total + n_obs64 * (1_int64 + n_reg64) * value_bytes
        if (n_instr64 > 0_int64) total = total + n_obs64 * n_instr64 * value_bytes
        if (header%has_weights) total = total + n_obs64 * value_bytes
        total = total + n_fe64 * n_obs64 * INT32_BYTES
        if (header%has_cluster) total = total + n_obs64 * INT32_BYTES
    end function compute_data_bytes

    subroutine ensure_io_success(ios, context)
        integer, intent(in) :: ios
        character(len=*), intent(in) :: context
        if (ios /= 0) then
            call fail('IO error while ' // trim(context))
        end if
    end subroutine ensure_io_success

    subroutine require_positive_int64(value, name)
        integer(int64), intent(in) :: value
        character(len=*), intent(in) :: name
        if (value <= 0_int64) then
            call fail(trim(name) // ' must be positive')
        end if
    end subroutine require_positive_int64

    subroutine require_nonnegative_int32(value, name)
        integer(int32), intent(in) :: value
        character(len=*), intent(in) :: name
        if (value < 0) then
            call fail(trim(name) // ' must be non-negative')
        end if
    end subroutine require_nonnegative_int32

    integer function safe_int(value, name) result(out)
        integer(int64), intent(in) :: value
        character(len=*), intent(in) :: name
        integer :: max_default
        integer(int64) :: max_allowed

        if (value < 0_int64) then
            call fail(trim(name) // ' must be non-negative')
        end if

        max_default = huge(0)
        max_allowed = int(max_default, int64)
        if (value > max_allowed) then
            call fail(trim(name) // ' exceeds maximum supported size for default integer kind')
        end if
        out = int(value)
    end function safe_int

    function int_to_string(value) result(buffer)
        integer(int32), intent(in) :: value
        character(len=32) :: buffer
        write(buffer, '(I0)') value
    end function int_to_string

    subroutine fail(message)
        character(len=*), intent(in) :: message
        call log_error(trim(message))
        error stop 1
    end subroutine fail

end module fe_data_io
