module fe_types
    use iso_fortran_env, only: int32, int64, real32, real64
    implicit none
    private

    integer, parameter, public :: PRECISION_FLOAT64 = 0_int32
    integer, parameter, public :: PRECISION_FLOAT32 = 1_int32

    type, public :: fe_dataset_header
        integer(int64) :: n_obs         = 0_int64
        integer(int32) :: n_regressors  = 0_int32
        integer(int32) :: n_instruments = 0_int32
        integer(int32) :: n_fe          = 0_int32
        logical        :: has_cluster   = .false.
        logical        :: has_weights   = .false.
        integer(int32) :: precision_flag = PRECISION_FLOAT64
        integer(int64) :: metadata_bytes = 0_int64
        character(len=:), allocatable :: depvar_name
        character(len=:), allocatable :: regressor_names(:)
        character(len=:), allocatable :: instrument_names(:)
        character(len=:), allocatable :: fe_names(:)
        character(len=:), allocatable :: cluster_name
        character(len=:), allocatable :: weight_name
    contains
        procedure :: precision_bytes => fe_header_precision_bytes
        procedure :: summary => fe_header_summary
    end type fe_dataset_header

    type, public :: fe_host_arrays
        real(real64), allocatable :: y(:)
        real(real64), allocatable :: W(:, :)
        real(real64), allocatable :: Z(:, :)
        integer(int32), allocatable :: fe_ids(:, :)
        integer(int32), allocatable :: cluster(:)
        real(real64), allocatable :: weights(:)
    end type fe_host_arrays

    public :: precision_kind_from_flag

contains

    pure integer function fe_header_precision_bytes(this) result(bytes)
        class(fe_dataset_header), intent(in) :: this

        select case (this%precision_flag)
        case (PRECISION_FLOAT32)
            bytes = storage_size(0.0_real32) / 8
        case default
            bytes = storage_size(0.0_real64) / 8
        end select
    end function fe_header_precision_bytes

    pure integer function precision_kind_from_flag(flag) result(kind_value)
        integer(int32), intent(in) :: flag

        select case (flag)
        case (PRECISION_FLOAT32)
            kind_value = real32
        case default
            kind_value = real64
        end select
    end function precision_kind_from_flag

    function fe_header_summary(this) result(message)
        class(fe_dataset_header), intent(in) :: this
        character(len=:), allocatable :: message
        character(len=256) :: buffer

        write(buffer, '("N=",I0,", K=",I0,", L=",I0,", FE dims=",I0,", clusters=",L1,", weights=",L1,", precision_flag=",I0)') &
            this%n_obs, this%n_regressors, this%n_instruments, this%n_fe, this%has_cluster, this%has_weights, this%precision_flag
        message = trim(buffer)
    end function fe_header_summary

end module fe_types
