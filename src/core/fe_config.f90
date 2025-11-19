module fe_config
    use iso_fortran_env, only: int32, real64
    implicit none
    private

    type, public :: fe_formula_term
        character(len=:), allocatable :: name
        logical :: is_categorical = .false.
    end type fe_formula_term

    type, public :: fe_formula_interaction
        type(fe_formula_term), allocatable :: factors(:)
    end type fe_formula_interaction

    type, public :: fe_runtime_config
        character(len=:), allocatable :: data_path
        real(real64) :: fe_tolerance = 1.0e-6_real64
        integer(int32) :: fe_max_iterations = 500
        logical :: use_gpu = .true.
        logical :: verbose = .false.
        integer(int32), allocatable :: cluster_fe_dims(:)
        integer(int32), allocatable :: iv_regressors(:)
        integer(int32), allocatable :: iv_instrument_cols(:)
        character(len=:), allocatable :: depvar_name
        character(len=:), allocatable :: formula_spec
        character(len=:), allocatable :: iv_regressor_names(:)
        character(len=:), allocatable :: iv_instrument_names(:)
        character(len=:), allocatable :: cluster_name_targets(:)
        character(len=:), allocatable :: fe_name_targets(:)
        character(len=:), allocatable :: fe_override_names(:)
        character(len=:), allocatable :: regressor_name_targets(:)
        integer(int32), allocatable :: regressor_selection(:)
        integer(int32), allocatable :: fe_selection(:)
        logical :: use_formula_design = .false.
        logical :: formula_has_categorical = .false.
        type(fe_formula_term), allocatable :: formula_terms(:)
        type(fe_formula_interaction), allocatable :: formula_interactions(:)
    end type fe_runtime_config

    public :: init_default_config
    public :: describe_config

contains

    subroutine init_default_config(cfg)
        type(fe_runtime_config), intent(out) :: cfg

        cfg%fe_tolerance = 1.0e-6_real64
        cfg%fe_max_iterations = 500
        cfg%use_gpu = .true.
        cfg%verbose = .false.
        cfg%data_path = 'data.bin'
        allocate(cfg%cluster_fe_dims(0))
        allocate(cfg%iv_regressors(0))
        allocate(cfg%iv_instrument_cols(0))
        allocate(character(len=1) :: cfg%fe_override_names(0))
        allocate(cfg%regressor_selection(0))
        allocate(cfg%fe_selection(0))
        allocate(cfg%formula_terms(0))
        allocate(cfg%formula_interactions(0))
        cfg%use_formula_design = .false.
        cfg%formula_has_categorical = .false.
    end subroutine init_default_config

    function describe_config(cfg) result(message)
        type(fe_runtime_config), intent(in) :: cfg
        character(len=:), allocatable :: message
        character(len=512) :: buffer
        character(len=:), allocatable :: cluster_str

        cluster_str = format_cluster_dims(cfg%cluster_fe_dims)
        call describe_to_buffer(buffer, cluster_str, cfg)
        message = trim(buffer)
    end function describe_config

    subroutine describe_to_buffer(buf, cluster_str, cfg)
        character(len=*), intent(inout) :: buf
        character(len=*), intent(in) :: cluster_str
        type(fe_runtime_config), intent(in) :: cfg
        character(len=*), parameter :: fmt = '("data_path=",A,", tol=",ES10.3,", max_iter=",I0,", use_gpu=",L1,' // &
            '", verbose=",L1,", cluster_fe=",A,", iv_cols=",A,", iv_z=",A,")")'

        write(buf, fmt) trim(cfg%data_path), cfg%fe_tolerance, cfg%fe_max_iterations, cfg%use_gpu, cfg%verbose, &
            trim(cluster_str), trim(format_cluster_dims(cfg%iv_regressors)), &
            trim(format_cluster_dims(cfg%iv_instrument_cols))
    end subroutine describe_to_buffer

    function format_cluster_dims(dims) result(out)
        integer(int32), intent(in) :: dims(:)
        character(len=:), allocatable :: out
        integer :: i
        if (size(dims) == 0) then
            out = '[]'
            return
        end if
        out = '['
        do i = 1, size(dims)
            out = out // trim(int_to_string(dims(i)))
            if (i < size(dims)) out = out // ','
        end do
        out = out // ']'
    end function format_cluster_dims

    function int_to_string(value) result(buf)
        integer(int32), intent(in) :: value
        character(len=16) :: buf
        write(buf, '(I0)') value
    end function int_to_string

end module fe_config
